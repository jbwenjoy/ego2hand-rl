"""Matched action tables from observed saved sources using the shared native plant."""
import gc
import json
import subprocess
import time
import tomllib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import freeze_sources,verify_source_snapshot
from .controlled_evaluation import write
from .saved_native_plant import saved_native_plant
from .saved_dynamic_arrays import restore_dynamic_arrays
from .gpu_data_snapshot import ResidentDataSnapshot
from .gpu_data_fork import fork_dynamic_data
from .gpu_model_snapshot import capture_model
from .native_pen_planner import NativePenPlanner
from .pen_native_planning_math import candidate_offsets,candidate_targets
from .warp_response_recorder import record_response
ROOT=Path(__file__).resolve().parents[3]


def main(a,cfg,cp,rp,vp,r,v):
    import torch
    import warp as wp
    import mujoco_warp as mw
    expected=[dict(mode='STATE_ONLY',dt=.005,anchors=[0]),dict(mode='STATE_PLUS_SUPPORT',dt=.005,anchors=[4])]
    if (cfg['cohorts']!=expected or cfg['amplitudes']!=[1,4] or cfg['repeats']!=3 or cfg['horizon_s']!=.02
            or cfg['paths']!=['ORIGINAL_57_CANDIDATE_GRAPH','INSTRUMENTED_SELECTED_SINGLE_STEP_GRAPH']
            or cfg['data_role']!='DIAGNOSTIC_NEVER_FIT' or verify_file_pin(v['run'])!=rp
            or r['orientation_objective']!='FULL_SO3' or r['candidate_offset_multiplier']!=4
            or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')):raise ValueError('Fixed observed-source amplitude protocol')
    a.output.mkdir();wp.init();wp.set_device('cuda:0');torch.set_num_threads(4);started=time.perf_counter()
    paths=sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[ROOT/'tools/replay_pen_visited_sources.py']
    receipt=freeze_sources(paths,a.output/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(a.output/'source_receipt.json',receipt)
    package=Path(mw.__file__).resolve().parent
    installed=freeze_sources(sorted(p for p in package.rglob('*.py') if not p.name.endswith('_test.py') and not p.name.startswith('test_')),
        a.output/'installed_source_snapshot',package,'installed-source-not-git');write(a.output/'installed_source_receipt.json',installed)
    intent=dict(schema='pen_matched_amplitude_sources:1',config=file_pin(cp),original_run=file_pin(rp),original_review=file_pin(vp),
        sources=[e['snapshot'] for e in receipt['records']],source_receipt=file_pin(a.output/'source_receipt.json'),installed_source_receipt=file_pin(a.output/'installed_source_receipt.json'),
        data_role=cfg['data_role'],neural_updates=0,scope='Two failure-selected observed sources, amplitude1/4, all57 candidates and freshly selected instrumented PD at3 repeats. No new task episode, expert, gradient, ego or speedup qualification.')
    write(a.output/'intent.json',intent)
    base=tomllib.loads(verify_file_pin(r['base_config']).read_text());scorecfg=tomllib.loads(verify_file_pin(r['config']).read_text())
    scorecfg.update({k:base[k] for k in ['max_penetration_m','max_self_penetration_m','max_joint_limit_excursion_rad','max_object_nonhand_normal_force_N']})
    anchors=[];replays=[];cases=[];keepers=[]
    for cohort in cfg['cohorts']:
        c=next(c for c in r['cases'] if c['name']==cohort['mode'] and c['dt']==cohort['dt']);k=cohort['anchors'][0];d=c['planning'][k]
        source=dict(np.load(verify_file_pin(d['source'])));old=dict(np.load(verify_file_pin(d['prediction'])));ref=np.load(verify_file_pin(c['initialization']['nominal_mean_wrench']))
        folder=a.output/f"{c['name']}_K{k}";folder.mkdir()
        m,dm,data,reconstruction=saved_native_plant(c,source,observed_layout=c['observed_Data_layout'])
        values,scalars,hashes=capture_model(dm);n=round(.02/c['dt'])
        np.savez_compressed(folder/'restored_model_arrays.npz',**values);write(folder/'restored_model_scalars.json',scalars)
        write(folder/'reconstruction.json',reconstruction)
        snapshot=ResidentDataSnapshot(data);wp.synchronize();np.savez_compressed(folder/'restored_source.npz',**snapshot.host_values());write(folder/'Data_scalars.json',snapshot.scalars)
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids=m.jnt_qposadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
        planner=NativePenPlanner(m,dm,data,qids,aids,n)
        with wp.ScopedStream(planner.stream):
            diagnostic,layout=fork_dynamic_data(data);was_gc=gc.isenabled();gc.disable()
            try:
                with wp.ScopedCapture(stream=planner.stream) as capture:mw.step(dm,diagnostic)
            finally:
                if was_gc:gc.enable()
        graph=capture.graph;keepers.append((m,dm,data,planner,diagnostic,graph))
        anchor_checks=dict(recorded_scalars_exact=snapshot.scalars==json.loads(verify_file_pin(c['observed_Data_layout']).read_text())['scalars'],
            restored_all_arrays=snapshot.audit(data)['all_array_bytes_equal'])
        for amplitude in cfg['amplitudes']:
            raw=old['raw_candidates'][0]+candidate_offsets(multiplier=amplitude)[k]
            if not np.isfinite(raw).all() or abs(raw).max()>1:raise ValueError('Original raw action bound')
            targets,ema,intended=candidate_targets(torch.tensor(raw,device='cuda'),torch.tensor(old['controller__processed_actions'],device='cuda'),
                torch.tensor(ref['reference_hand'][k+1],device='cuda'),torch.tensor(old['encoder_bias20'],device='cuda'),
                torch.tensor(old['PD_lower20'],device='cuda'),torch.tensor(old['PD_upper20'],device='cuda'))
            pd=targets.detach().cpu().numpy().copy();ema_cpu=ema.detach().cpu().numpy().copy();intended_cpu=intended.detach().cpu().numpy().copy()
            for repeat in range(3):
                restore_dynamic_arrays(data,source);wp.synchronize();planner.load_source()
                index,arrays,scores,cuda_ms=planner.search(targets,ema,torch.tensor(ref['COM_world_m'][k+1],device='cuda'),
                    torch.tensor(Rotation.from_quat(ref['quaternion_wxyz'][k+1][[1,2,3,0]]).as_matrix()[:,0],dtype=torch.float32,device='cuda'),
                    torch.tensor(ref['reference_hand'][k+1],device='cuda'),torch.tensor(old['support_target_world'],device='cuda'),
                    bool(d['nominal_wrench_valid']),float(old['support_weight']),scorecfg,reference_quaternion=torch.tensor(ref['quaternion_wxyz'][k+1],device='cuda'))
                path=folder/f'X{amplitude}_candidate_R{repeat}.npz';np.savez_compressed(path,**arrays,**{'score_'+key:val for key,val in scores.items()},
                    raw_candidates=raw,PD_targets=pd,EMA_candidates=ema_cpu,intended_PD=intended_cpu)
                audit=snapshot.audit(data)
                entry=dict(mode=c['name'],dt=c['dt'],k=k,amplitude=amplitude,repeat=repeat,selected=index,original_selected=d['selected'],
                    original_source=d['source'],original_prediction=d['prediction'],trace=file_pin(path),native_physics_ticks=57*n,candidate_cuda_ms=cuda_ms,
                    search_timing=dict(planner.last_search_timing),source_untouched=audit['all_array_bytes_equal'] and audit['scalar_equal'])
                replays.append(entry)
                with wp.ScopedStream(planner.stream):
                    restore_dynamic_arrays(diagnostic,source);before=ResidentDataSnapshot(diagnostic);wp.synchronize();exact=before.audit(data)
                    packet=record_response(m,dm,diagnostic,graph,aids,pd[index],n,instrument=True)
                path=folder/f'X{amplitude}_instrumented_R{repeat}.npz';np.savez_compressed(path,**packet)
                cases.append(dict(task='hand2-pen-spin',controller='INSTRUMENTED_SELECTED_SINGLE_STEP_GRAPH',name=f"{c['name']}_K{k}_X{amplitude}_R{repeat}",
                    mode=c['name'],dt=c['dt'],k=k,amplitude=amplitude,repeat=repeat,selected=index,trace=file_pin(path),
                    contact_label_schema='GPU_constraint_indexed_no_adhesion:1',host_topology=c['host_topology'],GPU_physical_context=c['GPU_physical_context'],model_layout=c['model_layout'],
                    fixed_model_arrays=c['fixed_model_arrays'],fixed_model_scalars=c['fixed_model_scalars'],original_source=d['source'],native_physics_ticks=n,
                    restored_source_exact=exact['all_array_bytes_equal'] and exact['scalar_equal']))
                write(a.output/'progress.json',dict(anchors=anchors,candidate_replays=replays,cases=cases))
        _,after_scalars,after_hashes=capture_model(dm)
        anchor_checks.update(model_unchanged=scalars==after_scalars and hashes==after_hashes,
            live_source_read_only=all(e['source_untouched'] for e in replays if e['mode']==c['name']),
            instrumented_source_exact=all(e['restored_source_exact'] for e in cases if e['mode']==c['name']))
        anchors.append(dict(mode=c['name'],dt=c['dt'],k=k,checks=anchor_checks,restored_source=file_pin(folder/'restored_source.npz'),
            restored_model_arrays=file_pin(folder/'restored_model_arrays.npz'),restored_model_scalars=file_pin(folder/'restored_model_scalars.json'),model_hashes_before=hashes,model_hashes_after=after_hashes,
            Data_scalars=file_pin(folder/'Data_scalars.json'),observed_Data_layout=c['observed_Data_layout'],reconstruction=file_pin(folder/'reconstruction.json'),diagnostic_layout=layout))
        print(json.dumps(dict(mode=c['name'],k=k,checks=anchor_checks)),flush=True)
        if not all(anchor_checks.values()):raise ValueError('Matched source isolation failed')
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(installed,live=True)
    result=dict(**intent,anchors=anchors,candidate_replays=replays,cases=cases,candidate_physics_ticks=sum(e['native_physics_ticks'] for e in replays),
        instrumented_physics_ticks=sum(e['native_physics_ticks'] for e in cases),collector_main_wall_s=time.perf_counter()-started)
    if result['candidate_physics_ticks']!=2736 or result['instrumented_physics_ticks']!=48:raise ValueError('Predeclared integration budget')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),candidate_ticks=2736,instrumented_ticks=48)))
