"""Matched saved-source20ms replay through original candidate and instrumented graphs."""
import gc
import json
import subprocess
import tomllib
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import freeze_sources,verify_source_snapshot
from .controlled_evaluation import write
from .gpu_model_snapshot import capture_model
from .saved_native_plant import saved_native_plant
from .collision_numerics import set_collision_numerics
from .saved_dynamic_arrays import restore_dynamic_arrays
from .gpu_data_fork import fork_dynamic_data
from .gpu_data_snapshot import ResidentDataSnapshot
from .native_pen_planner import NativePenPlanner
from .warp_response_recorder import record_response

ROOT=Path(__file__).resolve().parents[3]


def main(a):
    main_started=time.perf_counter()
    import mujoco
    import mujoco_warp as mw
    import torch
    import warp as wp
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());rp=(cp.parent/cfg['run']).resolve();vp=(cp.parent/cfg['review']).resolve()
    r=json.loads(rp.read_text());v=json.loads(vp.read_text())
    ccd=cfg.get('collision_numerics');profile=cfg.get('ccd_profile','ORIGINAL_RECONSTRUCTION')
    if ('ccd_profile' in cfg)!=('collision_numerics' in cfg):raise ValueError('CCD profile and explicit options must occur together')
    if ccd is not None and (profile not in ['BASELINE','TIGHT'] or ccd!={'ccd_iterations':50,'ccd_tolerance':1e-6 if profile=='BASELINE' else 1e-7}):
        raise ValueError('Fixed matched CCD profiles, no additional sweep')
    expected=[dict(mode='STATE_PLUS_VELOCITY_SUPPORT',dt=.005,anchors=[0,8,9,10]),dict(mode='STATE_ONLY',dt=.0025,anchors=[0,1,2])]
    if (cfg['cohorts']!=expected or cfg['repeats']!=3 or cfg['horizon_s']!=.02 or cfg['data_role']!='DIAGNOSTIC_NEVER_FIT'
            or cfg['paths']!=['ORIGINAL_57_CANDIDATE_GRAPH','INSTRUMENTED_SINGLE_STEP_GRAPH'] or verify_file_pin(v['run'])!=rp
            or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')):raise ValueError('Exact predeclared saved-source probe')
    a.output.mkdir();wp.init();wp.set_device('cuda:0');torch.set_num_threads(4)
    paths=sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[ROOT/'tools/replay_pen_visited_sources.py']
    receipt=freeze_sources(paths,a.output/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(a.output/'source_receipt.json',receipt)
    package=Path(mw.__file__).resolve().parent
    installed=freeze_sources(sorted(p for p in package.rglob('*.py') if not p.name.endswith('_test.py') and not p.name.startswith('test_')),
        a.output/'installed_source_snapshot',package,'installed-package-source-not-git')
    write(a.output/'installed_source_receipt.json',installed)
    sources=[q['snapshot'] for q in receipt['records']]
    intent=dict(schema='pen_visited_source_replay:2',config=file_pin(cp),original_run=file_pin(rp),original_review=file_pin(vp),
        source_receipt=file_pin(a.output/'source_receipt.json'),installed_source_receipt=file_pin(a.output/'installed_source_receipt.json'),ccd_profile=profile,collision_numerics=ccd,sources=sources,data_role=cfg['data_role'],neural_updates=0,
        scope='All7 predeclared anchors,3 repeats/path. Original57-candidate path plus independent instrumented20ms held-PD path. No live actual episode or training. Old missing Data scalars are explicitly reconstructed; lost original task-import list stays unavailable.')
    write(a.output/'intent.json',intent);anchors=[];candidate_replays=[];cases=[];keepers=[];setup_timings=[]
    base=tomllib.loads(verify_file_pin(r['base_config']).read_text());scorecfg=tomllib.loads(verify_file_pin(r['config']).read_text())
    scorecfg.update({k:base[k] for k in ['max_penetration_m','max_self_penetration_m','max_joint_limit_excursion_rad','max_object_nonhand_normal_force_N']})
    for cohort in cfg['cohorts']:
        c=next(c for c in r['cases'] if c['name']==cohort['mode'] and c['dt']==cohort['dt']);dt=c['dt'];n=round(.02/dt)
        folder=a.output/f"{c['name']}_{dt}";folder.mkdir()
        source0=dict(np.load(verify_file_pin(c['planning'][0]['source'])))
        model_setup_started=time.perf_counter()
        m,dm,data,reconstruction=saved_native_plant(c,source0)
        original_values,original_scalars,original_hashes=capture_model(dm)
        policy=set_collision_numerics(m,dm,ccd) if ccd is not None else None
        values,scalars,hashes=capture_model(dm)
        changed_arrays=sorted(k for k in hashes if hashes[k]!=original_hashes[k])
        changed_scalars=sorted(k for k in scalars if scalars[k]!=original_scalars[k])
        if set(changed_arrays)-{'opt.ccd_tolerance'} or changed_scalars:raise ValueError('Only declared CCD tolerance may differ')
        np.savez_compressed(folder/'profile_model_arrays.npz',**values);write(folder/'profile_model_scalars.json',scalars)
        reconstruction.update(original_model_exact_before_override=True,original_model_arrays=c['fixed_model_arrays'],original_model_scalars=c['fixed_model_scalars'],
            profile_model_arrays=file_pin(folder/'profile_model_arrays.npz'),profile_model_scalars=file_pin(folder/'profile_model_scalars.json'),
            changed_arrays=changed_arrays,changed_scalars=changed_scalars,collision_numerics=policy,
            model_exact_to_original=not changed_arrays and not changed_scalars,
            model_exact_scope='DECLARED_PROFILE_AFTER_EXPLICIT_CCD_OVERRIDE',
            Data_static_limit='Fresh compatible constructor capacities are not original observations. Exact169 array layout/bytes required.')
        write(folder/'reconstruction.json',reconstruction)
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids=m.jnt_qposadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
        planner_setup_started=time.perf_counter()
        planner=NativePenPlanner(m,dm,data,qids,aids,n)
        with wp.ScopedStream(planner.stream):
            diagnostic,diagnostic_layout=fork_dynamic_data(data)
            was_gc=gc.isenabled();gc.disable()
            try:
                with wp.ScopedCapture(stream=planner.stream) as capture:mw.step(dm,diagnostic)
            finally:
                if was_gc:gc.enable()
        graph=capture.graph;keepers.append((m,dm,data,planner,diagnostic,graph))
        wp.synchronize();setup_timings.append(dict(mode=c['name'],dt=dt,model_restore_and_snapshot_wall_s=planner_setup_started-model_setup_started,
            planner_graph_setup_including_compile_wall_s=time.perf_counter()-planner_setup_started))
        reference=np.load(verify_file_pin(c['initialization']['nominal_mean_wrench']))
        for k in cohort['anchors']:
            d=c['planning'][k];old=np.load(verify_file_pin(d['prediction']));source=dict(np.load(verify_file_pin(d['source'])))
            source_result=restore_dynamic_arrays(data,source);wp.synchronize();source_snapshot=ResidentDataSnapshot(data);wp.synchronize()
            anchor_folder=folder/f'A{k}';anchor_folder.mkdir()
            np.savez_compressed(anchor_folder/'reconstructed_source.npz',**source_snapshot.host_values())
            write(anchor_folder/'Data_scalars.json',source_snapshot.scalars)
            records=[]
            for repeat in range(cfg['repeats']):
                planner.load_source()
                index,arrays,scores,cuda_ms=planner.search(torch.tensor(old['PD_targets'],device='cuda'),torch.tensor(old['EMA_candidates'],device='cuda'),
                    torch.tensor(reference['COM_world_m'][k+1],device='cuda'),
                    torch.tensor(Rotation.from_quat(reference['quaternion_wxyz'][k+1][[1,2,3,0]]).as_matrix()[:,0],dtype=torch.float32,device='cuda'),
                    torch.tensor(reference['reference_hand'][k+1],device='cuda'),torch.tensor(old['support_target_world'],device='cuda'),
                    bool(d['nominal_wrench_valid']),float(old['support_weight']),scorecfg,
                    reference_velocity=torch.tensor(old['endpoint_velocity_world'],device='cuda'),velocity_valid=bool(old['endpoint_velocity_valid']),velocity_weight=float(old['velocity_weight']))
                path=anchor_folder/f'candidate_R{repeat}.npz';np.savez_compressed(path,**arrays,**{'score_'+key:val for key,val in scores.items()})
                after=source_snapshot.audit(data)
                entry=dict(mode=c['name'],dt=dt,k=k,repeat=repeat,original_selected=d['selected'],recomputed_selected=index,
                    original_prediction=d['prediction'],source=d['source'],trace=file_pin(path),candidate_cuda_ms=cuda_ms,
                    candidates=57,native_physics_ticks=57*n,search_timing=dict(planner.last_search_timing),source_untouched=after['all_array_bytes_equal'] and after['scalar_equal'])
                candidate_replays.append(entry);records.append(entry['source_untouched'])
            for repeat in range(cfg['repeats']):
                with wp.ScopedStream(planner.stream):
                    restore_dynamic_arrays(diagnostic,source);before=ResidentDataSnapshot(diagnostic);wp.synchronize()
                    response_started=time.perf_counter()
                    packet=record_response(m,dm,diagnostic,graph,aids,old['PD_targets'][d['selected']],n,instrument=True)
                    response_wall_s=time.perf_counter()-response_started
                path=anchor_folder/f'instrumented_R{repeat}.npz';np.savez_compressed(path,**packet)
                name=f"{c['name']}_A{k}_R{repeat}"
                cases.append(dict(task='hand2-pen-spin',controller='INSTRUMENTED_SINGLE_STEP_GRAPH',name=name,mode=c['name'],dt=dt,k=k,repeat=repeat,
                    role=cfg['data_role'],contact_label_schema='GPU_constraint_indexed_no_adhesion:1',trace=file_pin(path),
                    host_topology=c['host_topology'],GPU_physical_context=c['GPU_physical_context'],model_layout=c['model_layout'],
                    host_model_scope='Original topology/physical tables for EFC decoding; declared current numerical model saved in anchor reconstruction.',
                    original_prediction=d['prediction'],original_selected=d['selected'],original_source=d['source'],
                    restored_source_exact=before.audit(data)['all_array_bytes_equal'],native_physics_ticks=n,
                    instrumented_response_including_readback_wall_s=response_wall_s))
            _,after_scalars,after_hashes=capture_model(dm)
            checks=dict(model_exact_and_unchanged=after_scalars==scalars and after_hashes==hashes,
                all_candidate_sources_read_only=all(records),source_still_exact=source_snapshot.audit(data)['all_array_bytes_equal'],
                all_instrumented_sources_exact=all(e['restored_source_exact'] for e in cases if e['mode']==c['name'] and e['k']==k),
                complete_two_paths_three_repeats=len(records)==3 and sum(e['mode']==c['name'] and e['k']==k for e in cases)==3)
            anchors.append(dict(mode=c['name'],dt=dt,k=k,checks=checks,reconstruction=file_pin(folder/'reconstruction.json'),
                reconstructed_source=file_pin(anchor_folder/'reconstructed_source.npz'),Data_scalars=file_pin(anchor_folder/'Data_scalars.json'),
                Data_scalar_role='RECONSTRUCTED_NOT_ORIGINAL',diagnostic_layout=diagnostic_layout))
            write(a.output/'progress.json',dict(anchors=anchors,candidate_replays=candidate_replays,cases=cases))
            print(json.dumps(dict(mode=c['name'],dt=dt,k=k,checks=checks)),flush=True)
            if not all(checks.values()):raise ValueError('Saved-source reconstruction/isolation failed')
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(installed,live=True)
    result=dict(**intent,anchors=anchors,candidate_replays=candidate_replays,cases=cases,
        setup_timings=setup_timings,collector_main_wall_s=time.perf_counter()-main_started,
        timing_scope='Model restore/snapshots, graph setup(including compile), candidate CUDA, submit/wait and readback separated. Instrumented response includes per-step host reads. Final report write/process startup not included; not a training acceleration benchmark.',
        candidate_physics_ticks=sum(e['native_physics_ticks'] for e in candidate_replays),instrumented_physics_ticks=sum(e['native_physics_ticks'] for e in cases),
        passed=sum(sum(e['checks'].values()) for e in anchors),total=sum(len(e['checks']) for e in anchors))
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[result['passed'],result['total']],
        candidate_ticks=result['candidate_physics_ticks'],instrumented_ticks=result['instrumented_physics_ticks'])))
