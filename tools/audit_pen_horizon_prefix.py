"""Native H1/H5 prefix and independently executed scheduled-response audit."""
import argparse
import gc
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.saved_native_plant import saved_native_plant
from supermimic_research.runtime.saved_dynamic_arrays import restore_dynamic_arrays
from supermimic_research.runtime.gpu_data_snapshot import ResidentDataSnapshot
from supermimic_research.runtime.gpu_model_snapshot import capture_model
from supermimic_research.runtime.native_pen_planner import NativePenPlanner
from supermimic_research.runtime.pen_native_planning_math import candidate_sequence_targets
from supermimic_research.runtime.warp_response_recorder import record_response
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.pen_replay_fidelity import compare_response
from supermimic_research.learning.pen_stage import write


def load(pin):return dict(np.load(verify_file_pin(pin)))


def as_prediction(arrays,index,sl):
    return {key:arrays[name][index,sl] for key,name in [('post_qpos','Q'),('post_qvel','V'),('post_ctrl','U'),('post_GPU_time','T')]}


def main(a):
    import warp as wp
    import mujoco_warp as mw
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());pc=(cp.parent/cfg['prediction']).resolve();scorecfg=tomllib.loads(pc.read_text())
    bp=(pc.parent/scorecfg['base_config']).resolve();base=tomllib.loads(bp.read_text());scorecfg.update({k:base[k] for k in [
        'max_penetration_m','max_self_penetration_m','max_joint_limit_excursion_rad','max_object_nonhand_normal_force_N']})
    gates=tomllib.loads((pc.parent/scorecfg['gates']).read_text());rp=(cp.parent/cfg['parents']).resolve();parents=json.loads(rp.read_text())
    if (cfg['anchors']!=[0,10] or cfg['repeats']!=3 or cfg['fixed_candidates']!=[0,1,21,56] or not cfg['also_review_selected']
            or cfg['role']!='DIAGNOSTIC_NEVER_FIT' or scorecfg['prediction_controls']!=5 or a.output.exists()
            or not a.output.resolve().is_relative_to(ROOT/'runs')):raise ValueError('Fixed fresh H1/H5 prefix protocol')
    torch.set_num_threads(4);wp.init();wp.set_device('cuda:0');a.output.mkdir()
    receipt=freeze_sources(sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[Path(__file__).resolve(),cp,pc],
        a.output/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(a.output/'source_receipt.json',receipt)
    for key in ['installed_source_receipt','engine_source_receipt']:
        verify_source_snapshot(json.loads(verify_file_pin(parents[key]).read_text()),live=True)
    proposal=json.loads(verify_file_pin(parents['proposal']).read_text());selected=load(proposal['artifacts']['selected'])
    offsets=load(parents['candidates'])['raw_offsets'];rows=[];prefixes=[];sources=[];ticks=0;retained=[]
    write(a.output/'intent.json',dict(config=file_pin(cp),parents=file_pin(rp),prediction=file_pin(pc),
        role=cfg['role'],source_receipt=file_pin(a.output/'source_receipt.json'),scope=cfg['scope']))
    for c in parents['cases']:
        reference=load(c['initialization']['nominal_mean_wrench']);physical=load(c['GPU_physical_context']);dt=c['dt'];n=round(.02/dt)
        src=load(c['planning'][0]['source']);m,dm,data,reconstruction=saved_native_plant(c,src,observed_layout=c['observed_Data_layout'])
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids,dofs=m.jnt_qposadr[jids],m.jnt_dofadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
        planners={H:NativePenPlanner(m,dm,data,qids,aids,n*H,horizon_controls=H) for H in [1,5]}
        was_gc=gc.isenabled();gc.disable()
        try:
            with wp.ScopedCapture() as capture:mw.step(dm,data)
        finally:
            if was_gc:gc.enable()
        step_graph=capture.graph;retained.append((m,dm,data,planners,step_graph))
        for k in cfg['anchors']:
            folder=a.output/f'{c["name"]}_{dt}_k{k}';folder.mkdir();source=load(c['planning'][k]['source']);old=load(c['planning'][k]['prediction'])
            restore_dynamic_arrays(data,source);snapshot=ResidentDataSnapshot(data);wp.synchronize();_,ms,mh=capture_model(dm)
            raw=torch.tensor((2*selected['delta20']).astype(np.float32),device='cuda')+torch.tensor(offsets[k:k+5].transpose(1,0,2),device='cuda')
            future=torch.tensor(reference['reference_hand'][k+1:k+6],device='cuda')
            PD,ema,intended=candidate_sequence_targets(raw,torch.tensor(old['controller__processed_actions'],device='cuda'),future,
                torch.tensor(old['encoder_bias20'],device='cuda'),torch.tensor(old['PD_lower20'],device='cuda'),torch.tensor(old['PD_upper20'],device='cuda'))
            first_PD_exact=np.array_equal(PD[:,0].cpu().numpy(),old['PD_targets'])
            ref=dict(COM=torch.tensor(reference['COM_world_m'][k+1:k+6],device='cuda'),
                quaternion=torch.tensor(reference['quaternion_wxyz'][k+1:k+6],device='cuda'),
                hand=future,wrench=torch.tensor(reference['wrench_world'][k:k+5],dtype=torch.float32,device='cuda'),valid=reference['valid_interval'][k:k+5])
            from scipy.spatial.transform import Rotation
            ref['shaft']=torch.tensor(Rotation.from_quat(reference['quaternion_wxyz'][k+1:k+6][:,[1,2,3,0]]).as_matrix()[:,:,0],dtype=torch.float32,device='cuda')
            forecasts={};indices={}
            for H,planner in planners.items():
                planner.load_source();idx,arrays,scores,cuda_ms=planner.search(PD[:,0] if H==1 else PD,ema[:,0] if H==1 else ema,
                    ref['COM'][0],ref['shaft'][0],ref['hand'][0],ref['wrench'][0],bool(ref['valid'][0]),
                    scorecfg['support_weight'] if c['name']=='STATE_PLUS_SUPPORT' else 0.,scorecfg,
                    reference_quaternion=ref['quaternion'][0],reference_sequence=ref if H>1 else None)
                path=folder/f'H{H}.npz';np.savez_compressed(path,**arrays,PD_schedule=PD.cpu().numpy(),EMA_schedule=ema.cpu().numpy(),**{'score_'+key:value for key,value in scores.items()})
                forecasts[H]=arrays;indices[H]=idx;ticks+=57*n*H
            planner_isolation=snapshot.audit(data)
            for candidate in range(57):
                metrics,checks=compare_response(as_prediction(forecasts[5],candidate,slice(0,n)),as_prediction(forecasts[1],candidate,slice(0,n)),
                    forecasts[5]['W'][candidate,:n],forecasts[1]['W'][candidate,:n],qids,dofs,planners[1].qa,planners[1].da,gates)
                prefixes.append(dict(mode=c['name'],dt=dt,k=k,candidate=candidate,metrics=metrics,checks=checks))
            schedule=PD.cpu().numpy();candidates=sorted(set(cfg['fixed_candidates']+[indices[5]]))
            for candidate in candidates:
                for repeat in range(3):
                    snapshot.restore(data);wp.synchronize()
                    before=snapshot.audit(data)
                    packet=record_response(m,dm,data,step_graph,aids,schedule[candidate,0],n*5,control_schedule=schedule[candidate])
                    ticks+=n*5;path=folder/f'A{candidate:02d}_R{repeat}.npz';np.savez_compressed(path,**packet)
                    decoded,labels=decode_saved_case(c,packet,m,physical)
                    metrics,checks=compare_response(as_prediction(forecasts[5],candidate,slice(None)),packet,forecasts[5]['W'][candidate],
                        labels['world_COM_wrench'],qids,dofs,planners[1].qa,planners[1].da,gates)
                    contract=dict(restored_all_bytes=bool(before['all_array_bytes_equal'] and before['scalar_equal']),
                        source_qv_exact=np.array_equal(packet['pre_qpos'][0],source['qpos'][0]) and np.array_equal(packet['pre_qvel'][0],source['qvel'][0]),
                        scheduled_PD_exact=np.array_equal(packet['post_ctrl'][:,aids],np.repeat(schedule[candidate],n,axis=0)),
                        no_aid=not np.any(packet['post_qfrc_applied']) and not np.any(packet['post_xfrc_applied']),capacity=not np.any(packet['overflow']))
                    rows.append(dict(mode=c['name'],dt=dt,k=k,candidate=candidate,repeat=repeat,packet=file_pin(path),
                        EFC=decoded,metrics=metrics,fidelity=checks,contract={key:bool(v) for key,v in contract.items()}))
            snapshot.restore(data);wp.synchronize();isolation=snapshot.audit(data);_,ns,nh=capture_model(dm)
            contract=dict(first_PD_matches_original=first_PD_exact,
                planner_live_data_untouched=bool(planner_isolation['all_array_bytes_equal'] and planner_isolation['scalar_equal']),
                reconstructed_source_restored_after_independent_branches=bool(isolation['all_array_bytes_equal'] and isolation['scalar_equal']),model_unchanged=mh==nh and ms==ns)
            sources.append(dict(mode=c['name'],dt=dt,k=k,source=c['planning'][k]['source'],original_prediction=c['planning'][k]['prediction'],
                reconstruction=reconstruction,checks=contract,forecast_H1=file_pin(folder/'H1.npz'),forecast_H5=file_pin(folder/'H5.npz'),selected=indices))
            write(a.output/'progress.json',dict(sources=sources,prefixes=prefixes,responses=rows,native_integrations=ticks))
            print(json.dumps(dict(mode=c['name'],dt=dt,k=k,contracts=contract,responses=len(rows),native_integrations=ticks)),flush=True)
    verify_source_snapshot(receipt,live=True)
    passed=lambda blocks:sum(all(row.values()) for row in blocks)
    result=dict(schema='native_pen_horizon_prefix_audit:1',config=file_pin(cp),parents=file_pin(rp),prediction=file_pin(pc),
        source_receipt=file_pin(a.output/'source_receipt.json'),installed_source_receipt=parents['installed_source_receipt'],engine_source_receipt=parents['engine_source_receipt'],
        sources=sources,prefixes=prefixes,responses=rows,native_integrations=ticks,optimizer_updates=0,role=cfg['role'],
        counts=dict(sources=[passed([s['checks'] for s in sources]),len(sources)],prefixes=[passed([p['checks'] for p in prefixes]),len(prefixes)],
            EFC=[passed([p['EFC']['checks'] for p in rows]),len(rows)],fidelity=[passed([p['fidelity'] for p in rows]),len(rows)],contract=[passed([p['contract'] for p in rows]),len(rows)]),
        scope=cfg['scope']+' Failed full-schedule responses preclude dynamics/benefit qualification; prefix equivalence alone is not full100ms correctness.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),counts=result['counts'],native_integrations=ticks)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
