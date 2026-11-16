"""Recorded response-row physics audit; no native integration or fitting."""
import mujoco
import numpy as np
from .warp_packet_decode import decode_saved_case
from .controlled_pen_records import pack_case


def review_recorded_row(case,z,window,source,model,physical,cfg):
    metrics,decoded=decode_saved_case(case,z,model,physical)
    packet=pack_case(model,case,z,decoded,np.asarray(case['qids']),np.asarray(case['dofs']),np.asarray(case['aids']),z['initial_warmstart'],case['physics'])
    if packet['x'].shape!=(1,102) or packet['y'].shape!=(1,90):raise ValueError('One complete held-PD20ms response required')
    ns=source['native']['native']['data']['arrays'];initial=mujoco.MjData(model)
    mujoco.mj_setState(model,initial,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
    world=decoded['world_COM_wrench'].mean(0);R=source['source_qpos_R'];Rc=source['source_cached_R']
    truth=np.r_[R.T@world[:3],R.T@world[3:]];cached=np.r_[Rc.T@world[:3],Rc.T@world[3:]]
    checks=dict(source_qpos=np.array_equal(z['pre_qpos'][0],ns['qpos'][0]),source_qvel=np.array_equal(z['pre_qvel'][0],ns['qvel'][0]),
        actual_initial_warm=np.array_equal(z['initial_warmstart'],ns['qacc_warmstart'][0]),source_phase=int(z['pre_action_phase'][0])==source['phase'],
        held_PD=np.array_equal(z['post_ctrl'][:,case['aids']],np.tile(case['actual_PD20'],(8,1))),
        clock=bool(np.max(abs(z['post_GPU_time']-(float(ns['time'][0])+np.arange(1,9)*.0025)))<1e-6),
        fixed_mount=bool(np.array_equal(z['post_mocap_pos'],np.tile(initial.mocap_pos,(8,1,1))) and np.array_equal(z['post_mocap_quat'],np.tile(initial.mocap_quat,(8,1,1)))),
        true_source_BODY_label=bool(np.allclose(packet['y'][0,78:84],truth,atol=1e-10,rtol=1e-10)),
        resident_mean_wrench=bool(np.allclose(window['wrench_sum'][5]/8,cached,atol=1e-9,rtol=1e-9)),
        group_normal=bool(np.allclose(packet['y'][0,84:90],window['Fn6_sum']/8,atol=1e-9,rtol=1e-9)),
        finite=bool(all(np.isfinite(packet[k]).all() for k in ['x','y','heads'])),
        no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),role=case['role']==cfg['role'])
    checks.update({'EFC_'+k:bool(v) for k,v in metrics['checks'].items()})
    geometry=packet['geometry_flags'][0];legal=bool(np.all(geometry<=[cfg['object_penetration_m'],cfg['self_penetration_m'],cfg['joint_excursion_rad']]))
    valid=bool(case['qualified'] and packet['valid'][0] and all(checks.values()) and legal)
    return {k:packet[k][0] for k in ['x','y','heads']},dict(valid=valid,online_quality=case['qualified'],packet_quality=bool(packet['valid'][0]),geometry_legal=legal,geometry=geometry.tolist(),checks=checks,decoder_metrics=metrics)


def aligned_rows(rows,metadata):
    if len(rows)!=len(metadata):raise ValueError('Exact row/metadata alignment required')
    return {**{k:np.asarray([r[k] for r in rows],dtype=np.float64).reshape(-1,n) for k,n in [('x',102),('y',90),('heads',90)]},
        'valid':np.asarray([m['valid'] for m in metadata],dtype=np.bool_)}


def response_groups(metadata,sources,cfg):
    groups=[]
    for source in sources:
        ids=[i for i,r in enumerate(metadata) if r['source_name']==source['name']]
        # Include empty/incomplete declared groups, rather than silently reducing
        # the denominator when a source or a signed branch is unavailable.
        for direction in range(cfg['directions']):
            for epsilon in cfg['PD_eps_rad']:
                signs={str(sign):[i for i in ids if metadata[i]['profile']['direction']==direction and metadata[i]['profile']['epsilon']==epsilon and metadata[i]['profile']['sign']==sign] for sign in (1,-1)}
                complete=all(len(v)==cfg['repeats'] for v in signs.values());selected=signs['1']+signs['-1']
                groups.append(dict(source_name=source['name'],direction=direction,epsilon=epsilon,rows=signs,complete=complete,qualified=bool(complete and all(metadata[i]['valid'] for i in selected))))
    return groups
