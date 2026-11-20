"""Apply the frozen mug model and a fixed candidate recipe to annotated right-hand windows."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from evaluate_mug_matched_generation import plan,physics
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.ego.hand_frames import read_hand_window,reconstruct,transform_points
from supermimic_research.ego.trajectory_clock import resample_body_poses
from supermimic_research.mechanics.mug_sequence_ik import anchored_targets,solve_trajectory


def make_grid(seed,ik,baseline,preload,limits):
    rng=np.random.default_rng(seed)
    pattern=preload/np.max(abs(preload))
    padded=np.concatenate([ik,np.repeat(ik[-1:],2,axis=0)])
    grid=[]
    for i in range(len(ik)):
        center=padded[i:i+3]
        dirs=rng.normal(size=(3,20))
        dirs/=np.sqrt(np.mean(dirs**2,axis=1,keepdims=True))
        candidates=[np.tile(baseline,(3,1)),center]
        candidates.extend(center+a*pattern for a in (-.10,-.05,.05,.10))
        for v in dirs:candidates.extend([center+.04*v,center-.04*v])
        grid.append(np.clip(candidates,*limits.T))
    return np.stack(grid)


def trial(out,record,inputs,old,positive):
    path=verify_file_pin(record['archive'])
    meta=record['selected']
    frames=list(range(meta['start_frame'],meta['end_frame']+1))
    data=read_hand_window(path,frames,'right',9,meta['object_uid'])
    fk=reconstruct(data['profile'],data['angles22'],data['wrist'],'right')
    obj=data['objects']
    human=np.stack([transform_points(np.linalg.inv(t),tips[:5]) for t,tips in zip(obj,fk['landmarks21'])])
    # Constant left transforms (robot palm and human rest palm offsets) cancel
    # when forming the object trajectory relative to its first wrist-frame pose.
    wrist_obj=np.linalg.inv(data['wrist'])@obj
    anchored=old['anchor_world_from_BODY']@np.linalg.inv(wrist_obj[0])@wrist_obj
    sampled=resample_body_poses(data['timestamps_ns'],anchored,old['COM_BODY_m'])
    clock=sampled.pop('metadata')
    z={**old,**sampled}
    a=z['source_interval_alpha'][:,None,None];i=z['source_left_index']
    human_tips=(1-a)*human[i]+a*human[i+1]
    model_path=verify_file_pin(inputs['model'])
    m=mujoco.MjModel.from_binary_path(str(model_path))
    binding=physics.read_json(inputs['binding'])
    init=old['initial_qpos27'];d=mujoco.MjData(m);d.qpos[:]=init;mujoco.mj_forward(m,d)
    start=d.site_xpos[[m.site(f'right_finger{k}_tip').id for k in range(1,6)]].copy()
    target=anchored_targets(z['T_world_from_BODY'],human_tips,start)
    q,tips,ok=solve_trajectory(m,init,target)
    base=old['feedforward_PD_target20'][0];preload=base-init[:20]
    direct=np.clip(q[1:]+preload,*m.actuator_ctrlrange.T)
    folder=out/str(record['clip']);folder.mkdir()
    appendix=folder/'reference.npz'
    np.savez_compressed(appendix,frames=frames,source_time=data['timestamps_ns'],source_wrist=data['wrist'],
        source_obj=obj,source_tip_in_body=human,body=z['T_world_from_BODY'],t=z['elapsed_s'],q=q,target=target,actual_ik=tips)
    if positive:
        np.testing.assert_allclose(z['COM_world_m'],old['COM_world_m'],rtol=0,atol=1e-9)
        np.testing.assert_allclose(z['T_world_from_BODY'],old['T_world_from_BODY'],rtol=0,atol=1e-9)
        with np.load(verify_file_pin(physics.read_json(inputs['original_tip_report'])['trajectory'])) as archived:
            np.testing.assert_allclose(target,archived['targets'],rtol=0,atol=1e-8)
    stats=[]
    planning_ref={**z}
    for key in ('COM_world_m','quaternion_wxyz'):
        planning_ref[key]=np.concatenate([z[key],np.repeat(z[key][-1:],2,axis=0)])
    for method in ('direct','lookahead_8201','lookahead_8202'):
        t0=time.perf_counter()
        if method=='direct':
            cmd=direct
        else:
            grid=make_grid(int(method[-4:]),direct,base,preload,m.actuator_ctrlrange)
            traj=plan(model_path,binding,planning_ref,base,np.tile(base,(len(direct)+2,1)),grid,1.,1.)
            cmd=traj['ctrl']
        detail,trace=physics.replay(model_path,binding,z,base,cmd,1.,1.)
        if method!='direct':np.testing.assert_array_equal(traj['qpos'],trace['qpos'])
        f=folder/(method+'.npz');np.savez_compressed(f,**trace)
        stats.append(dict(arm=method,**detail,wall_s=time.perf_counter()-t0,controls=len(direct),trace=file_pin(f)))
    return dict(clip=record['clip'],window=meta,reference=file_pin(appendix),archive=record['archive'],clock=clock,
        source_sequence=data['records'][0]['sequence_id'],participant=data['records'][0]['participant_id'],
        source_interval_ns=[int(data['timestamps_ns'][0]),int(data['timestamps_ns'][-1])],IK_solved=sum(ok),
        target_tip_RMS_mm=1000*np.sqrt(np.mean(np.sum((tips-target)**2,axis=-1))),stats=stats)


def run(mode,out):
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]/'runs'
    latest=physics.read_json(file_pin(root/'mug_matched_generation_v1/intent.json'))
    inputs={**latest['inputs'],'original_tip_report':file_pin(root/'mug_tip_sequence_v1/report.json')}
    old=physics.load_arrays(inputs['reference'])
    if mode=='original':
        original=physics.read_json(file_pin(root/'right_mug_frames_v4/intent.json'))
        records=[dict(clip=1884,archive=original['clip'],selected=dict(start_frame=47,end_frame=107,object_uid='223371871635142'))]
    else:
        intake=physics.read_json(file_pin(root/'hot3d_adjacent_mugs_v1/report.json'))
        records=intake['rows']
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(mode=mode,inputs=inputs,records=records,code=file_pin(__file__),
        scope='All 61-frame windows selected by the independent existing intake; exact elapsed time truncated to whole 20ms controls. Same fixed-wrist rest grasp, nominal model, body-relative fingertip IK and absolute grid recipe. Evaluate every control of the reference. Only planning beyond the last frame repeats its target.'))
    rows=[]
    for record in records:
        if record['selected']:
            rows.append(trial(out,record,inputs,old,mode=='original'))
        else:rows.append(dict(clip=record['clip'],status='no_right_mug_window_passed_intake'))
        write_json(out/'progress.json',rows)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['original','adjacent']);p.add_argument('out',type=Path)
    args=p.parse_args();run(args.mode,args.out.resolve())
