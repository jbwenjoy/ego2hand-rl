"""Validate the new full-pose161 encoding against all recorded native observations."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mjlab.utils.lab_api.math import subtract_frame_transforms
from dex_supermimic.tasks.pen_spin.mdp.observations import make_axis_points,bounded_object_pose_to_points_wrist_local
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_pose_observation import pose9_world_in_wrist,pack_full_pose_policy,require_full_pose_contract,OBSERVATION_SCHEMA,ACTION_CONTRACT
from supermimic_research.learning.pen_stage import write


def load(pin):return dict(np.load(verify_file_pin(pin)))


def axis_feature(pose,mount,radius,bound):
    p,q=subtract_frame_transforms(mount[...,:3],mount[...,3:],pose[...,:3],pose[...,3:])
    return bounded_object_pose_to_points_wrist_local(torch.cat([p,q],-1),make_axis_points(radius_m=radius,dtype=pose.dtype),position_tanh_scale_m=bound).reshape(-1,6)


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());rp=(cp.parent/cfg['task_audit']).resolve();audit=json.loads(rp.read_text())
    if cfg['role']!='DIAGNOSTIC_NEVER_FIT' or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs') or audit['passed']!=audit['total']:
        raise ValueError('Fresh read-only feature replay of the reviewed cohort')
    a.output.mkdir();receipt=freeze_sources(sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[
        Path(__file__).resolve(),cp,ROOT/'src/dex_supermimic/tasks/pen_spin/mdp/observations.py',ROOT/'src/dex_supermimic/rl/dex_supermimic/rotations.py'],
        a.output/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(a.output/'source_receipt.json',receipt)
    radius=audit['policy_contract']['axis_radius_m'];bound=audit['policy_contract']['position_tanh_scale_m'];rows=[];vectors=[];sources=[];timepairs=[]
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32)
    for profile,run_pin in [('H1',audit['inputs']['baseline']),('H5',audit['inputs']['horizon'])]:
        run=json.loads(verify_file_pin(run_pin).read_text())
        for c in run['cases']:
            z=load(c['trace']);reference=load(c['initialization']['nominal_mean_wrench']);n=round(.02/c['dt'])
            model=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
            body=c['model_layout']['object_body'];palm=model.body('robot/right_palm_link').id;qa=c['model_layout']['object_qposadr']
            joints=[model.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)]
            for d in c['planning']:
                k=d['k'];s=load(d['source']);p=load(d['prediction']);old=tensor(z['pre_action_observations'][k:k+1])
                mount=tensor(np.r_[s['xpos'][0,palm],s['xquat'][0,palm]][None]);qpose=tensor(s['qpos'][:,qa:qa+7])
                cached=tensor(np.r_[s['xpos'][0,body],s['xquat'][0,body]][None])
                refpose=tensor(np.r_[reference['COM_world_m'][k+1],reference['quaternion_wxyz'][k+1]][None])
                qaxis=axis_feature(qpose,mount,radius,bound);cached_axis=axis_feature(cached,mount,radius,bound);refaxis=axis_feature(refpose,mount,radius,bound)
                current9=pose9_world_in_wrist(qpose,mount,position_bound=bound);ref9=pose9_world_in_wrist(refpose,mount,position_bound=bound)
                full=pack_full_pose_policy(old[:,:20],ref9,old[:,26:89],old[:,89:109],old[:,109:129],current9,old[:,135:155])
                old_object=old[:,129:135];error_q=float(torch.max(torch.abs(qaxis-old_object)));error_cached=float(torch.max(torch.abs(cached_axis-old_object)))
                error_ref=float(torch.max(torch.abs(refaxis-old[:,20:26])))
                velocity=tensor(s['qvel'][:,model.jnt_dofadr[joints]])*.25
                flags=dict(old155=old.shape==(1,155),new161=full.shape==(1,161),finite=bool(torch.isfinite(full).all()),
                    current_qpos_axis_agrees=error_q<=cfg['axis_feature_tolerance_m'],cached_axis_agrees=error_cached<=cfg['axis_feature_tolerance_m'],
                    reference_axis_agrees=error_ref<=cfg['axis_feature_tolerance_m'],source_qpos_exact=np.array_equal(s['qpos'][0],z['pre_qpos'][k*n]),
                    source_clock=float(s['time'][0])==(0. if k==0 else float(z['post_GPU_time'][k*n-1])),
                    phase=d['phase']==k+1 and int(z['pre_action_phase'][k])==k+1,
                    scaled_joint_velocity=bool(torch.max(torch.abs(velocity-old[:,109:129]))<=cfg['joint_velocity_tolerance_rad_s']),
                    last_action=np.array_equal(old[:,135:155].numpy(),p['controller__raw_actions']),
                    original_nonpose_channels_exact=torch.equal(torch.cat([old[:,:20],old[:,26:129],old[:,135:]],-1),
                        torch.cat([full[:,:20],full[:,29:132],full[:,141:]],-1)))
                rows.append(dict(profile=profile,mode=c['name'],dt=c['dt'],k=k,source=d['source'],prediction=d['prediction'],
                    checks={key:bool(v) for key,v in flags.items()},current_qpos_axis_max_error_m=error_q,cached_axis_max_error_m=error_cached,
                    reference_axis_max_error_m=error_ref,pose_qpos_vs_cached_max_m=float(torch.max(torch.abs(qpose[:,:3]-cached[:,:3])))))
                vectors.append(full.numpy()[0]);sources.append(qpose.numpy()[0]);timepairs.append([float(s['time'][0]),(k+1)*.02])
    metadata=dict(observation_schema=OBSERVATION_SCHEMA,policy_input_dim=161,action_dim=20,action_contract=ACTION_CONTRACT,
        frame='fixed wrist local, COM3 bounded then first two rotation MATRIX COLUMNS',state='FEATURE_REPLAY_ONLY_NOT_ACTOR_CHECKPOINT')
    require_full_pose_contract(metadata)
    np.savez_compressed(a.output/'features.npz',policy161=np.array(vectors),source_pose_world_wxyz=np.array(sources),source_and_reference_times_s=np.array(timepairs))
    angles=audit['policy_contract']['instantaneous_axial_roll_witness'];pose=tensor(sources[0][None]);mount=torch.tensor([0.,0,0,1,0,0,0])[None]
    R=Rotation.from_quat(pose[0,[4,5,6,3]].numpy());A=np.array([v['axial_roll_deg'] for v in angles])
    rotations=R*Rotation.from_euler('x',A[:,None],degrees=True)
    altered=tensor(np.c_[np.tile(pose.numpy()[0,:3],(len(A),1)),rotations.as_quat()[:,[3,0,1,2]]])
    f=pose9_world_in_wrist(altered,mount,position_bound=bound);delta=torch.linalg.vector_norm(f-f[0],dim=-1).numpy()
    verify_source_snapshot(receipt,live=True)
    checks={str(i)+'_'+key:val for i,r in enumerate(rows) for key,val in r['checks'].items()}
    checks['all285_sources']=len(rows)==285;checks['axial_full_pose_distinguishes']=bool(np.all(delta[1:]>0))
    result=dict(schema='pen_full_pose_policy_feature_replay:1',config=file_pin(cp),task_audit=file_pin(rp),source_receipt=file_pin(a.output/'source_receipt.json'),
        role=cfg['role'],rows=rows,metadata=metadata,features=file_pin(a.output/'features.npz'),
        passed=sum(checks.values()),total=len(checks),all_checks_passed=all(checks.values()),
        axial_roll_feature_L2_differences=delta.tolist(),native_integrations=0,optimizer_updates=0,scope=cfg['scope'])
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[result['passed'],result['total']],
        max_current_axis_error_m=max(r['current_qpos_axis_max_error_m'] for r in rows),max_cached_axis_error_m=max(r['cached_axis_max_error_m'] for r in rows),
        axial_feature_differences=result['axial_roll_feature_L2_differences'])))
    if not all(checks.values()):raise ValueError('Keep feature/frame/time discrepancies; no training readiness inferred')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
