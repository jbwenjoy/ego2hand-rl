"""Actual task breaches, fixed-wrist reconstruction and current policy contract audit."""
import argparse
import ast
import inspect
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
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.pen_task_contracts import tracking_series,first_exceed,rebase_fixed_wrist,normal_loads,candidate_pose_coverage,pose_components
from supermimic_research.runtime.controlled_evaluation import summarize_trace,write


def load(pin):return dict(np.load(verify_file_pin(pin)))


def literal(path,name):
    for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
        targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,ast.AnnAssign) else []
        if any(isinstance(t,ast.Name) and t.id==name for t in targets):return ast.literal_eval(node.value)
    raise ValueError('Explicit literal configuration field missing: '+name)


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());paths={k:(cp.parent/cfg[k]).resolve() for k in ['baseline','horizon','baseline_review','horizon_review','matrix']}
    docs={k:json.loads(p.read_text()) for k,p in paths.items()};runs=[docs['baseline'],docs['horizon']]
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs') or docs['matrix']['passed']!=docs['matrix']['total']:
        raise ValueError('Fresh unified diagnosis on the reviewed horizon matrix')
    bp=verify_file_pin(runs[0]['base_config']);base=tomllib.loads(bp.read_text())
    clip=(bp.parent/base['clips_root']/cfg['clip']).resolve();index=clip.parent/'index.json';catalog=json.loads(index.read_text())
    expected=next(e for e in catalog['clips'] if e['file']==clip.name)
    if file_pin(clip)['sha256']!=expected['sha256']:raise ValueError('Raw clip catalog hash changed')
    raw=dict(np.load(clip));a.output.mkdir()
    source_paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp]
    receipt=freeze_sources(source_paths,a.output/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(a.output/'source_receipt.json',receipt)
    constant=ROOT/'src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/hand2_constants.py'
    mount=np.array(literal(constant,'RIGHT_FIXED_WRIST_POSE_W'))
    rebased=rebase_fixed_wrist(raw['wrist_pose_w'],raw['object_pose_w'],mount)
    checks={};cases=[];tables=[];arrays={};by_key={};frames=[]
    for profile,run,review in zip(['H1','H5'],runs,[docs['baseline_review'],docs['horizon_review']]):
        checks[profile+'_review_binding']=verify_file_pin(review['run'])==paths['baseline' if profile=='H1' else 'horizon']
        for c in run['cases']:
            label=f'{profile}_{c["name"]}_{c["dt"]}';z=load(c['trace']);ref=load(c['initialization']['nominal_mean_wrench']);dt=c['dt'];n=round(.02/dt)
            m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
            for field,value in load(c['GPU_physical_context']).items():getattr(m,field)[:]=value[0]
            d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,d)
            palm=m.body('robot/right_palm_link').id;body=c['model_layout']['object_body'];qa=c['model_layout']['object_qposadr'];da=c['model_layout']['object_dofadr']
            physical_mount=np.r_[d.xpos[palm],d.xquat[palm]]
            frame_position=float(np.max(np.linalg.norm(rebased[:,:3]-ref['COM_world_m'],axis=1)))
            frame_rotation=float(np.max(pose_components(rebased[:,3:],ref['quaternion_wxyz'])['full_angle_rad']))
            mount_position=float(np.linalg.norm(physical_mount[:3]-mount[:3]));mount_angle=float(pose_components(physical_mount[None,3:],mount[None,3:])['full_angle_rad'][0])
            checks[label+'_fixed_mount']=mount_position<=cfg['frame_position_tolerance_m'] and mount_angle<=cfg['frame_rotation_tolerance_rad']
            checks[label+'_reference_rebase']=frame_position<=cfg['frame_position_tolerance_m'] and frame_rotation<=cfg['frame_rotation_tolerance_rad']
            checks[label+'_raw_hand_reference']=np.array_equal(raw['reference_qpos'].astype(np.float32),ref['reference_hand'])
            checks[label+'_original_clock']=len(z['time_s'])==369 and np.allclose(np.diff(z['time_s']),.02,rtol=0,atol=1e-14)
            series=tracking_series(z,dt);fn,nonhand=normal_loads(z,m,body)
            checks[label+'_hand_load_groups']=np.max(abs(fn.sum(1)-z['pre_hand_normal_N']))<1e-10
            checks[label+'_nonhand_load']=np.max(abs(nonhand-z['pre_object_nonhand_normal_N']))<1e-10
            for key in ['COM_RMS_m','COM_max_m','angle_RMS_rad','angle_max_rad']:
                value=series['COM_error_m'] if key.startswith('COM') else series['full_angle_rad']
                value=float(np.sqrt(np.mean(value**2))) if '_RMS_' in key else float(value.max())
                checks[label+'_'+key]=bool(np.isclose(value,c['summary']['metrics'][key],rtol=1e-12,atol=1e-12))
            measures=dict(COM_max_m=series['COM_error_m'],angle_max_rad=series['full_angle_rad'],shaft_10deg=series['shaft_angle_rad'],
                max_penetration_m=z['pre_penetration_m'],max_self_penetration_m=z['pre_self_penetration_m'],
                max_joint_limit_excursion_rad=z['post_joint_limit_excursion_rad'],max_object_nonhand_normal_force_N=z['pre_object_nonhand_normal_N'])
            first={key:first_exceed(v,base['angle_max_rad'] if key=='shaft_10deg' else base[key],series['nominal_time_s'],series['actual_GPU_time_s']) for key,v in measures.items()}
            event=first['angle_max_rad'];detail=None
            if event:
                j=event['index'];detail=dict(full_deg=float(np.rad2deg(series['full_angle_rad'][j])),shaft_deg=float(np.rad2deg(series['shaft_angle_rad'][j])),
                    reference_axis_twist_deg=float(np.rad2deg(series['reference_axis_twist_rad'][j])) if series['twist_defined'][j] else None,
                    hand_load_N=float(fn[j].sum()),Fn6_N=fn[j].tolist())
            jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
            source=load(c['planning'][0]['source']);packet0=load(c['planning'][0]['prediction'])
            lower,upper=packet0['PD_lower20'],packet0['PD_upper20']
            within=(ref['reference_hand']>=lower-2e-7)&(ref['reference_hand']<=upper+2e-7)
            collision=[dict(name=m.geom(g).name,type=int(m.geom_type[g]),size=m.geom_size[g].tolist(),body_position=m.geom_pos[g].tolist(),body_quaternion_wxyz=m.geom_quat[g].tolist())
                for g in range(m.ngeom) if m.geom_bodyid[g]==body and (m.geom_contype[g] or m.geom_conaffinity[g])]
            frame=dict(profile=profile,mode=c['name'],dt=dt,physical_mount_wxyz=physical_mount.tolist(),mount_position_difference_m=mount_position,mount_angle_difference_rad=mount_angle,
                rebase_max_COM_difference_m=frame_position,rebase_max_full_angle_rad=frame_rotation,reference_PD_outside_components=int((~within).sum()),
                initial_joint_velocity_RMS_rad_s=float(np.sqrt(np.mean(z['initial_qvel'][m.jnt_dofadr[jids]]**2))),
                initial_object_velocity_WORLD=z['initial_qvel'][da:da+3].tolist(),
                initial_object_omega_WORLD=(Rotation.from_quat(z['initial_qpos'][qa+3:qa+7][[1,2,3,0]]).apply(z['initial_qvel'][da+3:da+6])).tolist(),
                first_reference_velocity_WORLD=ref['velocity_world'][0].tolist(),initialization=c['initialization'],object_collision=collision,
                object_mass_kg=float(m.body_mass[body]),COM_BODY=m.body_ipos[body].tolist(),
                initial_joint_shift_max_rad=float(np.max(abs(z['initial_qpos'][m.jnt_qposadr[jids]]-ref['reference_hand'][0]))))
            frames.append(frame)
            cases.append(dict(profile=profile,mode=c['name'],dt=dt,duration=c['summary']['metrics']['actual_duration_s'],qualified=c['summary']['qualified'],
                first_breaches=first,first_full_angle_detail=detail,first_hand_load_below_0_1N=first_exceed(-fn.sum(1),-.1,series['nominal_time_s'],series['actual_GPU_time_s']),
                hand_contact_coverage_over_0_1N=float(np.mean(fn.sum(1)>.1)),twist_undefined_samples=int((~series['twist_defined']).sum()),
                original_summary=c['summary'],actual_EFC=review['EFC_counts'],selected_prefix_fidelity=review['selected_prefix_counts']))
            for name,value in series.items():arrays[label+'_'+name]=value
            arrays[label+'_Fn6_N']=fn
            by_key[(profile,c['name'],dt)]=(z,c)
            for decision in c['planning']:
                k=decision['k'];p=load(decision['prediction']);safe_first,safe_all,dist,angle=candidate_pose_coverage(p,ref,k,n,base['COM_max_m'],base['angle_max_rad'],qa)
                selected=decision['selected']
                tables.append(dict(profile=profile,mode=c['name'],dt=dt,k=k,H=int(p['prediction_controls']),
                    first_pose_and_full_forecast_safe=int(safe_first.sum()),all_pose_and_full_forecast_safe=int(safe_all.sum()),
                    selected_first_pose_safe=bool(safe_first[selected]),selected_all_pose_safe=bool(safe_all[selected]),
                    missed_existing_first_pose=bool(safe_first.any() and not safe_first[selected]),
                    selected_first_COM_m=float(dist[selected,0]),selected_first_full_angle_deg=float(np.rad2deg(angle[selected,0])),
                    selected_forecast_max_COM_m=float(dist[selected].max()),selected_forecast_max_full_angle_deg=float(np.rad2deg(angle[selected].max()))))
    common=[]
    for mode in ['STATE_ONLY','STATE_PLUS_SUPPORT']:
        for dt in [.005,.0025]:
            pair=[by_key[(profile,mode,dt)] for profile in ['H1','H5']];N=min(len(z['post_qpos']) for z,c in pair)
            summaries={}
            for profile,(z,c) in zip(['H1','H5'],pair):
                dynamic={k:v[:N] for k,v in z.items() if k.startswith('pre_') or k.startswith('post_') or k=='overflow'}
                summaries[profile]=summarize_trace(dynamic,{k:z[k] for k in ['time_s','COM_world_m','quaternion_wxyz']},dt,False,base)
            common.append(dict(mode=mode,dt=dt,native_ticks=N,duration_s=N*dt,summaries=summaries))
    from dex_supermimic.tasks.pen_spin.mdp.observations import make_axis_points,bounded_object_pose_to_points_wrist_local
    from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
    source=Path(inspect.getsourcefile(make_axis_points)).resolve();checks['unified_policy_mapper']=source.is_relative_to(ROOT/'src')
    command_source=ROOT/'src/dex_supermimic/tasks/pen_spin/mdp/commands.py'
    radius=literal(command_source,'axis_point_radius_m');bound=literal(command_source,'object_position_tanh_scale_m')
    angles=cfg['axis_roll_angles_deg'];p0=rebased[0];M=Rotation.from_quat(mount[[4,5,6,3]])
    R0=M.inv()*Rotation.from_quat(p0[[4,5,6,3]]);local_position=M.inv().apply(p0[:3]-mount[:3])
    rotations=R0*Rotation.from_euler('x',np.asarray(angles)[:,None],degrees=True)
    poses=np.c_[np.tile(local_position,(len(angles),1)),rotations.as_quat()[:,[3,0,1,2]]]
    features=bounded_object_pose_to_points_wrist_local(torch.tensor(poses,dtype=torch.float64),make_axis_points(radius_m=radius,dtype=torch.float64),position_tanh_scale_m=bound).numpy()
    feature_diff=np.max(abs(features-features[0]),axis=(1,2));witness=[dict(axial_roll_deg=ang,axis_feature_max_difference_m=float(delta),full_angle_deg=float(np.rad2deg(pose_components(q[None],poses[0,None,3:])['full_angle_rad'][0]))) for ang,delta,q in zip(angles,feature_diff,poses[:,3:])]
    checks['instantaneous_axis_roll_alias']=bool(feature_diff.max()<1e-12)
    try:DexSuperMimicCfg(task='hand2-pen-spin',algorithm='ppo',logger='none');ppo_guard=None
    except ValueError as error:ppo_guard=str(error)
    checks['PPO_pen_guard_retained']=ppo_guard=='PPO is currently restricted to hand2-reorient'
    np.savez_compressed(a.output/'actual_paths.npz',**arrays)
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_task_reference_and_learning_contract_audit:1',config=file_pin(cp),inputs={k:file_pin(v) for k,v in paths.items()},
        clip=file_pin(clip),catalog=file_pin(index),source_receipt=file_pin(a.output/'source_receipt.json'),
        checks={k:bool(v) for k,v in checks.items()},passed=int(sum(checks.values())),total=len(checks),cases=cases,frame_audits=frames,
        candidate_tables=tables,common_prefix=common,paths=file_pin(a.output/'actual_paths.npz'),
        policy_contract=dict(axis_feature_source=file_pin(source),axis_radius_m=radius,position_tanh_scale_m=bound,
            instantaneous_axial_roll_witness=witness,PPO_pen_guard=ppo_guard,
            observation_shape_from_actual_traces=sorted({by_key[key][0]['pre_action_observations'].shape[-1] for key in by_key}),
            scope='Other instantaneous channels held fixed. No assertion of global unobservability with history/contact feedback, or sole failure causation. Existing axis reward/observation does not directly implement full-SO3 tracking.'),
        native_integrations=0,optimizer_updates=0,scope=cfg['scope'])
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[result['passed'],result['total']],
        first_failures=[dict(profile=c['profile'],mode=c['mode'],dt=c['dt'],COM=c['first_breaches']['COM_max_m'],full=c['first_breaches']['angle_max_rad'],detail=c['first_full_angle_detail']) for c in cases],
        candidate_missed_first_pose=sum(p['missed_existing_first_pose'] for p in tables),policy=result['policy_contract'])))
    if not all(checks.values()):raise ValueError('Retain contract discrepancy; no qualified training path inferred')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
