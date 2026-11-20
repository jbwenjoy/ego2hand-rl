"""Prepare common, pre-action TRAIN starts for H=3 imagined Actor comparisons."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_reward_chain import A, I, ACTION, MOTION, ROBOT, CACHE, source_context
from supermimic_research.mechanics.pen_wrench_capacity import balanced_directions
from supermimic_research.runtime.pen_continuous_math_view_v1 import policy
from supermimic_research.strict_json import write_json


RUNS = ROOT / 'runs'
TRAIN_ROLES = {'guarded_finger_bank_v1': ('TRAIN_A', 'TRAIN_B'), 'guarded_finger_response_v1': ('TRAIN',)}
PHASES = (160, 180, 192, 200, 206, 210, 214, 218, 222, 226)
FEATURE = (A('env'), A('_native_full_pose_features'))
HORIZON = 3


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def rot(wxyz):
    np.testing.assert_allclose(np.linalg.norm(wxyz, axis=-1), 1., rtol=0, atol=1e-5)
    return Rotation.from_quat(wxyz[..., [1, 2, 3, 0]]).as_matrix()


def wrist_pose9(position, rotation, mount):
    wrist_world = rot(mount[3:]).T
    local = wrist_world @ (position - mount[:3])
    relative = wrist_world @ rotation
    return np.r_[.15*np.tanh(local/.15), relative[:,0], relative[:,1]]


def source_rows(report, role):
    """Select by the source ownership/phase declaration, independent of outcomes."""
    assert report['role'] == role
    result = {}
    for row in report['rows']:
        phase = row['phase']
        if row['role'] != role or phase not in PHASES:
            raise ValueError('Unexpected role or source phase')
        if phase not in result:
            result[phase] = dict(pin=row['source'], names=set(), check_packet=None)
        group = result[phase]
        if group['pin'] != row['source'] or row['name'] in group['names']:
            raise ValueError('Inconsistent source or duplicate intervention within phase')
        group['names'].add(row['name'])
        if row['name'] == 'sham':
            group['check_packet'] = row['packet']
    if set(result) != set(PHASES) or not all(len(g['names']) == 20 and g['check_packet'] for g in result.values()):
        raise ValueError('Incomplete source ownership manifest')
    return result


def reference_inputs():
    cap_pin=file_pin(RUNS/'pen_capacity_targets_v1/report.json')
    cap=read(cap_pin); intent=read(cap['intent']); clip=cap['refs']['clip_001']
    if intent['length_m'] != .10: raise ValueError('Capacity torque scale mismatch')
    catalog=read(intent['source'])['results']['clip_001']
    if catalog['source'] != clip['source']: raise ValueError('Different reference clip')
    with np.load(verify_file_pin(clip['series'])) as f: geometry=dict(f)
    with np.load(verify_file_pin(clip['source'])) as f: joints=f['reference_qpos'].copy()
    with np.load(verify_file_pin(catalog['series'])) as f: pose=np.c_[f['COM'],f['quat_wxyz']]
    np.testing.assert_array_equal(geometry['directions'],balanced_directions())
    assert joints.shape==(369,20) and pose.shape==(369,7) and geometry['capacity'].shape==(369,72)
    assert geometry['eligible'].shape==(368,) and geometry['stage'].shape==(368,) and geometry['mask'].shape==(369,)
    return geometry,joints,pose,dict(capacity_report=cap_pin,capacity_series=clip['series'],clip=clip['source'],fixed_poses=catalog['series'])


def source_data(snapshot, phase, geometry, clip_joints, clip_poses):
    if (snapshot['schema'],snapshot['worlds'],snapshot['metadata']['fixed_frame_features'])!=('controlled_pen_owned_state:4',1,'LIVE'):
        raise ValueError('Expected native W1 live-feature snapshot')
    p=snapshot['python'];t=p['tensors'];sc=p['scalars'];arrays=p['arrays'];model=snapshot['native']['model']['arrays'];d=snapshot['native']['data']['arrays']
    qids=t[FEATURE+(A('qids'),)].numpy();dofs=t[FEATURE+(A('dofs'),)].numpy();aids=t[ACTION+(A('_position_ctrl_ids'),)].numpy()
    local=t[ACTION+(A('_target_ids'),)].numpy();body=sc[FEATURE+(A('body'),)];palm=sc[FEATURE+(A('palm'),)];qa=sc[FEATURE+(A('qa'),)]
    jids=model['actuator_trnid'][aids,0];da=int(model['body_dofadr'][body])
    np.testing.assert_array_equal(qids,model['jnt_qposadr'][jids]);np.testing.assert_array_equal(dofs,model['jnt_dofadr'][jids])
    if len(np.unique(local))!=20 or np.any(local<0) or np.any(local>=20) or len(np.unique(aids))!=20 or model['jnt_qposadr'][model['body_jntadr'][body]]!=qa:
        raise ValueError('Invalid joint, actuator or object mapping')
    if bool(t[(A('pending'),)].any()) or not bool(t[ACTION+(A('_targets_valid'),)].all()):
        raise ValueError('Terminal or uninitialized source')
    case=dict(qids=qids.tolist(),dofs=dofs.tolist(),aids=aids.tolist(),model_layout=dict(object_qposadr=qa))
    ctx=source_context(dict(native=snapshot,phase=phase),case)
    frames=np.arange(phase,phase+HORIZON); starts=frames-1
    assert ctx['frames']==369 and frames[-1]<ctx['frames'] and sc[(A('physics_integrations'),)]==(phase-1)*8
    assert float(abs(d['time'][0]-(phase-1)*.02))<2e-4
    ref_joints=t[MOTION+(A('group'),A('joint_pos'))][frames].numpy()
    ref_links=t[MOTION+(A('group'),A('link_pos'))][frames].numpy().reshape(HORIZON,63)
    ref_pose=t[CACHE+(A('world_pose'),)][0,frames].numpy()
    np.testing.assert_array_equal(ref_joints,clip_joints[frames])
    np.testing.assert_allclose(ref_pose[:,:3],clip_poses[frames,:3],rtol=0,atol=1e-6)
    np.testing.assert_allclose(rot(ref_pose[:,3:]),rot(clip_poses[frames,3:]),rtol=0,atol=2e-5)
    mount=np.r_[d['xpos'][0,palm],d['xquat'][0,palm]]
    np.testing.assert_allclose(mount,t[CACHE+(A('mount'),)][0].numpy(),rtol=0,atol=1e-7)
    state=native_state55(d['qpos'][0],d['qvel'][0],qids,dofs,qa,da)
    object_R=np.column_stack((state[43:46],state[46:49],np.cross(state[43:46],state[46:49])))
    raw=t[ACTION+(A('_raw_actions'),)];ema=t[ACTION+(A('_processed_actions'),)]
    if float(raw.abs().max())>1+1e-7 or float(ema.abs().max())>.5+1e-7: raise ValueError('Controller memory exceeds envelope')
    q0=t[ROBOT+(A('default_joint_pos'),)][0,local].numpy();v0=t[ROBOT+(A('default_joint_vel'),)][0,local].numpy();bias=ctx['encoder_bias'].numpy()
    p9=np.stack([wrist_pose9(pose[:3],rot(pose[3:]),mount) for pose in ref_pose])
    feat=np.r_[ref_joints[0],p9[0],ref_links[0],state[:20]+bias-q0,.25*(state[20:40]-v0),wrist_pose9(state[40:43],object_R,mount),raw[0].numpy()]
    dyn=dict(phase=phase,arrays={k:d[k] for k in ('qpos','qvel','time','qacc_warmstart')},cached_R=d['xmat'][0,body],mount=torch.from_numpy(mount[None]),EMA=ema,last_raw=raw,trajectory_ids=t[MOTION+(A('trajectory_ids'),)])
    reconstructed=policy(snapshot,dyn,case)[0].numpy()
    np.testing.assert_allclose(feat,reconstructed,rtol=0,atol=2e-6)
    prefix=t[CACHE+(A('prefix'),)][0,frames].numpy()
    np.testing.assert_allclose(prefix,np.c_[ref_joints,p9,ref_links],rtol=0,atol=2e-6)
    prior=lambda key:t[(A('contact_views'),I(key))][0].numpy()
    if prior('counts')[0]!=8: raise ValueError('Source requires the complete preceding control window')
    prevw=prior('wrench_sum')[0]/8
    hist=np.r_[object_R.T@prevw[:3],object_R.T@prevw[3:],prior('Fn6_sum')/8]
    mass=float(model['body_mass'][0,body]);inertia=np.diag(model['body_inertia'][0,body]);gravity=model['opt.gravity'][0]
    np.testing.assert_array_equal(model['body_ipos'][0,body],[0,0,0]);np.testing.assert_array_equal(model['body_iquat'][0,body],[1,0,0,0])
    np.testing.assert_array_equal(arrays[(A('support_table'),A('mass'))],[mass]);np.testing.assert_array_equal(arrays[(A('support_table'),A('inertia'))][0],inertia.diagonal());np.testing.assert_array_equal(arrays[(A('support_table'),A('gravity'))][0],gravity)
    np.testing.assert_array_equal(t[ACTION+(A('_targets'),)][0].numpy(),d['ctrl'][0,aids])
    data=dict(state55=state,previous_raw=raw[0].numpy(),previous_ema=ema[0].numpy(),previous_pd=d['ctrl'][0,aids],
        observation161=reconstructed,mount_wxyz=mount,encoder_bias=bias,default_joint_pos=q0,default_joint_vel=v0,
        lower=ctx['lower'].numpy(),upper=ctx['upper'].numpy(),ref_joints=ref_joints,ref_pose_wxyz=ref_pose,ref_links=ref_links,
        ref_pose9_in_wrist=p9,ref_prefix92=prefix,phase=phase,ref_frame=frames,capacity_interval=starts,
        capacity_start=geometry['capacity'][starts],capacity_end=geometry['capacity'][frames],capacity_eligible=geometry['eligible'][starts],
        mask_start=geometry['mask'][starts],mask_end=geometry['mask'][frames],stage=geometry['stage'][starts],
        target_wrench_world=t[(A('support_table'),A('targets'))][0,starts].numpy(),target_wrench_valid=t[(A('support_table'),A('valid'))][starts].numpy(),
        mass_kg=mass,inertia_BODY=inertia,gravity_WORLD=gravity,history_load12=hist,warmstart26=d['qacc_warmstart'][0],
        qids=qids,dofs=dofs,aids=aids,local_joint_ids=local,object_qposadr=qa,object_dofadr=da,object_body=body)
    if any(not np.isfinite(v).all() for v in map(np.asarray,data.values()) if v.dtype.kind in 'fc'):
        raise ValueError('Nonfinite pre-action or reference quantity')
    return data,float(abs(feat-reconstructed).max())


@torch.no_grad()
def main(output):
    torch.set_num_threads(1)
    geometry,joints,poses,inputs=reference_inputs();sources=[];rows=[];seen=set()
    for bank,roles in TRAIN_ROLES.items():
        root=RUNS/bank;manifest_pin=file_pin(root/'report.json');intent_pin=file_pin(root/'intent.json')
        intent=read(intent_pin);manifest=read(manifest_pin);inputs[bank]=dict(manifest=manifest_pin,intent=intent_pin)
        if tuple(intent['phases'])!=PHASES:raise ValueError('Predeclared phases changed')
        actor_pins=intent.get('actors',intent.get('arms'))
        for role in roles:
            role_pin=manifest['arms'][role]; selected=source_rows(read(role_pin),role)
            actor_state=torch.load(verify_file_pin(actor_pins[role]),map_location='cpu',weights_only=False)
            cfg=actor_state['config'];actor=GaussianPPOActor(161,20,init_std=cfg['ppo_init_std'],std_range=(cfg['ppo_std_min'],cfg['ppo_std_max'])).eval()
            actor.load_state_dict(actor_state['actor'],strict=True)
            for phase in PHASES:
                entry=selected[phase];pin=entry['pin'];path=verify_file_pin(pin)
                if path.resolve().parent!=(root/role).resolve() or pin['sha256'] in seen:raise ValueError('Unexpected or repeated source identity')
                seen.add(pin['sha256']);s=torch.load(path,map_location='cpu',weights_only=False)
                row,obs_error=source_data(s,phase,geometry,joints,poses)
                mean=actor.distribution(torch.from_numpy(row['observation161'][None]).float(),validate_args=False).mean
                if bank=='guarded_finger_bank_v1':mean=mean.clamp(-1,1)
                # These two pre-action quantities are verification only. Never access the response y, validity or reward.
                with np.load(verify_file_pin(entry['check_packet'])) as packet:
                    np.testing.assert_allclose(row['state55'],packet['x'][0,:55],rtol=0,atol=1e-10)
                    actor_err=float(np.max(np.abs(mean.numpy()-packet['base_raw'])))
                    if not np.isfinite(actor_err) or actor_err>2e-5:raise ValueError(f'Native source Actor reconstruction differs by {actor_err}')
                sources.append(dict(index=len(rows),role=role,phase=phase,snapshot=pin,role_report=role_pin,check_packet=entry['check_packet'],
                    verification_actor=actor_pins[role],reference_signature=s['reference_signature'],host_model_digest=s['host_model_digest'],
                    max_obs_abs_error=obs_error,max_logged_actor_mean_abs_error=actor_err))
                rows.append(dict(role=role,**row))
    if len(rows)!=30 or len({r['reference_signature'] for r in sources})!=1 or len({r['host_model_digest'] for r in sources})!=1:
        raise ValueError('Source count/reference/model identity mismatch')
    output.mkdir(parents=True,exist_ok=False)
    pack={key:np.stack([r[key] for r in rows]) for key in rows[0]}
    pack.update(directions=geometry['directions'],length_scale_m=np.array(.10),dt_s=np.array(.02))
    path=output/'starts.npz';np.savez_compressed(path,**pack)
    with np.load(path,allow_pickle=False) as saved:
        for k,v in pack.items():np.testing.assert_array_equal(saved[k],v)
    write_json(output/'report.json',dict(schema='pen_train_only_imagined_starts:1',inputs=inputs,sources=sources,pack=file_pin(path),
        shapes={k:list(v.shape) for k,v in pack.items()},sources_count=len(rows),horizon=HORIZON,source_code=file_pin(__file__),
        capacity_eligible_slots=int(pack['capacity_eligible'].sum()),eligible_per_phase={str(p):int(pack['capacity_eligible'][pack['phase']==p].sum()) for p in PHASES},
        max_obs_abs_error=max(r['max_obs_abs_error'] for r in sources),max_logged_actor_mean_abs_error=max(r['max_logged_actor_mean_abs_error'] for r in sources),
        conventions='state55 is hand q20, hand qd20, world COM3, world rotation columns 0 and 1, world linear velocity3, world angular velocity3. For control at sensed phase p the Actor uses reference frame p (WORLD xyz+wxyz, link positions flattened in wrist frame) and the interval [p-1,p] capacity endpoints/eligibility/stage and nominal wrench. Indices are zero-based. For H=3, consume phase p+h, h=0..2. History is previous completed control mean wrench about moving COM, rotated from WORLD to current source BODY, then six normal-force means. Raw feedback is the previous outer-clamped raw action; EMA in joint radians. Capacity is shared 1N normal budget, torque / .10 m. target_wrench_world is the nominal motion prior, not a measured contact label.',
        calibration='Observation is unnormalized 161. Use the selected frozen seed Actor normalizer, not a fit on this pack. Controller outer clamp +/-1, scale .5, residual clamp +/- .6, EMA alpha .5, subtract encoder bias then clamp to lower/upper. Physics and calibration repeated per row.',
        scope='All 30 TRAIN phases selected by ownership and registration alone. No held-out snapshots, successor states, rewards, actor outputs or measured future contact labels in the pack. Actor inference and source-side sham fields used only for independent CPU checks. No new physics or updates.'))
    print(json.dumps(dict(pack=file_pin(path),sources=len(rows),horizon=HORIZON,eligible=int(pack['capacity_eligible'].sum()))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUNS/'pen_imagined_actor_starts_v1')
    main(parser.parse_args().output.resolve())
