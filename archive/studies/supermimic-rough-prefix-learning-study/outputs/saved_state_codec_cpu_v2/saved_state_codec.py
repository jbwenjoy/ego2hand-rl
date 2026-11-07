"""Reconstruct native state58 from recorded qpos/qvel, without physics.

The fixed palm is bound to this MJB's identity child of mocap_base. It is not
a generic substitution of mocap for palm. Native SimReader.state executes on
CPU tensor views; no Reader/Env is constructed or forward/step called.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .io import checked,loadnp,read,pin,canonical


def _need(value,message):
    if not value:raise ValueError(message)


def native_state58(qpos,qvel,model,palm_pose):
    import torch
    from dex_supermimic.rl.dex_supermimic.sim import SimReader
    from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat
    qpos=np.asarray(qpos,dtype=np.float32);qvel=np.asarray(qvel,dtype=np.float32)
    _need(qpos.ndim==2 and qvel.shape==(len(qpos),model.nv) and qpos.shape[1]==model.nq,
          'Wrong full qpos/qvel shapes')
    _need(np.isfinite(qpos).all() and np.isfinite(qvel).all(),'Nonfinite recorded state')
    palm=np.asarray(palm_pose,np.float32)
    _need(palm.shape==(7,) and np.isfinite(palm).all() and abs(np.linalg.norm(palm[3:])-1)<1e-5,
          'Wrong fixed palm pose')
    names=[f'robot/right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]
    ids=np.array([model.joint(name).id for name in names]);obj=model.joint('object/cube_freejoint').id
    qa=model.jnt_qposadr[ids];va=model.jnt_dofadr[ids]
    oq=int(model.jnt_qposadr[obj]);ov=int(model.jnt_dofadr[obj])
    data=SimpleNamespace(qpos=torch.from_numpy(qpos.copy()),qvel=torch.from_numpy(qvel.copy()))
    reader=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data)),
        robot=SimpleNamespace(indexing=SimpleNamespace(joint_q_adr=torch.from_numpy(qa.copy()),joint_v_adr=torch.from_numpy(va.copy()))),
        obj=SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=torch.arange(oq,oq+7),free_joint_v_adr=torch.arange(ov,ov+6))),
        palm_pos=torch.from_numpy(palm[:3].copy())[None].expand(len(qpos),-1),
        palm_rot=quat_to_mat(torch.from_numpy(palm[3:].copy())[None]).expand(len(qpos),-1,-1))
    value=SimReader.state(reader)
    _need(value.shape==(len(qpos),58) and bool(torch.isfinite(value).all()),'Native state58 reconstruction failed')
    return value


def bind_fixed_palm(model,bundle,raw):
    palm=model.body('robot/right_palm_link').id;parent=int(model.body_parentid[palm]);mocap=int(model.body_mocapid[parent])
    _need(model.body(parent).name=='robot/mocap_base' and int(model.body_parentid[parent])==0 and mocap==0,
          'Different bound palm/mocap parent structure')
    _need(int(model.body_jntnum[palm])==0 and np.array_equal(model.body_pos[palm],np.zeros(3))
          and np.array_equal(model.body_quat[palm],np.array([1.,0,0,0])),
          'Palm must be the fixed identity child, never a generic mount frame')
    source=bundle['spec']['source_pins']['contact_reference'];reference=loadnp(checked(source))
    pose=reference['wrist_pose_w'][0].astype(np.float32)
    _need(np.array_equal(reference['wrist_pose_w'],np.broadcast_to(reference['wrist_pose_w'][:1],(16,7))),
          'Palm reference moves')
    _need(raw['mocap_pos_w'].shape==(61,1,3) and raw['mocap_quat_wxyz'].shape==(61,1,4),'Wrong recorded mocap shape')
    _need(np.array_equal(raw['mocap_pos_w'].astype(np.float32),np.broadcast_to(pose[:3],(61,1,3)))
          and np.array_equal(raw['mocap_quat_wxyz'].astype(np.float32),np.broadcast_to(pose[3:],(61,1,4))),
          'Observed fixed parent and bound palm reference differ')
    return pose,{'palm_body_id':palm,'palm_body_name':'robot/right_palm_link','parent_body_id':parent,
        'parent_body_name':'robot/mocap_base','parent_mocap_id':mocap,'child_transform':'identity/no joint',
        'palm_reference':source,'FP32_pose':pose.tolist(),'all61_parent_pose_exact_to_bound_palm':True,
        'scope':'Structural fixed-frame reconstruction; native x58 was not itself recorded in these rollouts.'}


def executed_intervals(raw,events):
    _need(len(raw['raw_time_s'])==61 and len(events)==15 and
          all(type(e['phase']) is int and e['phase']==k for k,e in enumerate(events)),
          'Wrong state/control phase indexing')
    _need(np.max(np.abs(raw['raw_time_s']-np.arange(61)*.005))<=2e-6,'Wrong recorded clock')
    u=np.asarray([e['actualu'] for e in events],np.float32)
    _need(u.shape==(15,1,20) and np.isfinite(u).all(),'Wrong or nonfinite actualu15')
    u=u[:,0].copy()
    _need(np.asarray(raw['ctrl_rad']).shape==(61,20),'Wrong full control timeline')
    for k in range(15):
        _need(np.array_equal(raw['ctrl_rad'][4*k+1:4*k+5].astype(np.float32),np.broadcast_to(u[k],(4,20))),
              'Executed target does not match all four recorded substeps')
    return u,np.arange(0,61,4)


def load_saved_transitions(bundle,run,engineering_audit,expected_method,expected_pins=None):
    import mujoco
    import torch
    from .probe import GATES
    run=Path(run).resolve(strict=True);audit_path=Path(engineering_audit).resolve(strict=True)
    audit=read(audit_path);config=read(run/'config.json');evaluation=read(run/'evaluation.json')
    if expected_method=='A0':
        _need(audit.get('engineering_passed') is True and audit.get('status')=='PASS_ENGINEERING'
              and audit['input_pins']['worker_evaluation_record_only']==pin(run/'evaluation.json'),'A0 engineering audit mismatch')
    else:
        _need(expected_method in ('A1L','P_ONLY','A2') and audit.get('engineering_qualified') is True
              and type(audit.get('checks')) is dict and bool(audit['checks'])
              and all(v is True for v in audit['checks'].values()),'Policy engineering audit unqualified')
        _need(audit['evaluation']==pin(run/'evaluation.json') and audit['checkpoint']==config['checkpoint']
              and audit['config']['sha256']==pin(run/'config.json')['sha256'],'Policy audit/evaluation identity mismatch')
        _need(type(config['checkpoint_iteration']) is int and config['checkpoint_iteration']==2000,
              'Only fixed final2000 policy data belong to this diagnostic')
    _need(config['method']==expected_method and config['input_bundle']==bundle['descriptor']['input_bundle'],'Data arm/input identity differs')
    _need(evaluation.get('complete') is True and evaluation['physical_steps']==60
          and evaluation['control_intervals']==15 and evaluation['native_postinterval_forward_calls']==15
          and evaluation['resets_after_start']==0 and canonical(evaluation['criteria'])==canonical(GATES),
          'Require complete fixed native rollout, without selecting quality success')
    filenames=('raw_states.npz','control_events.json','nominal_model.mjb','startup_initialization_checks.json','capacity_audit.json')
    pins={name:pin(run/name) for name in filenames}
    if expected_pins is not None:
        _need(set(expected_pins)==set(filenames) and all(pins.get(name)==value for name,value in expected_pins.items()),
              'Registered saved data bytes changed')
    _need(pins['nominal_model.mjb']['sha256']==bundle['model']['sha256'],'Different recorded physical model')
    capacity=read(run/'capacity_audit.json')
    _need(capacity['status']=='OK' and capacity['read_count']==1 and capacity['read_after_step']==60
          and capacity['flags_by_world']==[0] and capacity['resets_during_probe']==0,'Capacity evidence mismatch')
    startup=read(run/'startup_initialization_checks.json')['checks']
    _need(bool(startup) and all(v is True for v in startup.values()),'Startup state unqualified')
    raw=loadnp(run/'raw_states.npz');events=read(run/'control_events.json')
    u,knots=executed_intervals(raw,events)
    _need(np.max(np.abs(raw['external_force_max_abs']))==0,'Exogenous data are unsupported')
    model=mujoco.MjModel.from_binary_path(str(run/'nominal_model.mjb'))
    palm,palm_receipt=bind_fixed_palm(model,bundle,raw)
    x=native_state58(raw['full_qpos'],raw['full_qvel'],model,palm)
    return {'x':x[knots[:-1]].clone(),'x_next':x[knots[1:]].clone(),'u':torch.from_numpy(u.copy()),
        'palm_pose':palm,'knots':knots,'receipt':{'schema':'rough_saved_native_state58_reconstruction:1',
          'method':expected_method,'run':str(run),'audit':pin(audit_path),'pins':pins,'palm':palm_receipt,
          'rows':15,'state_dim':58,'target_dim':20,'source_indices_k':knots[:-1].tolist(),
          'source_indices_kplus1':knots[1:].tolist(),'absolute_executed_u':True,
          'selfcollision_labels':'UNKNOWN_not_saved_sensor_found','physics_steps_added':0,
          'native_method':'SimReader.state on CPU qpos/qvel tensor views; no environment or reader constructor',
          'feature_status':'Reconstructed with exact installed method; not a recorded native feature replay',
          'source':pin(__file__)}}
