"""Strict source contract, three observed fields, and reconstructed mapping."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .io import sha,read

PROTOCOL=Path('D:/workspace/supermimic-observation-dynamic-wrench-study/proposals/short_dynamic_tracking_v1/protocol.json')
PROTOCOL_SHA='94ea7cbe03f3e2aaa7bb4f7c8860bae8e599cd4e8e823cbe47220f73ffe101a3'
OBS_FIELDS=('time_s','hand_landmarks_source_w','object_pose_source_w')
FROZEN_FIELDS=('time_s','source_to_fixed_wrist_R','source_to_fixed_wrist_translation_m','mapped_object_pose_w','morphology_normalized_target_points_w')
WEIGHTS={'bone_direction':1.,'native_hand_object_and_self_overlap':1.,'anchor_position':1.,'anchor_normal':.25,'q_to_canonical_q0':.03,'q_to_nominal':.001,'adjacent_q_difference':.03,'second_q_difference':.03}

def checked_protocol(path=PROTOCOL):
    path=Path(path)
    if sha(path)!=PROTOCOL_SHA:raise ValueError('Frozen94ea protocol drift')
    p=read(path)
    for name,pin in p['inputs'].items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Frozen input/source drift: '+name)
    s=p['next_CPU_task']['global_q_goal_solver']
    if s['residual_weights']!=WEIGHTS or (s['anchor_position_scale_m'],s['anchor_inset_m'],s['native_overlap_allowance_m'],s['native_overlap_scale_m'])!=(.001,.0003,.0005,.001):raise ValueError('Objective contract differs')
    if (s['knots'],s['joints'],s['max_nfev'],s['soft_wall_s'],s['owned_process_cap_s'])!=(16,20,80,110,120):raise ValueError('Budget/shape contract differs')
    return p

def checked_plan(path,*,run=False):
    p=read(path)
    if p['schema']!='prefix16_geometry_cpu_plan:1':raise ValueError('Plan schema')
    if run and p.get('optimization_authorized') is not True:raise ValueError('Root optimization authorization absent')
    protocol=p['protocol']
    if sha(protocol['path'])!=protocol['sha256'] or protocol['sha256']!=PROTOCOL_SHA:raise ValueError('Plan protocol binding')
    ref=checked_protocol(protocol['path'])
    if p['inputs']!=ref['inputs']:raise ValueError('Plan inputs must preserve all frozen protocol inputs')
    for name,pin in p['source_pins'].items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Plan source drift: '+name)
    expected={'frames':16,'joints':20,'max_nfev':80,'soft_wall_seconds':110,'owned_process_seconds':120,'automatic_retry':False}
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Formal fixed budget contract differs')
    root=Path(__file__).resolve().parents[2]
    required=[root/'scripts/build_prefix16_geometry_cpu.py',*sorted((root/'src/prefix_study').glob('*.py')),root/'vendor/runtime_io.py']
    required += [Path(ref['inputs'][k]['path']) for k in ('kinematics','native_filtered_geometry','material_geometry')]
    bound={str(Path(v['path']).resolve()).lower():v['sha256'] for v in p['source_pins'].values()}
    for file in required:
        if bound.get(str(file.resolve()).lower())!=sha(file):raise ValueError('Unbound executed dependency: '+str(file))
    return p,ref

def read_prefix_inputs(protocol,robot,*,read_array=None):
    """Never requests joint_pos_rad from the full539d artifact.

    read_array instrumentation enables a poison fixture proving that property.
    Explicit float64 reproduces the previous input/normalization convention.
    """
    log=[]
    def load(role,fields):
        path=protocol['inputs'][role]['path']
        with np.load(path,allow_pickle=False) as data:
            result={}
            for key in fields:
                log.append({'role':role,'field':key})
                value=data[key] if read_array is None else read_array(role,key,data)
                result[key]=np.array(value,dtype=np.float64,copy=True)
        return result
    obs=load('observation_fixture',OBS_FIELDS)
    frozen=load('frozen_full151_geometry',FROZEN_FIELDS)
    init=load('canonical_H123_q0',('joint_pos_rad','mapped_object_pose_w'))
    if obs['hand_landmarks_source_w'].shape!=(151,21,3) or obs['object_pose_source_w'].shape!=(151,7) or init['joint_pos_rad'].shape!=(1,20) or init['mapped_object_pose_w'].shape!=(1,7):raise ValueError('Source layout differs')
    if not all(np.isfinite(a).all() for d in (obs,frozen,init) for a in d.values()):raise ValueError('Nonfinite input')
    t=obs['time_s']
    if t.shape!=(151,) or np.max(np.abs(t-np.arange(151)*.02))>1e-12 or not np.array_equal(t,frozen['time_s']):raise ValueError('Canonical20ms clock')
    human=obs['hand_landmarks_source_w'][:16];sourcepose=obs['object_pose_source_w'][:16]
    if np.max(np.abs(np.linalg.norm(sourcepose[:,3:],axis=1)-1))>1e-6:raise ValueError('Improper source pose')
    sys.path.insert(0,str(Path(protocol['inputs']['kinematics']['path']).parent))
    from retarget_palmroot_cpu import hand_frames,directions
    R=np.einsum('ij,tkj->tik',robot.palm_frame,hand_frames(human))
    translation=robot.neutral[0]-np.einsum('tij,tj->ti',R,human[:,0])
    desired=np.einsum('tij,tkj->tki',R,directions(human))
    mapped_p=np.einsum('tij,tj->ti',R,sourcepose[:,:3])+translation
    mapped_R=R@Rotation.from_quat(sourcepose[:,[4,5,6,3]]).as_matrix()
    target=[]
    for units in desired:
        value=robot.neutral.copy()
        for finger in range(5):
            first=1+4*finger
            for bone in range(3):value[first+bone+1]=value[first+bone]+robot.lengths[3*finger+bone]*units[3*finger+bone]
        target.append(value)
    target=np.array(target);pose=frozen['mapped_object_pose_w'][:16].copy()
    checks={'Rt_rotation_reconstructed':np.max(np.abs(R-frozen['source_to_fixed_wrist_R'][:16]))<1e-12,'Rt_translation_reconstructed':np.max(np.abs(translation-frozen['source_to_fixed_wrist_translation_m'][:16]))<1e-12,'object_position_reconstructed':np.max(np.abs(mapped_p-pose[:,:3]))<1e-12,'object_rotation_reconstructed':np.max(np.abs(mapped_R-Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()))<1e-12,'normalized_targets_exact_reconstructed':np.array_equal(target,frozen['morphology_normalized_target_points_w'][:16]),'unit_bone_target_float64':np.max(np.abs(np.linalg.norm(desired,axis=-1)-1))<1e-12,'q0_pose_matches_mapped_initial':np.array_equal(init['mapped_object_pose_w'][0],pose[0]),'no_old_full_q_read':not any(x['role']=='frozen_full151_geometry' and x['field']=='joint_pos_rad' for x in log)}
    if not all(checks.values()):raise ValueError('Mapping contract failed: '+str(checks))
    q0=init['joint_pos_rad'][0].copy()
    if not np.all(q0>=robot.lower) or not np.all(q0<=robot.upper):raise ValueError('q0 must not be clipped')
    return {'time_s':t[:16].copy(),'q0':q0,'mapped_pose':pose,'desired_bones':desired,'normalized_target_points_w':target,'R':R,'translation':translation,'read_log':log,'checks':{k:bool(v) for k,v in checks.items()}}
