"""Pure saved-data helpers for permanently held-out tripod action diagnostics."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
ROLE='ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE'
RULES=dict(family_ids=[12,15],native_dt_s=[.0003125,.00015625],step_k=[0,3],contexts=8,
    action_axes=list(range(20)),signs=[-1,1],nominal_du_rad=.01,minimum_available_actual_du_rad=1e-6,
    boundary_rule='one-sided min(nominal,remaining true actuator ctrlroom); no halfroom, clip, rescue or symmetric shrink',
    planned_branches=328,baseline_branches=8,maximum_new_sides=320,maximum_native_steps=31488,
    all8_baselines_before_any_new_side=True,maximum_H_s=.02,restore_forward=0,NN_calls=0,training_updates=0,
    permanent_data_role=ROLE,automatic_retry=False)
FIELD_MAP=dict(x58='source_state58_F64',u20='actual_control20_F64',base_u20='baseline_control20_F64',y58='target_H_state58_F64',
    observed_prefix58='observed_prefix_target58_F64',delta52='target_delta52_F64',heads178='target_physical_heads178_F64',
    impulse22x6='impulse_source_BODY_moving_COM22x6_F64',meanFn22='mean_positive_Fn22_N_F64',coverage22='Fn_gt_0p1_coverage22_F64',
    generated_coverage22='generated_contact_coverage22_F64',triple_coverage='simultaneous_triple_coverage_F64',
    complete_H='complete_H_bool',finite='finite_bool',no_warning='no_warning_bool',labels_known='physical_labels_known_bool',
    actual_du='actual_du_F64',source_receipt='source_integration_F64',after_action_receipt='source_after_action_integration_F64',target_receipt='target_integration_F64')

def need(v,msg):
    if not v:raise ValueError(msg)
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Changed absolute pin');return Path(v['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(clean(value),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def save(path,arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
def npz(v):
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def receipt_meta(v):return {k:x for k,x in v.items() if k!='integration_state'}

def label_interval(trace,source,target_prefix,*,h,n,complete,finite,warning_numbers,D,maths,physical):
    """Actual22 future-H supervision; moving solverPRE COM, frozen source BODY."""
    N=len(trace['pre_tick']);known=bool(complete and finite and not np.any(warning_numbers))
    need(complete==(N==n) and 0<N<=n,'Only actual integer prefix/fullH')
    a=dict(trace,post_state58=trace['post_state58']);decoded=D.observed_window(a,source,h,maths)
    nan22=np.full(22,np.nan);J=np.full((22,6),np.nan);mean=nan22.copy();cov=nan22.copy();generated=nan22.copy();triple=float('nan');heads=np.full(178,np.nan);delta=np.full(52,np.nan)
    if known:
        need(decoded is not None,'Known actual contact operands');J=decoded['impulse_source_BODY_moving_COM22x6'];mean=decoded['mean_positive_Fn22_N'];cov=decoded['Fn_gt_0p1_coverage22'];generated=decoded['generated_contact_coverage22'];triple=decoded['simultaneous_triple_coverage']
        delta=maths['codec'].delta52(source,target_prefix)
        e=maths['encode'](source[None],target_prefix[None],J[None],np.array([h*n]),np.array(physical['gravity_world_m_s2']))
        heads=np.r_[e['joint_delta40'][0],e['COM_ballistic_residual3_m'][0],e['world_rotation_delta3_rad'][0],J.ravel()]
    return dict(target_H_state58_F64=target_prefix if complete else np.full(58,np.nan),observed_prefix_target58_F64=target_prefix,
        target_delta52_F64=delta,target_physical_heads178_F64=heads,impulse_source_BODY_moving_COM22x6_F64=J,
        mean_positive_Fn22_N_F64=mean,Fn_gt_0p1_coverage22_F64=cov,generated_contact_coverage22_F64=generated,simultaneous_triple_coverage_F64=np.asarray(triple),
        observed_prefix_impulse_source_BODY_moving_COM22x6_F64=np.full((22,6),np.nan) if decoded is None else decoded['impulse_source_BODY_moving_COM22x6'],
        observed_prefix_duration_s_F64=np.asarray(N*h),complete_H_bool=np.asarray(complete),finite_bool=np.asarray(finite),no_warning_bool=np.asarray(not np.any(warning_numbers)),physical_labels_known_bool=np.asarray(known))

def planned_side(u,axis,sign,bounds):
    room=float(bounds[axis,1]-u[axis]) if sign==1 else float(u[axis]-bounds[axis,0])
    requested=min(.01,room);v=u.copy();v[axis]=np.float64(u[axis]+sign*requested);du=float(v[axis]-u[axis])
    available=bool(room>=0 and sign*du>=1e-6 and np.isfinite(v).all() and np.all(v>=bounds[:,0]) and np.all(v<=bounds[:,1]))
    return dict(axis=int(axis),sign=int(sign),nominal_magnitude_rad=.01,remaining_room_rad=room,planned_magnitude_rad=requested,
        actual_du_F64=du,available=available,status='AVAILABLE' if available else 'UNAVAILABLE',
        unavailable_reason=None if available else 'actual_du_below_1e-6_or_numeric_bound_no_rescue',actual_ctrl20_F64=v.tolist(),
        actual_ctrl20_raw_sha256=sha(v),limited_by_bound=requested<.01,posthoc_clip=False)
