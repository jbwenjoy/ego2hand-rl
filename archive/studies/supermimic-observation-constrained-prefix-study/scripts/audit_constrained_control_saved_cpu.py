"""Saved-array independent controller audit. No production build/export imports."""
import hashlib,json
from functools import lru_cache
from pathlib import Path
import numpy as np
import mujoco
from scipy import sparse
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def basis(knots,t,d=0):
 last=int(np.flatnonzero(knots<knots[-1])[-1])
 @lru_cache(None)
 def n(i,p,order):
  if order>p:return 0.
  if not p:return float((knots[i]<=t<knots[i+1]) or (t==knots[-1] and i==last))
  a=knots[i+p]-knots[i];b=knots[i+p+1]-knots[i+1]
  if order:return (p/a*n(i,p-1,order-1) if a else 0.)-(p/b*n(i+1,p-1,order-1) if b else 0.)
  return ((t-knots[i])/a*n(i,p-1,0) if a else 0.)+((knots[i+p+1]-t)/b*n(i+1,p-1,0) if b else 0.)
 return np.array([n(i,3,d) for i in range(18)])
def main():
 base=ROOT/'outputs/constrained_prefix_control_export_v1';receipt=read(base/'receipt.json');pins=receipt['source_pins'];r=arrays(base/'reference_bspline.npz');u=arrays(base/'planned_control.npz');init=arrays(base/'init_requested.npz');checks=[]
 def check(n,ok,d=None):checks.append({'name':n,'pass':bool(ok),'details':d})
 def get(*path):
  val=receipt['control_metadata']
  for key in path:val=val[key]
  return u[val['npz_array']]
 def old(key):return get('fp32_old_export',key)
 pinrows=[{'name':n,'sha256':p['sha256'],'pass':sha(p['path'])==p['sha256']} for n,p in pins.items()]
 check('receipt7bb and all input/source pins',sha(base/'receipt.json')=='7bb31be6bd462fe5f4a9a53e25e9e488c4350a2f575903fbf06796b76d12b541' and all(x['pass'] for x in pinrows),len(pinrows))
 manif=read(pins['manifest']['path']);candidate=arrays(pins['candidate']['path']);terminal=arrays(pins['terminal_optimizer_return']['path']);proto=read(pins['protocol']['path'])
 check('340 terminal64aa candidate/newvariant bound',pins['candidate']['sha256']=='64aa2cfe44f20fee4546b07c4546916c95d57771404cb5fb4e39156206ad9b43' and np.array_equal(terminal['x'],candidate['optimizer_x']) and np.array_equal(r['control_points_rad'][1:].ravel(),terminal['x']) and manif['retarget_variant_id']=='H123_cubic_bspline_constrained_prefix16_v1')
 for kind,file in [('reference','reference_bspline.npz'),('init','init_requested.npz'),('control','planned_control.npz')]:
  m=read(base/(kind+'_manifest.json'))
  check(kind+' bundle identity/artifact/nullmeasurement',m['source_pins']==pins and m['artifact']['sha256']==sha(base/file) and m['source_identity']==proto['source_identity'] and m['retarget_variant_id']==manif['retarget_variant_id'] and m['init_measured'] is None and m['u_readback'] is None and m['physical_reference_qualified'] is False and m['training_authorized'] is False)
 P=r['control_points_rad'];q0=P[0];delta=P[1:]-q0;knots=r['Bspline_knots_s'];t16=r['time_s'];t61=r['planned_time61_s']
 def sample(t,d=0):return (q0 if not d else 0)+basis(knots,float(t),d)[1:]@delta
 err={}
 for times,keys in [(t16,('q_goal_rad','analytic_qd16_rad_s','analytic_qdd16_rad_s2')),(t61,('q61_rad','qd61_rad_s','qdd61_rad_s2'))]:
  for d,key in enumerate(keys):err[key]=float(abs(np.array([sample(t,d) for t in times])-r[key]).max())
 check('independent Cox basis q/d1/d2 16+61',err['q_goal_rad']<1e-12 and err['q61_rad']<1e-12 and max(err['analytic_qd16_rad_s'],err['qd61_rad_s'])<1e-9 and max(err['analytic_qdd16_rad_s2'],err['qdd61_rad_s2'])<1e-7,err)
 check('source64 P0 and16/61 saved candidate exact',np.array_equal(q0,candidate['q0_fixed_rad']) and np.array_equal(r['q_goal_rad'],candidate['q_goal_rad']) and np.array_equal(r['q61_rad'],candidate['planned_q_goal_rad']) and np.array_equal(r['mapped_object_pose_w'],candidate['mapped_object_pose_w']))
 dq=np.diff(r['q_goal_rad'],axis=0)/.02;native=np.r_[dq,dq[-1:]];q32=r['q_goal_rad'].astype(np.float32);d32=np.diff(q32,axis=0)/np.float32(.02);native32=np.r_[d32,d32[-1:]]
 check('native FD separate from analytic/designinit',np.array_equal(native,r['native_FD64_rad_s']) and np.array_equal(native32,r['native_FD32_rad_s']) and np.array_equal(init['qd_init_requested_FP32_rad_s'],r['analytic_qd16_rad_s'][0].astype(np.float32)) and not np.array_equal(native32[0],init['qd_init_requested_FP32_rad_s']))
 model=mujoco.MjModel.from_binary_path(pins['model']['path']);j=np.array([model.joint(f'robot/right_finger{f}_joint{k}').id for f in range(1,6) for k in range(1,5)]);a=np.array([np.flatnonzero(model.actuator_trnid[:,0]==i).item() for i in j]);lo,hi=model.jnt_range[j].T
 check('actual unitgear/affine/motor0',np.array_equal(model.actuator_gear[a],np.tile([1.,0,0,0,0,0],(20,1))) and np.all(model.actuator_gaintype[a]==mujoco.mjtGain.mjGAIN_FIXED) and np.all(model.actuator_biastype[a]==mujoco.mjtBias.mjBIAS_AFFINE) and np.all(model.actuator_dyntype[a]==mujoco.mjtDyn.mjDYN_NONE))
 hull_excess=float(np.maximum(np.maximum(lo-P,P-hi),0).max());check('controlpoint convexhull within original hardbox',hull_excess<=1e-12,hull_excess)
 linear=arrays(Path(pins['candidate']['path']).parent/'linear_contract.npz');A=sparse.csr_matrix((linear['A_CSR_data'],linear['A_CSR_indices'],linear['A_CSR_indptr']),shape=linear['A_shape']);lres=A@(terminal['x']-linear['x0'])-linear['b_centered']
 check('saved planning rows all nonpositive',np.all(lres<=0) and abs(float(lres.max())-receipt['max_linear_Ax_minus_b'])<1e-11,float(lres.max()))
 with np.load(pins['actual_context']['path'],allow_pickle=False) as z:kp=z['actuator_gainprm'][0,a,0];b=z['actuator_biasprm'][0,a,:3]
 design=read(pins['static_preload_design']['path']);pdu=np.array(design['u0_rad'],np.float32);tau=kp*pdu+(b[:,0]+b[:,1]*q0.astype(np.float32));Q=r['q_goal_rad'].astype(np.float32);V=r['analytic_qd16_rad_s'].astype(np.float32)
 proposed=(((tau-b[:,0])-b[:,1]*Q)-b[:,2]*V)/kp
 innerlo=np.maximum(lo,model.actuator_ctrlrange[a,0]).astype(np.float32)+np.float32(.02);innerhi=np.minimum(hi,model.actuator_ctrlrange[a,1]).astype(np.float32)-np.float32(.02);bounded=np.minimum(np.maximum(proposed,innerlo),innerhi)
 track=old('tracking_u_exec_15_FP32_rad');hold=old('hold_u_exec_15_FP32_rad');rate=np.diff(track.astype(float),axis=0)/.02;clamp=int(np.count_nonzero(track!=proposed[:15]));maxrate=float(abs(rate).max())
 check('raw FP32prior/proposal/C2 bitexact',np.array_equal(tau,old('tau_ff_static0_FP32_Nm')) and np.array_equal(proposed,old('u_pre_16_FP32_rad')) and np.array_equal(bounded,old('u_exec_16_diagnostic_FP32_rad')) and np.array_equal(track,bounded[:15]))
 check('no clamp15/sameu0/14rates and actual15 domain',clamp==0 and np.array_equal(hold,np.repeat(track[:1],15,axis=0)) and np.array_equal(init['u_planned0_FP32_rad'],track[0]) and np.array_equal(rate,old('actual_target_rate_14_rad_s')) and maxrate<=15 and np.all(track>=innerlo) and np.all(track<=innerhi),{'clamped_components':clamp,'planned14_max_rate_rad_s':maxrate})
 capslo=np.maximum(model.actuator_forcerange[a,0],model.jnt_actfrcrange[j,0]);capshi=np.minimum(model.actuator_forcerange[a,1],model.jnt_actfrcrange[j,1]);sgrid=(0.,.23,.71,1.);Bern=np.array([[(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3] for s in sgrid]);cert={}
 for name,commands in [('tracking',track),('hold',hold)]:
  F=[]
  for interval in range(15):
   vals=[]
   for s in sgrid:
    time=float(t16[interval]+s*(t16[interval+1]-t16[interval]));vals.append(kp.astype(float)*commands[interval].astype(float)+b[:,0]+b[:,1]*sample(time)+b[:,2]*sample(time,1))
   F.append(np.linalg.solve(Bern,np.array(vals)))
  F=np.array(F);saved=get('postC2_real_affine_Bernstein_coefficients',name);error=float(abs(F-saved).max());excess=float(np.maximum(np.maximum(capslo-F,F-capshi),0).max());cert[name]={'independent_coefficient_error_Nm':error,'real_affine_Bernstein_excess_Nm':excess,'min_cap_margin_Nm':float(np.minimum(F-capslo,capshi-F).min())}
  check(name+' finalU32 real Bernstein sufficientcaps',error<1e-10 and excess==0,cert[name])
 Q61=r['q61_rad'].astype(np.float32);V61=r['qd61_rad_s'].astype(np.float32);idx=np.minimum(np.arange(61)//4,14);floatcaps=[]
 for name,commands in [('tracking',track),('hold',hold)]:
  U=commands[idx];aff=kp*U+((b[:,0]+b[:,1]*Q61)+b[:,2]*V61);diff=(kp*(U-Q61)+b[:,0])+b[:,2]*V61
  for grouping,F in [('affine',aff),('difference',diff)]:
   saved=get('FP32_reference61_samples',name+'_rawPD_'+grouping+'61_FP32_Nm');bounds=[model.actuator_forcerange[a].astype(np.float32),model.jnt_actfrcrange[j].astype(np.float32)];ok=all(np.all(F>=cap[:,0]) and np.all(F<=cap[:,1]) for cap in bounds);floatcaps.append(ok)
   check(name+'61 FP32 '+grouping+' bitexact and caps',np.array_equal(F,saved) and ok)
 raw=arrays(pins['rawFD_design_arrays']['path']);R0=Rotation.from_quat(init['object_pose_init_requested_FP32_w'][[4,5,6,3]].astype(float)).as_matrix();body=model.body('object/cube').id;w=raw['raw_omega_world_rad_s'][0];v=raw['raw_v_world_m_s'][0]
 check('requestedinit analyticqd/worldbody convention notmeasured',np.array_equal(init['q_init_requested_FP32_rad'],q0.astype(np.float32)) and np.array_equal(init['object_pose_init_requested_FP32_w'],r['mapped_object_pose_w'][0].astype(np.float32)) and np.array_equal(init['object_angular_requested_FP32_body_rad_s'],(R0.T@w).astype(np.float32)) and np.array_equal(init['object_bodyorigin_v_requested_FP32_world_m_s'],(v-np.cross(w,R0@model.body_ipos[body])).astype(np.float32)) and receipt['init_measured'] is None and receipt['u_applied_measured'] is None)
 depth=r['saved_generated_depth61_m'];check('same saved depth/two thresholds one residual failure',np.array_equal(depth,candidate['operational_generated_maxdepth_m']) and float(depth.max())==receipt['max_saved_generated_overlap_m'] and depth.max()>.003)
 failed=[k for k,v in receipt['gates'].items() if not v];check('only planning+qualification geometry gates fail',set(failed)=={'saved_generated_depth61_planning_le2_999mm','saved_generated_depth61_qualification_le3mm'} and receipt['physical_reference_qualified'] is False and receipt['GPU_authorized'] is False and all(v==0 for v in receipt['forbidden_calls'].values()) and all(v==0 for v in receipt['optimizer_function_calls'].values()))
 out=ROOT/'outputs/constrained_control_independent_review_v1';out.mkdir(parents=True,exist_ok=False);result={'status':'PASS_SAVED_CONTROLLER_RECOMPUTATION_GEOMETRY_UNQUALIFIED' if all(x['pass'] for x in checks) else 'FAIL_SAVED_CONTROLLER_RECOMPUTATION','checks':checks,'source_pin_results':pinrows,'receipt_sha256':sha(base/'receipt.json'),'auditor_sha256':sha(__file__),'qhull_raw_max_excess_rad':hull_excess,'planned14_update_rate_max_rad_s':maxrate,'planned15_clamped_components':clamp,'postC2_real_affine_Bernstein':cert,'max_saved_generated_depth_m':float(depth.max()),'planning_failed_sample_indices':np.flatnonzero(np.any(depth>.002999,axis=1)).tolist(),'qualification3mm_failed_sample_indices':np.flatnonzero(np.any(depth>.003,axis=1)).tolist(),'failed_gates':failed,'numeric_failures_are_same_saved_geometry_at_two_thresholds':True,'LP_calls':0,'optimizer_calls':0,'geometry_queries':0,'physics_steps':0,'GPU_used':False,'production_build_export_calls':0,'physical_reference_qualified':False,'scope':'Independent Cox-deBoor and real-affine Bernstein/FP32 algebra only. No measured init/control/force, dynamic/WM/FMA qualification, or full151-motion success.'}
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[c['name'] for c in checks if not c['pass']],'rate':maxrate,'clamp':clamp,'hull_excess':hull_excess,'Bernstein':cert,'saveddepth':float(depth.max())},indent=2));assert all(x['pass'] for x in checks)
if __name__=='__main__':main()
