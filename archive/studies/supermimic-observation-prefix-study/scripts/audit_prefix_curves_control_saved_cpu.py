"""Independent saved-array/actual-PD audit; no production build/export calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
 base=ROOT/'outputs/prefix16_curves_control_export_v1';receipt=read(base/'receipt.json');r=arrays(base/'reference_curves.npz');u=arrays(base/'control_schedule.npz');init=arrays(base/'init_requested.npz')
 refs=receipt['source_pins'];candidate=arrays(refs['candidate']['path']);raw=arrays(refs['rawFD_design_arrays']['path']);m=mujoco.MjModel.from_binary_path(refs['model']['path'])
 checks=[]
 def check(name,ok,details=None):checks.append({'name':name,'pass':bool(ok),'details':details})
 pinrows=[{'name':n,'sha256':p['sha256'],'pass':sha(p['path'])==p['sha256']} for n,p in refs.items()]
 check('all12 source/input pins',len(pinrows)==12 and all(x['pass'] for x in pinrows))
 check('candidate1e67/source16 copied exactly',refs['candidate']['sha256']=='1e67d99b25b6d991fa8e5209f521ea5c269b3212a45b054fe7353ab5b7277ae4' and np.array_equal(r['q_goal_rad'],candidate['q_goal_rad']) and np.array_equal(r['mapped_object_pose_w'],candidate['mapped_object_pose_w']))
 manifests=[]
 for name,pin in receipt['bundles'].items():
  man=read(pin['path']);manifests.append(man)
  check(name+' manifest/artifact/source scope',sha(pin['path'])==pin['sha256'] and sha(man['artifact']['path'])==man['artifact']['sha256'] and man['candidate']==refs['candidate'] and man['source_pins']==refs and man['source_identity']['primary_pose_source_frame_range_half_open']==[200,216] and man['source_identity']['rawFD_derivative_support_source_range_half_open']==[200,218] and man['physical_reference_qualified'] is False and man['init_measured'] is None and man['training_authorized'] is False and man['GPU_authorized'] is False)
 q=r['q_goal_rad'];h=.02;delta=np.diff(q,axis=0);second=np.diff(delta,axis=0)
 qd=np.vstack(((3*delta[0]-delta[1])/(2*h),(delta[1:]+delta[:-1])/(2*h),(3*delta[-1]-delta[-2])/(2*h)))
 qdd=np.vstack(((2*second[0]-second[1])/h**2,second/h**2,(2*second[-1]-second[-2])/h**2))
 native=np.vstack((delta/h,delta[-1]/h));q32=q.astype(np.float32);d32=np.diff(q32,axis=0)/np.float32(h);native32=np.vstack((d32,d32[-1]))
 check('declared own16 FD stencils/native fields separate',np.array_equal(qd,r['q_stencil_knot_qd_rad_s']) and np.array_equal(qdd,r['q_stencil_knot_qdd_rad_s2']) and np.array_equal(native,r['native_FD_goal_qd_rad_s']) and np.array_equal(native32,r['native_FD_goal_qd_FP32_rad_s']))
 ids=np.array([m.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)]);aid=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j).item() for j in ids]);lo,hi=m.jnt_range[ids].T
 check('all16 raw q knots hardlegal',np.all(q>=lo) and np.all(q<=hi))
 # A separate polynomial-root oracle uses descending coefficients / np.roots.
 extrema=[];worst={'hard_excess_rad':0.};coeff=r['q_quintic_coefficients']
 for segment in range(15):
  for joint in range(20):
   c=coeff[segment,:,joint];roots=np.roots((np.arange(1,6)*c[1:])[::-1]);points=[0.,1.]+[float(z.real) for z in roots if abs(z.imag)<1e-10*(1+abs(z.real)) and 0<z.real<1]
   vals=np.polynomial.polynomial.polyval(points,c)
   for s,v in zip(points,vals):
    excess=max(float(lo[joint]-v),float(v-hi[joint]),0.)
    if excess>worst['hard_excess_rad']:worst={'hard_excess_rad':excess,'joint_index':joint,'joint_name':m.joint(ids[joint]).name,'segment':segment,'local_s':s,'time_s':float(segment*h+s*h),'q_rad':float(v),'side':'lower' if v<lo[joint] else 'upper'}
 lower0=abs(q[0]-lo)<1e-12;upper0=abs(q[0]-hi)<1e-12;outward=np.flatnonzero((lower0&(qd[0]<0))|(upper0&(qd[0]>0))).tolist()
 exported_range=read(base/'whole_q_range.json')
 check('whole quintic peak independently reproduced',abs(worst['hard_excess_rad']-exported_range['max_hard_excess_rad'])<1e-12 and outward==exported_range['initial_outward_boundary_joint_indices']==[3] and not exported_range['passed'])
 # Evaluate the stored polynomial directly at all61 ticks, without PrefixCurves.
 qt=[];vt=[]
 for time in np.arange(61)*.005:
  i=min(int(np.searchsorted(r['time_s'],time,side='right')-1),14);dt=r['time_s'][i+1]-r['time_s'][i];s=(time-r['time_s'][i])/dt
  qt.append(np.polynomial.polynomial.polyval(s,coeff[i]));vt.append(np.polynomial.polynomial.polyval(s,coeff[i,1:]*np.arange(1,6)[:,None])/dt)
 qt=np.array(qt);vt=np.array(vt)
 check('stored61 q/qd samples match polynomial',np.max(abs(qt-r['tick61__q_goal_rad']))<1e-12 and np.max(abs(vt-r['tick61__q_curve_qd_rad_s']))<1e-10)
 check('object rawFD first16 not recomputed',np.array_equal(r['mapped_object_pose_w'],raw['object_pose_mapped_w'][:16]) and all(np.max(abs(r[a]-raw[b][:16]))<1e-10 for a,b in [('knot16__object_v_world_m_s','raw_v_world_m_s'),('knot16__object_a_world_m_s2','raw_a_world_m_s2'),('knot16__object_omega_world_rad_s','raw_omega_world_rad_s'),('knot16__object_alpha_world_rad_s2','raw_alpha_world_rad_s2')]))
 with np.load(refs['actual_context']['path'],allow_pickle=False) as z:kp=z['actuator_gainprm'][0,aid,0].astype(np.float32);bias=z['actuator_biasprm'][0,aid,:3].astype(np.float32)
 outerlo=np.maximum(lo,m.actuator_ctrlrange[aid,0]).astype(np.float32);outerhi=np.minimum(hi,m.actuator_ctrlrange[aid,1]).astype(np.float32)
 preload=read(refs['static_preload_design']['path']);PDU=np.array(preload['u0_rad'],dtype=np.float32)
 tau=kp*PDU+(bias[:,0]+bias[:,1]*q[0].astype(np.float32));qk=r['knot16__q_goal_rad'].astype(np.float32);vk=r['knot16__q_curve_qd_rad_s'].astype(np.float32)
 proposal=(((tau-bias[:,0])-bias[:,1]*qk)-bias[:,2]*vk)/kp
 bounded=np.minimum(np.maximum(proposal,outerlo+np.float32(.02)),outerhi-np.float32(.02))
 schedule=u['tracking_u_exec_15_FP32_rad'];hold=u['hold_u_exec_15_FP32_rad'];rates=np.diff(schedule.astype(np.float64),axis=0)/.02;rate=float(abs(rates).max());clamp=int(np.count_nonzero(proposal[:15]!=schedule))
 check('actual context staticprior/proposed/C2 exact',np.array_equal(tau,u['tau_ff_static0_FP32_Nm']) and np.array_equal(proposal,u['u_pre_16_FP32_rad']) and np.array_equal(bounded,u['u_exec_16_diagnostic_FP32_rad']) and np.array_equal(bounded[:15],schedule))
 check('15 designed commands/sameu0/14 rate/clamp22',schedule.shape==(15,20) and hold.shape==(15,20) and np.array_equal(hold,np.repeat(schedule[:1],15,axis=0)) and np.array_equal(init['u_executed0_FP32_rad'],schedule[0]) and np.array_equal(rates,u['actual_target_rate_14_rad_s']) and rate==receipt['control_diagnostics']['max_actual_target_rate_rad_s'] and clamp==receipt['control_diagnostics']['clamped_components_actual15']==22)
 check('C2 bounds pass but planned rate fails15',np.all(schedule>=outerlo+np.float32(.02)) and np.all(schedule<=outerhi-np.float32(.02)) and rate>15)
 qf=r['tick61__q_goal_rad'][:60].astype(np.float32);vf=r['tick61__q_curve_qd_rad_s'][:60].astype(np.float32);which=np.arange(60)//4
 force_results={}
 for name,targets in [('track',schedule[which]),('hold',hold[which])]:
  affine=kp*targets+((bias[:,0]+bias[:,1]*qf)+bias[:,2]*vf);difference=(kp*(targets-qf)+bias[:,0])+bias[:,2]*vf
  prefix='reference_PD' if name=='track' else 'hold_reference_PD'
  check(name+' raw PD affine/difference predictions exact',np.array_equal(affine,u[prefix+'_affine_60_FP32_Nm']) and np.array_equal(difference,u[prefix+'_difference_60_FP32_Nm']))
  caps={}
  for capname,cap in [('actuator',m.actuator_forcerange[aid].astype(np.float32)),('joint',m.jnt_actfrcrange[ids].astype(np.float32))]:
   caps[capname]={}
   for grouping,F in [('affine',affine),('difference',difference)]:
    excess=np.maximum(np.maximum(cap[:,0]-F,F-cap[:,1]),0);caps[capname][grouping]={'max_excess_Nm':float(excess.max()),'violating_components':int(np.count_nonzero(excess))}
  force_results[name]=caps
  check(name+' reference PD cap failure retained',any(v['max_excess_Nm']>0 for a in caps.values() for v in a.values()))
 initR=Rotation.from_quat(init['object_pose_init_requested_FP32_w'][[4,5,6,3]].astype(float)).as_matrix();bodyw=(initR.T@init['object_angular_requested_FP32_world_rad_s'].astype(float)).astype(np.float32)
 check('designed init separate uses continuous qd not nativeFD',np.array_equal(init['q_init_requested_FP32_rad'],q[0].astype(np.float32)) and np.array_equal(init['qd_init_requested_FP32_rad_s'],r['q_curve_qd_rad_s'][0].astype(np.float32)) and not np.array_equal(init['qd_init_requested_FP32_rad_s'],native32[0]) and np.max(abs(bodyw-init['object_angular_requested_FP32_body_rad_s']))<1e-6 and receipt['new_initialization_measured'] is None)
 failed=[k for k,v in receipt['gates'].items() if not v]
 check('exact four numeric gates fail unchanged',set(failed)=={'whole_q_curve_hardrange','actual14_updates_rate_le15','actual60_ref_PD_predictions_within_both_caps','hold60_ref_PD_predictions_within_both_caps'} and receipt['status']=='CPU_EXPORT_FAILED_GATES_RETAINED_NO_TUNING' and receipt['physical_reference_qualified'] is False and receipt['GPU_authorized'] is False)
 out=ROOT/'outputs/prefix16_curves_control_independent_review_v1';out.mkdir(parents=True,exist_ok=False)
 result={'status':'PASS_INDEPENDENT_RECOMPUTATION_FOUR_PRE_GATES_FAIL' if all(x['pass'] for x in checks) else 'FAIL_INDEPENDENT_RECOMPUTATION',
  'checks':checks,'receipt_sha256':sha(base/'receipt.json'),'source_input_pins':pinrows,'auditor_sha256':sha(__file__),
  'whole_curve_worst':worst,'outward_initial_boundary_joint_indices':outward,'planned14_update_rate_max_rad_s':rate,'planned15_clamped_components':clamp,
  'reference_predicted_PD_excess_not_measured_saturation':force_results,'failed_numeric_gates':failed,
  'native_vs_continuous_initial_qd_max_abs_difference_rad_s':float(np.max(abs(native32[0]-init['qd_init_requested_FP32_rad_s']))),
  'scope':'Only saved NPZ/polynomial/actual-model-context algebra. No production build/export functions called, no optimizer or geometry/physics/GPU query. Designed executed-target fields are not measured controls; raw PD exceeds caps only in prediction.',
  'physical_reference_qualified':False,'init_measured':None,'new_physics_steps':0,'new_optimizer_calls':0,'new_geometry_queries':0,'GPU_used':False}
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[c['name'] for c in checks if not c['pass']],'whole_curve_worst':worst,'rate':rate,'clamp':clamp,'force_excess':force_results},indent=2));assert all(x['pass'] for x in checks)
if __name__=='__main__':main()
