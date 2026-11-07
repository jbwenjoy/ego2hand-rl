"""Read-only sequential N1 same-q QREF/preload comparison, no physics or LP."""
from pathlib import Path
import ast,csv,hashlib,json,os,re,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/h123_qref_preload_comparison_v1'
CASES={'QREF':ROOT/'outputs/h123_rest_baseline_v1','PRELOAD':ROOT/'outputs/h123_rest_baseline_preload_v1'}
NEW_AUDIT=ROOT/'scripts/audit_rest_hold_controls_cpu.py'
OLD_AUDIT=Path('D:/workspace/supermimic-observation-grounding-study/scripts/audit_rest_hold_cpu.py')
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
ACTIVE_FN=1e-6
JOINT_TO_PLOT=3 # Prespecified thumb joint4 at initial limit, not selected by result.
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def loadnp(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')
def quatmat(q):
 q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
 return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))
def stats(a):
 a=np.asarray(a,float);return {'max_abs':float(np.max(np.abs(a))),'RMS':float(np.sqrt(np.mean(a*a)))}
def writerows(path,rows):
 with Path(path).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadcase(label,directory):
 p=read(directory.with_name(directory.name+'_watchdog')/'plan_snapshot.json');raw=loadnp(directory/'raw_states.npz');solver=read(directory/'solver_samples.json');runtime=read(directory/'runtime.json');ev=read(directory/'evaluation.json');capacity=read(directory/'capacity_audit.json');context=loadnp(directory/'actual_context_applied.npz');rb=loadnp(directory/'actual_context_readbacks.npz');exp=loadnp(directory/'expected_context_fields.npz')
 assert all(sha(v['path'])==v['sha256'] for v in p['source_pins'].values())
 assert runtime['status']=='completed_engineering_new_rest_diagnostic' and runtime['physics_steps']==60 and ev['complete'] and capacity['status']=='OK' and capacity['overflow_bitmask']==0 and capacity['read_count']==1 and capacity['read_after_step']==60 and capacity['resets_after_initialization']==0
 assert len(solver)==60 and np.array_equal(raw['index'],np.arange(61))
 for f in FIELDS:assert np.array_equal(context[f],exp[f]) and np.array_equal(context[f],rb['after_copy__'+f]) and np.array_equal(context[f],rb['after_set_const_0__'+f])
 m=mujoco.MjModel.from_binary_path(p['model']['path']);obj=m.body('object/cube').id;og=set(np.flatnonzero(m.geom_bodyid==obj));names=read(p['firstframe']['path'])['joint_names'];jids=np.array([m.joint(n).id for n in names]);aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j).item() for j in jids])
 assert np.array_equal(m.actuator_gear[aids],np.tile([1.,0,0,0,0,0],(20,1))) and np.all(m.actuator_trntype[aids]==mujoco.mjtTrn.mjTRN_JOINT)
 assert np.all(m.actuator_gaintype[aids]==mujoco.mjtGain.mjGAIN_FIXED) and np.all(m.actuator_biastype[aids]==mujoco.mjtBias.mjBIAS_AFFINE) and not np.any(m.qpos0[m.jnt_qposadr[jids]])
 rawcontact=[json.loads(l) for l in (directory/'raw_contacts.jsonl').read_text().splitlines() if l.strip()];group={k:[c for c in rawcontact if c['step']==k] for k in range(61)}
 force=np.zeros((60,3));lever=np.zeros((60,3));direct=np.zeros((60,3));count=np.zeros(60,int);active=np.zeros(60,int);fingers=np.zeros((60,5),bool);normal_sum=np.zeros(60);timing=[];capture_errors=[];contact_events=[];t=np.zeros(60)
 for i,s in enumerate(solver):
  k=i+1;t[i]=s['solver_t_s'];assert s['phase']=='just_completed_solver_preintegration' and s['prestate_index']==i and s['poststate_index']==k
  assert t[i]==raw['raw_time_s'][i] and s['poststate_t_s']==raw['raw_time_s'][k] and raw['force_prestate_index'][k]==i
  assert np.array_equal(raw['actuator_force_native'][k],np.array(s['actuator_force_solver_native']))
  fs=[];ts=[];com=np.array(s['solver_object_COM_w'])
  for c in group[k]:
   assert c['phase']==s['phase'] and c['solver_t_s']==t[i] and c['poststate_t_s']==raw['raw_time_s'][k]
   g1,g2=c['geom']
   if not ((g1 in og)^(g2 in og)):continue
   hand=g2 if g1 in og else g1;hn=m.geom(hand).name
   assert hn.startswith('robot/'), 'Unexpected nonhand object contact'
   sign=1. if g2 in og else -1.;frame=np.array(c['frame_rows_w']);cf=np.array(c['force6_contact_frame']);f=sign*frame.T@cf[:3];dtau=sign*frame.T@cf[3:];ltau=np.cross(np.array(c['pos_w'])-com,f)
   force[i]+=f;lever[i]+=ltau;direct[i]+=dtau;count[i]+=1;fs.append(f);ts.append(dtau)
   positive=c['efc_address']>=0 and cf[0]>ACTIVE_FN
   if positive:
    active[i]+=1;normal_sum[i]+=cf[0];match=re.search(r'right_finger([1-5])_',hn)
    if match:fingers[i,int(match[1])-1]=True
   contact_events.append({'condition':label,'step':k,'solver_t_s':float(t[i]),'geom1':int(g1),'geom2':int(g2),'hand_geom':hn,'normal_force_N':float(cf[0]),'active':bool(positive),'Fx_N':float(f[0]),'Fy_N':float(f[1]),'Fz_N':float(f[2]),'COM_Mx_Nm':float(ltau[0]+dtau[0]),'COM_My_Nm':float(ltau[1]+dtau[1]),'COM_Mz_Nm':float(ltau[2]+dtau[2])})
  measured=s['measured_hand_object'];assert len(fs)==len(measured['force_w_N'])
  if fs:capture_errors.extend([np.max(np.abs(np.array(fs)-measured['force_w_N'])),np.max(np.abs(np.array(ts)-measured['torque_at_contact_w_Nm']))])
 assert not capture_errors or max(capture_errors)<1e-12
 kp=context['actuator_gainprm'][aids,0].astype(float);bias=context['actuator_biasprm'][aids].astype(float);limits=m.actuator_forcerange[aids].astype(np.float32).astype(float)
 qpre=raw['joint_rad'][:-1];vpre=raw['joint_velocity_rad_s'][:-1];u=raw['ctrl_rad'][:-1];actual=np.array([s['actuator_force_solver_native'] for s in solver])
 pred=kp*u+bias[:,0]+bias[:,1]*qpre+bias[:,2]*vpre;clipped=np.minimum(np.maximum(pred,limits[:,0]),limits[:,1])
 wrong=kp*u+bias[:,0]+bias[:,1]*raw['joint_rad'][1:]+bias[:,2]*raw['joint_velocity_rad_s'][1:];wrong=np.minimum(np.maximum(wrong,limits[:,0]),limits[:,1])
 k32=kp.astype(np.float32);b32=bias.astype(np.float32);pred32=k32*u.astype(np.float32)+((b32[:,0]+b32[:,1]*qpre.astype(np.float32))+b32[:,2]*vpre.astype(np.float32));pred32=np.minimum(np.maximum(pred32,limits[:,0].astype(np.float32)),limits[:,1].astype(np.float32))
 axis=quatmat(raw['object_pose_w'][:,3:])[:,:,0];angle=np.rad2deg(np.arccos(np.clip(axis@axis[0],-1,1)));position=np.linalg.norm(raw['object_pose_w'][:,:3]-raw['object_pose_w'][0,:3],axis=1);excess=np.maximum(np.maximum(raw['joint_lower_rad']-raw['joint_rad'],raw['joint_rad']-raw['joint_upper_rad']),0)
 gravity=m.opt.gravity.astype(np.float32).astype(float);mass=float(context['body_mass'][obj]);dt=float(np.float32(m.opt.timestep));linear=mass*np.diff(raw['object_twist_native'][:,:3],axis=0)/dt-force-mass*gravity
 return dict(label=label,plan=p,raw=raw,solver=solver,runtime=runtime,evaluation=ev,capacity=capacity,context=context,readbacks=rb,expected=exp,model=m,names=names,aids=aids,t=t,force=force,lever=lever,direct=direct,moment=lever+direct,ncontact=count,nactive=active,activefingers=fingers,normal_sum=normal_sum,actual=actual,pred=clipped,pred_unclipped=pred,pred32=pred32,wrong_post=wrong,position=position,angle=angle,excess=excess,contact_events=contact_events,mass=mass,weight=float(mass*np.linalg.norm(gravity)),linear_balance_residual=linear,force_cap=limits,context_capture_error=max(capture_errors) if capture_errors else 0.)

def main():
 started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
 manifest={'analysis_source':{'path':str(Path(__file__)),'sha256':sha(__file__)},'root_control_auditor':{'path':str(NEW_AUDIT),'sha256':sha(NEW_AUDIT)},'old_auditor':{'path':str(OLD_AUDIT),'sha256':sha(OLD_AUDIT)}}
 for label,directory in CASES.items():
  for f in directory.iterdir():
   if f.is_file():manifest[label+'__'+f.name]={'path':str(f),'sha256':sha(f)}
  watch=directory.with_name(directory.name+'_watchdog')
  for name in ['plan_snapshot.json','state.json']:manifest[label+'__watchdog_'+name]={'path':str(watch/name),'sha256':sha(watch/name)}
 for label,directory in CASES.items():
  plan=read(directory.with_name(directory.name+'_watchdog')/'plan_snapshot.json')
  for key,p in plan['source_pins'].items():manifest[label+'__pin_'+key]=p
 write(OUT/'source_input_pins_before_analysis.json',manifest)
 data={label:loadcase(label,directory) for label,directory in CASES.items()};a,b=data['QREF'],data['PRELOAD'];x,y=a['raw'],b['raw']
 same={key:np.array_equal(x[key][0],y[key][0]) for key in ['joint_rad','full_qpos','full_qvel','object_pose_w','object_twist_native','mocap_pos_w','mocap_quat_wxyz']}
 same.update(model=a['plan']['model']==b['plan']['model'],context_source=a['plan']['context']==b['plan']['context'],worker=a['plan']['source_pins']['worker']==b['plan']['source_pins']['worker'],supervisor=a['plan']['source_pins']['supervisor']==b['plan']['source_pins']['supervisor'],gates=a['plan']['gates']==b['plan']['gates'],raw_time_bitexact=np.array_equal(x['raw_time_s'],y['raw_time_s']))
 same.update({'actual8__'+k:np.array_equal(a['context'][k],b['context'][k]) for k in FIELDS});same.update({'readback__'+k:np.array_equal(a['readbacks'][k],b['readbacks'][k]) for k in a['readbacks']})
 assert all(same.values()),same
 targetdelta=y['fixed_target_FP32'].astype(float)-x['fixed_target_FP32'].astype(float);changed=np.count_nonzero(targetdelta)
 assert changed==19
 design=read(b['plan']['preload']['path']);witness=design['selected_mathematical_witness'];assert sha(b['plan']['preload']['path'])==b['plan']['preload']['sha256']
 assert np.array_equal(np.asarray(design['u0_rad'],np.float32),y['fixed_target_FP32']) and np.array_equal(y['control_proposed_once_FP32'],y['fixed_target_FP32'])
 summaries={};intervals=[];jointrows=[];arrays={};events=[]
 for label,d in data.items():
  z=d['raw'];pd_error=d['actual']-d['pred'];selected=np.array(witness['tau_act_math_Nm']);moment_norm=np.linalg.norm(d['moment'],axis=1)
  summaries[label]={'rest_hold_pass':d['evaluation']['rest_hold_pass'],'object_position_max_mm':float(d['position'].max()*1000),'axis_max_deg':float(d['angle'].max()),'actual_joint_excess_max_mrad':float(d['excess'].max()*1000),'penetration_max_mm':d['evaluation']['max_penetration_m']*1000,'contact_positive_solver_samples':int(np.count_nonzero(d['nactive'])),'no_positive_HO_solver_samples':int(np.count_nonzero(d['nactive']==0)),'maximum_positive_fingers':int(d['activefingers'].sum(axis=1).max()),'peak_HO_Fz_N':float(d['force'][:,2].max()),'peak_HO_Fz_over_weight':float(d['force'][:,2].max()/d['weight']),'peak_COM_moment_norm_Nm':float(moment_norm.max()),'contact_direct_torque_peak_norm_Nm':float(np.linalg.norm(d['direct'],axis=1).max()),'force_impulse_world_Ns':(float(np.float32(.005))*d['force'].sum(axis=0)).tolist(),'initial_solver_actual_actuation_minus_LP_tau_Nm':(d['actual'][0]-selected).tolist(),'actual_actuation_minus_fixed_LP_tau_all_steps':stats(d['actual']-selected),'PD_correct_prestate_clipped_FP64_error_Nm':stats(pd_error),'PD_correct_prestate_clipped_FP32_error_Nm':stats(d['actual']-d['pred32']),'PD_WRONG_poststate_error_Nm':stats(d['actual']-d['wrong_post']),'PD_clipped_components':int(np.count_nonzero(d['pred']!=d['pred_unclipped'])),'linear_balance_residual_N':stats(d['linear_balance_residual']),'q_and_qd_max':{'max_abs_q_change_from_initial_rad':float(np.abs(z['joint_rad']-z['joint_rad'][0]).max()),'max_abs_joint_velocity_rad_s':float(np.abs(z['joint_velocity_rad_s']).max())},'cost':{'worker_wall_s':d['runtime']['wall_seconds'],'watchdog_child_wall_s':read(CASES[label].with_name(CASES[label].name+'_watchdog')/'state.json')['elapsed_s'],'GPU_kernel_busy_s':None}}
  for key in ['t','force','lever','direct','moment','ncontact','nactive','activefingers','normal_sum','actual','pred','pred_unclipped','pred32','wrong_post','position','angle','excess','linear_balance_residual']:arrays[label+'__'+key]=d[key]
  arrays[label+'__joint_q']=z['joint_rad'];arrays[label+'__joint_qd']=z['joint_velocity_rad_s'];events+=d['contact_events']
  for i in range(60):
   intervals.append({'condition':label,'solver_step':i+1,'solver_prestate_s':float(d['t'][i]),'poststate_s':float(z['raw_time_s'][i+1]),'HO_contacts':int(d['ncontact'][i]),'positive_HO_contacts':int(d['nactive'][i]),'positive_fingers':int(d['activefingers'][i].sum()),'Fz_on_object_N':float(d['force'][i,2]),'COM_Mx_Nm':float(d['moment'][i,0]),'COM_My_Nm':float(d['moment'][i,1]),'COM_Mz_Nm':float(d['moment'][i,2]),'direct_torque_norm_Nm':float(np.linalg.norm(d['direct'][i])),'max_PD_prediction_error_Nm':float(np.abs(pd_error[i]).max())})
   for j,n in enumerate(d['names']):jointrows.append({'condition':label,'step':i+1,'solver_prestate_s':float(d['t'][i]),'joint':n,'q_pre_rad':float(z['joint_rad'][i,j]),'qd_pre_rad_s':float(z['joint_velocity_rad_s'][i,j]),'actual_u_rad':float(z['ctrl_rad'][i,j]),'actual_PD_force_Nm':float(d['actual'][i,j]),'predicted_PD_force_prestate_clipped_Nm':float(d['pred'][i,j]),'PD_prediction_error_Nm':float(pd_error[i,j])})
 np.savez_compressed(OUT/'comparison_arrays.npz',**arrays)
 writerows(OUT/'solver_intervals.csv',intervals);writerows(OUT/'joint_PD_timeseries.csv',jointrows);writerows(OUT/'contact_events.csv',events)
 writerows(OUT/'targets.csv',[{'joint':a['names'][j],'QREF_u_rad':float(x['fixed_target_FP32'][j]),'PRELOAD_u_rad':float(y['fixed_target_FP32'][j]),'delta_mrad':float(targetdelta[j]*1000),'changed':bool(targetdelta[j]!=0)} for j in range(20)])
 # The review checks semantic reuse; it neither modifies nor reexecutes the auditor.
 def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 old,new=functions(OLD_AUDIT),functions(NEW_AUDIT);common={k:old[k]==new[k] for k in old.keys()&new.keys()}
 source_review={'status':'READONLY_NO_BLOCKING_FINDING_FOR_THIS_CONTROL_EXTENT','old_source':manifest['old_auditor'],'new_source':manifest['root_control_auditor'],'common_function_AST_equal':common,'actual_expected_proposal_review':'QREF reads hash-bound candidate q32; PRELOAD reads plan-bound design after schema/source/candidate/q64/q32/pose/bounds/FP32 checks. Raw proposal header is compared to that expectation, never used to define it. Clamp count is proposal!=fixed, distinct from fixed!=initialq.','quality_measurement_scope':'quality/contact_reconstruction/capacity/matrix functions checked separately below; no physics or method thresholds changed.','findings':[]}
 for k in ['quality','contact_reconstruction','capacity_class','matrix']:assert common.get(k) is True,(k,common.get(k))
 write(OUT/'root_auditor_source_review.json',source_review)
 plot(data,targetdelta)
 source_unchanged={k:sha(v['path'])==v['sha256'] for k,v in manifest.items()};assert all(source_unchanged.values())
 report={'status':'COMPLETE_READONLY_SAME_Q_SEQUENTIAL_N1_COMPARISON','same_initial_and_physical_inputs':same,'actual_target_components_changed_from_prior_QREF':int(changed),'target_max_abs_delta_mrad':float(np.abs(targetdelta).max()*1000),'target_count_denominators':{'PRELOAD_vs_prior_C2_QREF':int(changed),'PRELOAD_fixed_vs_initial_q32':int(np.count_nonzero(y['fixed_target_FP32']!=y['q_reference0_FP32'])),'QREF_projection_clamped_components':int(np.count_nonzero(x['fixed_target_FP32']!=x['control_proposed_once_FP32'])),'PRELOAD_projection_clamped_components':int(np.count_nonzero(y['fixed_target_FP32']!=y['control_proposed_once_FP32']))},'conditions':summaries,'LP_witness':{'selected_model':design['selected_LP'],'static_normal_force_sum_N':witness['sum_normal_ray_weights_N'],'static_object_contact_wrench_world':witness['object_contact_wrench'],'scope':'Mathematical static allocation and torque at q32/qd0. It does not specify realized contact forces or remain the controller torque after q/qd evolve.'},'timing':'61 integration states0..0.3s; 60 force/contact/actuation rows at prestate0..0.295s. PD uses preintegration q and qvel, actual held ctrl, actual affine gain/bias and actuator cap. Startup sample is not counted twice. No angularEuler balance or unique causal attribution.','source_input_pins_before_analysis':manifest,'source_input_unchanged':source_unchanged,'root_auditor_review':{'path':str(OUT/'root_auditor_source_review.json'),'sha256':sha(OUT/'root_auditor_source_review.json')},'physics_steps':0,'LP_solves':0,'new_GPU_use':False,'Torch_or_Warp_imported':any(n in sys.modules for n in ['torch','warp','mujoco_warp']),'independent_training_seed_N':0,'per_condition_exploratory_N':1,'prospective_randomized_method_comparison':False,'physical_reference_qualified':False,'scope':'PRELOAD proposed after known earlier QREF outcomes; single same-q sequential engineering contrast. No method benefit, general grasp, safety, motion, realego, force-label or training conclusion. Existing quality gates and all old results retained.','CPU_wall_seconds_including_render_excluding_imports':time.perf_counter()-started}
 write(OUT/'report.json',report)
 (OUT/'REPORT.txt').write_text('Same-q sequential exploratory N1 each; no newphysics/LP/GPU.\n'+json.dumps({k:{n:v for n,v in s.items() if n in ['rest_hold_pass','object_position_max_mm','axis_max_deg','actual_joint_excess_max_mrad','penetration_max_mm','PD_correct_prestate_clipped_FP64_error_Nm','PD_WRONG_poststate_error_Nm','maximum_positive_fingers']} for k,s in summaries.items()},indent=2)+'\n19 target changes are versus earlier C2 QREF; preload differs from initialq in20components and itself has0 C2changes.\n',encoding='utf-8')
 print(json.dumps({'status':report['status'],'targets':report['target_count_denominators'],'conditions':{k:{n:v for n,v in s.items() if n in ['peak_HO_Fz_N','peak_COM_moment_norm_Nm','PD_correct_prestate_clipped_FP64_error_Nm','PD_WRONG_poststate_error_Nm','contact_positive_solver_samples']} for k,s in summaries.items()},'report_SHA256':sha(OUT/'report.json')},indent=2))

def plot(data,targetdelta):
 colors={'QREF':'#31688e','PRELOAD':'#b63679'};plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
 fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True);j=JOINT_TO_PLOT
 for label,d in data.items():
  c=colors[label];z=d['raw'];st=z['raw_time_s']*1000;t=d['t']*1000
  axes[0,0].plot(st,1000*d['position'],color=c,label=label);axes[0,1].plot(st,d['angle'],color=c,label=label);axes[0,2].plot(st,1000*d['excess'].max(axis=1),color=c,label=label)
  axes[1,0].plot(t,d['force'][:,2],color=c,label=label);axes[1,1].plot(t,1000*np.linalg.norm(d['moment'],axis=1),color=c,label=label);axes[1,2].step(t,d['activefingers'].sum(axis=1),where='post',color=c,label=label)
  axes[2,0].plot(st,z['joint_rad'][:,j],color=c,label=label);axes[2,1].plot(st,z['joint_velocity_rad_s'][:,j],color=c,label=label);axes[2,2].plot(t,d['actual'][:,j],color=c,label=label+' measured');axes[2,2].plot(t,d['pred'][:,j],color=c,ls='--',lw=1,label=label+' affine(pre)')
 for ax,title,y in zip(axes.flat,['Object position error','Directed shaft error','Maximum actual joint excess','Measured force on object','Measured total COM moment','Positive-force fingers','Thumb joint4 actual position','Thumb joint4 actual velocity','Thumb joint4 actuator force'],['mm','deg','mrad','Fz [N]','norm [mNm]','fingers','q [rad]','qd [rad/s]','Nm']):ax.set(title=title,ylabel=y,xlabel='time [ms]',xlim=(0,300));ax.grid(alpha=.2)
 for ax,value in [(axes[0,0],20),(axes[0,1],20),(axes[0,2],1)]:ax.axhline(value,color='0.5',ls=':',lw=1)
 axes[1,0].axhline(data['QREF']['weight'],color='0.5',ls=':',label='weight');axes[2,0].axhline(data['QREF']['raw']['joint_upper_rad'][j],color='0.5',ls=':',label='hard upper')
 axes[0,0].legend();axes[1,0].legend();axes[2,2].legend(fontsize=7);fig.suptitle('H123 canonical same-q exploratory rest | QREF earlier, PRELOAD later | N=1 each\nStates: 0-300ms; force/PD: solver prestate 0-295ms. Gates unchanged; no method-effect inference.',fontsize=12)
 fig.savefig(OUT/'sameq_rest_comparison.png');fig.savefig(OUT/'sameq_rest_comparison.svg');plt.close(fig)
 fig,ax=plt.subplots(2,3,figsize=(13,9),constrained_layout=True);qrange=max(np.abs(d['raw']['joint_rad']-d['raw']['joint_rad'][0]).max() for d in data.values());vrange=max(np.abs(d['raw']['joint_velocity_rad_s']).max() for d in data.values())
 for col,(label,d) in enumerate(data.items()):
  q=d['raw']['joint_rad']-d['raw']['joint_rad'][0];v=d['raw']['joint_velocity_rad_s'];im=ax[0,col].imshow(q.T,aspect='auto',origin='lower',extent=(0,300,.5,20.5),cmap='RdBu_r',vmin=-qrange,vmax=qrange);im2=ax[1,col].imshow(v.T,aspect='auto',origin='lower',extent=(0,300,.5,20.5),cmap='RdBu_r',vmin=-vrange,vmax=vrange);ax[0,col].set_title(label+' q minus initial [rad]');ax[1,col].set_title(label+' joint velocity [rad/s]');fig.colorbar(im,ax=ax[0,col],shrink=.75);fig.colorbar(im2,ax=ax[1,col],shrink=.75)
 ax[0,2].bar(np.arange(1,21),targetdelta*1000,color='#b63679');ax[0,2].set(title='Actual fixed u: PRELOAD minus QREF',xlabel='joint index (4 per finger)',ylabel='mrad');ax[0,2].axhline(0,color='0.4',lw=.5)
 for label,d in data.items():ax[1,2].plot(d['t']*1000,np.abs(d['actual']-d['pred']).max(axis=1),color=colors[label],label=label+' correct prestate');ax[1,2].plot(d['t']*1000,np.abs(d['actual']-d['wrong_post']).max(axis=1),color=colors[label],ls=':',label=label+' WRONG poststate')
 ax[1,2].set(title='PD prediction timing diagnostic',xlabel='solver prestate [ms]',ylabel='maximum force error [Nm]',yscale='log');ax[1,2].legend(fontsize=7)
 for a in [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]:a.set(xlabel='state time [ms]',ylabel='joint index (4 per finger)')
 fig.suptitle('All20 joint trajectories and control inputs | shared color ranges | descriptive N1 contrast',fontsize=12);fig.savefig(OUT/'all_joint_control_comparison.png');fig.savefig(OUT/'all_joint_control_comparison.svg');plt.close(fig)
if __name__=='__main__':main()
