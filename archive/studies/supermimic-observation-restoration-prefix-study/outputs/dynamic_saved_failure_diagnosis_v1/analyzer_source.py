"""Saved H0/T0 failure diagnosis only; no geometry, solver or physics calls."""
from pathlib import Path
import argparse,csv,hashlib,json,os,re
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':Path(p).resolve().as_posix(),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rotations(pose):return Rotation.from_quat(np.asarray(pose)[:,[4,5,6,3]]).as_matrix()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))


def main(output):
    output.mkdir(parents=True,exist_ok=False);base=ROOT/'outputs';export=base/'restored_prefix_control_export_v1';ref=load(export/'reference_bspline.npz');ctl=load(export/'planned_control.npz');proto=read(ROOT/'protocols/restoration-filter-prefix-v1.json')
    import mujoco
    forbidden={k:0 for k in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce','mj_kinematics')};original={k:getattr(mujoco,k) for k in forbidden}
    def deny(k):
        def fail(*a,**kw):forbidden[k]+=1;raise AssertionError('Saved-only diagnosis forbids '+k)
        return fail
    for k in forbidden:setattr(mujoco,k,deny(k))
    try:
        model=mujoco.MjModel.from_binary_path(proto['inputs']['model']['path']);obj=model.body('object/cube').id;og=set(np.flatnonzero(model.geom_bodyid==obj).tolist());hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])};jids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)]);names=[model.joint(int(j)).name for j in jids];oj=model.joint('object/cube_freejoint').id;ov=int(model.jnt_dofadr[oj]);g=model.opt.gravity.astype(np.float32).astype(float)
        Rref=ref['object_rotation_w'];nref=Rref[:,:,0];R0=Rref[0];checks={};case={};allarrays={};jointrows=[];steprows=[]
        for label in ('hold','track'):
            folder=base/f'dynamic_matched_{label}_v1';raw=load(folder/'raw_states.npz');request=load(folder/'requested_initialization_and_control.npz');context=load(folder/'actual_context_applied.npz');solver=read(folder/'solver_samples.json');contacts=[json.loads(s) for s in (folder/'raw_contacts.jsonl').read_text().splitlines()];evaluation=read(folder/'evaluation.json');capacity=read(folder/'capacity_audit.json')
            aids=request['position_ctrl_ids'];kp=context['actuator_gainprm'][aids,0].astype(float);b=context['actuator_biasprm'][aids,:3].astype(float);u=np.array([s['ctrl_solver_native'] for s in solver]);measured=np.array([s['actuator_force_solver_native'] for s in solver]);preq=raw['joint_rad'][:60];prev=raw['joint_velocity_rad_s'][:60];dt=np.diff(raw['raw_time_s']);times=raw['raw_time_s'];pretime=np.array([s['solver_t_s'] for s in solver]);R=rotations(raw['object_pose_w']);shaft=R[:,:,0]
            shaft_error=np.rad2deg(np.arccos(np.clip(np.sum(shaft*nref,axis=1),-1,1)));so3=np.rad2deg(np.arccos(np.clip((np.trace(Rref.transpose(0,2,1)@R,axis1=1,axis2=2)-1)/2,-1,1)));position_error=np.linalg.norm(raw['object_pose_w'][:,:3]-ref['object_COM_world_m'],axis=1);qerr=raw['joint_rad']-ref['q61_rad'];verr=raw['joint_velocity_rad_s']-ref['qd61_rad_s']
            actual_raw=kp*u+b[:,0]+b[:,1]*preq+b[:,2]*prev;nominal_raw=kp*u+b[:,0]+b[:,1]*ref['q61_rad'][:60]+b[:,2]*ref['qd61_rad_s'][:60];P_error=b[:,1]*qerr[:60];D_error=b[:,2]*verr[:60]
            actlo=model.actuator_forcerange[aids,0].astype(np.float32).astype(float);acthi=model.actuator_forcerange[aids,1].astype(np.float32).astype(float);expected_clipped=np.minimum(np.maximum(actual_raw,actlo),acthi);sat=(actual_raw<actlo)|(actual_raw>acthi);unsat=~sat
            F=np.zeros((60,3));T=np.zeros((60,3));directT=np.zeros((60,3));byfinger=np.zeros((60,5,6));normal_byfinger=np.zeros((60,5));generated=np.zeros(60,int);bearing=np.zeros(60,int);byfinger_count=np.zeros((60,5),int);env=np.zeros(60,int);helper_error=0.;object_pose_phase_error=0.
            for k in range(1,61):
                ss=solver[k-1];com=np.asarray(ss['solver_object_COM_w']);point_pose=np.asarray(ss['solver_object_body_pose_w']);Rs=rotations(point_pose[None])[0];helperF=np.asarray(ss['measured_hand_object']['force_w_N']);helperD=np.asarray(ss['measured_hand_object']['torque_at_contact_w_Nm']);helperP=np.asarray(ss['measured_hand_object']['pos_o_m']);helper_comtau=np.zeros(3)
                if helperF.size:helper_comtau=np.cross((Rs@helperP.T).T,helperF).sum(0)+helperD.sum(0)
                for c in (c for c in contacts if c['step']==k):
                    a,bg=c['geom'];is_obj=a in og or bg in og
                    if not is_obj:continue
                    other=bg if a in og else a;is_hand=other in hand
                    if not is_hand:env[k-1]+=1
                    sign=1. if bg in og else -1.;frame=np.asarray(c['frame_rows_w']);f6=np.asarray(c['force6_contact_frame']);force=sign*frame.T@f6[:3];direct=sign*frame.T@f6[3:];torque=np.cross(np.asarray(c['pos_w'])-com,force)+direct
                    F[k-1]+=force;T[k-1]+=torque;directT[k-1]+=direct
                    if is_hand:
                        generated[k-1]+=1;bearing[k-1]+=int(f6[0]>1e-8)
                        match=re.search(r'right_finger([1-5])_',model.geom(other).name)
                        if match:
                            finger=int(match.group(1))-1;byfinger[k-1,finger,:3]+=force;byfinger[k-1,finger,3:]+=torque;normal_byfinger[k-1,finger]+=f6[0];byfinger_count[k-1,finger]+=1
                if helperF.size:helper_error=max(helper_error,float(np.max(np.abs(F[k-1]-helperF.sum(0)))),float(np.max(np.abs(T[k-1]-helper_comtau))))
                object_pose_phase_error=max(object_pose_phase_error,float(np.max(np.abs(point_pose-raw['object_pose_w'][k-1]))))
            mass=float(context['body_mass'][obj]);Ri=Rotation.from_quat(context['body_iquat'][obj][[1,2,3,0]]).as_matrix();Ibody=Ri@np.diag(context['body_inertia'][obj])@Ri.T;Iwref=Rref[:60]@Ibody@Rref[:60].transpose(0,2,1);wr=ref['object_omega_world_rad_s'][:60];ar=ref['object_alpha_world_rad_s2'][:60];requiredF=mass*(ref['object_a_world_m_s2'][:60]-g);requiredT=np.einsum('tij,tj->ti',Iwref,ar)+np.cross(wr,np.einsum('tij,tj->ti',Iwref,wr))
            native_twist=raw['object_twist_native'];worldomega=np.einsum('tij,tj->ti',R,native_twist[:,3:]);linear_balance=mass*((native_twist[1:,:3]-native_twist[:-1,:3])/dt[:,None]-g);alpha_body=(native_twist[1:,3:]-native_twist[:-1,3:])/dt[:,None];body_torque=(Ibody@alpha_body.T).T+np.cross(native_twist[:-1,3:],(Ibody@native_twist[:-1,3:].T).T);angular_balance=np.einsum('tij,tj->ti',R[:60],body_torque)
            initialframe_impulse=np.cumsum(T*dt[:,None],axis=0)@R0;required_impulse=np.cumsum(requiredT*dt[:,None],axis=0)@R0;shaft_local=shaft@R0
            actforce_err=expected_clipped-measured;indices=np.argwhere(sat);sat_details=[{'solver_step':int(i+1),'solver_pre_time_s':float(pretime[i]),'joint_index':int(j),'joint':names[j],'unclamped_PD_model_Nm':float(actual_raw[i,j]),'reference_state_PD_model_Nm':float(nominal_raw[i,j]),'measured_postcap_force_Nm':float(measured[i,j]),'cap_FP32_lower_Nm':float(actlo[j]),'cap_FP32_upper_Nm':float(acthi[j]),'P_tracking_error_torque_Nm':float(P_error[i,j]),'D_tracking_error_torque_Nm':float(D_error[i,j])} for i,j in indices]
            firstfail=np.flatnonzero(shaft_error>20);primaryfail=np.arange(0,61,4)[shaft_error[::4]>20];global_q=np.unravel_index(np.abs(qerr).argmax(),qerr.shape);global_v=np.unravel_index(np.abs(verr).argmax(),verr.shape)
            prefixes=checks
            checks[label+'_complete_capacity_original_gates']=evaluation['complete'] and evaluation['physics_inference_qualified'] and capacity['status']=='OK' and capacity['overflow_bitmask']==0 and [k for k,v in evaluation['gates'].items() if not v]==['moving_shaft_angle_16']
            checks[label+'_force_step_prestate_clock_control_phase']=all(s['step']==i+1 and s['prestate_index']==i and s['poststate_index']==i+1 and s['control_interval_used']==i//4 and abs(s['solver_t_s']-times[i])<1e-12 and abs(s['poststate_t_s']-times[i+1])<1e-12 for i,s in enumerate(solver))
            checks[label+'_measured_force_rows_align_poststate']=np.array_equal(raw['actuator_force_native'][1:],measured)
            checks[label+'_ctrl_uses_solver_new_update_not_stale_prestate']=np.array_equal(u,raw['selected_planned15_FP32_rad'][np.arange(60)//4])
            checks[label+'_PD_model_error_decomposition_algebra']=float(np.max(np.abs(actual_raw-nominal_raw-P_error-D_error)))<1e-14
            checks[label+'_allobject_contacts_hand_no_env']=not np.any(env) and np.all(generated>0)
            checks[label+'_rawwrench_matches_logged_helper']=helper_error<1e-7
            checks[label+'_shaft_primary_and_fullSO3_metrics_match']=abs(shaft_error[::4].max()-evaluation['max_shaft_angle_16_deg'])<1e-10 and abs(so3[::4].max()-evaluation['max_SO3_error_16_deg'])<1e-10 and abs(position_error[::4].max()-evaluation['max_object_position_error_16_m'])<1e-12
            checks[label+'_zeroCOMoffset_for_linear_momentum_diagnostic']=not np.any(model.body_ipos[obj])
            case[label]={'inputs':{name:pin(folder/name) for name in ('raw_states.npz','raw_contacts.jsonl','solver_samples.json','requested_initialization_and_control.npz','actual_context_applied.npz','evaluation.json','capacity_audit.json','control_events.jsonl','runtime.json')},
                'original_gate':{'short_dynamic_pass':evaluation['short_dynamic_pass'],'failed_gates':[k for k,v in evaluation['gates'].items() if not v],'max_axis_primary_deg':float(shaft_error[::4].max()),'max_SO3_diagnostic_deg':float(so3[::4].max()),'max_position_primary_m':float(position_error[::4].max()),'first_axis_fail61_s':None if not len(firstfail) else float(times[firstfail[0]]),'first_axis_fail16_s':None if not len(primaryfail) else float(times[primaryfail[0]])},
                'q_tracking':{'position_rms_rad':rms(qerr),'position_abs_max_rad':float(np.abs(qerr).max()),'position_max_state_joint':[int(a) for a in global_q],'velocity_rms_rad_s':rms(verr),'velocity_abs_max_rad_s':float(np.abs(verr).max()),'velocity_max_state_joint':[int(a) for a in global_v],'terminal_position_abs_max_rad':float(np.abs(qerr[-1]).max())},
                'PD_diagnostic':{'predicted_unclamped_abs_max_Nm':float(np.abs(actual_raw).max()),'measured_postcap_abs_max_Nm':float(np.abs(measured).max()),'reference_state_raw_abs_max_Nm':float(np.abs(nominal_raw).max()),'predicted_saturated_joint_steps':int(sat.sum()),'total_joint_steps':1200,'saturation_details':sat_details,'clip_model_vs_measured_max_Nm':float(np.abs(actforce_err).max()),'unclipped_model_vs_measured_unsaturated_max_Nm':float(np.abs((actual_raw-measured)[unsat]).max()),'raw_PD_is_model_not_saved_preclip_measurement':True,'classification':'unclamped real-arithmetic actual-state PD outside the expected FP32 individual actuator cap; boundaries/roundoff retained, no new acceptance threshold'},
                'object_motion':{'actual_final_shaft_vs_initial_deg':float(np.rad2deg(np.arccos(np.clip(shaft[-1]@shaft[0],-1,1)))),'desired_final_shaft_vs_initial_deg':float(np.rad2deg(np.arccos(np.clip(nref[-1]@nref[0],-1,1)))),'actual_final_shaft_initialframe':shaft_local[-1].tolist(),'desired_final_shaft_initialframe':(nref[-1]@R0).tolist(),'actual_world_omega_end_rad_s':worldomega[-1].tolist(),'desired_world_omega_end_rad_s':ref['object_omega_world_rad_s'][-1].tolist()},
                'contact_wrench':{'generated_hand_object_perstep_minmax':[int(generated.min()),int(generated.max())],'forcebearing_fn_gt1e8_perstep_minmax':[int(bearing.min()),int(bearing.max())],'forcebearing_threshold_N':1e-8,'zero_total_contact_force_steps':(np.flatnonzero(np.linalg.norm(F,axis=1)==0)+1).tolist(),'force_mean_w_N':F.mean(0).tolist(),'gravity_support_mg_N':float(-mass*g[2]),'torque_mean_w_Nm':T.mean(0).tolist(),'reference_required_torque_mean_w_Nm':requiredT.mean(0).tolist(),'torque_impulse_initialframe_Nms':initialframe_impulse[-1].tolist(),'reference_required_impulse_initialframe_Nms':required_impulse[-1].tolist(),'raw_vs_helper_wrench_max_component_error':helper_error,'solver_body_pose_vs_raw_prepose_max_component_error':object_pose_phase_error,'reference_wrench_comparison_scope':'Actual wrench is about actual solver COM; design demand is about reference COM with reference inertia/orientation. Both world components; this difference is descriptive, not a contact-force tracking objective or cause proof.'},
                'one_step_momentum_diagnostic':{'linear_balance_force_rms_N':rms(F-linear_balance),'linear_balance_force_abs_max_N':float(np.abs(F-linear_balance).max()),'bodyEuler_torque_rms_Nm':rms(T-angular_balance),'bodyEuler_torque_abs_max_Nm':float(np.abs(T-angular_balance).max()),'scope':'Saved velocity finite differences and known mass/inertia/gravity, no native qacc recomputation. Discrete integration/FP32 error remains; not force or solver-accuracy qualification.'}}
            for j,name in enumerate(names):jointrows.append({'condition':label,'joint':name,'q_error_rms_rad':rms(qerr[:,j]),'q_error_abs_max_rad':float(np.abs(qerr[:,j]).max()),'qd_error_rms_rad_s':rms(verr[:,j]),'qd_error_abs_max_rad_s':float(np.abs(verr[:,j]).max()),'raw_PD_exceeds_individual_cap_steps':int(sat[:,j].sum()),'raw_PD_abs_max_Nm':float(np.abs(actual_raw[:,j]).max()),'postcap_measured_abs_max_Nm':float(np.abs(measured[:,j]).max()),'reference_PD_abs_max_Nm':float(np.abs(nominal_raw[:,j]).max())})
            for i in range(60):steprows.append({'condition':label,'step':i+1,'solver_pre_time_s':float(pretime[i]),'poststate_time_s':float(times[i+1]),'axis_error_at_post_deg':float(shaft_error[i+1]),'position_error_post_m':float(position_error[i+1]),'q_rms_pre_rad':rms(qerr[i]),'qd_rms_pre_rad_s':rms(verr[i]),'raw_PD_cap_exceed_jointcount':int(sat[i].sum()),'HO_generated_contacts':int(generated[i]),'HO_forcebearing_contacts':int(bearing[i]),**{f'contact_F_{a}_N':float(F[i,j]) for j,a in enumerate('xyz')},**{f'contact_tau_{a}_Nm':float(T[i,j]) for j,a in enumerate('xyz')}})
            arrays={'time_s':times,'solver_pre_time_s':pretime,'axis_error_deg':shaft_error,'SO3_error_deg':so3,'position_error_m':position_error,'q_error_rad':qerr,'qd_error_rad_s':verr,'actual_raw_PD_model_Nm':actual_raw,'reference_state_raw_PD_model_Nm':nominal_raw,'measured_postcap_actuator_Nm':measured,'actual_raw_PD_clip_prediction_Nm':expected_clipped,'P_error_torque_Nm':P_error,'D_error_torque_Nm':D_error,'predicted_saturation_mask':sat,'contact_force_w_N':F,'contact_torque_COM_w_Nm':T,'contact_direct_torque_w_Nm':directT,'contact_wrench_byfinger':byfinger,'contact_normal_force_byfinger_N':normal_byfinger,'generated_byfinger':byfinger_count,'generated_HO_count':generated,'forcebearing_HO_count':bearing,'reference_required_F_w_N':requiredF,'reference_required_tau_COM_w_Nm':requiredT,'discrete_linear_balance_F_w_N':linear_balance,'discrete_bodyEuler_tau_w_Nm':angular_balance,'shaft_in_initialframe':shaft_local,'contact_angular_impulse_initialframe_Nms':initialframe_impulse,'reference_angular_impulse_initialframe_Nms':required_impulse,'world_omega_actual_rad_s':worldomega}
            allarrays.update({label+'__'+k:v for k,v in arrays.items()});case[label]['_raw']=raw
        H,T=case['hold']['_raw'],case['track']['_raw'];matched={k:bool(np.array_equal(H[k][:5],T[k][:5])) for k in ('joint_rad','joint_velocity_rad_s','object_pose_w','object_twist_native')};checks['identical_actual_prefix_through20ms_before_update']=all(matched.values());checks['forbidden_queries_zero']=not any(forbidden.values())
        for label in case:case[label].pop('_raw')
        np.savez_compressed(output/'diagnostic_arrays.npz',**allarrays)
        for filename,rows in [('all40_joint_diagnostics.csv',jointrows),('all120_solver_diagnostics.csv',steprows)]:
            with (output/filename).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary={'status':'SAVED_DYNAMIC_FAILURE_DIAGNOSIS_COMPLETE','checks':{k:bool(v) for k,v in checks.items()},'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'cases':case,'same_actual_prefix0_to20ms':matched,'reference':pin(export/'reference_bspline.npz'),'control':pin(export/'planned_control.npz'),'model':proto['inputs']['model'],'analyzer':pin(__file__),
                 'supported':['Both objects rotate substantially but shaft direction paths depart from the desired path; scalar rotation amplitude alone does not describe failure.','T0 reduces joint-position reference error in these runs, but large joint-velocity discrepancies remain. Actual-state PD predictions explain measured postcap actuator values within smallFP32/FMA residuals.','Only one T0 joint-step demands clipping; H0 none. This is a transient at the first20ms target update, not persistent actuator saturation.','Contact forces persist and mean vertical force is near object weight; net COM moments differ from the reference rigid-body demand and are consistent with saved actual velocity changes at the discrete-step scale.','Actual initial states/prefix through20ms are bitwise matched; subsequent intervention differs. Results remain known-history sequentialN1, not method population performance.'],
                 'not_identified':['No claim that the single clipped step caused the final axis error.','No isolation of contact slip/rolling, anatomical mapping/anchor error, passive finger dynamics, contact stiffness/friction or trajectory feasibility as the unique cause; relative hand-contact velocities and complete hand generalized-force decomposition were not logged.','Reference-state rawforce caps and geometric feasibility do not guarantee actual dynamic tracking; the precomputed target schedule has no object-orientation feedback and transports a static load prior, not a changing dynamic contact-force solution.','No global motion impossibility, WM/contact-prediction success, full151motion or hardware conclusion. No new contact labels or tuned parameters.'],
                 'phase_contract':'Solver sample stepk uses rawstate[k-1] q/qd and solver.ctrl for the new interval. The force, rawcontacts and COM are preintegration solver quantities; rawstate[k] is postintegration. Startup step0 excluded from all60 wrench/impulse sums.',
                 'forbidden_native_calls':forbidden,'geometry_queries':0,'optimizer_calls':0,'physics_steps':0,'GPU_used':False,'original_grades_changed':False}
        write(output/'report.json',summary)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'hold':'#d95f02','track':'#1b9e77'}
        for name in ('hold','track'):
            a={k[len(name)+2:]:v for k,v in allarrays.items() if k.startswith(name+'__')};ax[0,0].plot(a['time_s']*1000,a['axis_error_deg'],label='H0' if name=='hold' else 'T0',color=colors[name]);ax[0,1].plot(a['shaft_in_initialframe'][:,1],a['shaft_in_initialframe'][:,2],label=name,color=colors[name]);ax[0,1].scatter(a['shaft_in_initialframe'][-1,1],a['shaft_in_initialframe'][-1,2],color=colors[name]);ax[1,1].plot((a['solver_pre_time_s']+.005)*1000,a['contact_angular_impulse_initialframe_Nms'][:,2]*1000,label=name,color=colors[name])
        nr=nref@R0;ax[0,1].plot(nr[:,1],nr[:,2],color='#7570b3',label='reference');ax[0,1].scatter(nr[-1,1],nr[-1,2],color='#7570b3');ax[0,1].scatter([0],[0],color='black',s=16,label='common initial shaft')
        ax[0,0].axhline(20,color='0.4',ls=':',label='20° primary gate');ax[0,0].set(title='Moving directed-shaft error; original FAIL retained',xlabel='Poststate time [ms]',ylabel='deg');ax[0,0].legend(fontsize=8)
        ax[0,1].set(title='Shaft direction path in initial object frame',xlabel='signed y component',ylabel='signed z component');ax[0,1].legend(fontsize=8)
        t=allarrays['track__solver_pre_time_s']*1000;j=10
        for key,label,ls in [('actual_raw_PD_model_Nm','actual-state unclamped PD','-'),('measured_postcap_actuator_Nm','measured postcap force','--'),('reference_state_raw_PD_model_Nm','reference-state prediction',':')]:ax[1,0].plot(t,allarrays['track__'+key][:,j],label=label,ls=ls)
        ax[1,0].axhline(-.25095,color='0.5',lw=.8,ls=':');ax[1,0].set(title='T0 finger3_joint3: one transient clip at20ms',xlabel='Solver prestate time [ms]',ylabel='Nm');ax[1,0].legend(fontsize=7)
        ax[1,1].plot((allarrays['track__solver_pre_time_s']+.005)*1000,allarrays['track__reference_angular_impulse_initialframe_Nms'][:,2]*1000,label='reference demand',color='#7570b3');ax[1,1].set(title='COM angular impulse about initial-frame z',xlabel='Step end time [ms]',ylabel='mNm·s');ax[1,1].legend(fontsize=8)
        for a in ax.flat:a.grid(alpha=.2)
        fig.suptitle('Saved matched short dynamic probes: direction error, not absence of rotation\nForces use preintegration state/COM; no new query, simulation or causal isolation',fontsize=12)
        for suffix in ('png','svg'):fig.savefig(output/f'dynamic_failure_diagnostics.{suffix}',dpi=160)
        plt.close(fig)
        print(json.dumps({'status':summary['status'],'passed':summary['passed'],'total':summary['total'],'failed':[k for k,v in checks.items() if not v],'cases':{k:{'q':v['q_tracking'],'PD':v['PD_diagnostic'],'motion':v['object_motion'],'balance':v['one_step_momentum_diagnostic']} for k,v in case.items()}},indent=2));return 0 if all(checks.values()) else 1
    finally:
        for k,v in original.items():setattr(mujoco,k,v)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/dynamic_saved_failure_diagnosis_v1');args=ap.parse_args();raise SystemExit(main(args.output))
