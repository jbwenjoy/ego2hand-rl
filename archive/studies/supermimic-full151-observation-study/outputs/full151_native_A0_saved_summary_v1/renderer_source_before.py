"""Saved full151 native A0 science figure. No Env/FK/Reader/WM/device calls."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def table(p,columns,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)


def rotation(q):
    q=np.asarray(q,float)
    if q.shape[-1:]!=(4,) or not np.isfinite(q).all() or np.max(np.abs(np.linalg.norm(q,axis=-1)-1))>1e-6:raise ValueError('Malformed saved wxyz quaternion')
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def main(source,out):
    if out.exists():raise FileExistsError('Fresh figure output required')
    inputs={}
    def use(name,file):
        p=source/file;inputs[name]=pin(p);return p
    evaluation=json.loads(use('evaluation','evaluation.json').read_text('utf-8'))
    if evaluation.get('complete') is not True or evaluation.get('physical_steps')!=600 or evaluation.get('control_intervals')!=150:raise ValueError('Expected complete601/600/150 episode, regardless task quality')
    raw=arrays(use('raw','raw_states.npz'));requested=arrays(use('requested_init_control','requested_initialization_and_control.npz'))
    solver=json.loads(use('solver','solver_samples.json').read_text('utf-8'));events=json.loads(use('events','control_events.json').read_text('utf-8'))
    descriptor=json.loads(use('controller_descriptor','controller_descriptor.json').read_text('utf-8'))
    refpin=descriptor['root_descriptor']['source_pins']['reference_arrays'];refpath=Path(refpin['path'])
    if sha(refpath)!=refpin['sha256']:raise ValueError('Frozen reference source changed')
    inputs['reference']=pin(refpath);ref=arrays(refpath)
    checks={}
    if len(solver)!=600 or len(events)!=150 or raw['object_pose_w'].shape!=(601,7) or raw['joint_rad'].shape!=(601,20):raise ValueError('Raw coverage/shape mismatch')
    t=np.arange(601)*.005;primary=np.arange(0,601,4);pre=t[:-1];post=t[1:]
    if not np.array_equal(raw['canonical_time_s'],t) or not np.array_equal(ref['planned_time601_s'],t):raise ValueError('Canonical601 clocks differ')
    if any(s['step']!=k+1 or s['prestate_index']!=k or s['poststate_index']!=k+1 for k,s in enumerate(solver)):raise ValueError('Solver phase/index mismatch')
    pos=np.linalg.norm(raw['object_pose_w'][:,:3]-ref['object_COM_world_m'],axis=1)
    R=rotation(raw['object_pose_w'][:,3:]);shaft=np.rad2deg(np.arccos(np.clip(np.sum(R[:,:,0]*ref['object_rotation_w'][:,:,0],axis=1),-1,1)))
    qexcess=np.maximum(np.maximum(raw['joint_lower_rad']-raw['joint_rad'],raw['joint_rad']-raw['joint_upper_rad']),0)
    penetration=np.array([[row['penet_category_m'][name] for name in ('hand_object','hand_self','other')] for row in solver],float)
    forces=np.array([row['actuator_force_solver_native'] for row in solver],float)
    lo,hi=raw['force_lower_Nm'],raw['force_upper_Nm']
    if forces.shape!=(600,20) or penetration.shape!=(600,3) or np.any(lo>=0) or np.any(hi<=0):raise ValueError('Bad raw force/category shape or caps')
    ratio=np.where(forces>=0,forces/hi,forces/lo)
    charge=np.zeros(20);charge_history=[]
    for dt,row in zip(np.diff(raw['raw_time_s']),ratio):
        charge=np.clip(charge+dt*(np.minimum(row,1)-charge/2),0,1);charge_history.append(charge.copy())
    charges=np.asarray(charge_history)
    ctrl_indices=np.r_[0,np.arange(5,598,4)];actual150=raw['ctrl_rad'][ctrl_indices]
    event150=np.asarray([e['actualu'][0] for e in events]);planned=raw['selected_planned150_FP32_rad']
    if not np.array_equal(actual150,event150) or not np.array_equal(actual150,planned.astype(float)):raise ValueError('Measured control events/raw/planned targets differ')
    rates=np.diff(actual150,axis=0)/.02;updates=np.arange(1,150)*.02
    expected=raw['expected_raw_FP32_time_s'];clockdiff=raw['raw_time_s']-t
    if not np.array_equal(expected,requested['expected_clock601_FP32_s'].astype(float)):raise ValueError('Saved FP32 expected-clock sources differ')
    clockbits=np.array_equal(raw['raw_time_s'].astype(np.float32).view(np.uint32),expected.astype(np.float32).view(np.uint32))
    numeric={'position151_max_m':float(pos[primary].max()),'position601_max_m':float(pos.max()),'shaft151_max_deg':float(shaft[primary].max()),'shaft601_max_deg':float(shaft.max()),'penetration600_max_m':float(penetration.max()),'actualq601_excess_max_rad':float(qexcess.max()),'actual149_target_rate_max_rad_s':float(np.abs(rates).max()),'load600_ratio_max':float(ratio.max()),'load600_charge_max':float(charges.max()),'clock601_nominal_drift_max_s':float(np.abs(clockdiff).max()),'clock601_FP32_bits_exact':bool(clockbits),'startup_ctrl_to_first150_target_jump_max_rad':float(np.abs(actual150[0]-raw['ctrl_rad'][0]).max())}
    for name,actual,key,tol in [('position151',numeric['position151_max_m'],'max_object_position_error_151_m',1e-12),('position601',numeric['position601_max_m'],'max_object_position_error_601_m',1e-12),('shaft151',numeric['shaft151_max_deg'],'max_shaft_angle_151_deg',1e-10),('shaft601',numeric['shaft601_max_deg'],'max_shaft_angle_601_deg',1e-10),('penetration600',numeric['penetration600_max_m'],'max_penetration_m',1e-12),('actualq601',numeric['actualq601_excess_max_rad'],'max_actual_joint_excess_rad',1e-12),('rate149',numeric['actual149_target_rate_max_rad_s'],'max_actual_target_rate_rad_s',1e-12),('load600',numeric['load600_ratio_max'],'max_actuator_load_ratio',1e-12),('charge600',numeric['load600_charge_max'],'max_load_charge',1e-12)]:
        checks[name+'_saved_metric_match']=abs(actual-evaluation[key])<=tol
    checks['clock_FP32_exact_and_saved_drift_match']=bool(clockbits and abs(numeric['clock601_nominal_drift_max_s']-evaluation['clock_audit']['max_abs_nominal_drift_s'])<1e-15)
    checks['full601_600_150_coverage']=len(t)==601 and len(solver)==600 and len(events)==150
    checks['task_FAIL_retained']=evaluation['native_A0_pass'] is False
    checks['all150_measured_target_matches']=bool(np.array_equal(actual150,event150) and raw['ctrl_exact'].all())
    if not all(checks.values()):raise ValueError('Saved metric/coverage inconsistency')
    if not all(np.isfinite(a).all() for a in (pos,shaft,qexcess,penetration,ratio,charges,rates,clockdiff)):raise ValueError('Nonfinite plotted data')
    out.mkdir(parents=True);(out/'renderer_source_before.py').write_bytes(Path(__file__).read_bytes())
    table(out/'poststates601.csv',['state_index','canonical_time_s','raw_FP32_time_s','primary151_knot','position_error_m','directed_shaft_error_deg','max_actual_q_hard_excess_rad','nominal_clock_drift_s','raw_expected_FP32_clock_error_s'],[[k,t[k],raw['raw_time_s'][k],k%4==0,pos[k],shaft[k],qexcess[k].max(),clockdiff[k],raw['raw_time_s'][k]-expected[k]] for k in range(601)])
    table(out/'solver600.csv',['step','prestate_index','poststate_index','canonical_solver_pre_time_s','raw_solver_pre_time_s','canonical_post_time_s','HO_generated_penetration_m','self_generated_penetration_m','other_generated_penetration_m','max_actuator_load_ratio','max_integrated_load_charge']+[f'joint{j:02d}_load_ratio' for j in range(20)]+[f'joint{j:02d}_load_charge' for j in range(20)],[[k+1,k,k+1,pre[k],solver[k]['solver_t_s'],post[k],*penetration[k],float(ratio[k].max()),float(charges[k].max()),*ratio[k],*charges[k]] for k in range(600)])
    table(out/'actual_control_updates149.csv',['new_interval','scheduled_update_time_s','first_poststate_readback_index','raw_first_poststate_readback_time_s','max_abs_rate_rad_s']+[f'joint{j:02d}_actual_rate_rad_s' for j in range(20)],[[k+1,updates[k],int(ctrl_indices[k+1]),float(raw['raw_time_s'][ctrl_indices[k+1]]),float(np.abs(rates[k]).max()),*rates[k]] for k in range(149)])
    table(out/'actual_targets150.csv',['interval','preaction_canonical_time_s','first_poststate_ctrl_index']+[f'joint{j:02d}_actual_u_rad' for j in range(20)],[[k,k*.02,int(ctrl_indices[k]),*actual150[k]] for k in range(150)])
    table(out/'original13_gates.csv',['gate','passed'],list(evaluation['gates'].items()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    fig.suptitle('Native full151 A0 — complete episode engineering / TASK FAIL',fontsize=16,fontweight='bold',color='#a42b2b',y=.98)
    fig.text(.5,.944,'150 intervals ·600 solver rows ·601 poststates ·151 reference knots; no quality-based truncation, reset or assist',ha='center',fontsize=10)
    ax=axes[0,0];ax.plot(t,pos*1000,lw=1,color='#225f96',label='all601 poststates');ax.plot(t[primary],pos[primary]*1000,'.',ms=2.5,color='#ba7930',label='151 knot gate samples');ax.axhline(20,color='#ad3232',ls='--',lw=1,label='original20 mm limit')
    ax.set(title=f'Position FAIL: max151={1000*pos[primary].max():.3f} mm',ylabel='object center error (mm)',xlabel='canonical episode time (s)',xlim=(0,3));ax.legend(fontsize=7,loc='upper left')
    ax=axes[0,1];ax.plot(t,shaft,lw=1,color='#225f96',label='all601 poststates');ax.plot(t[primary],shaft[primary],'.',ms=2.5,color='#ba7930',label='151 knot gate samples');ax.axhline(20,color='#ad3232',ls='--',lw=1,label='original20 degree limit')
    ax.set(title=f'Directed shaft FAIL: max151={shaft[primary].max():.3f}°\nall601 diagnostic max={shaft.max():.3f}°',ylabel='directed shaft error (degrees)',xlabel='canonical episode time (s)',xlim=(0,3));ax.legend(fontsize=7,loc='upper left')
    ax=axes[0,2]
    for j,(name,color) in enumerate(zip(('hand–object','hand–self','other'),('#225f96','#a66b2b','#6c8070'))):ax.plot(pre,penetration[:,j]*1000,lw=.9,color=color,label=name)
    ax.axhline(3,color='#ad3232',ls='--',lw=1,label='original3 mm limit')
    ax.set(title=f'Penetration FAIL: max600={1000*penetration.max():.3f} mm',ylabel='generated overlap aggregate (mm)',xlabel='solver preintegration time (s)',xlim=(0,3));ax.legend(fontsize=7,loc='upper left')
    ax=axes[1,0];ax.scatter(np.repeat(updates,20),np.abs(rates).ravel(),s=2,color='#a4aab0',alpha=.4);ax.plot(updates,np.abs(rates).max(1),'o',ms=2.7,color='#225f96',label='max over20 joints /149 updates');ax.axhline(15,color='#ad3232',ls='--',lw=1,label='original15 rad/s target-rate limit')
    ax.set(title=f'Actual target rate PASS: max={np.abs(rates).max():.6f} rad/s',ylabel='absolute actual target rate (rad/s)',xlabel='scheduled native update time (s)',xlim=(0,3),ylim=(-.3,17));ax.legend(fontsize=7,loc='upper right')
    inset=ax.inset_axes([.13,.14,.78,.25]);inset.plot(t,qexcess.max(1)*1000,color='#5f6b70',lw=.8);inset.axhline(1,color='#ad3232',ls=':',lw=.8);inset.set(xlim=(0,3),ylim=(-.02,1.07),yticks=[0,1],xticks=[],ylabel='q excess\n(mrad)');inset.tick_params(labelsize=7);inset.yaxis.label.set_size(7);inset.text(.98,.6,f'601-state max {1000*qexcess.max():.6g} mrad; limit1',ha='right',transform=inset.transAxes,fontsize=7)
    ax=axes[1,1];ax.plot(pre,ratio.max(1),color='#225f96',lw=.8,label='600 preintegration max load ratios');ax.plot(post,charges.max(1),color='#af782e',lw=1.1,label='600 integrated max charge updates');ax.axhline(1.01,color='#ad3232',ls='--',lw=1,label='load ratio limit1.01');ax.axhline(.8,color='#af782e',ls=':',lw=1,label='charge limit0.8')
    ax.set(title=f'Load/charge PASS: {ratio.max():.6f} / {charges.max():.6f}',ylabel='dimensionless load ratio / charge',xlabel='canonical solver pre/post time (s)',xlim=(0,3),ylim=(-.03,1.15));ax.legend(fontsize=7,loc='upper left')
    ax=axes[1,2];ax.plot(t,clockdiff*1e6,color='#225f96',lw=1.1,label='601 raw-time minus canonical time');ax.axhline(50,color='#ad3232',ls='--',lw=1,label='frozen full151 drift bound50 us');ax.axhline(2,color='#888',ls=':',lw=.8,label='legacy2 us diagnostic only')
    ax.set(title='Clock PASS: all601 expected FP32 bits exact',ylabel='raw minus nominal time (µs)',xlabel='canonical episode time (s)',xlim=(0,3),ylim=(-1,57));ax.text(.98,.77,f'final drift {1e6*clockdiff[-1]:.6f} µs\nexpected-FP32 residual:0',ha='right',transform=ax.transAxes,fontsize=8);ax.legend(fontsize=7,loc='upper left')
    for ax in axes.flat:ax.grid(alpha=.16)
    failed=[k for k,v in evaluation['gates'].items() if not v]
    fig.text(.04,.025,'Original13 gates:10 pass /3 fail (151-knot position,151-knot directed shaft,600-solver generated penetration). All601 q states,150 targets and149 updates retained.\nZero penetration aggregate means no generated overlap observed, not positive clearance. Contacts/forces use preintegration phase; state tracking uses postintegration phase.\nOne known-rough legacy source[200,351), one A0 episode; no Actor/WM/training/CHORD or generalization claim. Capacity applies to its observed reset/episode window; constructor work was unobserved.',fontsize=9,va='bottom')
    fig.tight_layout(rect=(.02,.10,.99,.92),h_pad=2,w_pad=2)
    fig.savefig(out/'full151_native_A0_summary.png',dpi=180);fig.savefig(out/'full151_native_A0_summary.svg');plt.close(fig)
    if any(k in sys.modules for k in ('torch','warp','mujoco','scipy')):raise AssertionError('Renderer imported a model/device runtime')
    if any(sha(p['path'])!=p['sha256'] for p in inputs.values()):raise ValueError('Inputs changed during rendering')
    result={'schema':'full151_native_A0_saved_figure_summary:1','status':'ENGINEERING_COMPLETE_TASK_FAIL_SAVED_SUMMARY','source':pin(__file__),'inputs':inputs,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'numeric':numeric,'original_gates':evaluation['gates'],'failed_original_gates':failed,'denominators':{'raw_states':601,'primary_goal_knots':151,'solver_samples':600,'actual_targets':150,'actual_target_updates':149},'new_Env_Reader_FK_queries_WM_physics_GPU':0,'visual_QA':'PENDING_ACTUAL_VIEW','scope':'Figures re-express saved measurements; Contact independent raw audit and Super process/source audit are separate. No regrade or promotion to physical-reference/training qualification.','artifacts':[pin(p) for p in sorted(out.iterdir()) if p.is_file()]}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n','utf-8');print(json.dumps({'PNG':str(out/'full151_native_A0_summary.png'),'summary':pin(out/'summary.json'),'saved_consistency':[result['passed'],result['total']]}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:main(Path(a.input).resolve(),Path(a.output).resolve())
    else:print(json.dumps({'status':'NOT_RUN','source':pin(__file__),'scope':'Default does not read results or render'}))
