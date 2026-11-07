"""Saved-only A0 / matched primary2000 policy descriptive comparison.

Requires complete saved rollouts and their independent engineering audits.
NumPy/Matplotlib/JSON only; no Env, Torch, Warp, Actor, physics or new rollout.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin,checked,loadnp,canonical
from rough_prefix.bundle_v2 import load_bundle,rotation
from rough_prefix.probe import GATES


def same32(a,b):
    return np.array_equal(np.asarray(a,np.float32).view(np.uint32),np.asarray(b,np.float32).view(np.uint32))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--a1l-audit',type=Path,default=ROOT/'outputs/policy_saved_audit_primary_a1l_v1/report.json')
    ap.add_argument('--p-only-audit',type=Path,default=ROOT/'outputs/policy_saved_audit_primary_p_only_v1/report.json')
    ap.add_argument('--a2-audit',type=Path,default=None)
    a=ap.parse_args()
    if any(name in sys.modules for name in ('torch','warp','mujoco_warp')):raise RuntimeError('Saved-only CPU plotting boundary violated')
    bundle=load_bundle(ROOT/'outputs/input_bundle_v1/bundle.json');ref=bundle['arrays']['reference']
    entries=[('Native A0 (zero residual)','native_a0_v2',ROOT/'outputs/native_a0_saved_audit_v2/report.json',None,None,'#69717b'),
             ('A1L / 2000 iterations','eval_primary_a1l_seed0_v1',a.a1l_audit,2000,'A1L','#2865a7'),
             ('P_ONLY / 2000 iterations','eval_primary_p_only_seed0_v1',a.p_only_audit,2000,'P_ONLY','#d98222')]
    if a.a2_audit is not None:
        entries.append(('A2 / 2000 iterations','eval_primary_a2_seed0_v1',a.a2_audit,2000,'A2','#23896c'))
    sources={'bundle':bundle['descriptor']['input_bundle'],**{'input__'+key:value for key,value in bundle['descriptor']['source_pins'].items()}}
    rows=[];traces=[];arrays_for_plot=[];initials=[];primary_budgets={};clock_arrays=[]
    for label,directory,audit_path,iteration,expected_method,color in entries:
        run=ROOT/'outputs'/directory
        audit=read(audit_path);v=read(run/'evaluation.json');raw=loadnp(run/'raw_states.npz');solver=read(run/'solver_samples.json');events=read(run/'control_events.json');config=read(run/'config.json');capacity=read(run/'capacity_audit.json')
        if iteration is None:
            if audit.get('engineering_passed') is not True or audit.get('status')!='PASS_ENGINEERING' or not audit.get('checks') or any(row.get('passed') is not True for row in audit['checks']) or audit['input_pins']['worker_evaluation_record_only']!=pin(run/'evaluation.json'):raise ValueError('A0 independent engineering audit not qualified/bound')
        else:
            if audit.get('schema')!='rough_native_policy_saved_CPU_audit:1' or audit.get('engineering_qualified') is not True or not audit.get('checks') or any(value is not True for value in audit['checks'].values()) or audit.get('checkpoint')!=config['checkpoint'] or audit.get('evaluation')!=pin(run/'evaluation.json'):raise ValueError('Policy independent engineering audit not qualified/bound')
            checked(audit['config'])
            if audit['config']['sha256']!=pin(run/'config.json')['sha256']:raise ValueError('Run config copy differs from audited external config')
            if config['method']!=expected_method or config['checkpoint_iteration']!=iteration or config['seed']!=0:raise ValueError('Wrong checkpoint length/method/seed')
            training_path=checked(config['training_config']);training=read(training_path)
            if training['role']!='primary_matched' or training['method']!=expected_method or training['seed']!=0 or training['training']['max_iterations']!=2000 or training['input_bundle']!=bundle['descriptor']['input_bundle']:raise ValueError('Only matching primary2000 training/source is allowed')
            primary_budgets[expected_method]=training['training']
            if len({canonical(value) for value in primary_budgets.values()})!=1:raise ValueError('Primary data/update budgets differ')
            sources[directory+'/primary_training_config']=config['training_config']
        if v.get('complete') is not True or len(raw['raw_time_s'])!=61 or len(solver)!=60 or len(events)!=15 or v['physical_steps']!=60 or v['native_postinterval_forward_calls']!=15 or v['control_intervals']!=15 or v['resets_after_start']!=0 or canonical(v['criteria'])!=canonical(GATES):raise ValueError('Require complete61/60/15 native cadence and unchanged quality criteria')
        if capacity['status']!='OK' or capacity['flags_by_world']!=[0] or capacity['read_count']!=1 or capacity['read_after_step']!=60 or capacity['resets_during_probe']!=0:raise ValueError('Saved capacity not qualified')
        if pin(run/'nominal_model.mjb')['sha256']!=bundle['model']['sha256']:raise ValueError('Different nominal physical model')
        initial=loadnp(run/'actual_context_readbacks.npz')
        for field in ('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm'):
            for stage in ('after_copy','after_set_const_0'):
                actual=initial[stage+'__'+field];wanted=np.broadcast_to(bundle['context'][field][0].astype(actual.dtype),actual.shape)
                if not np.array_equal(actual,wanted):raise ValueError('Different actual physical context')
        time=raw['raw_time_s'];pose=raw['object_pose_w'];q=raw['joint_rad'];R=rotation(pose[:,3:])
        if np.max(np.abs(time-np.arange(61)*.005))>GATES['clock_absolute_tolerance_s']:raise ValueError('Raw clock differs from original gate')
        clock_arrays.append((label,time.copy()))
        pos=np.linalg.norm(pose[:,:3]-ref['object_COM_world_m'],axis=1)*1000
        shaft=np.degrees(np.arccos(np.clip(np.sum(R[:,:,0]*ref['object_rotation_w'][:,:,0],axis=1),-1,1)))
        applied=np.asarray([event['actualu'] for event in events],float).reshape(15,20)
        target_delta=np.diff(np.vstack((raw['ctrl_rad'][0],applied)),axis=0)
        rates=np.abs(target_delta).max(axis=1)/.02
        depth=np.asarray([s['penetration_m'] for s in solver])*1000
        excess=np.maximum(np.maximum(raw['joint_lower_rad']-q,q-raw['joint_upper_rad']),0).max(axis=1)*1000
        force=np.asarray([s['actuator_force_solver_native'] for s in solver])
        load=np.where(force>=0,force/raw['force_upper_Nm'],force/raw['force_lower_Nm']).max(axis=1)
        knots=np.arange(0,61,4)
        recomputed={'position16_mm':float(pos[knots].max()),'position61_mm':float(pos.max()),'shaft16_deg':float(shaft[knots].max()),'shaft61_deg':float(shaft.max()),'penetration_mm':float(depth.max()),'actual_q_excess_mrad':float(excess.max()),'target_rate_including_startup_rad_s':float(rates.max()),'startup_target_jump_rad':float(np.abs(target_delta[0]).max()),'startup_target_rate_rad_s':float(rates[0]),'max_load_ratio':float(load.max())}
        compare={'position16_mm':v['max_object_position_error_16_m']*1000,'position61_mm':v['max_object_position_error_61_m']*1000,'shaft16_deg':v['max_shaft_angle_16_deg'],'shaft61_deg':v['max_shaft_angle_61_deg'],'penetration_mm':v['max_penetration_m']*1000,'actual_q_excess_mrad':v['max_actual_joint_excess_rad']*1000,'target_rate_including_startup_rad_s':v['max_actual_target_rate_rad_s'],'max_load_ratio':v['max_actuator_load_ratio']}
        if any(not np.isclose(recomputed[key],value,atol=1e-9,rtol=1e-10) for key,value in compare.items()):raise ValueError('Saved grade/raw metric mismatch')
        rows.append({'condition':label,'trained_method':expected_method,'training_seed':None if iteration is None else 0,'training_iterations':iteration,'quality_pass':v.get('native_policy_pass',v.get('native_A0_pass')),'failed_quality_gates':[key for key,value in v['gates'].items() if not value],**recomputed,'saved_evaluation':pin(run/'evaluation.json'),'independent_engineering_audit':pin(audit_path)})
        arrays_for_plot.append({'label':label,'color':color,'time':time,'shaft':shaft,'position':pos,'solver_time':np.asarray([s['solver_t_s'] for s in solver]),'depth':depth,'rate':rates,'excess':excess,'load':load})
        initials.append({'condition':label,**{key:raw[key][0].copy() for key in ('full_qpos','full_qvel','mocap_pos_w','mocap_quat_wxyz','ctrl_rad')}})
        for k in range(61):traces.append({'condition':label,'raw_index':k,'poststate_time_s':float(time[k]),'position_error_mm':float(pos[k]),'directed_shaft_deg':float(shaft[k]),'actual_q_excess_mrad':float(excess[k])})
        for name in ('raw_states.npz','solver_samples.json','control_events.json','evaluation.json','config.json','capacity_audit.json','controller_descriptor.json','actual_context_readbacks.npz'):
            sources[directory+'/'+name]=pin(run/name)
        sources[directory+'/independent_audit']=pin(audit_path)
    # Initial body angular velocity can differ by documented API arithmetic
    # <=1e-6. Report actual values; do not rename this a full-state bitwise pair.
    initial_comparisons=[]
    for item in initials[1:]:
        zero=initials[0]
        initial_comparisons.append({'condition':item['condition'],'reference_condition':zero['condition'],'FP32_exact':{key:same32(item[key],zero[key]) for key in ('full_qpos','full_qvel','mocap_pos_w','mocap_quat_wxyz','ctrl_rad')},'max_abs_full_velocity_difference':float(np.abs(item['full_qvel']-zero['full_qvel']).max())})
    clock_comparisons=[{'condition':label,'reference_condition':clock_arrays[0][0],'max_abs_time_difference_s':float(np.abs(value-clock_arrays[0][1]).max())} for label,value in clock_arrays[1:]]
    a.output.mkdir(parents=True,exist_ok=False);(a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14.2,8.2),layout='constrained')
    for item in arrays_for_plot:
        style={'color':item['color'],'label':item['label'],'lw':1.6,'alpha':.9}
        for ax,key in ((axes[0,0],'shaft'),(axes[0,1],'position'),(axes[1,1],'excess')):
            ax.plot(item['time'],item[key],**style)
            if key!='excess':ax.plot(item['time'][::4],item[key][::4],ls='none',marker='.',ms=3,color=item['color'])
        axes[0,2].plot(item['solver_time'],item['depth'],**style)
        axes[1,0].plot(np.arange(15)*.02,item['rate'],ls='none',marker='o',ms=3.7,color=item['color'],label=item['label'])
        axes[1,2].plot(item['solver_time'],item['load'],**style)
    settings=[(axes[0,0],'Directed shaft error','deg',20),(axes[0,1],'Object position error','mm',20),(axes[0,2],'Solved overlap, pre-integration','mm',3),(axes[1,0],'Applied target differences, including startup','rad/s',15),(axes[1,1],'Actual joint hard-limit excess','mrad',1),(axes[1,2],'Measured actuator load ratio','ratio',1.01)]
    for ax,title,ylabel,threshold in settings:
        ax.axhline(threshold,color='#b74444',ls='--',lw=1,label='Original limit');ax.set(title=title,xlabel='Time (s)',ylabel=ylabel,xlim=(0,.3));ax.set_ylim(bottom=0);ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Native prefix16 | A0 and matched primary2000 policies, seed0\nSingle source, single rollout per condition; descriptive comparison, no method-benefit or replication claim',fontsize=12)
    fig.savefig(a.output/'primary_native_diagnostics.png',dpi=180);fig.savefig(a.output/'primary_native_diagnostics.svg');plt.close(fig)
    csv_rows=[{**{key:value for key,value in row.items() if key not in ('saved_evaluation','independent_engineering_audit','failed_quality_gates')},'failed_quality_gates':';'.join(row['failed_quality_gates'])} for row in rows]
    for name,data in (('summary.csv',csv_rows),('state_traces.csv',traces)):
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    report={'schema':'rough_matched_primary_native_saved_comparison:1','status':'SAVED_ONLY_DESCRIPTIVE','rows':rows,'initial_comparisons':initial_comparisons,'source_pins':sources,'plotter':pin(__file__),'input_signature_sha256':bundle['descriptor']['input_signature_sha256'],'retarget_variant_id':bundle['descriptor']['retarget_variant_id'],'quality_gates':GATES,'source_lineage_N':1,'independent_ego_N':0,'primary_training_seed_N':1,'per_method_training_seed_N':{entry[4]:1 for entry in entries if entry[4] is not None},'physical_reference_qualified':False,'physics_steps_added':0,'Env_or_Runner_constructed':False,'Torch_imported':'torch' in sys.modules,'Warp_imported':'warp' in sys.modules,'sample_counts':{'raw_states':61,'reference_knots':16,'solver_samples':60,'applied_targets':15,'target_differences_including_startup':15},'timing':'Raw states are sampled after integration before each of15Manager forwards; solved overlap/force use solver pre-integration time. Rate markers only, including startup ctrl0->firstlearnedu0 over declared20ms convention, not instantaneous derivative.','scope':'Same frozen source/bundle and matched primary2000 data/update settings, seed0 per method; same budget does not mean identical on-policy data or compute cost. A0 is zero residual. Descriptive single-rollout comparison, not independent replication, overall method benefit, speed generalization, safety or full3s/ego/contact-gradient qualification.'}
    report.update(primary_budgets=primary_budgets,primary_budgets_identical=True,clock_comparisons=clock_comparisons,quality_outcome_selection=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'output':str(a.output),'conditions':[row['condition'] for row in rows]},indent=2))


if __name__=='__main__':main()
