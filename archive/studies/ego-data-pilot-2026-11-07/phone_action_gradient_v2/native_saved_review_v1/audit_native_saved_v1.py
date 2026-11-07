"""Independent saved action matrix review: no simulator, producer math or NN imports."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,importlib.util,json,math,sys
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent.parent
HELPER=P/'phone_episode_acquisition_v1/independent_dataset_review_v1/audit_saved_dataset_v1.py'
HELPER_SHA='3af410ac90fba95b8eb0774706747daa301e3ce9938999016b04549d786be1cf'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA
spec=importlib.util.spec_from_file_location('frozen_independent_phone_math',HELPER);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
pin=A.pin;checked=A.checked;read=A.read;load=A.load;write=A.write;require=A.require;bits=A.bits;close=A.close
def ctrl_segment(before,after,u):
    require(before.dtype==after.dtype==u.dtype==np.float64 and before.shape==after.shape==(300,) and u.shape==(20,),'Full300/raw20 ABI')
    locations=[i for i in range(281) if after[i:i+20].tobytes()==u.tobytes()]
    require(len(locations)==1,'Unique empirical exported ctrl20 segment')
    i=locations[0];expected=before.copy();expected[i:i+20]=u
    bits(after,expected,'Only actual ctrl segment may change; all other integration values unchanged')
    return i
def receipt(meta,array,tick,h):
    require(array.dtype==np.float64 and array.shape==(300,) and meta['size']==300 and meta['signature']==16383,'Actual full integration state spec300')
    require(A.digest(array)==meta['sha256'] and meta['global_tick']==tick,'Full integration SHA/tick')
    close(meta['native_time_s'],tick*h,'native_source_target_clock')
def fixture():
    a=np.arange(300,dtype=float)/1000;u=np.arange(20,dtype=float)+.125;b=a.copy();b[81:101]=u
    require(ctrl_segment(a,b,u)==81,'Synthetic control-only context update')
    b[200]+=1
    try:ctrl_segment(a,b,u)
    except ValueError:pass
    else:raise ValueError('Unexpected opaque context mutation not rejected')
    return dict(schema='phone_native_saved_control_context_fixture:1',passed=True,cases=1,
        scope='Only contiguous command replacement/outside-context rawbyte preservation, no model or physical claim.')
def audit(report_path):
    r=read(report_path);require(r['schema']=='phone_full20_native_action_branch_result:1' and r['status']=='COMPLETE_BASELINE_MATCHED_FULL20_NATIVE_FINITE_ACTIONS','Complete actual native action matrix')
    p=read(checked(r['protocol']));require(p['schema']=='phone_full20_native_action_branch_protocol:1','Actual native protocol namespace')
    for item in [*p['inputs'].values(),*p['source_pins'].values()]:checked(item)
    require(p['source_pins']['runner']['sha256']=='e9860442712f57ea794f03cf6ef7fe8bf70d51dccdcb2d79f85bc1cdcf99606d','Frozen executed native runner')
    selection=read(checked(p['inputs']['selection']));design=read(checked(p['inputs']['design']));addendum=read(checked(p['inputs']['all20_addendum']))
    require(p['inputs']['selection']['sha256']=='23a0b10cef90133d1e909de346a49aae019a24881d4b72ed0c7a894755edea51','All20 predeclared selection')
    require(p['inputs']['operands']==selection['operands']==addendum['reused_operands'],'Original operands unchanged')
    op=load(p['inputs']['operands']);acquired=read(checked(p['inputs']['acquisition_report']));records=read(checked(r['records']))
    previous_review=read(checked(design['dataset_saved_independent']));require(previous_review['engineering_qualified'] is True,'Original episode data engineering audit')
    require(len(selection['states'])==12 and len(records)==488 and selection['available_sides']==476,'All states and branches preserved')
    expected=[];unavailable=[]
    for s in selection['states']:
        require(len(s['perturbations'])==40 and s['split']=='validation','All20/frozen VAL states')
        expected.append((s,None))
        for item in s['perturbations']:
            (expected if item['available'] else unavailable).append((s,item))
    require(len(expected)==488 and len(unavailable)==4,'All476 plus12 /4unavailable')
    require(Counter(v['direction_joint']<8 for s,v in expected if v is not None)=={True:188,False:288},'Original8 and added12 coverage')
    totals=Counter();dtcounts=defaultdict(Counter);ctrl_offsets=set();data=[];recomputed=[];source_cache={};baseline={};identities=set()
    baseline_ids=[];side_ids=[];contact_change=Counter();maxdrift=0.;maxangle=0.;maxpen=0.;maxexc=0.
    for index,(rec,ex) in enumerate(zip(records,expected)):
        s,planned=ex;sid=s['state_id'];h=s['native_dt_s'];n=s['substeps'];tick=s['source_before_action_receipt']['global_tick']
        require(rec['branch_id']==index and rec['state_id']==sid,'Exact branch order/ID')
        for key in ['family_id','episode_id','source_sample_id','split','interval_index','source_branch']:require(rec[key]==s[key],'Branch selected source join '+key)
        require(rec['dt_s']==h and rec['perturbation']==planned and rec['kind']==('baseline' if planned is None else 'finite_action_side'),'Prespecified action branch identity')
        z=load(rec['output']);N=len(z['post_qpos27']);require(N==n==rec['returned_substeps'],'Actual all branches full H64/128')
        require(rec['complete_H20ms'] is True and rec['finite'] is True and rec['usable_observable59'] is True and rec['termination']=='COMPLETE_INTERVAL','No actual partial/masked branch hidden')
        u=np.array(s['baseline_ctrl20_F64'] if planned is None else planned['actual_ctrl20_F64'],dtype=float)
        bits(z['actual_control20'],u,'Actual exact action plan, no clamp')
        ranges=np.array(acquired['model_audits'][str(h)]['actuator_ctrlrange20x2']);require(np.all(u>=ranges[:,0]) and np.all(u<=ranges[:,1]),'True native control limits')
        if planned is not None:
            j=planned['direction_joint'];sgn=planned['sign'];base=np.array(s['baseline_ctrl20_F64']);keep=np.arange(20)!=j
            bits(u[keep],base[keep],'Other19 command values unchanged')
            margin=ranges[j,1]-base[j] if sgn>0 else base[j]-ranges[j,0];hh=min(.01,.5*margin)
            proposal=base.copy();proposal[j]+=sgn*hh;bits(u,proposal,'Independent one-sided halfmargin action arithmetic')
            du=float(u[j]-base[j]);require(abs(du)>=1e-6 and du==planned['actual_signed_du_F64_rad'],'Actual signed F64 control denominator')
            du32=float(np.float64(np.float32(u[j]))-np.float64(np.float32(base[j])))
            require(du32==planned['actual_signed_du_FP32_rad'] and (du32==0)==planned['FP32_input_collapsed_numeric'],'Recorded FP32 input resolution without NN')
            identities.add((sid,j,planned['side']));side_ids.append(index)
        else:baseline_ids.append(index)
        bits(z['source_integration_state'],op['source_integration_F64'][sid],'Every branch same original before-action300')
        receipt(rec['source_receipt'],z['source_integration_state'],tick,h);receipt(rec['source_after_action_receipt'],z['source_after_action_integration_state'],tick,h)
        receipt(rec['target_receipt'],z['target_integration_state'],tick+n,h)
        require(rec['source_receipt']['sha256']==s['source_before_action_receipt']['sha256'],'Original before-action receipt pin')
        ctrl_offsets.add(ctrl_segment(z['source_integration_state'],z['source_after_action_integration_state'],u))
        model=acquired['model_audits'][str(h)];ipos=np.array(model['body_ipos_body_m']);qrange=np.array(model['joint_range20x2'])
        source=A.state58(op['source_qpos27_F64'][sid],op['source_qvel26_F64'][sid],ipos)
        close(z['source_state58'],source,'source58_raw_quaternion_COM_world');bits(z['source_state58'],op['source_state58_F64'][sid],'Exact same source state58 operand')
        q0=op['initial_qpos27_F64'][sid];x0=A.state58(q0,np.zeros(26),ipos);close(x0,op['initial_state58_F64'][sid],'Original episode hardstop COM/quaternion baseline')
        y=A.state58(z['post_qpos27'],z['post_qvel26'],ipos);close(z['post_state58'],y,'all_POST_state58_raw_recomputed');close(z['observed_prefix_target58'],y[-1],'Observed physical endpoint')
        require(np.array_equal(z['pre_tick'],np.arange(tick,tick+n)) and np.array_equal(z['post_tick'],np.arange(tick+1,tick+n+1)),'Original global integer PRE/POST ticks')
        close(z['pre_native_time_s'],z['pre_tick']*h,'solver_PRE_time');close(z['post_native_time_s'],z['post_tick']*h,'native_POST_time')
        d=A.delta52(source,y[-1]);close(z['physical_world_delta52'],d,'independent_physical_world_delta52')
        close(A.exp3(d[43:46])@source[43:52].reshape(3,3),y[-1,43:52].reshape(3,3),'independent_full_orientation_Exp_roundtrip')
        pad,other,table,pen,selfpen,wrench=A.contact_math(z['contact_rows24'],z['pre_tick'],np.vstack([source[40:43],y[:-1,40:43]]))
        for key,value in [('pre_pad_normal_N2',pad),('pre_phone_nonpad_normal_N',other),('pre_table_normal_N',table),('pre_penetration_m',pen),('pre_self_penetration_m',selfpen),('pre_contact_wrench_about_COM_world6',wrench)]:close(z[key],value,key)
        coverage=np.count_nonzero(pad>.1,axis=0)/n;minimum=pad.min(0);endpoint=pad[-1];force5=np.r_[minimum,endpoint,other.max()]
        obs=np.r_[d,coverage,np.log1p(force5)]
        close(z['observable59'],obs,'all59_independent_observables');sm=rec['summary']
        close(sm['pad_coverage2'],coverage,'summary_coverage');close(sm['pad_min_normal_N2'],minimum,'summary_min_normal');close(sm['pad_endpoint_normal_N2'],endpoint,'summary_lastPRE_normal');close(sm['phone_nonpad_max_normal_N'],other.max(),'summary_nonpad_max')
        first=[next((int(t) for t,f in zip(z['pre_tick'],pad[:,j]) if f<=.1),-1) for j in range(2)]
        require(sm['first_pad_le_0p1N_global_PRE_tick2']==first and sm['endpoint_force_global_PRE_tick']==tick+n-1,'Original coverage event and lastPRE label phase')
        close(sm['contact_impulse_world6'],[math.fsum(wrench[:,j])*h for j in range(6)],'Contact impulse4 PREphase')
        require(all(np.isfinite(z[key]).all() for key in ['post_qpos27','post_qvel26','post_state58','pre_energy2','pre_actuator_force20','pre_qfrc_actuator20','contact_rows24']) and not any(rec['warning_numbers']),'Visible finite/rawwarning mask independently checked')
        drift=np.linalg.norm(y[:,40:43]-x0[40:43],axis=1);angle=2*np.arccos(np.clip(abs(z['post_qpos27'][:,23:27]@q0[23:27]),0,1))
        exc=np.max(np.maximum(qrange[:,0]-z['post_qpos27'][:,:20],0)+np.maximum(z['post_qpos27'][:,:20]-qrange[:,1],0),axis=1)
        require(np.all(drift<=.15) and np.all(angle<=math.pi/2) and np.all(pen<=.005) and np.all(exc<=.05),'No original-episode safety event ignored')
        maxdrift=max(maxdrift,float(drift.max()));maxangle=max(maxangle,float(angle.max()));maxpen=max(maxpen,float(pen.max()));maxexc=max(maxexc,float(exc.max()))
        if sid not in source_cache:source_cache[sid]=load(s['source_trace'])
        if planned is None:
            original=source_cache[sid]
            for key in ['post_qpos27','post_qvel26','post_state58','pre_pad_normal_N2','pre_phone_nonpad_normal_N','pre_table_normal_N','pre_energy2','contact_rows24','pre_native_time_s','post_native_time_s']:
                bits(z[key],original[key],'Matched original baseline every saved tick '+key)
            bits(z['target_integration_state'],op['original_target_integration_F64'][sid],'Baseline full300 target bytes')
            bits(z['post_qpos27'][-1],op['original_target_qpos27_F64'][sid],'Baseline original targetqpos')
            bits(z['post_qvel26'][-1],op['original_target_qvel26_F64'][sid],'Baseline original targetqvel')
            bits(z['observable59'],op['baseline_observable59_F64'][sid],'Baseline59 captured labels exactbytes')
            require(rec['source_after_action_receipt']['sha256']==s['original_source_after_action_receipt']['sha256'] and rec['target_receipt']['sha256']==s['original_target_receipt']['sha256'],'Baseline original endpoint receipt hashes')
            declared=read(report_path.parent/f'state{sid:02d}_baseline_checks.json');require(declared['passed'] is True and all(v is True for v in declared['checks'].values()),'Source baseline flags match independently verified raw equality')
            baseline[sid]=dict(obs=obs,record_id=index,trace=z)
        else:
            require(sid in baseline,'Actual own-state baseline occurs before its perturbations')
            signature=lambda a:sorted((int(t),int(g1),int(g2),int(dim),int(addr>=0)) for t,g1,g2,dim,addr in a[:,:5])
            changed=signature(z['contact_rows24'])!=signature(baseline[sid]['trace']['contact_rows24'])
            contact_change['changed' if changed else 'same']+=1
        recomputed.append(obs);data.append(dict(branch_id=index,state_id=sid,kind=rec['kind'],direction_joint=None if planned is None else planned['direction_joint'],side=None if planned is None else planned['side'],actualdu_F64=None if planned is None else planned['actual_signed_du_F64_rad'],dt_s=h,complete=True,eligible=True,contact_rows=len(z['contact_rows24'])))
        dtcounts[h]['MjData']+=1;dtcounts[h]['integration_steps']+=n;dtcounts[h]['step_contact_force_reads']+=len(z['contact_rows24']);dtcounts[h]['receipt_reads']+=4
    require(len(ctrl_offsets)==1,'Stable empirical native ctrl segment offset across488 branches')
    require(identities=={(s['state_id'],p['direction_joint'],p['side']) for s in selection['states'] for p in s['perturbations'] if p['available']},'No branch loss/extra/replacement')
    # Independently verify all4 exclusions, not merely absence in output.
    for s,item in unavailable:
        j=item['direction_joint'];u=np.array(s['baseline_ctrl20_F64']);bounds=np.array(acquired['model_audits'][str(s['native_dt_s'])]['actuator_ctrlrange20x2'])
        margin=bounds[j,1]-u[j] if item['sign']>0 else u[j]-bounds[j,0];trial=u.copy();trial[j]+=item['sign']*min(.01,.5*margin)
        require(abs(trial[j]-u[j])<1e-6 and item['unavailable_reason']=='ACTUAL_F64_STEP_BELOW_1e-6_OR_BOUND_VIOLATION','Declared four insufficient-step sides preserved')
    for h,count in dtcounts.items():
        count.update(MJB_reads=1,initial_forward=0,initial_contact_force_reads=0,observer_forward=0,NN_training_GPU=0)
        count['contact_force_reads']=count['step_contact_force_reads'];audit=r['audits'][str(h)]
        require(dict(count)==audit['model']['counts'],'Exact per-dt physical counters from saved lengths')
        require(audit['model']['model_pin']==p['inputs']['model_mjb'] and audit['model']['passed'] is True and audit['model']['model_digest']==acquired['model_audits'][str(h)]['model_digest'],'Immutable same effective model/source digest scope')
        require(audit['probe_counts']==dict(branches_attempted=count['MjData'],branches_returned=count['MjData'],state_sets=count['MjData'],restore_forward=0),'All direct state-set branch counters')
        require(audit['restored_integration_is_full_MjData_qualified'] is False,'Keep fullMjData qualification false')
        totals.update(count)
    require(dict(totals)==r['counts'] and totals['integration_steps']==46848 and r['partial_branches']==0 and r['usable_observables']==488,'All global completed counters/zero partial')
    termpath=report_path.parent.parent/'process_native_run1/terminal.json';terminal=read(termpath)
    require(terminal['returncode']==0 and terminal['protocol']==r['protocol'] and terminal['runner']==p['source_pins']['runner'] and terminal['automatic_retry'] is False,'Actual once source/protocol terminal')
    for key in ['stdout','stderr']:checked(terminal[key])
    require(terminal['stderr']['bytes']==0,'Empty actual stderr')
    order_all12_first=max(baseline_ids)<min(side_ids)
    require(order_all12_first is False,'Record known per-context interleaving, never rewrite global precheck order')
    require(not any(name in sys.modules for name in ['mujoco','torch','warp']),'No native/NN imports')
    observations=np.stack(recomputed);secants=[]
    for d0 in data:
        if d0['kind']!='baseline':secants.append((observations[d0['branch_id']]-baseline[d0['state_id']]['obs'])/d0['actualdu_F64'])
    np.savez_compressed(H/'independently_recomputed_F64.npz',observable59_F64=observations,native_side_secant59_F64=np.stack(secants),
        baseline_branch_ids_i64=np.array(baseline_ids,dtype=np.int64),side_branch_ids_i64=np.array(side_ids,dtype=np.int64))
    write(H/'all_branch_recomputed_summary.json',data)
    data_checks={key:True for key in ['frozen_source_protocol_inputs_model_and_terminal','all12_states_488_branches_188plus288_and4exclusions',
        'single_axis_control_19unchanged_actualdu_no_clipping','all_source_contexts_match_original_beforeaction300',
        'afteraction_changes_only_unique_ctrl20_segment','all12_baselines_fulltarget_and_allticks_exactbytes',
        'all_POST_state58_COM_world_and_fullorientation','all59_physical_delta_coverage_logforce_recomputed',
        'original_episode_hardstop_reference_and_PRE_POST_clocks','all_contacts_min_coverage_endpoint_and_impulses',
        'full_counts_zero_forward_NN_and_preserved_partial_scope','no_new_model_physics_LP_NN_or_learning_in_review']}
    checks=data_checks|{'protocol_order_all12_baselines_before_any_new_side':False}
    result=dict(schema='phone_full20_native_action_saved_independent:1',status='PASS_MATCHED_CONTEXT_DATA_WITH_ORDER_DEVIATION',
        engineering_qualified=True,matched_context_data_qualified=True,protocol_conformance=False,
        protocol_order_all12_baselines_before_any_new_side=False,per_state_baseline_before_own_sides=True,
        checks=checks,passed=sum(checks.values()),total=len(checks),data_checks=data_checks,
        producer_report=pin(report_path),protocol=r['protocol'],selection=p['inputs']['selection'],operands=p['inputs']['operands'],
        records=r['records'],source_pins=dict(auditor=pin(Path(__file__)),math_helper=pin(HELPER),source_freeze=pin(H/'SOURCE_FREEZE_v1.json'),fixture=pin(H/'pure_fixture_v1.json')),
        terminal=pin(termpath),counts=dict(totals),states=12,baselines=12,primary8_sides=188,secondary12_sides=288,all20_sides=476,unavailable_sides=4,
        control_segment_index_zero_based=int(next(iter(ctrl_offsets))),outside_control280_F64_values_unchanged=True,
        native_contact_signature_comparison=dict(contact_change),max_F64_arithmetic_errors=dict(A.ERRORS),
        max_original_safety_descriptors=dict(COM_drift_m=maxdrift,angle_rad=maxangle,penetration_m=maxpen,joint_excursion_rad=maxexc),
        independent_observables=pin(H/'independently_recomputed_F64.npz'),branch_summaries=pin(H/'all_branch_recomputed_summary.json'),
        order_deviation='Actual execution is baseline(s)->sides(s)->next state, whereas design text required all12 baselines first. All12 independently reproduce original contexts, so matched-context values are usable; exact prospective order is not satisfied. No rerun or rewrite.',
        scope='Pure saved arithmetic/bytes. mjSTATE_INTEGRATION vectors are not all MjData derived buffers; effective model immutability/zeroaid/getState meaning and missing raw qacc finite guard remain source-backed. No independent re-integration or analytic physical Jacobian claim.',
        qualification=dict(physical_analytic_Jacobian=False,holding=False,policy_benefit=False,real_ego_dynamics=False,new_learning=False),
        actual_new_model_FK_LP_steps_NN_training_calls=0)
    write(H/'report.json',result);return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');ap.add_argument('--report',type=Path);a=ap.parse_args()
    if a.fixture:
        v=fixture();write(H/'pure_fixture_v1.json',v);print(json.dumps(dict(fixture=pin(H/'pure_fixture_v1.json'),passed=True)));return
    require(a.report is not None and not (H/'report.json').exists() and not (H/'failure.json').exists(),'Fresh explicit audit, no automatic retry')
    try:
        r=audit(a.report);print(json.dumps(dict(report=pin(H/'report.json'),status=r['status'],passed=r['passed'],total=r['total'],engineering_qualified=r['engineering_qualified'],protocol_conformance=r['protocol_conformance'],counts=r['counts'])))
    except BaseException as err:
        write(H/'failure.json',dict(schema='phone_native_action_saved_review_failure:1',error=repr(err),source=pin(Path(__file__)),producer_report=pin(a.report),automatic_retry=False,new_native_steps=0));raise
if __name__=='__main__':main()
