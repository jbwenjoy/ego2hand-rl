"""Bounded same-state TRAIN intervention collection, never NN training.

The frozen NativeSavedActionProbe/Reader supply the entire native loop. All72
original-action baselines must replay all captured ticks/targets bit-exactly
before any new side. Default is JSON/hash only: no NPZ open or native import.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
import numpy as np

SCHEMA='phone_causal_TRAIN_action_acquisition_CPU_protocol:1'
PROBE_SHA='b10bd019326f51c48c5d471c64defd6026df255754fb14549860b6899e8b7d31'
READER_SHA='877aa4c5d51bff09e0a6824804eef3f2b335a8dbda26dcc3d0235bb3af57b841'
PARTITIONS={'CAUSAL_FIT_KNOWN_TRAIN_STATE','HELDOUT_ACTION_RESPONSE_CHECK_KNOWN_TRAIN_STATE'}
DT=(.00015625,.0003125)

def require(x,message):
    if not x:raise ValueError(message)
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):
    p=Path(v['path']);require(p.is_absolute(),'Absolute pinned file path')
    require(pin(p)==v,'Pinned file identity/bytes changed: '+str(p));return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    return v
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(v),f,indent=2,allow_nan=False)
def save(p,**arrays):
    with Path(p).open('xb') as f:np.savez_compressed(f,**arrays)
def compact(r):return {k:v for k,v in r.items() if k!='integration_state'}
def loadmod(path):
    sp=importlib.util.spec_from_file_location('causal_acquisition_frozen_native_probe',path)
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m


def preflight(protocol):
    protocol=Path(protocol).resolve(strict=True);p=read(protocol)
    require(p['schema']==SCHEMA,'New acquisition protocol schema')
    require(type(p['CPU_run_authorized']) is bool and p['training_authorized'] is False and p['fresh_final_acquisition_authorized'] is False,'Explicit bounded native-only authorization')
    for r in [*p['inputs'].values(),*p['sources'].values()]:checked(r)
    require(p['sources']['runner']==pin(__file__),'Exact current runner source')
    require(p['sources']['native_probe']['sha256']==PROBE_SHA and p['sources']['reader']['sha256']==READER_SHA,'Frozen trusted native path')
    design=read(checked(p['inputs']['causal_design_report']))
    require(design['schema']=='phone_causal_train_metadata_design:1' and design['status']=='GENERATED_VERIFIABLE_TRAIN_INTERVENTION_AND_NEW_FINAL_PLANS_NOT_ACQUIRED','Original design header')
    for name in ['acquisition_report','dataset_audit','acquisition_design']:
        require(p['inputs'][name]==design['inputs'][name],'Exact original design parent: '+name)
    require(p['inputs']['restore_contract']==design['inputs']['restore_saved_fixture'],'Same limited restore evidence')
    for key,file in [('contexts','contexts72.json'),('branch_plan','all_baselines_then_sides_plan.json'),('restore_arrays','source_before_action_restore_arrays.npz'),('partition_policy','exposure_and_partition_policy.json')]:
        require(p['inputs'][key]==design['outputs'][file],'Design semantic input membership: '+key)
    acquired=read(checked(p['inputs']['acquisition_report']))
    limits=np.asarray(read(checked(p['inputs']['acquisition_design']))['actual_limits']['actuator_ctrlrange20x2_F64'],dtype=np.float64)
    require(limits.shape==(20,2) and np.isfinite(limits).all(),'Frozen true actuator bounds')
    audit=read(checked(p['inputs']['dataset_audit']))
    require(acquired['schema']=='phone_episode_acquisition_result:1' and acquired['status']=='COMPLETE_GROUPED_NATIVE_ACQUISITION','Original saved native data')
    require(audit['engineering_qualified'] is True and audit['dataset_report']==p['inputs']['acquisition_report'] and all(v is True for v in audit['checks'].values()),'Original engineering review only')
    require(p['inputs']['identities']==acquired['sample_identities'] and p['inputs']['episodes']==acquired['episodes'],'Raw identity/episode joins')
    restore=read(checked(p['inputs']['restore_contract']))
    require(restore['schema']=='native_saved_action_restore_actual_fixture:1' and restore['status']=='PASS_FOUR_STATE_384STEP_ORIGINAL_ACTION_REPLAY' and restore['probe']==p['sources']['native_probe'] and all(v is True for v in restore['checks'].values()),'Existing limited restore qualification, not full hidden-state claim')
    contexts=read(checked(p['inputs']['contexts']));plan=read(checked(p['inputs']['branch_plan']))
    policy=read(checked(p['inputs']['partition_policy']))
    require(len(contexts)==72 and len(plan)==2952,'All72 sources/2952 records incl10 unavailable')
    require(policy['known_state_response_fit_family_ids']==[0,8,16,24,32,40] and policy['known_state_heldout_action_check_family_ids']==[5,13,21,29,37,45],'Original family partitions unchanged')
    require(policy['original_VAL476_diagnostic_branches_allowed_in_TRAIN'] is False and policy['fresh_final_not_acquired_or_fit'] is True,'No VAL leakage/fresh relabel')
    aliases={}
    for i,c in enumerate(contexts):
        require(c['context_id']==c['restore_array_index']==i and c['original_partition']=='train' and c['partition'] in PARTITIONS,'Dense context/knownTRAIN partition')
        require(c['native_h_s'] in DT and c['n_substeps'] in (64,128) and c['native_h_s']*c['n_substeps']==.02,'Fixed20ms integer native ticks')
        require(c['before_action_receipt']['signature']==16383 and c['before_action_receipt']['size']==300,'Full current integration spec')
        safety=c['hard_stops']
        require(safety['nonfinite'] is True and safety['any_native_warning'] is True and
            safety['COM_drift_from_episode_initial_m']==.15 and safety['orientation_from_episode_initial_rad']==np.pi/2 and
            safety['max_native_penetration_m']==.005 and safety['max_joint_limit_excursion_rad']==.05,'Unchanged Reader stop thresholds/original episode origin')
        expected='CAUSAL_FIT_KNOWN_TRAIN_STATE' if c['original_family_id'] in policy['known_state_response_fit_family_ids'] else 'HELDOUT_ACTION_RESPONSE_CHECK_KNOWN_TRAIN_STATE'
        require(c['partition']==expected and c['original_family_id'] in policy['known_state_response_fit_family_ids']+policy['known_state_heldout_action_check_family_ids'],'No family reassignment')
        key=(c['native_h_s'],c['before_action_receipt']['sha256']);aliases.setdefault(key,set()).add(c['partition'])
    require(all(len(v)==1 for v in aliases.values()),'No cross-partition full-state aliases')
    available=0;steps=0
    for i,b in enumerate(plan):
        c=contexts[b['context_id']]
        require(b['branch_id']==i and b['source_before_action_array_index']==c['context_id'] and b['partition']==c['partition'] and b['new_response_family_id']==c['new_response_family_id'],'Plan exact order/context/family')
        u=np.asarray(b['endpoint_u20_F64'],dtype=np.float64);base=np.asarray(c['baseline_u20_F64'],dtype=np.float64)
        require(u.shape==(20,) and np.isfinite(u).all() and sha(u)==b['endpoint_u_raw_sha256'],'Frozen absolute endpoint bytes')
        require(np.all(u>=limits[:,0]) and np.all(u<=limits[:,1]),'True hardware bounds, no clipping')
        if i<72:
            require(b['kind']=='BASELINE' and b['context_id']==i and bits(u,base),'All72 baselines first')
            steps+=c['n_substeps']
        else:
            require(b['kind']=='FINITE_SIDE' and type(b['available']) is bool,'Retain all2880 side records')
            j=b['joint_index'];sign=b['sign'];off=np.arange(20)!=j
            require(0<=j<20 and sign in (-1,1) and bits(u[off],base[off]),'Only declared coordinate may change')
            require(b['context_id']==(i-72)//40 and j==((i-72)%40)//2 and sign==(-1 if (i-72)%2==0 else 1),'Complete context/joint/sign mechanical order')
            du=float(u[j]-base[j]);du32=float(np.float32(u[j])-np.float32(base[j]))
            require(du==b['actual_delta_u_F64_rad'] and du32==b['represented_delta_u_FP32_rad'],'Actual F64 and encoded FP32 deltas')
            require(b['available']==bool(sign*du>=1e-6),'Frozen availability threshold, no rescue or clipping')
            if b['available']:available+=1;steps+=c['n_substeps']
    require(available==2870 and steps==282432 and p['budget']['maximum_native_steps']==steps,'Exact executable count/budget')
    out=Path(p['output']);require(out.is_absolute() and not out.exists(),'Fresh absolute output')
    return p,contexts,plan,acquired,out


def prepare_operands(p,contexts):
    """Read saved current state, all baseline targets and original stop origins.

    Native is not used here. Raw qpos/qvel come from initial or prior POST,
    never from a guessed INTEGRATION slice or quaternion matrix inversion.
    """
    identities=read(checked(p['inputs']['identities']));byid={r['sample_id']:r for r in identities}
    byep={(r['episode_id'],r['interval_index']):r for r in identities}
    with np.load(checked(p['inputs']['restore_arrays']),allow_pickle=False) as z:restore={k:z[k].copy() for k in z.files}
    arrays={k:[] for k in ['source_integration_F64','initial_qpos27_F64','initial_state58_F64','source_qpos27_F64','source_qvel26_F64','source_state58_F64','original_target_integration_F64','original_target_qpos27_F64','original_target_qvel26_F64','baseline_observable59_F64']}
    states=[];initials={};reads=1
    for c in contexts:
        i=c['context_id'];r=byid[c['source_sample_id']]
        require(r['trace']==c['source_trace'] and r['episode_id']==c['source_episode_id'] and r['split']=='train' and r['training_eligible'] is True,'Selected qualified actual source trace')
        with np.load(checked(c['source_trace']),allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
        reads+=1
        if c['source_episode_id'] not in initials:
            with np.load(checked(c['original_episode_initial']),allow_pickle=False) as z:initials[c['source_episode_id']]={k:z[k].copy() for k in ['qpos27','qvel26','state58']}
            reads+=1
        init=initials[c['source_episode_id']]
        if c['interval_index']==0:source=init
        else:
            prev=byep[(c['source_episode_id'],c['interval_index']-1)]
            require(prev['complete_H20ms'] is True and prev['target_receipt']['sha256']==c['before_action_receipt']['sha256'],'Previous full target = current before action')
            with np.load(checked(prev['trace']),allow_pickle=False) as z:source=dict(qpos27=z['post_qpos27'][-1].copy(),qvel26=z['post_qvel26'][-1].copy())
            reads+=1
        before=restore['source_integration_state_F64'][i]
        require(before.dtype==np.float64 and before.shape==(300,) and sha(before)==c['before_action_receipt']['sha256'] and bits(before,trace['source_integration_state']),'Full300 original source bytes')
        require(bits(restore['source_after_action_integration_state_F64'][i],trace['source_after_action_integration_state']) and sha(trace['source_after_action_integration_state'])==c['after_action_receipt']['sha256'],'After-action300 exact')
        require(bits(restore['x58_F64'][i],trace['source_state58']) and bits(restore['baseline_u20_F64'][i],trace['actual_control20']),'Original current x/u bytes')
        require(sha(trace['target_integration_state'])==r['target_receipt']['sha256'],'Original target300 receipt')
        pad=trace['pre_pad_normal_N2'];force=np.r_[np.min(pad,axis=0),pad[-1],np.max(trace['pre_phone_nonpad_normal_N'])]
        observable=np.r_[trace['physical_world_delta52'],np.mean(pad>.1,axis=0),np.log1p(force)]
        vals=[before,init['qpos27'],init['state58'],source['qpos27'],source['qvel26'],trace['source_state58'],trace['target_integration_state'],trace['post_qpos27'][-1],trace['post_qvel26'][-1],observable]
        for name,v in zip(arrays,vals):arrays[name].append(v)
        states.append(dict(state_id=i,source_before_action_receipt=c['before_action_receipt'],original_source_after_action_receipt=c['after_action_receipt'],original_target_receipt=r['target_receipt']))
    return states,{k:np.asarray(v) for k,v in arrays.items()},reads


def persist_branch(out,b,c,value):
    idx=b['branch_id'];path=out/'branches'/f'branch_{idx:04d}.npz'
    base=np.asarray(c['baseline_u20_F64'],dtype=np.float64);u=value['actual_control20']
    arrays=dict(value['trace'])|dict(source_state58=value['source_state58'],actual_control20=u,actual_control20_FP32=u.astype(np.float32),
        actual_delta_control20_F64=u-base,represented_delta_control20_FP32=u.astype(np.float32)-base.astype(np.float32),
        observed_prefix_target58=value['target_state58'],target_at_H_state58=value['target_state58'] if value['complete_H20ms'] else np.full(58,np.nan),
        physical_world_delta52=value['physical_world_delta52'],observable59=value['observable59'],
        source_integration_state=value['source_receipt']['integration_state'],source_after_action_integration_state=value['source_after_action_receipt']['integration_state'],target_integration_state=value['target_receipt']['integration_state'],
        warning_numbers=value['warning_numbers'])
    save(path,**arrays)
    return dict(branch_id=idx,context_id=c['context_id'],kind=b['kind'],partition=c['partition'],new_response_family_id=c['new_response_family_id'],original_family_id=c['original_family_id'],
        source_sample_id=c['source_sample_id'],original_partition='train',source_episode_id=c['source_episode_id'],interval_index=c['interval_index'],source_branch=c['source_branch'],
        native_h_s=c['native_h_s'],n_substeps=c['n_substeps'],output=pin(path),status='RETURNED_NATIVE_BRANCH',executed=True,
        plan=b,complete_H20ms=value['complete_H20ms'],training_eligible=value['training_eligible'],finite=value['finite'],termination=value['termination'],executed_substeps=value['returned_substeps'],
        source_receipt=compact(value['source_receipt']),source_after_action_receipt=compact(value['source_after_action_receipt']),target_receipt=compact(value['target_receipt']),
        summary=value['summary'],warning_numbers=value['warning_numbers'],force_phase=value['force_phase'],
        semantics='Hypothetical native same-state branch, original two designated pads15/28; not a held robot trajectory or holding success')


def aggregate(out,plan,contexts,records):
    """All2952 records, including unavailable and partials; separate partitions."""
    byid={v['branch_id']:v for v in records};outputs={};baseline_x={};reads=0
    for c in contexts:
        with np.load(checked(byid[c['context_id']]['output']),allow_pickle=False) as z:baseline_x[c['context_id']]=z['source_state58'].copy()
        reads+=1
    for partition in sorted(PARTITIONS):
        chosen=[b for b in plan if b['partition']==partition];arr={k:[] for k in ['branch_id_i64','context_id_i64','original_family_id_i64','native_h_s_F64','n_substeps_i64','available_bool','executed_bool','executed_substeps_i64','complete_H_bool','training_eligible_bool','x58_F64','u20_F64','u20_FP32','actual_du20_F64','represented_du20_FP32','observed_prefix_y58_F64','y58_at_H_F64','target_delta52_F64','contact_summary_F64']}
        for b in chosen:
            c=contexts[b['context_id']];r=byid[b['branch_id']];u=np.asarray(b['endpoint_u20_F64'],dtype=np.float64);base=np.asarray(c['baseline_u20_F64'],dtype=np.float64)
            x=np.full(58,np.nan);prefix=x.copy();y=x.copy();delta=np.full(52,np.nan);s7=np.full(7,np.nan)
            # Unavailable records retain x from the matching baseline, not a fake outcome.
            x=baseline_x[c['context_id']]
            if r['executed']:
                with np.load(checked(r['output']),allow_pickle=False) as z:prefix=z['observed_prefix_target58'].copy();y=z['target_at_H_state58'].copy();delta=z['physical_world_delta52'].copy()
                reads+=1
                s=r['summary'];s7=np.r_[s['pad_coverage2'],s['pad_min_normal_N2'],s['pad_endpoint_normal_N2'],s['phone_nonpad_max_normal_N']]
            vals=[b['branch_id'],c['context_id'],c['original_family_id'],c['native_h_s'],c['n_substeps'],b.get('available',True),r['executed'],r.get('executed_substeps',0),r.get('complete_H20ms',False),r.get('training_eligible',False),x,u,u.astype(np.float32),u-base,u.astype(np.float32)-base.astype(np.float32),prefix,y,delta,s7]
            for name,v in zip(arr,vals):arr[name].append(v)
        path=out/(partition+'.npz');save(path,**{k:np.asarray(v) for k,v in arr.items()});outputs[partition]=pin(path)
    return outputs,reads


def execute(protocol):
    protocol=Path(protocol).resolve(strict=True);p,contexts,plan,acquired,out=preflight(protocol)
    require(p['CPU_run_authorized'] is True,'Root must explicitly authorize a fresh execution protocol')
    out.mkdir();(out/'branches').mkdir();(out/'baseline_checks').mkdir();write(out/'protocol.json',p);write(out/'all_plan_records.json',plan)
    started=time.perf_counter();records=[];probes={};phase='prepare_saved_operands';active_branch=None
    try:
        states,operands,input_NPZ_reads=prepare_operands(p,contexts)
        probe=loadmod(checked(p['sources']['native_probe']))
        for h in DT:
            probes[h]=probe.NativeSavedActionProbe(checked(p['inputs']['model_mjb']),dt_s=h,expected_effective_model_digest=acquired['model_audits'][str(h)]['model_digest'])
        require(len({v.reader.shared_physics_digest for v in probes.values()})==1,'Only native dt differs, not physics parameters')
        baseline_results=[]
        for b in plan[:72]:
            c=contexts[b['context_id']];active_branch=b;phase=f'baseline_{b["branch_id"]}'
            with np.load(checked(c['source_trace']),allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
            input_NPZ_reads+=1
            value=probes[c['native_h_s']].run(states[c['context_id']],operands,b['endpoint_u20_F64'])
            record=persist_branch(out,b,c,value);records.append(record)
            checks=probe.baseline_checks(value,states[c['context_id']],operands,trace=original)
            for key,v in value['trace'].items():checks['all_captured_ticks_'+key]=bits(v,original[key])
            checks['source_x58']=bits(value['source_state58'],original['source_state58'])
            checks['actual_u20']=bits(value['actual_control20'],original['actual_control20'])
            checks['target58_and_world_delta52']=bits(value['target_state58'],original['observed_prefix_target58']) and bits(value['physical_world_delta52'],original['physical_world_delta52'])
            checks['target_global_tick_and_clock']=value['target_receipt']['global_tick']==states[c['context_id']]['original_target_receipt']['global_tick'] and value['target_receipt']['native_time_s']==states[c['context_id']]['original_target_receipt']['native_time_s']
            result=dict(context_id=c['context_id'],branch_id=b['branch_id'],checks=checks,all_passed=all(checks.values()),branch_output=record['output'])
            write(out/'baseline_checks'/f'baseline_{c["context_id"]:02d}.json',result);baseline_results.append(result)
            require(result['all_passed'],'Baseline replay mismatch: ALL new sides remain forbidden')
            if len(records)%12==0:print(json.dumps(dict(phase='baseline',completed=len(records),all_returned_baselines_bitexact=True)),flush=True)
        # Global barrier: not one side has been attempted until all72 passed.
        require(len(baseline_results)==72 and all(v['all_passed'] for v in baseline_results),'Global72-baseline barrier')
        write(out/'all72_baseline_barrier.json',dict(schema='phone_all72_before_any_side_replay_barrier:1',all72_passed=True,side_attempts_at_barrier=0,baseline_results=baseline_results))
        for b in plan[72:]:
            c=contexts[b['context_id']];active_branch=b;phase=f'side_{b["branch_id"]}'
            if not b['available']:
                records.append(dict(branch_id=b['branch_id'],context_id=c['context_id'],partition=c['partition'],new_response_family_id=c['new_response_family_id'],original_family_id=c['original_family_id'],kind=b['kind'],plan=b,status='UNAVAILABLE_RETAINED_NOT_EXECUTED',executed=False,executed_substeps=0,training_eligible=False,complete_H20ms=False));continue
            value=probes[c['native_h_s']].run(states[c['context_id']],operands,b['endpoint_u20_F64'])
            records.append(persist_branch(out,b,c,value))
            if len(records)%240==0:print(json.dumps(dict(phase='side',records_retained=len(records))),flush=True)
        audits={str(h):v.audit() for h,v in probes.items()}
        counts={k:sum(v.reader.counts[k] for v in probes.values()) for k in next(iter(probes.values())).reader.counts}
        probe_counts={k:sum(v.counts[k] for v in probes.values()) for k in next(iter(probes.values())).counts}
        require(len(records)==2952 and probe_counts['branches_returned']==2942 and counts['integration_steps']<=282432,'All planned records and bounded actual steps')
        require(counts['initial_forward']==counts['observer_forward']==probe_counts['restore_forward']==0,'No observer/reset/restore forward')
        write(out/'branches.json',records);arrays,derived_NPZ_reads=aggregate(out,plan,contexts,records)
        result=dict(schema='phone_causal_TRAIN_native_action_acquisition_result:1',status='COMPLETE_ALL72_BASELINES_THEN_FULL20_SIDES_RETAINED',protocol=pin(protocol),sources=p['sources'],inputs=p['inputs'],
            context_count=72,plan_records=2952,baseline_branches=72,side_branches_returned=2870,unavailable_sides_retained=10,all72_baselines_passed_before_any_side=True,
            counts=counts,probe_counts=probe_counts,model_audits=audits,input_NPZ_reads=input_NPZ_reads,own_output_NPZ_reads_for_partition_arrays=derived_NPZ_reads,
            complete_H_branches=sum(bool(v.get('complete_H20ms')) for v in records),partial_executed_branches=sum(v['executed'] and not v['complete_H20ms'] for v in records),
            explicit_training_eligible_branches=sum(bool(v.get('training_eligible')) for v in records),records=pin(out/'branches.json'),arrays=arrays,baseline_barrier=pin(out/'all72_baseline_barrier.json'),
            exposure_policy=read(checked(p['inputs']['partition_policy'])),elapsed_s=time.perf_counter()-started,
            NN_training_AD_GPU_calls=0,fresh_final_families_acquired=0,original_VAL_TEST_native_branch_sources=0,
            restored_full_MjData_or_Markov_or_true_Jacobian_qualified=False,holding_or_ego_or_CHORD_benefit_qualified=False,
            scope='Known original TRAIN states, family-separated response fit/action-check only. Original two designated pads15/28 unchanged. Native decoded PRE contact/force labels and POST state targets, not real human force. Unavailable/partial/safety events retained; future fullH contacts never deployable input. No learned update or physical holding success qualification.')
        write(out/'report.json',result);print(json.dumps(dict(status=result['status'],counts=counts,branches=2942,partial=result['partial_executed_branches'],report=str(out/'report.json'))),flush=True)
        return 0
    except BaseException as err:
        fail=dict(schema='phone_causal_TRAIN_acquisition_failure:1',phase=phase,error=repr(err),traceback=traceback.format_exc(),active_plan_record=active_branch,
            returned_records=records,counters={str(h):v.reader.counts for h,v in probes.items()},automatic_retry=False,
            scope='All returned traces/partial hardstop results preserved. If a frozen helper raised before return, only terminal data can be saved; no missing tick trace is fabricated.')
        if active_branch is not None:
            current=probes.get(contexts[active_branch['context_id']]['native_h_s'])
            if current is not None and current.reader.d is not None:
                r=current.reader;receipt=r.receipt();path=out/'unexpected_failure_terminal_native_state.npz'
                save(path,integration_state=receipt['integration_state'],qpos27=r.d.qpos.copy(),qvel26=r.d.qvel.copy(),qacc26=r.d.qacc.copy(),ctrl20=r.d.ctrl.copy(),warning_numbers=np.array(r.d.warning.number))
                fail['unexpected_failure_terminal_state']=pin(path);fail['terminal_receipt']=compact(receipt)
        write(out/'failure.json',fail)
        print(json.dumps(dict(status='FAILED_PRESERVED_NO_RETRY',phase=phase,error=repr(err))),flush=True)
        return 1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.run:return execute(a.protocol)
    p,c,b,r,out=preflight(a.protocol)
    print(json.dumps(dict(status='METADATA_ONLY_NO_NPZ_NATIVE_TORCH',CPU_run_authorized=p['CPU_run_authorized'],contexts=len(c),plan_records=len(b),executable_branches=2942,unavailable=10,maximum_steps=282432,output=str(out))))
    return 0

if __name__=='__main__':raise SystemExit(main())
