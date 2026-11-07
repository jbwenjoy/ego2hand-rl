"""Author saved-artifact review; independent value comparisons, no native/Torch.

Rechecks all returned branch/group schemas, controls/receipts/masks and all72
baseline raw traces. Does not replay native dynamics or claim a second author.
"""
from pathlib import Path
import argparse,csv,hashlib,json,traceback
import numpy as np

RUNNER_SHA='c852a57993567bee071648df81ae15ea57be59826f9a2a1b1238a846c03b32ee'
PARTITIONS=['CAUSAL_FIT_KNOWN_TRAIN_STATE','HELDOUT_ACTION_RESPONSE_CHECK_KNOWN_TRAIN_STATE']
TRACE_KEYS=['pre_tick','post_tick','pre_native_time_s','post_native_time_s','post_qpos27','post_qvel26','post_state58',
 'pre_pad_normal_N2','pre_phone_nonpad_normal_N','pre_table_normal_N','pre_penetration_m','pre_self_penetration_m',
 'pre_energy2','pre_contact_wrench_about_COM_world6','pre_actuator_force20','pre_qfrc_actuator20','contact_rows24']

def need(x,msg):
    if not x:raise ValueError(msg)
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):need(pin(v['path'])==v,'Pin mismatch: '+v['path']);return Path(v['path'])
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True)
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def summary7(r):
    s=r['summary'];return np.r_[s['pad_coverage2'],s['pad_min_normal_N2'],s['pad_endpoint_normal_N2'],s['phone_nonpad_max_normal_N']].astype(np.float64)


def audit(producer_report,output,terminal=None):
    producer_report=Path(producer_report).resolve(strict=True);out=Path(output).resolve();need(not out.exists(),'Fresh review output');out.mkdir()
    counts=dict(own_branch_NPZ_reads=0,original_baseline_NPZ_reads=0,partition_NPZ_reads=0,native_model_FK_forward_steps=0,Torch_NN_AD_training_calls=0)
    try:
        r=read(producer_report);p=read(checked(r['protocol']));root=producer_report.parent
        need(r['schema']=='phone_causal_TRAIN_native_action_acquisition_result:1' and r['status']=='COMPLETE_ALL72_BASELINES_THEN_FULL20_SIDES_RETAINED','Completed actual acquisition only')
        need(r['sources']['runner']['sha256']==RUNNER_SHA,'Frozen c852 producer')
        for v in [*r['sources'].values(),*r['inputs'].values()]:checked(v)
        need(p['CPU_run_authorized'] is True and p['training_authorized'] is False and p['fresh_final_acquisition_authorized'] is False,'Native-only Root authorization')
        need(Path(p['output']).samefile(root),'Actual output path join')
        contexts=read(checked(r['inputs']['contexts']));plan=read(checked(r['inputs']['branch_plan']));policy=read(checked(r['inputs']['partition_policy']))
        records=read(checked(r['records']));barrier=read(checked(r['baseline_barrier']))
        need(len(contexts)==72 and len(plan)==len(records)==2952,'Every planned record retained')
        need(barrier['schema']=='phone_all72_before_any_side_replay_barrier:1' and barrier['all72_passed'] is True and barrier['side_attempts_at_barrier']==0 and len(barrier['baseline_results'])==72,'Saved global barrier')
        for i,b in enumerate(barrier['baseline_results']):
            need(b['context_id']==b['branch_id']==i and b['all_passed'] is True and all(v is True for v in b['checks'].values()),'All baseline check receipts pass')
            need(read(root/'baseline_checks'/f'baseline_{i:02d}.json')==b,'Individual receipt = barrier entry')
        groups={k:None for k in PARTITIONS}
        for k in groups:
            with np.load(checked(r['arrays'][k]),allow_pickle=False) as z:groups[k]={n:z[n].copy() for n in z.files}
            counts['partition_NPZ_reads']+=1
        expected_group_keys={'branch_id_i64','context_id_i64','original_family_id_i64','native_h_s_F64','n_substeps_i64','available_bool','executed_bool','executed_substeps_i64','complete_H_bool','training_eligible_bool','x58_F64','u20_F64','u20_FP32','actual_du20_F64','represented_du20_FP32','observed_prefix_y58_F64','y58_at_H_F64','target_delta52_F64','contact_summary_F64'}
        index={};group_counts={}
        for group,z in groups.items():
            need(set(z)==expected_group_keys,'Exact grouped array schema')
            n=len(z['branch_id_i64']);need(n==1476,'36contexts times41 records per partition')
            for k,a in z.items():
                dt=np.int64 if k.endswith('_i64') else np.bool_ if k.endswith('_bool') else np.float32 if k.endswith('_FP32') else np.float64
                need(a.dtype==dt and len(a)==n,'Group typed first dimension: '+k)
            for j,bid in enumerate(z['branch_id_i64']):need(int(bid) not in index,'No duplicate/group-crossed branch');index[int(bid)]=(group,j)
            group_counts[group]=dict(plan_records=n,executed=int(z['executed_bool'].sum()),complete_H=int(z['complete_H_bool'].sum()),reader_label_valid=int(z['training_eligible_bool'].sum()))
        need(set(index)==set(range(2952)),'Both groups form exact original plan partition')
        checks={k:True for k in ['all_plan_identities_and_original_TRAIN_family_groups','global72_baseline_receipts_and_raw_traces',
            'actual_controls_duF64_FP32_and_source_receipts','per_tick_schema_clock_contact_rows_and_prefix',
            'partial_unavailable_and_Reader_validity_masks','all_partition_arrays_match_branch_captures',
            'counts_and_unchanged_effective_models_zero_observer_forward']}
        issues=[];rows=[];baseline_x={};steps=contact_count=executed=partial=eligible=0
        def mark(name,ok,bid,detail):
            if not ok:checks[name]=False;issues.append(dict(branch_id=bid,check=name,detail=detail))
        for bid,(b,record) in enumerate(zip(plan,records)):
            c=contexts[b['context_id']];group,j=index[bid];z=groups[group];kind=b['kind'];base=np.asarray(c['baseline_u20_F64'],dtype=np.float64);u=np.asarray(b['endpoint_u20_F64'],dtype=np.float64)
            identity=(record['branch_id']==b['branch_id']==bid and record['context_id']==b['context_id'] and record['plan']==b and record['partition']==c['partition']==group and record['new_response_family_id']==c['new_response_family_id'] and record['original_family_id']==c['original_family_id'] and c['original_partition']=='train')
            mark('all_plan_identities_and_original_TRAIN_family_groups',identity,bid,'Record/plan/context/source role join')
            expected_family=policy['known_state_response_fit_family_ids'] if group==PARTITIONS[0] else policy['known_state_heldout_action_check_family_ids']
            mark('all_plan_identities_and_original_TRAIN_family_groups',c['original_family_id'] in expected_family,bid,'Family role')
            available=b.get('available',True);did=record['executed'];need(did==available,'Only predeclared available branches executed')
            ex=0;ncontact=0;x=baseline_x.get(c['context_id']);prefix=np.full(58,np.nan);y=prefix.copy();delta=np.full(52,np.nan);s7=np.full(7,np.nan)
            if did:
                executed+=1
                with np.load(checked(record['output']),allow_pickle=False) as f:v={k:f[k].copy() for k in f.files}
                counts['own_branch_NPZ_reads']+=1
                need(set(v)==set(TRACE_KEYS)|{'source_state58','actual_control20','actual_control20_FP32','actual_delta_control20_F64','represented_delta_control20_FP32','observed_prefix_target58','target_at_H_state58','physical_world_delta52','observable59','source_integration_state','source_after_action_integration_state','target_integration_state','warning_numbers'},'Exact branch NPZ key ABI')
                x=v['source_state58'];prefix=v['observed_prefix_target58'];y=v['target_at_H_state58'];delta=v['physical_world_delta52'];s7=summary7(record)
                ex=record['executed_substeps'];steps+=ex;ncontact=len(v['contact_rows24']);contact_count+=ncontact
                actualok=bits(v['actual_control20'],u) and bits(v['actual_control20_FP32'],u.astype(np.float32)) and bits(v['actual_delta_control20_F64'],u-base) and bits(v['represented_delta_control20_FP32'],u.astype(np.float32)-base.astype(np.float32))
                for key,receipt in [('source_integration_state',record['source_receipt']),('source_after_action_integration_state',record['source_after_action_receipt']),('target_integration_state',record['target_receipt'])]:
                    actualok=actualok and v[key].dtype==np.float64 and v[key].shape==(300,) and sha(v[key])==receipt['sha256'] and receipt['signature']==16383 and receipt['size']==300
                actualok=actualok and record['source_receipt']['sha256']==c['before_action_receipt']['sha256']
                mark('actual_controls_duF64_FP32_and_source_receipts',actualok,bid,'Endpoint controls and full300 raw receipts')
                start=c['before_action_receipt']['global_tick'];tick=np.arange(start,start+ex,dtype=np.int64)
                schemaok=(1<=ex<=c['n_substeps'] and bits(v['pre_tick'],tick) and bits(v['post_tick'],tick+1))
                for key,shape in [('post_qpos27',(ex,27)),('post_qvel26',(ex,26)),('post_state58',(ex,58)),('pre_pad_normal_N2',(ex,2)),('pre_energy2',(ex,2)),('pre_contact_wrench_about_COM_world6',(ex,6)),('pre_actuator_force20',(ex,20)),('pre_qfrc_actuator20',(ex,20)),('contact_rows24',(ncontact,24))]:
                    schemaok=schemaok and v[key].dtype==np.float64 and v[key].shape==shape
                schemaok=schemaok and bits(prefix,v['post_state58'][-1]) and float(v['pre_native_time_s'][0])==c['before_action_receipt']['native_time_s'] and float(v['post_native_time_s'][-1])==record['target_receipt']['native_time_s'] and record['target_receipt']['global_tick']==start+ex
                contact=v['contact_rows24'];schemaok=schemaok and (not ncontact or (np.isin(contact[:,0],tick).all() and np.equal(contact[:,:5],np.floor(contact[:,:5])).all()))
                mark('per_tick_schema_clock_contact_rows_and_prefix',schemaok,bid,'Typed raw tick/phase capture schema')
                valid=bool(ex==c['n_substeps'] and record['finite'] and not np.any(v['warning_numbers']))
                maskok=(record['complete_H20ms']==(ex==c['n_substeps']) and record['training_eligible']==valid and np.array_equal(v['warning_numbers'],record['warning_numbers']))
                if not record['complete_H20ms']:partial+=1;maskok=maskok and np.isnan(y).all() and np.isnan(delta).all()
                else:maskok=maskok and bits(y,prefix)
                if not valid:maskok=maskok and np.isnan(v['observable59']).all() and np.isnan(delta).all()
                else:eligible+=1;maskok=maskok and all(np.isfinite(v[k]).all() for k in ['source_state58','actual_control20','post_qpos27','post_qvel26','post_state58','contact_rows24','physical_world_delta52','observable59'])
                mark('partial_unavailable_and_Reader_validity_masks',maskok,bid,'Explicit wholeH/finite/no-warning masks; no synthesized partial target')
                if bid<72:
                    baseline_x[c['context_id']]=x.copy()
                    with np.load(checked(c['source_trace']),allow_pickle=False) as f:original={k:f[k].copy() for k in f.files}
                    counts['original_baseline_NPZ_reads']+=1
                    replay=all(bits(v[k],original[k]) for k in TRACE_KEYS)
                    replay=replay and bits(x,original['source_state58']) and bits(u,original['actual_control20']) and bits(delta,original['physical_world_delta52']) and bits(prefix,original['observed_prefix_target58'])
                    replay=replay and all(bits(v[k],original[k]) for k in ['source_integration_state','source_after_action_integration_state','target_integration_state'])
                    replay=replay and barrier['baseline_results'][bid]['branch_output']==record['output']
                    mark('global72_baseline_receipts_and_raw_traces',replay,bid,'All captured baseline bytes vs original selected TRAIN')
                else:mark('actual_controls_duF64_FP32_and_source_receipts',bits(x,baseline_x[c['context_id']]),bid,'All sides have the same current state as their baseline')
            else:
                mark('partial_unavailable_and_Reader_validity_masks',record['status']=='UNAVAILABLE_RETAINED_NOT_EXECUTED' and record['executed_substeps']==0 and record['training_eligible'] is False and 'output' not in record,bid,'Unavailable retained, no outcome file')
            expected=dict(branch_id_i64=bid,context_id_i64=c['context_id'],original_family_id_i64=c['original_family_id'],native_h_s_F64=c['native_h_s'],n_substeps_i64=c['n_substeps'],available_bool=available,executed_bool=did,executed_substeps_i64=ex,complete_H_bool=record.get('complete_H20ms',False),training_eligible_bool=record.get('training_eligible',False),x58_F64=x,u20_F64=u,u20_FP32=u.astype(np.float32),actual_du20_F64=u-base,represented_du20_FP32=u.astype(np.float32)-base.astype(np.float32),observed_prefix_y58_F64=prefix,y58_at_H_F64=y,target_delta52_F64=delta,contact_summary_F64=s7)
            grouped=all(same(z[k][j],np.asarray(value,dtype=z[k].dtype)) for k,value in expected.items())
            mark('all_partition_arrays_match_branch_captures',grouped,bid,'Every grouped feature/label/mask from the matching branch')
            rows.append(dict(branch_id=bid,context_id=c['context_id'],partition=group,original_family_id=c['original_family_id'],executed=did,complete_H=record.get('complete_H20ms',False),reader_label_valid=record.get('training_eligible',False),executed_substeps=ex,generated_contacts=ncontact))
        nc=r['counts'];pc=r['probe_counts']
        countok=(executed==2942 and steps==nc['integration_steps']<=282432 and partial==r['partial_executed_branches'] and eligible==r['explicit_training_eligible_branches'] and nc['MjData']==pc['branches_returned']==pc['branches_attempted']==pc['state_sets']==2942 and nc['MJB_reads']==2 and nc['contact_force_reads']==nc['step_contact_force_reads']==contact_count and nc['initial_forward']==nc['observer_forward']==nc['initial_contact_force_reads']==pc['restore_forward']==0)
        audits=r['model_audits'];countok=countok and len({v['model']['shared_physics_digest_without_dt'] for v in audits.values()})==1 and all(v['model']['passed'] is True and v['restored_integration_is_full_MjData_qualified'] is False for v in audits.values())
        mark('counts_and_unchanged_effective_models_zero_observer_forward',countok,-1,'Sum raw step/contact rows, fresh state sets and source-scoped model audits')
        write(out/'all2952_record_review.json',rows)
        with (out/'all2952_record_review.csv').open('x',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        terminal_pin=None
        if terminal is not None:
            t=read(terminal);need(t['returncode']==0 and t['protocol']==r['protocol'],'Root actual terminal exit0/protocol join');terminal_pin=pin(terminal)
        result=dict(schema='phone_causal_TRAIN_acquisition_author_saved_review:1',status='PASS_SAVED_GROUPS_BASELINE_BARRIER_SCHEMA_NOT_MODEL_OR_POLICY_QUALIFICATION' if all(checks.values()) else 'FAIL_SAVED_REVIEW_PRESERVED_NO_RETRY',
            reviewer_is_source_author=True,producer_report=pin(producer_report),protocol=r['protocol'],auditor=pin(__file__),terminal=terminal_pin,
            checks=checks,passed=sum(checks.values()),total=len(checks),issues=issues,group_counts=group_counts,
            counts=counts,raw_steps=steps,raw_generated_contact_rows=contact_count,baseline_raw_trace_comparisons=72,plan_records=2952,executed_branches=executed,unavailable_records=10,partial_executed_branches=partial,Reader_valid_branches=eligible,
            outputs={q.name:pin(q) for q in out.iterdir() if q.is_file()},
            scope='Author saved-value/group/phase/schema audit, zero native/Torch. All72 raw baseline traces rechecked. Barrier chronology source-supported plus saved receipt, not independent instrumentation. Reader finite qacc/modeleffective audit remain producer/source evidence; no model reload, native force recomputation, physics success or hidden-state qualification. Action-check remains outside FIT; fresh12families still unacquired.')
        write(out/'report.json',result);print(json.dumps(dict(status=result['status'],passed=result['passed'],total=result['total'],steps=steps,report=str(out/'report.json'))))
        return 0 if all(checks.values()) else 1
    except BaseException as e:
        write(out/'failure.json',dict(status='SAVED_REVIEW_FAILED_PRESERVED_NO_RETRY',error=repr(e),traceback=traceback.format_exc(),counts=counts));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--producer-report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--terminal',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:
        print(json.dumps(dict(status='DEFAULT_NO_ARRAY_OR_NATIVE_READ',producer_report=str(a.producer_report.resolve()),output=str(a.output.resolve()))))
    else:raise SystemExit(audit(a.producer_report,a.output,a.terminal))
