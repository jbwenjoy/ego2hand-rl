"""Bounded source-author saved integrity; no native or label-force re-solve."""
from pathlib import Path
from collections import defaultdict
import ast,hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent;B=H.parent

def need(v,msg):
    if not v:raise ValueError(msg)
def pin(f):
    f=Path(f).resolve(strict=True);b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):need(pin(v['path'])==v,'Current immutable pin');return Path(v['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def npz(v):
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)

def run():
    rp=pin(B/'native_completed_v1/report.json');r=read(rp['path']);tp=pin(B/'process_run1/terminal.json');term=read(tp['path'])
    need(term['returncode']==0 and term['protocol']==r['protocol'] and term['automatic_retry'] is False and term['new_learning_started'] is False,'Actual terminal0 and same protocol')
    p=read(checked(r['protocol']));need(term['source']==p['sources']['runner'] and r['data_role']==p['permanent_data_role']=='ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE','Actual source/heldout role')
    for v in [*p['inputs'].values(),*p['sources'].values()]:checked(v)
    checked(term['stderr']);need(term['stderr']['bytes']==0,'Empty actual stderr')
    plan=read(checked(r['selection']));operands=npz(r['operands']);rows=read(checked(r['branches']));bar=read(checked(r['baseline_barrier']))
    need(len(rows)==328 and [z['branch_id'] for z in rows]==list(range(328)) and len(plan['states'])==8,'All planned IDs/order retained')
    task=read(checked(r['task_bundle']));bounds=np.asarray(task['actuator_ctrlrange20x2'],dtype=np.float64)
    planned={z['branch_id']:z for st in plan['states'] for z in st['perturbations']}
    need(bar['all8_passed'] is True and bar['baseline_count']==8 and bar['new_sides_executed_before_barrier']==0 and len(bar['baselines'])==8,'Actual persisted global baseline barrier')
    need(sum(z['integration_steps'] for z in bar['counts'].values())==768 and sum(z['MjData'] for z in bar['counts'].values())==8,'Barrier native accounting has only8 original-action branches')
    baseline_proofs=[]
    for i,v in enumerate(bar['baselines']):
        z=read(checked(v));need(z['context_id']==z['branch_id']==i and z['passed'] is True and all(q is True for q in z['checks'].values()),'Each actual baseline saved check positive')
        need(z['raw_output']==rows[i]['raw_output'] and z['output']==rows[i]['output'],'Proof/output joins');baseline_proofs.append(v)
    # Extract only literal trace-key names from the frozen Reader, without importing it.
    tree=ast.parse(checked(p['sources']['native_reader']).read_text());tracekeys=None
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in n.targets) and isinstance(n.value,ast.DictComp):
            tracekeys=ast.literal_eval(n.value.generators[0].iter)+['contact_rows24'];break
    need(tracekeys is not None,'Frozen exact native trace ABI')
    contacts=steps=0;per_h=defaultdict(lambda:dict(branches=0,steps=0,contacts=0));source_groups=defaultdict(list);action_groups=defaultdict(list);partial=unknown=outside=0
    baseline_original={st['context_id']:npz(st['source_trace']) for st in plan['states']}
    for state in plan['states']:
        sid=state['context_id'];source=npz(state['qpos_source']);initial=npz(state['episode_initial']['arrays']);initial_task=npz(state['initial_task'])
        q=source['qpos27'] if state['step_k']==0 else source['post_qpos27'][-1]
        v=source['qvel26'] if state['step_k']==0 else source['post_qvel26'][-1]
        context=source['integration_state'] if state['step_k']==0 else source['target_integration_state']
        need(bits(q,operands['source_qpos27_F64'][sid]) and bits(v,operands['source_qvel26_F64'][sid]) and bits(context,operands['source_integration_F64'][sid]),'Source q/v/300 directly from original reset or previousPOST, no guessed slices')
        need(bits(context,baseline_original[sid]['source_integration_state']),'Original current interval before-action join')
        need(bits(initial['qpos27'],operands['initial_qpos27_F64'][sid]) and bits(initial['state58'],initial_task['state58_F64']) and bits(initial_task['state58_F64'],operands['initial_state58_F64'][sid]) and not np.any(initial['qvel26']),'Original episode safety origin retained, not current-context recentering')
    for rec in rows:
        bid=rec['branch_id'];sid=rec['context_id'];state=plan['states'][sid];base=np.asarray(state['baseline_ctrl20_F64'],dtype=np.float64)
        need(rec['state_id']==rec['baseline_branch_id']==sid and rec['family_id']==state['family_id'] and rec['episode_id']==state['episode_id'] and rec['source_sample_id']==state['source_sample_id'] and rec['step_k']==state['step_k'],'Perbranch source manifest identity')
        need(rec['data_role']==r['data_role'] and rec['permanently_excluded_from_train_and_monitorselection'] is True and rec['absolute_control_hypothetical_not_policy_action'] is True,'Never fit or policy-effect data')
        need(rec['status']=='EXECUTED','Actual plan has0 unavailable, no silent lost row')
        raw=npz(rec['raw_output']);out=npz(rec['output'])
        need(all(k in out and bits(a,out[k]) for k,a in raw.items()),'Every durable raw array unchanged in labeled output')
        need(bits(out['source_state58_F64'],operands['source_state58_F64'][sid]) and bits(out['baseline_control20_F64'],base),'Same source x and base control')
        before=out['source_integration_F64'];after=out['source_after_action_integration_F64'];target=out['target_integration_F64'];u=out['actual_control20_F64'];offset=state['source_control_segment_offset']
        need(bits(before,operands['source_integration_F64'][sid]) and bits(before[offset:offset+20],operands['source_previous_u20_F64'][sid]),'Original before-action context/previousu')
        need(bits(after[offset:offset+20],u) and bits(before[:offset],after[:offset]) and bits(before[offset+20:],after[offset+20:]),'Only measured control segment differs after action')
        need(u.dtype==np.float64 and u.shape==(20,) and bits(u,np.asarray(rec['actual_u20_F64'])) and np.all(u>=bounds[:,0]) and np.all(u<=bounds[:,1]),'Actual legal F64 u, no clipping')
        if bid<8:
            need(rec['kind']=='baseline' and bits(u,base),'Baseline originalaction')
            need(bits(target,operands['original_target_integration_F64'][sid]),'All8 target full300 byte reproduction')
            original=baseline_original[sid]
            need(all(bits(raw[k],original[k]) for k in tracekeys),'All8 every native trace array byte reproduction')
            for k in out:
                if 'baseline_'+k in operands:need(bits(out[k],operands['baseline_'+k][sid]),'Baseline actual labels byte-join prepared source labels')
        else:
            side=planned[bid];need(rec['perturbation']==side and side['available'] and rec['axis']==side['axis'] and rec['sign']==side['sign'],'All20 signed-side plans unchanged')
            changed=np.flatnonzero(u.view(np.uint64)!=base.view(np.uint64));need(np.array_equal(changed,[side['axis']]),'Exactly one changed control component')
            du=float(u[side['axis']]-base[side['axis']]);need(bits(np.asarray(du),out['actual_du_F64']) and du==rec['actual_du_F64']==side['actual_du_F64'] and abs(du)>=1e-6,'True F64 actualdu, not nominal denominator')
            need(bits(u,np.asarray(side['actual_ctrl20_F64'])) and side['posthoc_clip'] is False,'Frozen branch control bytes/no rescue')
            outside+=not side['Actor_EMA_reachable_descriptor']
        h=rec['dt_s'];n=state['n_substeps'];N=len(out['pre_tick']);start=state['source_before_action_receipt']['global_tick']
        need(N==rec['returned_substeps'] and np.array_equal(out['pre_tick'],np.arange(start,start+N)) and np.array_equal(out['post_tick'],np.arange(start+1,start+N+1)),'Integer PRE/POST interval phase')
        for key,vec,tick in [('source_receipt',before,start),('source_after_action_receipt',after,start),('target_receipt',target,start+N)]:
            m=rec[key];need(vec.dtype==np.float64 and vec.shape==(300,) and m['signature']==16383 and m['size']==300 and m['global_tick']==tick and sha(vec)==m['sha256'],'Raw300 receipt shape/clock/hash')
        need(rec['source_receipt']['native_time_s']==rec['source_after_action_receipt']['native_time_s']==float(out['pre_native_time_s'][0]) and rec['target_receipt']['native_time_s']==float(out['post_native_time_s'][-1]),'Actual PRE/POST native-clock joins, not rounded tick reconstruction')
        complete=N==n;known=bool(complete and rec['finite'] and rec['no_warning'])
        need(bool(out['complete_H_bool'])==rec['complete_H20ms']==complete and bool(out['physical_labels_known_bool'])==rec['labels_known']==known and not np.any(rec['warning_numbers']),'Actual mask consistency')
        need(bits(out['observed_prefix_target58_F64'],out['post_state58'][-1]),'Observed prefix ends actual last POST')
        if not complete:
            need(np.isnan(out['target_H_state58_F64']).all() and np.isnan(out['target_delta52_F64']).all(),'Partial cannot invent fullH labels');partial+=1
        if known:
            for key,size in [('source_state58_F64',58),('target_H_state58_F64',58),('target_delta52_F64',52),('target_physical_heads178_F64',178),('mean_positive_Fn22_N_F64',22),('Fn_gt_0p1_coverage22_F64',22)]:
                need(out[key].shape==(size,) and np.isfinite(out[key]).all(),'Known complete typed target '+key)
            need(bits(out['target_H_state58_F64'],out['observed_prefix_target58_F64']) and bits(out['target_delta52_F64'],out['raw_native_delta52_F64']),'H state/delta labels join actual native target')
            need(out['impulse_source_BODY_moving_COM22x6_F64'].shape==(22,6) and np.isfinite(out['impulse_source_BODY_moving_COM22x6_F64']).all(),'Known pair22 shape/finites')
            need(bits(out['target_physical_heads178_F64'][46:],out['impulse_source_BODY_moving_COM22x6_F64'].ravel()),'178 head preserves exact22x6 layout')
            cov=out['Fn_gt_0p1_coverage22_F64'];tri=float(out['simultaneous_triple_coverage_F64'])
            need(np.all((cov>=0)&(cov<=1)) and np.all(out['mean_positive_Fn22_N_F64']>=0) and 0<=tri<=min(cov[[5,9,13]])+1e-15,'Coverage/positiveFn/triple range only; no independent force relabel')
        else:unknown+=1;need(np.isnan(out['impulse_source_BODY_moving_COM22x6_F64']).all(),'Unknown labels remain masked')
        need('legacy_two_pad_observable59_NOT_tripod_GT' in out and all('legacy' not in v for v in r['field_map'].values()),'Legacy59 excluded from canonical mappings')
        cr=out['contact_rows24'];need(cr.ndim==2 and cr.shape[1]==24 and np.isin(cr[:,0],out['pre_tick']).all(),'Raw24contacts have valid solverPRE ticks')
        steps+=N;contacts+=len(cr);per_h[str(h)]['branches']+=1;per_h[str(h)]['steps']+=N;per_h[str(h)]['contacts']+=len(cr)
        source_groups[(rec['source_receipt']['sha256'],h)].append(bid);action_groups[(rec['source_after_action_receipt']['sha256'],sha(u),h,n)].append(bid)
    alias=read(checked(r['alias_review']));need(alias['same_source_context_groups']==[v for v in source_groups.values() if len(v)>1] and alias['exact_full_afteraction_action_groups']==[v for v in action_groups.values() if len(v)>1],'All alias groups retained, no deduplication')
    need(steps==r['counts']['integration_steps']==31488 and contacts==r['counts']['step_contact_force_reads']==48072,'All native ticks/contact rows accounted')
    need(r['counts']['MjData']==328 and r['counts']['MJB_reads']==2 and r['counts']['receipt_reads']==1312 and r['counts']['initial_forward']==r['counts']['observer_forward']==r['restore_forward']==0,'Original integration budget/no newForward counters')
    for h,a in r['audits'].items():need(a['model']['passed'] is True and a['probe_counts']['state_sets']==a['probe_counts']['branches_returned']==164 and a['probe_counts']['restore_forward']==0,'Actual perdt frozen model/probe audit')
    need('torch' not in sys.modules and 'mujoco' not in sys.modules,'Review has no heavy/native imports')
    checks={k:True for k in ['terminal_protocol_current_source_pins','all8_saved_global_barrier_counts_and_baseline_bits','all328_raw_to_labeled_array_bytes','all_source300_and_ctrl_only_afteraction_joins',
        'all20_axis_actualdu_bounds_and_plan_masks','actual_PRE_POST_ticks_receipt_hashes_and_targets','complete_unknown_partial_masks_and_typed22_layout','all_aliases_heldout_role_and_no_dedup','31488steps48072contacts1312receipts_noForward','zero_reviewer_native_NN_or_force_relabel']}
    result=dict(schema='phone_tripod_heldout_action_response_saved_review:1',status='PASS_SAVED_HELDOUT_TRIPOD_RESPONSE_INTEGRITY_NOT_INDEPENDENT_PHYSICS',engineering_qualified=True,
        native_report=rp,producer_report=rp,terminal=tp,protocol=r['protocol'],selection=r['selection'],branches=r['branches'],baseline_barrier=r['baseline_barrier'],field_map=r['field_map'],checks=checks,passed=len(checks),total=len(checks),
        contexts=8,executed_branches=328,known_complete_branches=328,partial_branches=partial,unknown_label_branches=unknown,partial_actual_cases_exercised=bool(partial),
        permanently_heldout=True,Actor_EMA_unreachable_side_descriptors=outside,per_dt=dict(per_h),counts=dict(saved_NPZ_reads=689,new_native_NN_FK_steps=0,force_impulse_redecodes=0),
        source=pin(__file__),scope=['Reviewer authored the diagnostic source; Root executed native sampling. This is source-author saved integrity with no independent physics-label reproof.',
        'All8 baseline exact native arrays/300 and persisted labels are checked from saved files; the global-order conclusion also uses frozen separate-loop source plus saved barrier counters, not a separate instrumented native-call stream.',
        'No new native model/forward/step, Torch/NN, label force reconstruction, training or parameter updates. Windows separately owns response numerics/momentum checks.',
        'All328 actual branches complete/known. Partial handling is source contract, not exercised by these actual cases.',
        'Absolute-control hypothetical interventions are not necessarily Actor+EMA reachable. All328 stay permanently outside fitting and checkpoint monitor selection. No fullMjData/Markov/analyticJacobian/policy benefit/holding qualification.'])
    write(H/'report.json',result);print(json.dumps(dict(report=pin(H/'report.json'),passed=len(checks),total=len(checks))))

if __name__=='__main__':run()
