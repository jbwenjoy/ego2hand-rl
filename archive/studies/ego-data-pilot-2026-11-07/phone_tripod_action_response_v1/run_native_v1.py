"""Eight matched baselines globally precede any held-out finite-action side.

No top-level native/NN imports. Never trains, modifies a model or recenters the
original episode safety origin. Native restoration uses the frozen probe.
"""
from pathlib import Path
import argparse,importlib.util,json,time,traceback
from collections import defaultdict
import numpy as np
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('tripod_response_runtime_common',H/'common_v1.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)

def metadata(path):
    p=C.read(path);C.need(p['schema']=='phone_tripod_heldout_action_response_CPU_protocol:1' and p['rules']==C.RULES and p['permanent_data_role']==C.ROLE,'Exact held-out diagnostic contract')
    C.need(type(p['CPU_run_authorized']) is bool and p['NN_training_GPU_authorized'] is False and p['automatic_retry'] is False,'Explicit native-only no retry/no fit')
    for v in [*p['inputs'].values(),*p['sources'].values()]:C.checked(v)
    C.need(p['sources']['runner']==C.pin(__file__) and p['sources']['common']==C.pin(H/'common_v1.py'),'Current source bytes')
    r=C.read(C.checked(p['inputs']['acquisition_report']));plan=C.read(C.checked(p['inputs']['selection']))
    C.need(plan['rules']==C.RULES and plan['acquisition_report']==p['inputs']['acquisition_report'] and plan['task_bundle']==p['inputs']['task_bundle'],'Selection/source bindings')
    C.need(r['sample_identities']==p['inputs']['source_identities'] and r['episode_records']==p['inputs']['source_episodes'],'Source path comes from producer manifests')
    C.need([(v['family_id'],v['native_dt_s'],v['step_k']) for v in plan['states']]==[(f,h,k) for f in [12,15] for h in [.0003125,.00015625] for k in [0,3]],'All mechanical8 contexts, no response selection')
    C.need(plan['planned_branches']==328 and len(plan['states'])==8,'Complete predeclared plan')
    for state in plan['states']:
        C.need(state['state_id']==state['context_id']==state['baseline_branch_id'] and state['split']=='TRIPOD_MONITOR' and state['source_identity']['valid'],'MON current prealive complete source')
        for v in [state['source_trace'],state['initial_task'],state['qpos_source'],state['episode_initial']['arrays']]:C.checked(v)
        C.need([(v['axis'],v['sign']) for v in state['perturbations']]==[(j,sgn) for j in range(20) for sgn in [-1,1]],'All40 side masks retained')
    for name,parentkey in [('source_integrity_review','producer_report'),('source_label_review','collector_report')]:
        proof=C.read(C.checked(p['inputs'][name]));C.need(proof['engineering_qualified'] is True and all(v is True for v in proof['checks'].values()) and proof[parentkey]==p['inputs']['acquisition_report'],'Positive existing source saved review')
    task=C.read(C.checked(p['inputs']['task_bundle']));C.need(task['model']==p['inputs']['model_mjb'] and task['task_designated_geom_ids']==[15,28,41],'Same model/new tripod roles')
    out=Path(p['output']);C.need(out.parent.resolve()==H and not out.exists(),'Fresh result directory, no overwrite/retry')
    return p,plan,r,out

def baseline_checks(value,state,operands,original,labels):
    sid=state['state_id'];checks=dict(complete_H=value['complete_H20ms'] is True,finite_no_warning_labels=bool(value['training_eligible']),
        source_before_300=value['source_receipt']['sha256']==state['source_before_action_receipt']['sha256'],
        source_after_original_action_300=value['source_after_action_receipt']['sha256']==state['original_source_after_action_receipt']['sha256'],
        target_300=C.bits(value['target_receipt']['integration_state'],operands['original_target_integration_F64'][sid]),
        source_state58=C.bits(value['source_state58'],operands['source_state58_F64'][sid]),actual_original_control=C.bits(value['actual_control20'],operands['baseline_ctrl20_F64'][sid]),
        target_qpos27=C.bits(value['trace']['post_qpos27'][-1],operands['original_target_qpos27_F64'][sid]),
        target_qvel26=C.bits(value['trace']['post_qvel26'][-1],operands['original_target_qvel26_F64'][sid]),
        no_restore_observer_initial_forward=all(v==0 for v in value['forward_counters'].values()))
    for key,a in value['trace'].items():checks['all_native_ticks__'+key]=key in original and C.bits(a,original[key])
    for key,a in labels.items():checks['canonical_tripod_label_bits__'+key]=C.bits(a,operands['baseline_'+key][sid])
    return checks

def run(protocol_path):
    p,plan,parent,out=metadata(protocol_path);C.need(p['CPU_run_authorized'] is True,'Root-reviewed actual CPU diagnostic only')
    probe_module=C.mod('tripod_response_frozen_probe',C.checked(p['sources']['probe']))
    D=C.mod('tripod_response_frozen_labels',C.checked(p['sources']['label_adapter']));maths=D.load_math();physical=C.read(C.checked(p['inputs']['physical_parameters']))
    operands=C.npz(p['inputs']['operands']);out.mkdir();C.write(out/'protocol.json',p);began=time.perf_counter()
    probes={};records={};baseline_receipts=[];phase='construct_two_models';last_raw=None
    def persist(state,bid,kind,value,side=None):
        nonlocal last_raw
        sid=state['state_id'];raw=dict(value['trace'],source_state58_F64=value['source_state58'],actual_control20_F64=value['actual_control20'],
            baseline_control20_F64=operands['baseline_ctrl20_F64'][sid],source_integration_F64=value['source_receipt']['integration_state'],
            source_after_action_integration_F64=value['source_after_action_receipt']['integration_state'],target_integration_F64=value['target_receipt']['integration_state'],
            observed_prefix_target58_F64=value['target_state58'],raw_native_delta52_F64=value['physical_world_delta52'],
            legacy_two_pad_observable59_NOT_tripod_GT=value['observable59'],actual_du_F64=np.asarray(0. if side is None else side['actual_du_F64']))
        rawpath=out/f'branch_{bid:03d}_raw.npz';C.save(rawpath,raw);last_raw=C.pin(rawpath)
        # Full native raw is fsynced before derived-label arithmetic.
        labels=C.label_interval(value['trace'],value['source_state58'],value['target_state58'],h=state['native_dt_s'],n=state['n_substeps'],complete=value['complete_H20ms'],finite=value['finite'],warning_numbers=value['warning_numbers'],D=D,maths=maths,physical=physical)
        offset=state['source_control_segment_offset'];before=value['source_receipt']['integration_state'];after=value['source_after_action_receipt']['integration_state']
        C.need(C.bits(before,operands['source_integration_F64'][sid]) and C.bits(after[offset:offset+20],value['actual_control20']) and C.bits(before[:offset],after[:offset]) and C.bits(before[offset+20:],after[offset+20:]),'Every branch restores same before-action300; only measured ctrl segment changes')
        path=out/f'branch_{bid:03d}.npz';C.save(path,raw|labels)
        record=dict(branch_id=bid,context_id=sid,state_id=sid,baseline_branch_id=sid,kind=kind,status='EXECUTED',data_role=C.ROLE,
            family_id=state['family_id'],episode_id=state['episode_id'],source_sample_id=state['source_sample_id'],step_k=state['step_k'],dt_s=state['native_dt_s'],n_substeps=state['n_substeps'],
            axis=None if side is None else side['axis'],sign=0 if side is None else side['sign'],actual_du_F64=0. if side is None else side['actual_du_F64'],
            baseline_u20_F64=state['baseline_ctrl20_F64'],actual_u20_F64=value['actual_control20'].tolist(),perturbation=side,raw_output=last_raw,output=C.pin(path),
            source_receipt=C.receipt_meta(value['source_receipt']),source_after_action_receipt=C.receipt_meta(value['source_after_action_receipt']),target_receipt=C.receipt_meta(value['target_receipt']),
            complete_H20ms=value['complete_H20ms'],finite=value['finite'],no_warning=not bool(np.any(value['warning_numbers'])),labels_known=bool(labels['physical_labels_known_bool']),
            returned_substeps=value['returned_substeps'],termination=value['termination'],warning_numbers=value['warning_numbers'],
            permanently_excluded_from_train_and_monitorselection=True,legacy59_scope='frozen probe two-pad auxiliary only; not any new tripod GT',
            absolute_control_hypothetical_not_policy_action=True)
        C.write(out/f'branch_{bid:03d}.json',record);records[bid]=record
        return labels
    try:
        for h in C.RULES['native_dt_s']:
            probes[h]=probe_module.NativeSavedActionProbe(C.checked(p['inputs']['model_mjb']),dt_s=h,expected_effective_model_digest=parent['model_audits'][str(h)]['model_digest'])
        # GLOBAL BARRIER: deliberately separate loops, no side code in this loop.
        for state in plan['states']:
            sid=state['state_id'];phase=f'baseline_{sid}';original=C.npz(state['source_trace'])
            value=probes[state['native_dt_s']].run(state,operands,state['baseline_ctrl20_F64'])
            labels=persist(state,sid,'baseline',value);checks=baseline_checks(value,state,operands,original,labels)
            item=dict(context_id=sid,branch_id=sid,checks=checks,passed=all(checks.values()),source_trace=state['source_trace'],raw_output=records[sid]['raw_output'],output=records[sid]['output'])
            C.write(out/f'baseline_{sid:02d}_checks.json',item);baseline_receipts.append(C.pin(out/f'baseline_{sid:02d}_checks.json'))
        barrier=dict(schema='phone_tripod_all8_baseline_barrier:1',all8_passed=all(C.read(v['path'])['passed'] for v in baseline_receipts),baselines=baseline_receipts,
            baseline_count=8,new_sides_executed_before_barrier=0,counts={str(h):dict(v.reader.counts) for h,v in probes.items()},scope='Empirical8 matched actions/selected full300 and all native trace bytes; not fullMjData or arbitrary-action restore proof')
        C.write(out/'baseline_barrier.json',barrier);C.need(barrier['all8_passed'],'Baseline barrier failed: no new action side may run')
        for state in plan['states']:
            for side in state['perturbations']:
                bid=side['branch_id'];phase=f'side_{bid}'
                if not side['available']:
                    records[bid]=dict(branch_id=bid,context_id=state['state_id'],state_id=state['state_id'],baseline_branch_id=state['state_id'],kind='finite_action_side',status='UNAVAILABLE',data_role=C.ROLE,
                        axis=side['axis'],sign=side['sign'],actual_du_F64=side['actual_du_F64'],perturbation=side,output=None,raw_output=None,returned_substeps=0,complete_H20ms=False,labels_known=False,permanently_excluded_from_train_and_monitorselection=True)
                    C.write(out/f'branch_{bid:03d}.json',records[bid]);continue
                value=probes[state['native_dt_s']].run(state,operands,side['actual_ctrl20_F64'])
                C.need(C.bits(value['actual_control20'],np.array(side['actual_ctrl20_F64'],dtype=np.float64)),'Planned F64 control bytes, no clipping or rescue')
                persist(state,bid,'finite_action_side',value,side)
        ordered=[records[i] for i in range(328)];C.write(out/'branches.json',ordered)
        groups=defaultdict(list);sourcegroups=defaultdict(list)
        for rec in ordered:
            if rec['status']!='EXECUTED':continue
            st=plan['states'][rec['context_id']];sourcegroups[(rec['source_receipt']['sha256'],st['native_dt_s'])].append(rec['branch_id'])
            key=(rec['source_after_action_receipt']['sha256'],C.sha(np.array(rec['actual_u20_F64'])),st['native_dt_s'],st['n_substeps'])
            groups[key].append(rec['branch_id'])
        C.write(out/'aliases.json',dict(exact_full_afteraction_action_groups=[v for v in groups.values() if len(v)>1],same_source_context_groups=[v for v in sourcegroups.values() if len(v)>1],no_deduplication=True,scope='Known local MON contexts; no independent trajectory or random-row cohort claim'))
        audits={str(h):pr.audit() for h,pr in probes.items()};counts={k:sum(pr.reader.counts[k] for pr in probes.values()) for k in next(iter(probes.values())).reader.counts}
        C.need(all(v['model']['passed'] for v in audits.values()) and counts['integration_steps']<=31488 and counts['initial_forward']==counts['observer_forward']==0,'Fixed model/no restore or observer forward/budget')
        n_exec=sum(v['status']=='EXECUTED' for v in ordered);C.need(n_exec==8+plan['available_sides'],'All available/unavailable plans retained')
        result=dict(schema='phone_tripod_heldout_action_response_result:1',status='COMPLETED_ALL8_BASELINES_THEN_HELDOUT_FULL20_FINITE_ACTIONS',data_role=C.ROLE,
            protocol=C.pin(protocol_path),selection=p['inputs']['selection'],operands=p['inputs']['operands'],acquisition_report=p['inputs']['acquisition_report'],task_bundle=p['inputs']['task_bundle'],
            branches=C.pin(out/'branches.json'),baseline_barrier=C.pin(out/'baseline_barrier.json'),alias_review=C.pin(out/'aliases.json'),field_map=C.FIELD_MAP,
            contexts=8,planned_branches=328,executed_branches=n_exec,baseline_branches=8,new_action_branches=plan['available_sides'],unavailable_sides=plan['unavailable_sides'],
            partial_branches=sum(v['status']=='EXECUTED' and not v['complete_H20ms'] for v in ordered),labels_known_branches=sum(v['labels_known'] for v in ordered),
            canonical_pair_geom_ids=list(D.PAIRS),designated_geom_ids=list(D.PADS),actuator_bounds_source=C.read(C.checked(p['inputs']['selection']))['actuator_bounds_source'],
            counts=counts,audits=audits,restore_forward=0,all8_baselines_before_any_new_side=True,new_NN_learning_parameter_updates=0,
            permanent_training_eligible=False,scope='Absolute-control hypothetical finite actions, not necessarily reachable by present Actor residual+EMA. True contact22 labels use signed world force and moving PRE COM in frozen source BODY; triple is genuine simultaneous3pad. No actual Actor, training, trueJacobian, policy benefit, fullMjData or holding qualification.',elapsed_s=time.perf_counter()-began)
        C.write(out/'report.json',result);return result
    except BaseException as e:
        forensic={}
        for h,probe in probes.items():
            r=probe.reader
            if getattr(r,'d',None) is not None:
                d=r.d;rr=r.receipt();path=out/f'failure_native_{h}.npz';C.save(path,{k:np.array(getattr(d,k),copy=True) for k in ['qpos','qvel','ctrl','qacc','qacc_warmstart','qfrc_applied','xfrc_applied']}|dict(integration_state=rr['integration_state'],warning_numbers=np.array(d.warning.number,copy=True)));forensic[str(h)]=dict(raw=C.pin(path),receipt=C.receipt_meta(rr))
        C.write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),records=records,baseline_receipts=baseline_receipts,last_raw=last_raw,raw_forensic=forensic,
            counts={str(h):pr.reader.counts for h,pr in probes.items()},automatic_retry=False,new_side_execution_requires_global8baselinebarrier=True));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.run:r=run(a.protocol);print(json.dumps(dict(report=C.pin(Path(C.read(a.protocol)['output'])/'report.json'),counts=r['counts'])))
    else:
        p,plan,r,out=metadata(a.protocol);print(json.dumps(dict(status='METADATA_ONLY_NO_NATIVE_NN_NPZ_VALUES',contexts=8,planned_branches=328,available_sides=plan['available_sides'],maximum_native_steps=31488,output=str(out))))
