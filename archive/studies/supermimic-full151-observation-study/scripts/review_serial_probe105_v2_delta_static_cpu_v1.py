"""Read source/AST only; never imports the reviewed production modules."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/full151105_serial_worker_v2_delta_static_review_v1'
OLD=ROOT/'src/full151_gradient/serial_probe105_v1.py'
NEW=ROOT/'src/full151_gradient/serial_probe105_v2.py'

def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    OUT.mkdir(exist_ok=False)
    old,new=OLD.read_text(),NEW.read_text();a,b=ast.parse(old),ast.parse(new)
    def funcs(t):return {n.name:n for n in ast.walk(t) if isinstance(n,ast.FunctionDef)}
    x,y=funcs(a),funcs(b);checks={}
    def check(k,v):checks[k]=bool(v)
    check('old_executed792_source_exact',pin(OLD)['sha256']=='79207ad27252157036915360696442924176b22692464a3ea855c879656c104d')
    check('new_formed8c24_source_exact',pin(NEW)['sha256']=='8c24d06f0b2c702f5848da633a654b4f4c5079e53316164ec40bdd18b314f5c6')
    unchanged=['dump_inventory','allocate_artifacts','baseline_errors','object_terms','controlled_object_score_FP32','object_terms_f64','backend_call_trace','topology','mark','checktime','save','counted','reward_capture','raw','process','substep','interval','drift_at']
    for name in unchanged:check('AST_unchanged_'+name,ast.dump(x[name],include_attributes=False)==ast.dump(y[name],include_attributes=False))
    def src(name):return ast.get_source_segment(new,y[name])
    validate=src('validate_config');plan=src('execution_plan');schedule=src('branch_schedule');run=src('run_probe');completion=src('completion_scope');review=src('validate_review_manifest');header=src('_review_header');current=src('reviewed_foreign_inventory')
    check('config2_and_fixed_proposal2_scalar_contract',"full151105_same_warp_physical_probe_run:2" in validate and "full151105_same_Warp_control_gradient_diagnostic_proposal:2" in validate and "'scalar_contract':SCALAR_CONTRACT" in validate)
    check('reviewNone_only_discoveryall_no_flat_fallback',"config['probe_mode']=='inventory_discovery_all' and config.get('foreign_inventory_review') is not None" in validate and "if config['probe_mode']=='serial_full':validate_review_manifest(config)" in validate and "elif config.get('foreign_inventory_review') is not None:raise" in validate)
    check('discovery_two_complete_donors_8captures_1200steps0branches',all(s in plan for s in ("long=full or discovery","'donors':PATHS if long else PATHS[:1]","'donor_intervals':150 if long else 1","'snapshot_count':8 if long else 1","'episode_substeps':1808 if full else 1200 if discovery else 16","'branch_count':152 if full else 0 if discovery else 3")))
    check('all152slots_discovery_not_scheduled',"scheduled=probe_mode=='serial_full' or (probe_mode=='engineering_restore_smoke' and s==0 and slot['type']=='baseline')" in schedule and "status='NOT_RUN'" in schedule)
    check('discovery_never_enters_branchloop',"for k in capture_phases:\n                                    if mode=='inventory_discovery_all':continue" in run and "with tx.branch('branch_'+str(bid)) as restore:" in run)
    check('discovery_snapshots_pending_not_restorepassed',"foreign_review_status='REVIEW_PENDING',foreign_review_qualified=False" in run and "native_restore_qualification_scope='NOT_RUN_INVENTORY_ONLY'" in run and "status='REVIEW_PENDING' if mode=='inventory_discovery_all'" in run)
    check('capture_roundtrip_and_native_validation_not_swallowed',all(s in run for s in ("Transaction105.__init__", "tx.verify_current();record['integration_roundtrip_bitexact']=True", "if not np.array_equal(integration().view(np.uint32),integ.view(np.uint32)):raise", "record['inventory_archive']=dump_inventory")) and "except BaseException as exc:" in run and "Secondary discovery-save failure" in run)
    instrument=src('instrument_transaction_calls')
    check('actual_begin_finish_native_restore_calls_instrumented',all(s in instrument for s in ("(tx,'begin','transaction_begin')","(tx,'finish','transaction_finish')","(tx.native,'restore','native_snapshot_restore')","report[_label+'_attempted']+=1","result=_original(*args,**kwargs)","report[_label+'_returned']+=1")))
    check('discovery_final_nonvacuous_no_restore_assertion',"if discovery and (baseline_gates or arrays['restore_qualified'].any() or arrays['restore_inventory_bitexact'].any() or any(report[k] for k in ('transaction_begin_attempted','transaction_finish_attempted','native_snapshot_restore_attempted')))" in run)
    check('integration_counts16gets8sets_same_value_scope',"integration_set_attempted']!=planned_snapshots" in run and "integration_set_returned']!=planned_snapshots" in run and "expected_gets=2*planned_snapshots+planned_branches" in run and "integration_get_attempted']!=expected_gets" in run and "state_write_scope='Native initialization/reset/actual8 setup plus same-value INTEGRATION roundtrips; never zero-state-writes claim'" in run)
    check('discovery_qualification_flags_nonvacuous_false',"'gradient_qualified':False,'whole_graph_hidden_state_qualified':False" in completion and "baseline_all_passed=False if discovery" in run and "foreign_inventory_qualified=False if discovery" in run and "restore_inventory_qualified=False if discovery" in run and "full_serial_qualified=mode=='serial_full',longclock_qualified=mode=='serial_full'" in run)
    check('donorclock_is_separate_from_restored_longclock',"donor_clock_qualified=mode in ('serial_full','inventory_discovery_all')" in run and "DONOR_ONLY_NO_RESTORED_LONGCLOCK_QUALIFICATION" in run)
    check('full_review2_exact8labels_worker_transaction_bundle',"full151105_foreign_inventory_per_snapshot_source_review:2" in header and "'worker':pin(__file__)" in header and "'transaction':pin(transaction105.__file__)" in header and "set(tables)!=set(SNAPSHOT_LABELS)" in header and "review.get('allowed_snapshot_labels')!=list(SNAPSHOT_LABELS)" in header)
    check('review_requires_actual_pinned_description_topology_eachlabel',all(s in review for s in ("checked(item['inventory']['description'])","checked(item['inventory']['topology'])","description.get('label')!=label","topology.get('worker_source')!=review['worker']","topology.get('transaction_source')!=review['transaction']")))
    check('review_exact_opaque_set_unique_no_wildcards',"len(set(paths))!=len(paths)" in src('_allowed_set') and "'*' not in v[k]" in src('_allowed_set') and "len(actual)!=len(item['allowed_identity_only'])" in review and "_allowed_set(item['allowed_identity_only'])" in current)
    check('review_checks_currentlabel_no_snapshot_fallback',"if config['probe_mode']!='serial_full':raise" in current and "label=tx.label" in current and "item=review['per_snapshot'][label]" in current)
    check('new_full_baseline_still_gated_not_hidden_certificate',"if row['baseline_repeat']==2 and not all(baseline_pass):raise" in run and "if len(baseline_pass)!=3 or not all(baseline_pass):raise" in run and "full_native_restore_qualified=False" in run and "wholehidden graphstate UNKNOWN" in run)
    check('no_new_Runner_construction_or_model_forward_call',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('DexSuperMimicRunner','Actor','WorldModel') for n in ast.walk(b)))
    check('physics_no_autoreset_terminal_guard_stays_original',"(k==149)" in src('interval') and "time_steps.tolist()!=[k+1]" in src('interval') and "no_u150=True" in src('interval'))
    check('source_only_process_imports_no_native',not any(n.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for n in sys.modules))
    (OUT/'serial_probe105_v1.py.snapshot').write_bytes(OLD.read_bytes());(OUT/'serial_probe105_v2.py.snapshot').write_bytes(NEW.read_bytes())
    (OUT/'v1_to_v2.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='executed_v1_79207',tofile='formed_v2_8c24')),encoding='utf-8')
    report={'schema':'full151105_serial_worker_v2_delta_static_review:1','status':'STATIC_SOURCE_NO_KNOWN_BLOCKING' if all(checks.values()) else 'STATIC_REVIEW_BLOCKING_OR_HARNESS_FAILURE_RETAINED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'reviewer':pin(__file__),'source_pins':{'executed_worker1':pin(OLD),'formed_worker2':pin(NEW),'transaction105':pin(NEW.with_name('transaction105.py'))},'ast_unchanged_functions':unchanged,'line_refs':{k:{'start':y[k].lineno,'end':y[k].end_lineno} for k in ('validate_config','_review_header','validate_review_manifest','reviewed_foreign_inventory','execution_plan','instrument_transaction_calls','completion_scope','run_probe')},'scope':['This is source/AST inspection, not production import, metadata execution, actual Env, snapshot, restore, solver, WM, Actor or GPU test.','Inventory-all performs two sequential fresh donors with eight clone/capture plus same-value INTEGRATION set/get checks. No baseline or intervention response is produced; 152 branch slots remain NOT_RUN. These same-value state writes are not zero-state-write or restoration evidence.','Discovery restore exclusion is the fixed branch-loop skip plus nonzero actual-call counter failure at completion; the counter wrapper itself does not reject before invoking the wrapped method. The present reviewed control flow has no discovery call path to begin/finish/native.restore. No generic interception proof is claimed.','The source-validated per-snapshot review must still be built from actual eight-label discovery artifacts and independent foreign dependency reviews. Their hidden graph scratch remains UNKNOWN. New schema and source hash prevent reusing the phase0 flat-list manifest.','Legacy engineering_restore_smoke remains accepted with None for discovery behavior but cannot pass branch qualification in worker2 because only serial_full may consume a review; the executed worker1 condition03 remains immutable and is the separate empirical front-end prerequisite.','A later full run requires a new Root-authorized config/protocol/lifecycle and each snapshot three fresh baselines before directions. This source review does not authorize physical execution or grant actual restore, full-serial, hidden-state or physical-gradient qualification.'],'native_execution_or_framework_import':False,'physical_gradient_qualified':False,'whole_hidden_state_qualified':False}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')},indent=2))
    return 0 if all(checks.values()) else 1

if __name__=='__main__':raise SystemExit(main())
