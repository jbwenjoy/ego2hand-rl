"""Pure-CPU join of frozen C2v2 metrics and independent sticky-capacity evidence.

Never changes v1, a frozen source, task thresholds, or physical data. A zero
capacity flag is not solver convergence, contact accuracy or a safety guarantee.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def integer(x):return type(x) is int

def capacity_class(cap):
    """Internal consistency only; identity/hash/timing binding is separate."""
    if cap.get('status')=='UNKNOWN' and cap.get('overflow_bitmask') is None and cap.get('flags_by_world') is None:return 'UNKNOWN'
    values=cap.get('flags_by_world');mask=cap.get('overflow_bitmask')
    if not isinstance(values,list) or not values or any(not integer(v) or v<0 for v in values) or not integer(mask):return 'INVALID'
    total=0
    for value in values:total|=value
    expected='OK' if total==0 else 'OVERFLOW'
    if mask!=total or cap.get('status')!=expected:return 'INVALID'
    return expected

def binding_checks(cap,job,plan,plan_sha,condition,runtime,health,meta,rollout_sha,control_sha,capacity_sha):
    try:
        observed=datetime.fromisoformat(cap['observed_utc']);finished=datetime.fromisoformat(runtime['finished_utc']);started=datetime.fromisoformat(runtime['started_utc'])
        clock=started<=observed<=finished
    except (KeyError,TypeError,ValueError):clock=False
    return {'sidecar_schema':cap.get('schema')=='C2_sticky_capacity_audit:2',
            'job_plan_worker_identity':cap.get('job_id')==job['job_id'] and cap.get('plan_sha256')==plan_sha and cap.get('worker_sha256')==plan['source_pins']['worker']['sha256'],
            'checkpoint_reference_identity':cap.get('checkpoint_sha256')==plan['checkpoint']['sha256'] and cap.get('reference_sha256')==plan['reference']['sha256'],
            'single_read_after150_intervals':integer(cap.get('read_count')) and cap['read_count']==1 and integer(cap.get('read_at_control_intervals')) and cap['read_at_control_intervals']==150,
            'one_actual_world':integer(cap.get('actual_world_count')) and cap['actual_world_count']==1 and (cap.get('flags_by_world') is None or len(cap['flags_by_world'])==1),
            'no_reset_scope':integer(cap.get('resets_during_rollout')) and cap['resets_during_rollout']==meta['resets_during_rollout']==0,
            'actual_backend_field':cap.get('backend_field')=='wp_data.overflow',
            'read_timestamp_inside_worker':clock,
            'raw_artifact_linkage':cap.get('rollout_npz_sha256')==rollout_sha and cap.get('control_npz_sha256')==control_sha,
            'runtime_full_copy_identical':runtime.get('capacity_audit')==cap,
            'health_full_copy_identical':health.get('capacity_audit')==cap,
            'sidecar_SHA_in_condition_and_health':condition.get('capacity_audit_sha256')==health.get('capacity_audit_sha256')==capacity_sha,
            'runtime_job_plan_and_horizon':runtime.get('status')=='completed' and runtime.get('job_id')==job['job_id'] and runtime.get('plan_sha256')==plan_sha and runtime.get('control_intervals')==150,
            'condition_sources_and_identity':condition.get('job_id')==job['job_id'] and condition.get('plan_sha256')==plan_sha and condition.get('source_hashes')=={k:v['sha256'] for k,v in plan['source_pins'].items()},
            'capacity_fields_consistent':capacity_class(cap)!='INVALID'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,default=ROOT/'configs/target-inset-capacity-v2.json');p.add_argument('--analysis',type=Path,default=ROOT/'outputs/analysis_capacity_v2');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:return self_test()
    out=args.analysis;out.mkdir(parents=True,exist_ok=True);plan=read(args.plan);plan_sha=sha(args.plan);manager_path=Path(plan['output_root'])/'state.json';manager=read(manager_path) if manager_path.exists() else {'status':'PENDING'}
    header={'schema':'C2_capacity_independent_join:2','join_source_sha256':sha(__file__),'plan_sha256':plan_sha,'gpu_used':False,'training_seed_N':1,'v1_rows_pooled':False,'v1_qualification_changed':False,'m0':None,'a_brake':None,'parameter_enablement':False}
    if manager['status']!='completed_9_jobs' or not (out/'comparison.json').exists():
        write(out/'capacity_review.json',{**header,'status':'PENDING_TERMINAL_OR_BASE_ANALYSIS','manager_status':manager['status'],'capacity_qualified_task_counts':None});print('PENDING_TERMINAL_OR_BASE_ANALYSIS');return
    base=read(out/'comparison.json')
    if base['status']!='COMPLETE' or base['plan_sha256']!=plan_sha or manager['plan_sha256']!=plan_sha:raise ValueError('Base/manager does not bind this complete v2plan')
    source_checks={name:sha(pin['path'])==pin['sha256'] for name,pin in plan['source_pins'].items()}
    inputs={name:sha(plan[name]['path'])==plan[name]['sha256'] for name in ('checkpoint','reference','reference_metadata')}
    if not all(source_checks.values()) or not all(inputs.values()):raise ValueError('Frozen source/input changed')
    jobs=plan['jobs'];rows=[]
    if len(jobs)!=9 or len({j['job_id'] for j in jobs})!=9:raise ValueError('Expected nine unique predeclared jobs')
    for job in jobs:
        identity={k:job[k] for k in ('job_id','block_id','position','margin_mrad','repeat_id')}
        try:
            actual=next(r for r in base['results'] if r['job_id']==job['job_id']);mj=next(r for r in manager['jobs'] if r['job_id']==job['job_id']);folder=Path(job['output_dir']);cap_path=folder/'capacity_audit.json';cap=read(cap_path)
            condition=read(folder/'condition.json');runtime=read(Path(job['wrapper_metadata_dir'])/'runtime.json');meta=read(folder/'rollout.json');health=mj['health']
            raw_sha,control_sha=sha(folder/'rollout.npz'),sha(folder/'control_targets_50hz.npz')
            checks=binding_checks(cap,job,plan,plan_sha,condition,runtime,health,meta,raw_sha,control_sha,sha(cap_path))
            checks['base_original_metric_control_contracts']=actual['status']=='VALID' and actual['hashes']['rollout.npz']==raw_sha and actual['hashes']['control_targets_50hz.npz']==control_sha
            checks['manager_complete_exit0']=mj['status']=='completed' and mj['process']['status']=='completed' and mj['process']['exit_code']==0
            checks['manager_original8_artifact_hashes']=len(health['artifact_sha256'])==8 and all(sha(folder/name)==expected==actual['hashes'].get(name) for name,expected in health['artifact_sha256'].items())
            cls=capacity_class(cap);joined='BOUND_ZERO_OVERFLOW' if all(checks.values()) and cls=='OK' else 'BOUND_OVERFLOW' if all(checks.values()) and cls=='OVERFLOW' else 'UNKNOWN' if all(checks.values()) and cls=='UNKNOWN' else 'INVALID_CAPACITY_RECORD'
            qualified=actual['task_success'] if joined=='BOUND_ZERO_OVERFLOW' else None
            row={**identity,'status':joined,'checks':checks,'observed_frozen_task_success':actual['task_success'],'capacity_qualified_observed_task_success':qualified,'overflow_bitmask':cap.get('overflow_bitmask'),'flags_by_world':cap.get('flags_by_world'),'read_count':cap.get('read_count'),'read_at_control_intervals':cap.get('read_at_control_intervals'),'resets_during_rollout':cap.get('resets_during_rollout'),'metrics':actual['metrics'],'failed_task_gates':actual['failed_task_gates'],'hashes':{'capacity_audit_json':sha(cap_path),'runtime_json':sha(Path(job['wrapper_metadata_dir'])/'runtime.json'),'condition_json':sha(folder/'condition.json'),'rollout_npz':raw_sha,'control_targets_npz':control_sha}}
        except (OSError,KeyError,ValueError,StopIteration,TypeError) as error:row={**identity,'status':'INVALID_OR_MISSING_CAPACITY','observed_frozen_task_success':None,'capacity_qualified_observed_task_success':None,'reason':str(error)}
        rows.append(row)
    conditions={}
    for margin in (0,5,20):
        group=[r for r in rows if r['margin_mrad']==margin];zero=sum(r['status']=='BOUND_ZERO_OVERFLOW' for r in group);complete=len(group)==zero==3
        conditions[str(margin)]={'jobs':len(group),'zero_overflow_qualified_jobs':zero,'observed_frozen_task_passes':base['conditions'][str(margin)]['task_passes'],'capacity_qualified_observed_task_passes':sum(r['capacity_qualified_observed_task_success'] is True for r in group) if complete else None,'qualification_complete_for_all3':complete}
    data={**header,'status':'COMPLETE_CAPACITY_QUALIFIED_OBSERVED_METRICS' if all(r['status']=='BOUND_ZERO_OVERFLOW' for r in rows) else 'CAPACITY_QUALIFICATION_INCOMPLETE','source_pin_checks':source_checks,'input_pin_checks':inputs,'manager_sha256':sha(manager_path),'base_comparison_sha256':sha(out/'comparison.json'),'rows':rows,'conditions':conditions,'plot_qa':'PENDING_VISUAL_REVIEW','scope':'The one post-physics sticky flag establishes no recorded backend capacity overflow since reset for this observed3s run only. It does not establish convergence/accuracy, state safety, hardware, new-motion/ego generalization, WM qualification or actuator-only braking. v1 UNKNOWN remains unchanged; no pooling or best-of-two selection.'}
    write(out/'capacity_review.json',data)
    with (out/'capacity_join.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['job_id','block_id','margin_mrad','status','observed_frozen_task_success','capacity_qualified_observed_task_success','overflow_bitmask','flags_by_world','read_count','read_at_control_intervals','resets_during_rollout'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in rows)
    lines=[f"Capacity join status: {data['status']}",'v2 only; no v1 pooling, replacement or best-run selection. Same training seed1/evaluation0; training-seed N remains1.']
    for margin,c in conditions.items():lines.append(f"{margin}mrad: observed passes={c['observed_frozen_task_passes']}/3; capacity-zero qualified jobs={c['zero_overflow_qualified_jobs']}/3; capacity-qualified observed passes={c['capacity_qualified_observed_task_passes']}/3 (null if any capacity unqualified).")
    lines.append(data['scope']);(out/'CAPACITY_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':data['status'],'conditions':conditions},indent=2))

def self_test():
    class Cases(unittest.TestCase):
        def test_zero(self):self.assertEqual(capacity_class({'status':'OK','flags_by_world':[0],'overflow_bitmask':0}),'OK')
        def test_nonzero_not_zero(self):self.assertEqual(capacity_class({'status':'OVERFLOW','flags_by_world':[2],'overflow_bitmask':2}),'OVERFLOW')
        def test_unknown_not_zero(self):self.assertEqual(capacity_class({'status':'UNKNOWN','flags_by_world':None,'overflow_bitmask':None}),'UNKNOWN')
        def test_false_is_not_integer_zero(self):self.assertEqual(capacity_class({'status':'OK','flags_by_world':[False],'overflow_bitmask':0}),'INVALID')
        def test_status_mask_inconsistent(self):self.assertEqual(capacity_class({'status':'OK','flags_by_world':[1],'overflow_bitmask':0}),'INVALID')
        def test_empty_is_not_zero(self):self.assertEqual(capacity_class({'status':'OK','flags_by_world':[],'overflow_bitmask':0}),'INVALID')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases));out=ROOT/'outputs/capacity_join_cpu_checks_v2';out.mkdir(parents=True,exist_ok=True)
    write(out/'checks.json',{'scope':'Pure CPU synthetic classification tests only; no measured v2capacity qualification','tests_run':result.testsRun,'passed':result.wasSuccessful(),'join_source_sha256':sha(__file__)})
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
