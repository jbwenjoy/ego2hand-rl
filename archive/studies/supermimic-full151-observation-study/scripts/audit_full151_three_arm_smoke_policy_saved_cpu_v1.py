"""Fixed three-arm schema2 smoke evaluation saved audit; no model imports."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/full151_v2_three_arm_smoke_policy_saved_review_v1'
CHILD=ROOT/'scripts/audit_full151_policy_saved_cpu_v3.py'
METHODS=('A1L','P_ONLY','A2')


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=False)
    sources={'cohort_auditor':pin(__file__),'per_run_auditor':pin(CHILD)}
    reports={};processes={};checks={};selection={}
    for method in METHODS:
        run=ROOT/f'outputs/full151_v2_policy_eval_smoke40_{method}_seed0_v1'
        config=ROOT/f'configs/full151-v2-policy-eval-smoke40_{method}_seed0-v2.json'
        health=ROOT/f'outputs/full151_v2_smoke40_{method}_health_v1/report.json'
        selection[method]={'run_dir':str(run),'config':pin(config),'health':pin(health)}
        command=[sys.executable,str(CHILD),'--run-dir',str(run),'--health',str(health),'--output',str(OUT/method),'--expected-method',method]
        t=time.perf_counter()
        process=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        processes[method]={'command':command,'exit_code':process.returncode,'wall_seconds':time.perf_counter()-t,'stdout':process.stdout,'stderr':process.stderr}
        report=read(OUT/method/'report.json');reports[method]=report
        for key,value in report['checks'].items():checks[method+'__'+key]=value
        checks[method+'__audit_normal_completion']=process.returncode==0 and report['status']=='PASS_SAVED_AUDIT_ORIGINAL_QUALITY_PRESERVED'
        checks[method+'__exact_authoritative_config_pin']=selection[method]['config']['sha256']==pin(run/'config.json')['sha256']
        checks[method+'__fixed_final40_not_selected_by_quality']=report['checkpoint_iteration']==40 and report['training_role']=='engineering_smoke' and report['actual_checkpoint']==read(config)['checkpoint']
    checks['auditor_sources_unchanged']=all(pin(v['path'])==v for v in sources.values())
    with (OUT/'metrics.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['method','native_policy_pass','position151_max_mm','directed_shaft151_max_deg','depth600_max_mm','joint_excess601_max_mrad','target_rate150_max_rad_s','load_ratio_max','load_charge_max','failed_gates'])
        for method in METHODS:
            r=reports[method];m=r['recomputed_metrics'];g=r['recomputed_gates']
            writer.writerow([method,all(g.values()),m['max_object_position_error_151_m']*1000,m['max_shaft_angle_151_deg'],m['max_penetration_m']*1000,
                m['max_actual_joint_excess_rad']*1000,m['max_actual_target_rate_rad_s'],m['max_actuator_load_ratio'],m['max_load_charge'],';'.join(r['failures'])])
    report={'schema':'full151_v2_three_arm_smoke_policy_saved_review:1','status':'PASS_SAVED_ENGINEERING_QUALITY_FAILURES_PRESERVED' if all(checks.values()) else 'SAVED_AUDIT_FAILURE_RETAINED',
        'engineering_qualified':all(checks.values()),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_pins':sources,'fixed_selection':selection,
        'run_audits':{k:pin(OUT/k/'report.json') for k in METHODS},'audit_processes':processes,
        'quality_separate':{k:{'native_policy_pass':bool(all(v['recomputed_gates'].values())),'gates':v['recomputed_gates'],'metrics':v['recomputed_metrics'],'failed_gates':v['failures']} for k,v in reports.items()},
        'metrics_csv':pin(OUT/'metrics.csv'),'wall_seconds':time.perf_counter()-started,
        'scope':['Three fixed final40 checkpoints, trainingseed0/evaluationseed0 and one process per arm; dependent same-reference smoke diagnostics, no method-benefit estimate.',
            'No new Actor/WM/checkpoint-tensor load, Env, native model, FK, collision, solver, physics or GPU. Only saved NumPy/JSON/rawcontact/source ZIP evidence.',
            'All13 original gates recomputed separately from engineering qualification; full601states/600solverrows/150controlintervals/150rewardcaptures retained.',
            'Saved Actor before/after digests and runtime assertions are rebound source evidence, not independently replayed Actor inference or tensor snapshots.',
            'Native selfcollision15 comes from saved reward_snapshot; raw hand-object contacts are not substituted for its labels. Construction/reset/graph API counts remain UNKNOWN.']}
    with (OUT/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json'),
        'quality':{k:v['failed_gates'] for k,v in report['quality_separate'].items()}},indent=2))
    return 0 if all(checks.values()) else 1


if __name__=='__main__':raise SystemExit(main())
