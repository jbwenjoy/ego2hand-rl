"""Saved-only enrollment of the fixed schema2 three-arm engineering smoke."""
from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
METHODS=['A1L','P_ONLY','A2']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(v):
    assert set(v)=={'path','sha256'} and Path(v['path']).is_absolute()
    actual=pin(v['path']);assert actual['sha256']==v['sha256'] and os.path.samefile(actual['path'],v['path'])
    return Path(v['path'])
def alltrue(r):
    assert r['checks'] and all(v is True for v in r['checks'].values())
    assert r['passed']==r['total']==len(r['checks'])
def write_new(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    pairpath=ROOT/'outputs/full151_v2_three_arm_smoke_pairing_v2/report.json'
    pair=read(pairpath);alltrue(pair)
    assert pair['schema']=='full151_matched_three_arm_smoke_saved_review:1'
    assert pair['status']=='PASS_SAVED_THREE_ARM_SMOKE_PAIRING_NOT_TASK_QUALITY'
    assert pair['same_initial_four_parameter_fingerprints'] is True
    protocol=pin(ROOT/'protocols/full151-v2-three-arm-smoke40-v1.json')
    health={};evaluations={}
    for method in METHODS:
        hp=ROOT/f'outputs/full151_v2_smoke40_{method}_health_v1/report.json'
        h=read(hp);alltrue(h);cfg=read(checked(h['config']))
        assert h['method']==method and h['iteration']==40 and h['final_iteration_scope'] is True
        assert h['status']=='PASS_SAVED_ENGINEERING_HEALTH' and h['role']=='engineering_smoke'
        assert cfg['schema']=='full151_knownrough_native_training_run:2' and cfg['method']==method
        assert cfg['protocol']['sha256']==protocol['sha256'] and os.path.samefile(cfg['protocol']['path'],protocol['path'])
        health[method]=pin(hp)
        assert pair['health'][method]['sha256']==health[method]['sha256']
        checked(h['checkpoint'])
        run=ROOT/f'outputs/full151_v2_policy_eval_smoke40_{method}_seed0_v1'
        runtime=read(run/'runtime.json');e=read(run/'evaluation.json')
        identity=read(run.with_name(run.name+'_watchdog')/'engineering_identity.json');alltrue({'checks':identity['checks'],'passed':sum(identity['checks'].values()),'total':len(identity['checks'])})
        assert identity['engineering_identity_qualified'] is True and runtime['status']=='completed_native_policy_evaluation'
        assert runtime['method']==method and e['complete'] is True and e['sampled_state_rows']==601 and e['sampled_solver_rows']==600 and e['physical_steps']==600 and e['control_intervals']==150
        evaluations[method]={'evaluation':pin(run/'evaluation.json'),'OS_identity':pin(run.with_name(run.name+'_watchdog')/'engineering_identity.json'),'gates_passed':sum(e['gates'].values()),'gates_total':len(e['gates']),'failed_gates':[k for k,v in e['gates'].items() if not v], 'position_max_mm':1000*e['max_object_position_error_151_m'],'shaft_max_deg':e['max_shaft_angle_151_deg'],'penetration_max_mm':1000*e['max_penetration_m'],'target_rate_max_rad_s':e['max_actual_target_rate_rad_s'],'task_pass':e['native_policy_pass']}
    enrollment={'schema':'full151_matched_three_arm_smoke_enrollment:1','status':'PASS_ENGINEERING_THREE_ARM_SMOKE_NOT_TASK_QUALITY','methods':METHODS,'same_initial_four_parameter_fingerprints':True,'saved_pair_review':pin(pairpath),'health':health,'smoke_protocol':protocol,'full3s_smoke_evaluations':evaluations,'stage_source':pin(__file__),'scope':'Engineering and rawinitial pairing only. All3 task FAIL; no prior benefit, no reference/contact/physical qualification. Primary three budgets preregistered; only A2 starts now; no automatic A1L/P_ONLY queue.','pairing_v1_failure_retained':pin(ROOT/'outputs/full151_v2_three_arm_smoke_pairing_v1/report.json')}
    out=ROOT/'outputs/full151_v2_three_arm_smoke_enrollment_v1.json'
    write_new(out,enrollment);print(json.dumps({'enrollment':pin(out),'evaluations':evaluations},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
