"""Retain completed diagnostic evidence without copying binary trajectories."""
from pathlib import Path
import hashlib,json,shutil,subprocess

ROOT=Path(__file__).resolve().parents[1]

def pin(path):
    path=Path(path).resolve(strict=True);raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

out=ROOT/'provenance/control_authority'
assert not out.exists()
a=read(ROOT/'runs/control_authority_v1/report.json')
t=read(ROOT/'runs/contact_torque_box_v1/report.json')
r=read(ROOT/'runs/control_authority_review_v3/report.json')
assert (a['passed'],a['total'])==(8,8)
assert (t['passed'],t['total'])==(9,9)
assert (r['passed'],r['total'])==(9,9)
assert len(a['cases'])==130 and a['individual_cases_passing']==0
assert a['paired_schedules_passing']==[] and a['counts']['integration_steps']==75011
assert read(ROOT/'runs/control_authority_v1/terminal.json')['returncode']==0
assert r['authority_report']==pin(ROOT/'runs/control_authority_v1/report.json')
assert r['torque_report']==pin(ROOT/'runs/contact_torque_box_v1/report.json')
assert r['auditor']==pin(ROOT/'tools/review_control_authority.py')
for record in t['records']:assert record['raw']==pin(record['raw']['path'])
native=[x for x in r['details']['support_weight_sensitivity'] if x['cone']=='native_mju_decodePyramid_condim3']
assert len(native)==6
for dt in [.0003125,.00015625]:
    bydomain={x['domain']:x for x in native if x['dt_s']==dt}
    assert bydomain['first_H_EMA_half_scale']['support_weight_interval']['maximum']['success'] is False
    assert abs(bydomain['steady_residual_scale']['support_weight_interval']['maximum']['support_weight_N']-.16781047280218364)<1e-12
    assert abs(bydomain['full_actuator_target_range_analytical_only']['support_weight_interval']['maximum']['support_weight_N']-.4929919941383698)<1e-12
tests=ROOT/'runs/control_authority_full_tests_v2.log'
assert 'Ran 58 tests' in tests.read_text(encoding='utf8') and tests.read_text(encoding='utf8').rstrip().endswith('OK')
assert not subprocess.check_output(['git','diff','--name-only','9b38f97','--','src','configs','archive'],cwd=ROOT).strip()

retained=[]
out.mkdir()
for relative,name in [
    ('runs/control_authority_v1/report.json','authority_report.json'),
    ('runs/control_authority_v1/intent.json','authority_intent.json'),
    ('runs/control_authority_v1/terminal.json','authority_terminal.json'),
    ('runs/control_authority_v1/zero_barrier.json','zero_barrier.json'),
    ('runs/control_authority_v1/initial_contact_force_balance.json','ideal_force_balance.json'),
    ('runs/contact_torque_box_v1/intent.json','torque_box_intent.json'),
    ('runs/contact_torque_box_v1/report.json','torque_box_report.json'),
    ('runs/control_authority_review_v1/report.json','saved_review_v1_historical.json'),
    ('runs/control_authority_review_v1/auditor_source.py','saved_review_v1_historical_source.py'),
    ('runs/control_authority_review_v2/report.json','saved_review_v2_historical.json'),
    ('runs/control_authority_review_v3/report.json','saved_review_v3.json'),
    ('runs/control_authority_full_tests_v2.log','unit_tests.log'),
    ('runs/freeze_control_authority.py','freeze_milestone.py'),
]:
    source=ROOT/relative;destination=out/name;item=pin(source)
    shutil.copyfile(source,destination)
    assert destination.read_bytes()==source.read_bytes()
    retained.append(dict(original_path=item['path'],retained_path=str(destination.relative_to(ROOT)).replace('\\','/'),bytes=item['bytes'],sha256=item['sha256']))
historical=read(out/'saved_review_v1_historical.json')['auditor']
assert pin(out/'saved_review_v1_historical_source.py')['sha256']==historical['sha256']

old=ROOT.parent/'ego-data-pilot-2026-11-07'
history=[]
for relative in [
    'functional_tripod_support_v1/completed_v1/report.json',
    'functional_tripod_loaded_entry_v1/solve_compression_geometry_v2.py',
    'functional_tripod_loaded_dynamics_v1/completed_run1/report.json',
    'functional_tripod_loaded_dynamics_v1/saved_review_v1/report.json',
]:history.append(pin(old/relative))

summary=dict(schema='supermimic_research.control_authority_milestone:1',frozen_runtime_commit='9b38f97',
    authority_integrity=[8,8],torque_box_integrity=[9,9],saved_review_v3_integrity=[9,9],unit_tests=58,
    cases=130,passing_cases=0,paired_schedules_passing=0,
    exit_time_s_range=[min(c['native_summary']['finite_steps_saved']*c['dt_s'] for c in a['cases']),max(c['native_summary']['finite_steps_saved']*c['dt_s'] for c in a['cases'])],
    native_dynamic_counts=a['counts'],static_torque_counts=t['counts'],saved_review_counts=r['counts'],
    native_pyramid_frozen_reset_support=native,actual_phone_weight_N=1.962,
    new_learning_stage_started=False,new_checkpoint=False,goal_complete=False,goal_status='active',
    fitting_role='NO_FITTING_CONTROL_AUTHORITY_DIAGNOSTIC',policy_benefit_qualified=False,
    conclusion='Finite legal-action cases all failed. Native-pyramid fixed-reset numerical static capacity is below gravity even with a freely varied dry-friction envelope; no global dynamic impossibility claim.',
    next_plan='docs/research-plan-v0.39.md',retained=retained,
    external_binary_rule='Raw traces/actions/reset arrays remain under runs, pinned by retained reports. They are not fitting data.',
    prior_geometry_preload_sources_reviewed=history)
with (out/'summary.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(dict(summary=pin(out/'summary.json'),retained=len(retained),report=pin(out/'saved_review_v3.json'))))
