"""Retain the completed local initialization attempt and relaxed saved plans."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def pin(path):
    path=Path(path).resolve(strict=True);raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

report=read(ROOT/'runs/initialization_search_v1/report.json');review=read(ROOT/'runs/initialization_search_review_v1/report.json')
assert (report['passed'],report['total'])==(8,8) and (review['passed'],review['total'])==(10,10)
assert report['qualified_candidates']==[] and len(report['cases'])==6
assert report['counts']['integration_steps']==3508 and report['static_counts']['static_forward']==919
assert read(ROOT/'runs/initialization_search_v1/terminal.json')['returncode']==0
assert review['report']==pin(ROOT/'runs/initialization_search_v1/report.json')
intent=read(report['intent']['path'])
for item in list(intent['sources'].values())+list(intent['frozen_runtime_sources'].values()):assert pin(item['path'])==item
for item in report['candidates']:
    assert item['geometry_pass'] and not item['qualified']
    for key in ('proposal_log','reset_state','geometry','reference_deviation'):assert pin(item[key]['path'])==item[key]
test_path=ROOT/'runs/initialization_full_tests.log'
assert 'Ran 62 tests' in test_path.read_text(encoding='utf8') and test_path.read_text(encoding='utf8').rstrip().endswith('OK')
frozen_paths=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
assert not subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen_paths],cwd=ROOT).strip()
old=ROOT.parent/'ego-data-pilot-2026-11-07'
convex_origin=pin(old/'geometry_query_v2/convex_query.py');convex_new=pin(ROOT/'src/supermimic_research/mechanics/convex.py')
assert convex_origin['sha256']==convex_new['sha256']=='64306b1174c4ef914d1a3f1d6e99188fcf2835d483dbeb518617baaf00e49729'
origins=dict(schema='supermimic_research.mechanics_origins:1',convex_query=dict(original=convex_origin,maintained=convex_new,byte_identical=True),
    collision_audit=dict(maintained=pin(ROOT/'src/supermimic_research/mechanics/collision.py'),
        adapted_eligibility_and_shape_algorithm=pin(old/'retarget_gravity_v3/evaluator_v3.py'),
        adapted_segment_distance_algorithm=pin(old/'retarget_geometry_v2/solve_contact_first_v2.py'),
        method='Ordinary package imports and explicit model/data; no old-path imports, discovery or runtime source selection.'),
    initialization=dict(maintained=pin(ROOT/'src/supermimic_research/mechanics/initialization.py'),method='New raw native-force score and bounded38-variable search; frozen online source closure unchanged.'))

out=ROOT/'provenance/initialization';assert not out.exists();out.mkdir();retained=[]
def keep(relative,name):
    source=ROOT/relative;destination=out/name;item=pin(source);shutil.copyfile(source,destination)
    assert destination.read_bytes()==source.read_bytes()
    retained.append(dict(original_path=item['path'],retained_path=str(destination.relative_to(ROOT)).replace('\\','/'),bytes=item['bytes'],sha256=item['sha256']))
for relative,name in [
    ('runs/initialization_search_v1/intent.json','search_intent.json'),
    ('runs/initialization_search_v1/report.json','search_report.json'),
    ('runs/initialization_search_v1/terminal.json','search_terminal.json'),
    ('runs/initialization_search_v1/zero_barrier.json','zero_barrier.json'),
    ('runs/initialization_search_v1/search_seeds.json','search_seeds.json'),
    ('runs/initialization_search_review_v1/report.json','saved_review.json'),
    ('runs/relaxed_COM_relocation_v1/report.json','relaxed_COM_plan_v1_historical.json'),
    ('runs/relaxed_COM_relocation_v1/planner_source.py','relaxed_COM_planner_v1_historical.py'),
    ('runs/relaxed_COM_relocation_v2/report.json','relaxed_COM_plan_v2.json'),
    ('runs/initialization_full_tests.log','unit_tests.log'),
    ('runs/protocols/phone_online_resume1.toml','reproduction_input_configuration.toml'),
    ('runs/freeze_initialization.py','freeze_milestone.py'),
]:keep(relative,name)
for i in range(2):
    for name in ('candidate','geometry','reference_deviation'):keep(f'runs/initialization_search_v1/search_seed{i}/{name}.json',f'seed{i}_{name}.json')
old_planner=read(out/'relaxed_COM_plan_v1_historical.json')['sources']['planner']
assert pin(out/'relaxed_COM_planner_v1_historical.py')['sha256']==old_planner['sha256']
with (out/'origins.json').open('x',encoding='utf8') as f:json.dump(origins,f,indent=2,allow_nan=False)
summary=dict(schema='supermimic_research.initialization_milestone:1',frozen_runtime_commit='9b38f97',search_integrity=[8,8],saved_review=[10,10],unit_tests=62,
    native_counts=report['counts'],static_counts=report['static_counts'],review_static_counts=review['fresh_static_counts'],
    original_zero_cases=2,candidates=2,candidate_native_cases=4,qualified_candidates=[],new_task_bundles=0,new_learning_stage_started=False,new_checkpoint=False,
    candidate_summary=[dict(name=c['name'],calls=c['search']['calls'],stop=c['search']['stop'],cost=c['search']['fresh_final_cost'],geometry_pass=c['geometry_pass'],
        initial_vertical_net_force_N=c['initial_net_generalized26'][22]) for c in report['candidates']],
    relocation_and_moment_free_plan=read(out/'relaxed_COM_plan_v2.json')['records'],goal_complete=False,goal_status='active',
    data_role='NO_FITTING_INITIALIZATION_DEVELOPMENT',policy_benefit_qualified=False,
    conclusion='Both initial geometries pass but all four native cases fail. Local score minimization stalls; relaxed fixed-pocket COM relocation fails, while dropping object moment permits sufficient upward force. Next stage must change contact patches/lever arms, not claim global impossibility or train on this failed initializer.',
    next_plan='docs/research-plan-v0.40.md',retained=retained,
    external_rule='Raw NPZ states/traces and all916 proposal rows stay under runs and retain their declared no-fitting role; pins are in retained reports.',
    reproduction_configuration_scope='Post-completion copy of the actual commanded configuration, also pinned through the prior authority intent; not a newly authored contemporaneous search intent.')
with (out/'summary.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(dict(summary=pin(out/'summary.json'),retained=len(retained),origins=pin(out/'origins.json'))))
