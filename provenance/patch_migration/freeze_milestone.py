"""Retain both declared searches, source snapshots, review and ego intake."""
from pathlib import Path
import json,hashlib,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'provenance/patch_migration'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve(strict=True);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
assert not (out/'summary.json').exists()
v1=read(ROOT/'runs/patch_migration_v1/report.json');v2=read(ROOT/'runs/patch_migration_v2/report.json');review=read(ROOT/'runs/patch_migration_review_v1/report.json');ego=read(ROOT/'runs/ego_known_mesh_inventory_v1/report.json')
assert (v1['passed'],v1['total'])==(10,10) and (v2['passed'],v2['total'])==(10,10)
assert (review['passed'],review['total'])==(11,11)
assert v1['qualified']==v2['qualified']==[] and (len(v1['cases']),len(v2['cases']))==(2,4)
assert sum(x['proposal_calls'] for x in v1['restoration_counts'])==136
assert sum(x['proposal_calls'] for x in v2['restoration_counts'])==840
assert (v1['counts']['integration_steps'],v2['counts']['integration_steps'])==(1171,2238)
assert v2['geometry_query_cache']['hits']==124998 and v2['geometry_query_cache']['calls']==179139
for version in ('v1','v2'):assert read(ROOT/f'runs/patch_migration_{version}/terminal.json')['returncode']==0
for item in read(out/'v1_source_map.json'):
    assert pin(ROOT/item['snapshot'])['sha256']==item['original']['sha256']
for item in read(ROOT/'runs/patch_migration_v2/intent.json')['sources'].values():assert pin(item['path'])==item
for item in ego['input_pins']:assert pin(item['path'])==item
old_audit=read(ROOT/'provenance/initialization/saved_review.json')['auditor']
assert pin(ROOT/'provenance/initialization/saved_review_auditor_v1_source.py')['sha256']==old_audit['sha256']
assert read(ROOT/'runs/initialization_search_review_v2/report.json')['passed']==10
log=ROOT/'runs/patch_migration_full_tests.log'
assert 'Ran 68 tests' in log.read_text(encoding='utf8') and log.read_text(encoding='utf8').rstrip().endswith('OK')
frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
assert not subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT).strip()
retained=[]
def keep(relative,name):
    source=ROOT/relative;target=out/name;assert not target.exists();item=pin(source);shutil.copyfile(source,target)
    assert target.read_bytes()==source.read_bytes()
    retained.append(dict(original_path=item['path'],retained_path=str(target.relative_to(ROOT)).replace('\\','/'),bytes=item['bytes'],sha256=item['sha256']))
for version in ('v1','v2'):
    for field in ('intent','report','terminal','compiled_axes','zero_barrier'):keep(f'runs/patch_migration_{version}/{field}.json',f'{version}_{field}.json')
    for i in range(8):
        for field in ('restored_geometry','qualification'):keep(f'runs/patch_migration_{version}/seed{i:02d}/{field}.json',f'{version}_seed{i:02d}_{field}.json')
for field in ('screen','final_geometry','reference_deviation'):keep(f'runs/patch_migration_v2/seed04/{field}.json',f'v2_seed04_{field}.json')
for relative,name in [
    ('runs/patch_migration_review_v1/report.json','saved_review.json'),
    ('runs/initialization_search_review_v2/report.json','prior_initialization_declared_closure_review.json'),
    ('runs/ego_known_mesh_inventory_v1/report.json','ego_intake_report.json'),
    ('runs/patch_migration_full_tests.log','unit_tests.log'),
    ('runs/freeze_patch_migration.py','freeze_milestone.py'),
]:keep(relative,name)
for clip in ('001852','001856','001884'):keep(f'runs/ego_known_mesh_inventory_v1/clip-{clip}.json',f'ego_clip-{clip}.json')
mug=next(x for clip in ego['reports'] for x in clip['best_per_object'] if x['object_id']==9)
assert (mug['hand'],mug['start_frame'],mug['end_frame'])==('right',47,107)
summary=dict(schema='supermimic_research.patch_migration_milestone:1',parent_commit='5429cff',frozen_runtime_commit='9b38f97',
    search_checks_v1=[10,10],search_checks_v2=[10,10],saved_review=[11,11],prior_dependency_closure_review=[10,10],unit_tests=68,
    v1=dict(restoration_proposals=136,native_cases=2,new_candidate_native_cases=0,counts=v1['counts']),
    v2=dict(restoration_proposals=840,native_cases=4,new_candidate_native_cases=2,counts=v2['counts'],force_counts=v2['force_counts'],cache=v2['geometry_query_cache']),
    qualified_initializations=[],new_learning_stage_started=False,new_checkpoint=False,goal_complete=False,goal_status='active',policy_benefit_qualified=False,
    fitting_role='NO_FITTING_PATCH_MIGRATION_DEVELOPMENT',ego_intake_role=ego['role'],next_ego_candidate=dict(clip='001884',**mug),
    candidate_not_qualified_for_contact_or_robot_dynamics=True,
    inspected_RGB_preview=pin(ROOT.parent/'ego-data-pilot-2026-11-07/outputs/hot3d_three_clip_preview.png'),
    next_plan='docs/research-plan-v0.41.md',retained=retained,
    binary_rule='Raw NPZ states/native traces and all proposal logs remain under runs, pinned by reports; none promoted to fitting.',
    audit_scope='Source bytes, selected-state geometry, final native static reset, saved controls/final COM-angle/gates/missing tails; not full per-tick metric independent recomputation or an independent native solver.')
with (out/'summary.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(dict(summary=pin(out/'summary.json'),retained=len(retained),candidate=mug)))
