"""Retain geometry/frame priors, failed attempts, numerical QA and asset index."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'provenance/mug_asset'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve(strict=True);data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
mesh=read(ROOT/'runs/mug_mesh_v1/report.json');final=read(ROOT/'runs/mug_asset_validation_v2/report.json');review=read(ROOT/'runs/mug_asset_saved_review_v1/report.json')
old=read(ROOT/'runs/mug_asset_validation_v1/report.json');comparison=read(ROOT/'runs/mug_contact_prior_v1/report.json')
assert (final['passed'],final['total'])==(9,9) and final['qualified_numerical_asset_prototype']
assert (review['passed'],review['total'])==(10,10) and (old['passed'],old['total'])==(7,9)
assert read(ROOT/'runs/mug_decomposition_v1/terminal.json')['timed_out'] is True
for version,pieces in [('v2',64),('v3',128)]:
    report=read(ROOT/f'runs/mug_decomposition_{version}/report.json')
    assert len(report['components'])==pieces and read(ROOT/f'runs/mug_decomposition_{version}/terminal.json')['returncode']==0
    for item in [report['raw'],*map(lambda row:row['obj'],report['components'])]:assert pin(item['path'])==item
assert pin(mesh['source']['path'])==mesh['source'] and mesh['source']['sha256']=='26b2884fa61a9b9870ae089de04ea9bccdf8eb8e1467bdd185e7d514ae259195'
for item in (final['asset_mjb'],mesh['raw'],final['intent']):assert pin(item['path'])==item
for item in read(ROOT/'runs/mug_asset_validation_v2/intent.json')['sources'].values():assert pin(item['path'])==item
snapshots=[
    (ROOT/'runs/mug_decomposition_v1/intent.json','tool',ROOT/'runs/mug_decomposition_v1/executed_tool.py'),
    (ROOT/'runs/mug_decomposition_v2/intent.json','tool',ROOT/'runs/mug_decomposition_v2/executed_tool.py'),
]
for intent,key,source in snapshots:assert pin(source)['sha256']==read(intent)[key]['sha256']
v1pins=read(ROOT/'runs/mug_asset_validation_v1/intent.json')['sources']
assert pin(ROOT/'runs/mug_asset_validation_v1/executed_tool.py')['sha256']==v1pins['tools/validate_mug_asset.py']['sha256']
assert pin(ROOT/'runs/mug_asset_validation_v1/executed_compound.py')['sha256']==v1pins['src/supermimic_research/assets/compound.py']['sha256']
testlog=ROOT/'runs/mug_asset_full_tests.log'
assert 'Ran 75 tests' in testlog.read_text(encoding='utf8') and testlog.read_text(encoding='utf8').rstrip().endswith('OK')
frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
assert not subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT).strip()
assert not out.exists();out.mkdir();retained=[]
def keep(relative,name):
    source=ROOT/relative;target=out/name;item=pin(source);shutil.copyfile(source,target)
    assert target.read_bytes()==source.read_bytes()
    retained.append(dict(original_path=item['path'],retained_path=str(target.relative_to(ROOT)).replace('\\','/'),bytes=item['bytes'],sha256=item['sha256']))
for version in ('v1','v2','v3'):
    for name in ('intent','terminal'):keep(f'runs/mug_decomposition_{version}/{name}.json',f'decomposition_{version}_{name}.json')
    keep(f'runs/mug_decomposition_{version}/worker.log',f'decomposition_{version}_worker.log')
    if version!='v1':keep(f'runs/mug_decomposition_{version}/report.json',f'decomposition_{version}_report.json')
    if version!='v3':keep(f'runs/mug_decomposition_{version}/executed_tool.py',f'decomposition_{version}_source.py')
for version in ('v1','v2'):
    for name in ('intent','report'):keep(f'runs/mug_asset_validation_{version}/{name}.json',f'validation_{version}_{name}.json')
for relative,name in [
    ('runs/mug_mesh_v1/report.json','mesh_report.json'),('runs/mug_mesh_v1/source_sections.png','source_sections.png'),('runs/mug_mesh_v1/compound128_sections.png','compound128_sections.png'),
    ('runs/mug_gauges_v1/protocol.json','source_gauge_protocol.json'),('runs/mug_contact_prior_v1/intent.json','contact_prior_intent.json'),('runs/mug_contact_prior_v1/report.json','contact_prior_report.json'),
    ('runs/mug_asset_validation_v1/executed_tool.py','validation_v1_source.py'),('runs/mug_asset_validation_v1/executed_compound.py','compound_v1_source.py'),
    ('runs/mug_asset_validation_v2/physical_prior.json','physical_prior.json'),('runs/mug_asset_validation_v2/asset_contract.json','asset_contract.json'),('runs/mug_asset_validation_v2/artifacts.toml','artifacts.toml'),
    ('runs/mug_asset_validation_v2/mug_asset.xml','mug_asset.xml'),('runs/mug_asset_validation_v2/drop_scene.xml','drop_scene.xml'),('runs/mug_asset_saved_review_v1/report.json','saved_review.json'),
    ('runs/mug_asset_requirements.txt','asset_environment.txt'),('runs/mug_asset_full_tests.log','unit_tests.log'),('runs/freeze_mug_asset.py','freeze_milestone.py'),
]:keep(relative,name)
sys_path=__import__('sys').path;sys_path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry
inspection=ArtifactRegistry(ROOT/'runs/mug_asset_validation_v2/artifacts.toml').inspect(verify=True)
assert len(inspection)==8 and all(x.get('verified') for x in inspection)
with (out/'registry_review.json').open('x',encoding='utf8') as stream:json.dump(inspection,stream,indent=2)
summary=dict(schema='supermimic_research.mug_asset_milestone:1',parent_commit='f4aaab5',frozen_learning_runtime_commit='9b38f97',
    source_mesh=mesh['source'],source_frame_bake=mesh['bake'],source_nonmanifold_edges_after_exact_weld=mesh['exact_weld_topology']['nonmanifold_edges'],
    decomposition_calls=3,completed_decompositions=2,timed_out_decompositions=1,final_parts=128,final_backend_concavity_threshold_not_certified=True,
    first_prototype_checks=[7,9],final_prototype_checks=[9,9],saved_review=[10,10],unit_tests=75,artifact_registry_verified=8,
    sampled_geometry=final['geometry'],physical_prior=final['physical_prior'],contact_time_prior_s=final['contact_time_prior_s'],
    drop_metrics=[x['metrics'] for x in final['drops']],dt_pair=review['dt_pair'],native_primary_void_and_surface_controls_pass=True,
    total_drop_steps_this_stage=old['counts']['isolated_drop_steps']+final['counts']['isolated_drop_steps']+comparison['native_steps'],
    final_object_only_native_layout=dict(nq=7,nv=6,nu=0),Hand2_loaded=False,calibrated=False,robot_holding_qualified=False,ego_retargeting_qualified=False,
    new_neural_training=False,new_checkpoint=False,goal_complete=False,goal_status='active',next_plan='docs/research-plan-v0.42.md',
    numerical_review_scope=review['scope'],source_snapshots_scope='Executed timeout/64-part/20ms source copies retained. Active shared code has explicit recipe/contact parameters, not separate study implementations.',
    retained=retained,binary_rule='Original GLB, OBJ, component/state NPZ and native MJB remain under external pilot/runs and are SHA-pinned. Archived XML/index copies are evidence; runtime index is in runs/mug_asset_validation_v2.',
    unmeasured_prior_rule='Surface geometry does not identify mass/COM/inertia/friction/compliance. No global geometric-error or real hardware accuracy certificate.')
with (out/'summary.json').open('x',encoding='utf8') as stream:json.dump(summary,stream,indent=2,allow_nan=False)
print(json.dumps(dict(summary=pin(out/'summary.json'),retained=len(retained),prototype_checks=[9,9],robot_holding_qualified=False)))
