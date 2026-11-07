"""One-time SOURCE/JSON preparation. No Torch, arrays, NN or native calls."""
from pathlib import Path
import ast, hashlib, json, runpy, sys
H=Path(__file__).resolve().parent;P=H.parent
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
source=H/'train_shared_WM_A1_A2.py';ast.parse(source.read_text(encoding='utf8'))
module=runpy.run_path(str(source));R=module['RECIPE']
assert 'torch' not in sys.modules
task=read(P/'phone_geometry_actor_integration_v1/protocol_CPUFalse_v2.json')
parent=read(P/'phone_geometry_momentum_wm_v1/protocol_CPUTrue_resume1.json')
sources=dict(trainer=pin(source),WM_trainer=parent['sources']['trainer'])
for name in ['structured_codec','momentum','rotation','scalar_codec','state_codec','FK','directional','geometry_constants','geometry_arrays','direction_arrays']:sources[name]=parent['sources'][name]
for name in ['task','imagination','reader_replay','reference','Actor_Critic','Actor_common','replay','chord_contacts','weighted_support']:sources[name]=task['sources'][name]
recipe=dict(schema='phone_tripod_sharedWM_matched_recipe:1',recipe=R,
    first_learning_checkpoint='Common100 emits JSON only. Paired100 is first resumable updated model/optimizer checkpoint; health+ETA then exit for Root handoff/Goal pause.',
    actor_imagination='All153 complete prealive TRAIN starts may start model H2; these are not129 native H2 groundtruth windows.',
    scope='Shared WM offline matched A1/A2 pilot. No native collection inside trainer; policy-native closed loop and original2s double-dt evaluation remain future requirements.',
    counts_first100=dict(shared_WM_optimizer_steps=200,Actor_optimizer_steps_per_arm=100,Critic_optimizer_steps_per_arm=100,WM_imagination_FW=1600,Actor_FW=1600,Critic_TD_FW=400,Critic_imagination_bootstrap_FW=800,monitor_WM_FW=1,health_FW=10,initial_fresh_NN_ctors=5,initial_target_critic_deepcopies=2,health_fresh_NN_ctors=5),
    no_new_training_or_NN_at_preparation=True)
write(H/'recipe_v1.json',recipe)
inputs=dict(WM_result=pin(P/'phone_geometry_momentum_wm_v1/cpu_resume1/result.json'),
    WM_health=pin(P/'phone_geometry_momentum_wm_v1/cpu_resume1/health_5000.json'),
    task_bundle=task['task_bundle'],old_original_TRAIN=parent['inputs']['original_acquisition'],
    old_pair_labels=parent['inputs']['original_pair_report'],geometry_catalog=parent['inputs']['geometry_catalog'],
    dataset=pin(P/'phone_tripod_replay_dataset_v1/completed_v1/report.json'),
    dataset_review=pin(P/'phone_tripod_replay_dataset_v1/saved_review_v1/report.json'))
protocol=dict(schema=module['SCHEMA'],role='shared_WM_offline_matched_phone_A1_tracking_vs_A2_engineered_proxy_pilot',
    CPU_run_authorized=False,GPU_run_authorized=False,native_online_authorized=False,device='cuda:0',
    output=str((H/'gpu_run1').resolve()),recipe=R,recipe_pin=pin(H/'recipe_v1.json'),inputs=inputs,sources=sources,
    resume_checkpoint=None,resume_health=None,
    data_scope='TRIPOD_TRAIN153/12families and oldTRAIN684 only fit. MONITOR52/4families WM health only. No CHECK/fresh-final/oldTEST opens. New16families known initial simulation, all32episodes failed boundary holding; no ego/policy/holding qualification.',
    terminal_semantics='Boundary POST-H task loss absorbs TD. Collection truncation bootstraps. Any-native-tick loss is descriptive only. Unknown/partial targets never promoted.',
    contact_scope='Newtripod15/28/41 task; canonical22 actual labels. A2 material/normal proxy is engineered and not true contact force, native4ray support or closest normal.',
    checkpoint_scope='State dictionaries, five raw optimizers, named pre/postclip gradients, target critics/norms, sampler and CPU/Python/NumPy/selectedCUDA RNG, fixed log prefix. No whole-module pickle.',
    preparation=pin(__file__))
write(H/'protocol_CPUFalse_v1.json',protocol)
p,*_=module['metadata_preflight'](H/'protocol_CPUFalse_v1.json')
assert 'torch' not in sys.modules
write(H/'metadata_CPUFalse_v1.json',dict(schema='phone_tripod_matched_training_metadata:1',status='PASS_METADATA_ONLY_NOT_NN_RUNTIME',protocol=pin(H/'protocol_CPUFalse_v1.json'),recipe=pin(H/'recipe_v1.json'),sources=sources,inputs=inputs,NN_constructors_FW_backward_optimizer_native_GPU_calls=0,torch_imported=False,source_array_payload_reads=0))
print(json.dumps(dict(protocol=pin(H/'protocol_CPUFalse_v1.json'),trainer=pin(source),recipe=pin(H/'recipe_v1.json'),metadata=pin(H/'metadata_CPUFalse_v1.json'),Torch_imported=False)))
