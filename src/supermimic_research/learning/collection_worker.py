"""CPU-only actor collection worker; no learning or old Python imports."""
from pathlib import Path
import os
from supermimic_research.artifacts import file_pin, verify_file_pin
from . import online_contract as C


def preflight(protocol_path):
    path = Path(protocol_path).resolve(strict=True); p = C.read(path)
    C.need(p['schema'] == C.COLLECTION and type(p['native_authorized']) is bool, 'Explicit collection protocol')
    config_path = verify_file_pin(p['configuration_file'])
    _,c,registry = C.configuration(config_path)
    C.need(c == p['configuration'], 'Immutable executed configuration')
    C.need(file_pin(registry.manifest) == p['artifact_manifest'], 'Executed artifact manifest')
    sources = C.source_pins()
    C.need(C.fingerprint(sources) == C.fingerprint(p['sources']), 'Actual unified collection sources')
    root = C.root_metadata(c,registry)
    health = C.actor_metadata(p['actor_input'],p['collection_cycle'],c,registry,sources,root)
    from supermimic_research.runtime.native_collection import profile_plan
    C.need(C.read(registry.resolve_pin(p['profile_plan'])) == profile_plan(), 'Fixed deterministic exploration profile')
    out = Path(p['output'])
    runs = Path(__file__).resolve().parents[3]/'runs'
    C.need(out.is_absolute() and out.resolve().is_relative_to(runs.resolve()) and not out.exists(), 'Fresh output within unified runs')
    return path,p,c,registry,root,health


def inspect(protocol_path):
    path,p,c,registry,root,health = preflight(protocol_path)
    import sys
    return dict(status='COLLECTION_METADATA_ONLY',protocol=file_pin(path),source_stage=p['actor_input']['stage_id'],
        collection_cycle=p['collection_cycle'],native_authorized=p['native_authorized'],output=p['output'],
        Torch_imported='torch' in sys.modules,MuJoCo_imported='mujoco' in sys.modules,NN_native_calls=0)


def run(protocol_path):
    path,p,c,registry,root,health = preflight(protocol_path)
    C.need(p['native_authorized'] is True and c['online']['native_authorized'] is True, 'Actual native execution configuration')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    C.need(not torch.cuda.is_initialized(), 'CPU worker must not initialize CUDA')
    from . import runner as R,imagination as IM,world_model as WM
    from supermimic_research.runtime.native_collection import collect_matched_round,profile_plan
    ck=torch.load(registry.resolve_pin(p['actor_input']['checkpoint']),map_location='cpu',weights_only=False)
    C.validate_actor_payload(ck,p['actor_input'],health,root,c,p['sources'])
    actors={}
    for mode in R.MODES:
        pair=IM.make_fresh_actor_critic(seed=1733,device='cpu')
        actor=pair['actor']; actor.load_state_dict(ck['arms'][mode]['stateactor'],strict=True)
        C.need(WM.raw_tree_equal(actor.state_dict(),ck['arms'][mode]['stateactor']) and
            WM.raw_tree_equal(actor.norm.state_dict(),ck['arms'][mode]['norm']['actor']), 'Raw finite-source Actor and normalization state')
        C.need(all(torch.isfinite(v).all() for v in actor.state_dict().values()), 'All serialized Actor tensors finite')
        actor.eval(); actor.requires_grad_(False); actors[mode]=actor
    geometry=R._geometry(registry,c)
    tasks={mode:R._task(registry,c,mode,geometry) for mode in R.MODES}
    provenance=dict(actor_input=p['actor_input'],root_checkpoint=root['checkpoint'],task_bundle=registry.pin(c['artifacts']['task']),
        collection_cycle=p['collection_cycle'],profile_plan=p['profile_plan'],protocol=file_pin(path),sources=p['sources'])
    out=Path(p['output']);out.parent.mkdir(parents=True,exist_ok=True)
    result=collect_matched_round(actors_by_mode=actors,tasks_by_mode=tasks,geometry=geometry,
        model_path=registry.resolve(c['artifacts']['model']),physical=C.read(registry.resolve(c['artifacts']['physical'])),
        plan=profile_plan(),provenance=provenance,output=out)
    C.need(not torch.cuda.is_initialized(), 'Collection remained CPU-only')
    receipt=dict(schema='supermimic_research.collection_worker_receipt:1',protocol=file_pin(path),
        report=file_pin(out/'report.json'),own_CPU_checkpoint_reads=1,NN_constructors=4,unused_Critic_constructors=2,
        WM_FW=0,Critic_FW=0,optimizer_constructors=0,backward=0,learning_updates=0,CUDA_initialized=False)
    C.write(out/'worker_receipt.json',receipt)
    return dict(report=file_pin(out/'report.json'),dataset_report=result['dataset_report'],
        worker_receipt=file_pin(out/'worker_receipt.json'),rows=result['rows'])
