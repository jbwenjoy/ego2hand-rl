"""Metadata and lineage for the explicit new online experiment.

No Torch, simulator or checkpoint deserialization. Frozen offline1000 supplies
weights/normalizers; only this implementation's own checkpoint resumes Adam.
"""
from pathlib import Path
import json
from supermimic_research.artifacts import file_pin
from . import runner as R

CHECKPOINT = 'supermimic_research.online_checkpoint:1'
HEALTH = 'supermimic_research.online_checkpoint_health:1'
PROTOCOL = 'supermimic_research.online_protocol:1'
COLLECTION = 'supermimic_research.collection_protocol:1'
ROOT_CHECKPOINT = 'phone_tripod_sharedWM_A1A2_training_checkpoint:1'
ROOT_KEYS = ('root_result','root_protocol','root_health','root_terminal')


def need(condition, message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    import os
    path = Path(path)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def positive(health):
    checks = health.get('checks')
    return isinstance(checks, dict) and bool(checks) and all(v is True for v in checks.values())


def content(pin): return pin['bytes'], pin['sha256']


def source_pins():
    root = Path(__file__).resolve().parents[3]
    return dict(R.source_pins(), cli=file_pin(root/'src/supermimic_research/cli.py'),
                entry=file_pin(root/'scripts/research.py'))


def fingerprint(pins): return {key: content(value) for key, value in pins.items()}


def science_config(config):
    result = R._science_config(config)
    result['online'].pop('native_authorized', None)
    return result


def configuration(path):
    path, c, registry = R.configuration(path)
    online = c.get('online', {})
    need((online.get('collection_cycles'), online.get('updates_per_cycle')) == (3,100), 'Three fixed collection/update cycles')
    need([online.get(k) for k in ['WM_online_rows','WM_old_original_rows','WM_old_tripod_rows']] == [4,2,2], 'Explicit WM 4online+2old+2tripod')
    need((online.get('root_paired_updates'), online.get('root_WM_updates')) == (1000,1100), 'Fixed final1000 root lineage')
    t = c['training']
    need((t['seed'],t['common_WM_updates'],t['updates'],t['horizon'],t['first_checkpoint'],t['cpu_threads']) == (1734,0,300,2,10,4), 'Explicit online seed/budget/first-health contract')
    need(t['microbatch'] in (2,8) and type(t['microbatch']) is int, 'Qualified micro2 or explicitly qualified micro8')
    need(t['stop_after_first_healthy'] is True, 'First healthy online checkpoint must stop before further updates')
    need(type(t['authorized']) is bool and type(online['native_authorized']) is bool, 'Typed run configuration')
    need(t['device'] in ('cpu','cuda:0'), 'Explicit CPU or cuda:0')
    need(c['sampling']['WM_batch'] == c['sampling']['Actor_batch'] == 8, 'Matched batch8')
    need(c['model']['hidden'] == [128,128] and c['model']['geometry_chunk_rows'] == 4, 'Root WM architecture and cache rows')
    need(c['optimizer'] == dict(WM_lr=.0001,actor_lr=.0001,critic_lr=.0003), 'Declared five fresh Adam learning rates')
    need(c['artifacts']['warmstart'] == c['artifacts']['root_result'], 'Only fixed final1000 initializes the new stage')
    need(registry.entries[c['artifacts']['root_result']].role == 'fixed_final1000_result', 'Root result role')
    need(registry.entries[c['artifacts']['root_result']].sha256 is not None, 'Root result is now content pinned')
    return path,c,registry


def root_metadata(c, registry):
    pins = {key:registry.pin(c['artifacts'][key]) for key in ROOT_KEYS}
    result, protocol, health, terminal = [read(registry.resolve_pin(pins[k])) for k in ROOT_KEYS]
    need(result['schema'] == 'phone_tripod_matched_supermimic_training_result:1' and result['status'] == 'COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION', 'Completed fixed offline root')
    need((result['paired_round'], result['shared_WM_updates']) == (1000,1100), 'Complete root phase')
    need(health['schema'] == 'phone_tripod_matched_learning_checkpoint_health:1' and positive(health), 'Positive serialized root health')
    need((health['paired_round'],health['shared_WM_updates']) == (1000,1100) and health['checkpoint'] == result['checkpoint'], 'Root health/result join')
    need(terminal['returncode'] == 0 and terminal['protocol'] == pins['root_protocol'], 'Actual successful final resume protocol')
    need(terminal['source'] == protocol['sources']['trainer'], 'Actual frozen root source')
    need(protocol['CPU_run_authorized'] is True and protocol['GPU_run_authorized'] is True and protocol['device'] == 'cuda:0' and protocol['native_online_authorized'] is False, 'Actual offline root mode')
    for key,n in [('WM_optimizer_steps',1100),('Actor_optimizer_steps',2000),('Critic_optimizer_steps',2000)]:
        need(result['counts'][key] == health['counts'][key] == n, 'Complete offline counts '+key)
    for pin in protocol['sources'].values():
        if pin is not None: registry.resolve_pin(pin)
    for pin in protocol['inputs'].values():
        if pin is not None: registry.resolve_pin(pin)
    registry.resolve_pin(protocol['recipe_pin']); registry.resolve_pin(result['checkpoint'])
    need(content(protocol['inputs']['task_bundle']) == content(registry.pin(c['artifacts']['task'])), 'Same root task')
    need(content(protocol['inputs']['geometry_catalog']) == content(registry.pin(c['artifacts']['geometry'])), 'Same root geometry')
    need(content(protocol['inputs']['dataset']) == content(registry.pin(c['artifacts']['tripod_dataset'])), 'Same TRAIN/MON partition manifest')
    bundle = read(registry.resolve(c['artifacts']['task']))
    need(content(bundle['model']) == content(registry.pin(c['artifacts']['model'])), 'Same native model asset')
    return dict(**pins, checkpoint=result['checkpoint'], protocol=protocol)


def validate_root_payload(ck, root):
    need(ck['schema'] == ROOT_CHECKPOINT and (ck['paired_round'],ck['shared_WM_updates']) == (1000,1100), 'Exact offline1000 payload, never own resume')
    need(ck['protocol'] == root['root_protocol'], 'Actual final root payload protocol')
    protocol = root['protocol']
    for key in ('recipe','inputs','sources'): need(ck[key] == protocol[key], 'Root payload '+key)
    need(set(ck['arms']) == set(R.MODES) and 'WM_loss_scales' in ck, 'Two root arms and frozen loss scales')


def cursor(update):
    need(type(update) is int and 1 <= update <= 300, 'Online update1..300')
    return dict(cycle=(update-1)//100+1,update_in_cycle=(update-1)%100+1,
                global_paired_updates=update,total_lineage_paired_updates=1000+update,
                shared_WM_lineage_updates=1100+update)


def expected_learning_calls(update, microbatch):
    cursor(update)
    need(type(microbatch) is int and microbatch in (2,8),'Declared qualified microbatch')
    micro=8//microbatch
    return dict(WM_supervised_FW=update,WM_backward=update,WM_optimizer_steps=update,
        Actor_FW=4*micro*update,WM_imagination_FW=4*micro*update,Critic_real_FW=4*update,
        Critic_imagination_FW=2*micro*update,Actor_backward_microbatches=2*micro*update,
        Actor_optimizer_steps=2*update,Critic_backward=2*update,Critic_optimizer_steps=2*update)


def selected_inputs(c, registry, root):
    pins = R._artifact_pins(registry,c)
    pins.update({key:registry.pin(c['artifacts'][key]) for key in ('model','physical',*ROOT_KEYS)})
    return dict(pins,root_checkpoint=root['checkpoint'])


def validate_own_header(health, checkpoint_pin, c, sources, root, registry):
    need(health['schema'] == HEALTH and positive(health) and health['checkpoint'] == checkpoint_pin, 'Positive own online health')
    update = health['update']
    need(type(update) is int and update in (10,100,200,300), 'Only declared online health points')
    need(all(type(v) is int for v in health['cursor'].values()) and health['cursor'] == cursor(update), 'Typed exact online cursor')
    need(all(type(v) is int for v in health['counts'].values()), 'Typed exact online counters')
    need(health['counts'] == dict(WM_updates=update,paired_updates=update), 'Online-only optimizer counts')
    need(all(type(v) is int for v in health['learning_calls'].values()) and
        health['learning_calls'] == expected_learning_calls(update,c['training']['microbatch']), 'Online lineage learning-call budget')
    need(health['root_checkpoint'] == health['normalizer_origin_checkpoint'] == root['checkpoint'], 'Frozen root normalizer origin')
    need(science_config(health['configuration']) == science_config(c), 'Same own numerical recipe')
    need(fingerprint(health['provenance']['sources']) == fingerprint(sources), 'Same own source implementation')
    need(fingerprint(selected_inputs(c,registry,root)) == fingerprint(health['provenance']['inputs']), 'Same selected data/task/geometry/root inputs')


def actor_metadata(actor_input, cycle, c, registry, sources, root):
    need(type(cycle) is int and 1 <= cycle <= 3, 'Finite collection cycle')
    need(set(actor_input) == {'kind','stage_id','checkpoint','health','training_protocol'}, 'Explicit actor source fields')
    need(actor_input['stage_id'] == cycle-1 and type(actor_input['stage_id']) is int, 'Previous completed source stage')
    for key in ('checkpoint','health','training_protocol'): registry.resolve_pin(actor_input[key])
    if cycle == 1:
        need(actor_input == dict(kind='offline1000',stage_id=0,checkpoint=root['checkpoint'],health=root['root_health'],training_protocol=root['root_protocol']), 'First collection fixed final1000')
        return None
    need(actor_input['kind'] == 'own_online', 'Only own complete-stage Actor supplies later collection')
    h = read(registry.resolve_pin(actor_input['health']))
    validate_own_header(h,actor_input['checkpoint'],c,sources,root,registry)
    need(h['update'] == 100*(cycle-1) and h['protocol'] == actor_input['training_protocol'], 'First checkpoint10 cannot supply next collection')
    seen = {actor_input['checkpoint']['sha256']}; last = h['update']
    for _ in range(4):
        cp,hp = h['parent_checkpoint'],h['parent_health']
        registry.resolve_pin(cp); registry.resolve_pin(hp)
        need(cp['sha256'] not in seen, 'No cyclic parent lineage'); seen.add(cp['sha256'])
        if cp == root['checkpoint']:
            need(hp == root['root_health'], 'Exact root health endpoint'); break
        h = read(registry.resolve_pin(hp));validate_own_header(h,cp,c,sources,root,registry)
        need(h['update'] < last, 'Strictly decreasing parent updates');last=h['update']
    else: raise ValueError('Finite own lineage must reach root')
    return read(registry.resolve_pin(actor_input['health']))


def validate_actor_payload(ck, actor_input, health, root, c, sources):
    if actor_input['kind'] == 'offline1000': validate_root_payload(ck,root); return
    need(ck['schema'] == CHECKPOINT and ck['update'] == health['update'] and ck['protocol'] == actor_input['training_protocol'], 'Own complete-stage payload')
    need(ck['root_checkpoint'] == ck['normalizer_origin_checkpoint'] == root['checkpoint'], 'Root payload ancestry')
    need(ck['provenance'] == health['provenance'] and science_config(ck['configuration']) == science_config(c), 'Own payload source/config binding')
    need(fingerprint(ck['provenance']['sources']) == fingerprint(sources), 'Current collector source fingerprints')
    for key in ('cursor','counts','learning_calls','parent_checkpoint','parent_health','collections','pool_identity_map','normalizer_origin_checkpoint'):
        need(ck[key] == health[key], 'Own payload/health join '+key)
    need(all(type(v) is int for v in ck['cursor'].values()) and all(type(v) is int for v in ck['counts'].values()), 'Typed own payload cursors')


def verify_collections(receipts, registry):
    need([r['cycle'] for r in receipts] == list(range(1,len(receipts)+1)) and len(receipts) <= 3, 'Ordered collections without repeats/skips')
    total=0
    for receipt in receipts:
        values = {key:read(registry.resolve_pin(receipt[key])) for key in ('protocol','terminal','report','dataset_report')}
        p,t,r,d = [values[k] for k in ('protocol','terminal','report','dataset_report')]
        need(p['schema'] == COLLECTION and p['collection_cycle'] == r['collection_cycle'] == receipt['cycle'], 'Collection protocol/report/cursor join')
        need(t['returncode'] == 0 and t['protocol'] == r['protocol'] == receipt['protocol'], 'Successful collection child/protocol join')
        need(t['source'] == p['sources']['learning/collection_worker.py'] and fingerprint(t['sources']) == fingerprint(p['sources']), 'Actual child source binding')
        need(t['command'][-3:] == ['--protocol',receipt['protocol']['path'],'--run'], 'Actual collection command arguments')
        need(r['dataset_report'] == receipt['dataset_report'] and r['actor_input'] == receipt['actor_input'] == p['actor_input'], 'Actual Actor/pool source join')
        need(r['episodes'] == 16 and 0 < r['rows'] <= 320 and 0 < r['counts']['integration_steps'] <= 30720 and r['counts']['observer_forward'] == 0, 'Finite matched native budget')
        need(d['schema'] == 'phone_tripod_online_common_replay_dataset:1' and d['data_training_ready'] is True, 'Qualified online TRAIN pool')
        need(d['collection_cycle'] == receipt['cycle'], 'Actual cycle dataset')
        worker=read(registry.resolve_pin(receipt['worker_receipt']))
        need(worker['protocol']==receipt['protocol'] and worker['report']==receipt['report'], 'Actual worker/report join')
        need(worker['CUDA_initialized'] is False and worker['own_CPU_checkpoint_reads']==1 and worker['NN_constructors']==4 and worker['unused_Critic_constructors']==2, 'CPU-only one-load two-Actor worker')
        need(all(type(worker[k]) is int and worker[k]==0 for k in ('WM_FW','Critic_FW','optimizer_constructors','backward','learning_updates')), 'Collection has no learning or WM/Critic forward')
        registry.resolve_pin(d['common_pool']); total += r['counts']['integration_steps']
    need(total <= 92160, 'Total native budget without refills')


def inspect(config_path):
    import sys
    path,c,registry=configuration(config_path); root=root_metadata(c,registry)
    return dict(status='ONLINE_METADATA_ONLY_NEW_STAGE',configuration=file_pin(path),root={k:v for k,v in root.items() if k!='protocol'},
        sources=source_pins(),device=c['training']['device'],training_authorized=c['training']['authorized'],
        native_authorized=c['online']['native_authorized'],cycles=3,updates_per_cycle=100,first_checkpoint=10,
        microbatch=c['training']['microbatch'],fresh_Adam=5,normalizer_refit=False,NN_native_calls=0,
        Torch_imported='torch' in sys.modules,MuJoCo_imported='mujoco' in sys.modules)
