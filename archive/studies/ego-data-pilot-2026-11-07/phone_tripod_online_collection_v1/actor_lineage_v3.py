"""Exact own-checkpoint metadata/payload lineage; JSON/NumPy only on import.

No checkpoint is deserialized here. The caller loads exactly the declared own
checkpoint once after this metadata validation, and validates its payload
before constructing Actor modules. Norm-origin identity is source/health-chain
evidence; it does not pretend to independently reload the root normalizers.
"""
from pathlib import Path
import hashlib,json

SOURCE_KEYS=('task','reader_replay','directional','imagination','Actor_Critic','Actor_common','paired_trainer')
ONLINE_PROTOCOL='phone_tripod_online_matched_learning_protocol:1'
ONLINE_CHECKPOINT='phone_tripod_online_matched_checkpoint:1'
ONLINE_HEALTH='phone_tripod_online_matched_checkpoint_health:1'
ROOT_CHECKPOINT='phone_tripod_sharedWM_A1A2_training_checkpoint:1'
ROOT_HEALTH='phone_tripod_matched_learning_checkpoint_health:1'
ROOT_FIELDS=('root_training_protocol','root_result','root_health','root_terminal')

def need(v,msg):
    if not v:raise ValueError(msg)

def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def checked(v):
    need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Changed absolute lineage pin');return Path(v['path'])

def read(v):return json.loads(checked(v).read_text(encoding='utf8'))

def positive(h):return bool(h.get('checks')) and all(v is True for v in h['checks'].values())

def subset(p):return {key:p['sources'][key] for key in SOURCE_KEYS}

def root_metadata(p):
    values={key:p['inputs'][key] for key in ROOT_FIELDS}
    if any(v is None for v in values.values()):return None
    tp,r,h,t=[read(values[key]) for key in ROOT_FIELDS]
    seed=read(p['inputs']['offline_seed_protocol'])
    need(r['schema']=='phone_tripod_matched_supermimic_training_result:1' and r['status']=='COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION','Actual complete offline1000 root')
    need(r['paired_round']==1000 and r['shared_WM_updates']==1100,'Exact offline root phase')
    need(h['schema']==ROOT_HEALTH and h['paired_round']==1000 and h['shared_WM_updates']==1100 and h['checkpoint']==r['checkpoint'] and positive(h),'Healthy serialized offline root')
    checked(r['checkpoint'])
    need(t['returncode']==0 and t['protocol']==values['root_training_protocol'] and r.get('protocol',t['protocol'])==values['root_training_protocol'],'True final resume protocol, not initial protocol')
    need(tp['inputs']==seed['inputs'] and tp['sources']==seed['sources'] and tp['recipe']==seed['recipe'],'Root same offline experiment/data/recipe/source')
    need(t['source']==tp['sources']['trainer']==p['sources']['paired_trainer'],'Actual original paired trainer')
    need(tp['inputs']['task_bundle']==p['inputs']['task_bundle'],'Same tripod task')
    for key in SOURCE_KEYS:
        sourcekey='trainer' if key=='paired_trainer' else key
        need(tp['sources'][sourcekey]==p['sources'][key],'Frozen root interface '+key)
    for k,n in [('WM_optimizer_steps',1100),('Actor_optimizer_steps',2000),('Critic_optimizer_steps',2000)]:need(r['counts'][k]==n,'Complete root counts '+k)
    return dict(checkpoint=r['checkpoint'],health=values['root_health'],protocol=values['root_training_protocol'],result=values['root_result'],terminal=values['root_terminal'])

def online_header(h,cp,root,p,protocol,stage,*,complete=True):
    """Validate all counters, recipe and source provenance without any PT load."""
    need(h['schema']==ONLINE_HEALTH and positive(h) and h['checkpoint']==cp,'Healthy own online checkpoint')
    need(h['stage_id']==h['stage_cycle']==stage and type(stage) is int and 1<=stage<=2,'Source cycle, not future cycle')
    need(protocol['schema']==ONLINE_PROTOCOL,'Only own online matched protocol')
    recipe=protocol['recipe'];need(recipe['updates_per_cycle']==100 and recipe['collection_cycles']==3,'Fixed online source budget')
    need(read(protocol['recipe_pin'])['recipe']==recipe and h['recipe_pin']==protocol['recipe_pin'],'Pinned online numerical recipe')
    u=h['update_in_cycle'];g=(stage-1)*100+u
    need(type(u) is int and u in [10,100] and (not complete or u==100),'Only complete100 supplies collection; healthy10 can be a resume ancestor')
    need(h['global_paired_updates']==g and h['total_lineage_paired_updates']==1000+g and h['shared_WM_updates']==1100+g,'Online-only and lineage counters')
    need(h['offline1000_root']==h['norm_origin_checkpoint']==root['checkpoint'],'Frozen root norm identity')
    need(h['collector_sources']==subset(p),'Same seven frozen task/factory/memory sources')
    need(protocol['sources']['collector']==p['sources']['collector'],'Same v3 collection implementation')
    for key in SOURCE_KEYS:need(protocol['sources'][key]==p['sources'][key],'Same online direct source '+key)
    for key in ROOT_FIELDS:need(protocol['inputs'][key]==p['inputs'][key],'Same offline root artifact '+key)

def validate_actor_metadata(p):
    cycle=p['collection_cycle'];a=p['actor_input']
    need(type(cycle) is int and 1<=cycle<=3,'Finite three-cycle collection contract')
    need(set(a)=={'kind','stage_id','checkpoint','health','training_protocol'} and type(a['stage_id']) is int and a['stage_id']==cycle-1,'Source checkpoint cycle is previous completed cycle')
    need(a['kind']==('offline1000' if cycle==1 else 'online_stage'),'Declared checkpoint kind, not arbitrary PT')
    root=root_metadata(p)
    pending=root is None or any(a[k] is None for k in ['checkpoint','health','training_protocol'])
    if pending:
        need(p['CPU_run_authorized'] is False,'Pending lineage forbids actual collection')
        return dict(pending=True,kind=a['kind'],source_stage_id=a['stage_id'],collection_cycle=cycle,root=root)
    for key in ['checkpoint','health','training_protocol']:checked(a[key])
    h=read(a['health']);tp=read(a['training_protocol'])
    if a['kind']=='offline1000':
        need(a['checkpoint']==root['checkpoint'] and a['health']==root['health'] and a['training_protocol']==root['protocol'],'Stage0 exact final offline1000 only')
    else:
        need(h['protocol']==a['training_protocol'],'Actual online execution protocol')
        online_header(h,a['checkpoint'],root,p,tp,a['stage_id'])
        # The original online update source/recipe is one experiment across cycles.
        previous_cp=h['parent_checkpoint'];previous_health=h['parent_health'];last_global=h['global_paired_updates'];seen={a['checkpoint']['sha256']}
        for _ in range(7):
            checked(previous_cp);need(previous_cp['sha256'] not in seen,'No cyclic checkpoint lineage');seen.add(previous_cp['sha256'])
            if previous_cp==root['checkpoint']:
                need(previous_health==root['health'],'Exact root health endpoint');break
            prev=read(previous_health)
            prev_protocol=read(prev['protocol'])
            need(prev['recipe_pin']==h['recipe_pin'] and prev_protocol['recipe']==tp['recipe'] and prev_protocol['recipe_pin']==tp['recipe_pin'] and prev_protocol['sources']==tp['sources'],'Same frozen experiment across an actual resume protocol')
            online_header(prev,previous_cp,root,p,prev_protocol,prev['stage_cycle'],complete=False)
            need(prev['global_paired_updates']<last_global,'Parent updates strictly precede child; resume10 is not next-stage input')
            last_global=prev['global_paired_updates'];previous_cp,previous_health=prev['parent_checkpoint'],prev['parent_health']
        else:raise ValueError('Lineage must reach frozen offline root within finite declared stages')
    return dict(pending=False,kind=a['kind'],source_stage_id=a['stage_id'],collection_cycle=cycle,root=root,health=h)

def validate_actor_payload(ck,p,meta):
    """Called after one own CPU load and before all Actor/Critic constructors."""
    a=p['actor_input'];tp=read(a['training_protocol'])
    need(ck['protocol']==a['training_protocol'] and ck['task_bundle']==p['inputs']['task_bundle'],'Exact executed payload protocol/task')
    need(ck['recipe']==tp['recipe'] and ck['inputs']==tp['inputs'] and ck['sources']==tp['sources'],'Full payload recipe/input/source binding')
    need(set(ck['arms'])=={'A1_tracking','A2_engineered_material_proxy'},'Both matched arms')
    if a['kind']=='offline1000':
        need(ck['schema']==ROOT_CHECKPOINT and ck['paired_round']==1000 and ck['shared_WM_updates']==1100,'Only final root offline artifact')
    else:
        h=meta['health'];need(ck['schema']==ONLINE_CHECKPOINT,'Only own online matched artifact')
        for key in ['stage_cycle','update_in_cycle','global_paired_updates','total_lineage_paired_updates','shared_WM_updates',
                    'recipe_pin','parent_checkpoint','parent_health','offline1000_root','norm_origin_checkpoint','collector_sources']:
            need(ck[key]==h[key],'Online payload/health field '+key)
        need(ck['stage_cycle']==a['stage_id'] and ck['update_in_cycle']==100,'Only completed online source stage')
    return meta
