"""Pure JSON lineage/AST contract. Synthetic headers are never PT-loaded."""
from pathlib import Path
import ast,copy,importlib.util,json,subprocess,sys
H=Path(__file__).resolve().parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run():
    O=load('v3_pure_collector',H/'collect_online_v3.py');L=load('v3_pure_lineage',H/'actor_lineage_v3.py')
    p=O.read(H/'protocol_CPUFalse_v3.json');need=O.need
    need(O.metadata_preflight(H/'protocol_CPUFalse_v3.json')[3] is True,'Pending root is CPUFalse metadata only')
    F=H/'source_contract_v3_fixture';need(not F.exists(),'One fresh pure fixture');F.mkdir()
    def save(name,v):
        q=F/name;q.write_text(json.dumps(v,indent=2),encoding='utf8');return O.pin(q)
    def fake_weight(name):
        q=F/name;q.write_bytes(('SYNTHETIC_HEADER_ONLY_NOT_A_CHECKPOINT '+name).encode());return O.pin(q)
    seed=O.read(p['inputs']['offline_seed_protocol']['path']);root_protocol=save('root_protocol.json',seed)
    root_cp=fake_weight('root_tag.bin')
    rh=dict(schema=L.ROOT_HEALTH,paired_round=1000,shared_WM_updates=1100,checkpoint=root_cp,checks={'fixture_header_only':True})
    root_health=save('root_health.json',rh)
    root_result=save('root_result.json',dict(schema='phone_tripod_matched_supermimic_training_result:1',status='COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION',paired_round=1000,shared_WM_updates=1100,checkpoint=root_cp,counts=dict(WM_optimizer_steps=1100,Actor_optimizer_steps=2000,Critic_optimizer_steps=2000)))
    root_terminal=save('root_terminal.json',dict(returncode=0,protocol=root_protocol,source=seed['sources']['trainer']))
    p['inputs'].update(root_training_protocol=root_protocol,root_result=root_result,root_health=root_health,root_terminal=root_terminal)
    p['actor_input'].update(checkpoint=root_cp,health=root_health,training_protocol=root_protocol)
    need(L.validate_actor_metadata(p)['pending'] is False,'Synthetic exact-root header joins')
    recipe=dict(updates_per_cycle=100,collection_cycles=3,fixture_only=True)
    recipe_pin=save('online_recipe.json',dict(schema='phone_tripod_online_matched_learning_recipe:1',recipe=recipe))
    tp=dict(schema=L.ONLINE_PROTOCOL,recipe=recipe,recipe_pin=recipe_pin,sources=dict(p['sources']),inputs={k:p['inputs'][k] for k in L.ROOT_FIELDS})
    online_protocol=save('online_protocol.json',tp)
    def health(stage,u,parent_cp,parent_health,name):
        cp=fake_weight(name+'_tag.bin');g=(stage-1)*100+u
        h=dict(schema=L.ONLINE_HEALTH,stage_id=stage,stage_cycle=stage,update_in_cycle=u,global_paired_updates=g,total_lineage_paired_updates=1000+g,shared_WM_updates=1100+g,
            protocol=online_protocol,recipe_pin=recipe_pin,checkpoint=cp,parent_checkpoint=parent_cp,parent_health=parent_health,
            offline1000_root=root_cp,norm_origin_checkpoint=root_cp,collector_sources=L.subset(p),checks={'fixture_header_only':True})
        return cp,save(name+'.json',h),h
    cp10,h10,_=health(1,10,root_cp,root_health,'cycle1_10')
    cp100,h100,_=health(1,100,cp10,h10,'cycle1_100')
    cp200,h200,h=health(2,100,cp100,h100,'cycle2_100')
    q=copy.deepcopy(p);q['collection_cycle']=3;q['actor_input']=dict(kind='online_stage',stage_id=2,checkpoint=cp200,health=h200,training_protocol=online_protocol)
    meta=L.validate_actor_metadata(q);need(meta['pending'] is False,'Completed cycle2 via samecycle10 and previouscycle100 reaches root')
    payload=dict(schema=L.ONLINE_CHECKPOINT,protocol=online_protocol,task_bundle=p['inputs']['task_bundle'],recipe=recipe,inputs=tp['inputs'],sources=tp['sources'],arms={m:{} for m in O.MODES})
    for key in ['stage_cycle','update_in_cycle','global_paired_updates','total_lineage_paired_updates','shared_WM_updates','recipe_pin','parent_checkpoint','parent_health','offline1000_root','norm_origin_checkpoint','collector_sources']:payload[key]=h[key]
    L.validate_actor_payload(payload,q,meta)
    rejected=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError):rejected.append(name);return
        raise AssertionError('Expected rejection '+name)
    bad=copy.deepcopy(p);bad['collection_cycle']=2;bad['actor_input']=dict(kind='online_stage',stage_id=1,checkpoint=cp10,health=h10,training_protocol=online_protocol)
    reject('firsthealthy10_cannot_launch_next_collection',lambda:L.validate_actor_metadata(bad))
    bad=copy.deepcopy(q);bad['actor_input']['kind']='arbitrary_pt';reject('arbitrary_checkpoint_kind',lambda:L.validate_actor_metadata(bad))
    bad_payload=copy.deepcopy(payload);bad_payload['norm_origin_checkpoint']=cp100;reject('payload_norm_origin_mismatch',lambda:L.validate_actor_payload(bad_payload,q,meta))
    a=ast.parse((H/'collect_online_v2.py').read_text());b=ast.parse((H/'collect_online_v3.py').read_text())
    fns=lambda t:{v.name:ast.dump(v,include_attributes=False) for v in t.body if isinstance(v,ast.FunctionDef)}
    old,new=fns(a),fns(b);unchanged=[name for name in old.keys()&new.keys() if old[name]==new[name]]
    need(all(name in unchanged for name in ['canonical_online_row','profile_plan','tensor_tree_digest','load_common_pool','make_common_replay']),'Frozen label/profile/read-only/replay functions unchanged')
    child=subprocess.run([sys.executable,str(H/'collect_online_v3.py'),'--protocol',str(H/'protocol_CPUFalse_v3.json')],capture_output=True,text=True)
    need(child.returncode==0 and 'torch' not in sys.modules and 'mujoco' not in sys.modules,'Actual default CLI and no heavy imports')
    checks={k:True for k in ['pending_CPUFalse_contract','root1000_header_binding','completed_online_parent_chain_includes_resume10','payload_source_recipe_health_binding',
        'reject10_current_arbitrarykind_wrongnormorigin','preserved_raw_label_profile_readonly_and_replay_AST','actual_default_metadata_CLI_exit0','zero_NN_native_checkpoint_deserialization']}
    result=dict(schema='phone_tripod_online_collection_source_contract:3',status='PASS_JSON_AST_SOURCE_NOT_NN_NATIVE_OR_HEALTH_QUALIFICATION',checks=checks,passed=len(checks),total=len(checks),
        rejected_cases=rejected,unchanged_functions=sorted(unchanged),protocol=O.pin(H/'protocol_CPUFalse_v3.json'),sources={n:O.pin(H/n) for n in ['collect_online_v3.py','actor_lineage_v3.py','prepare_protocol_v3.py','check_source_contract_v3.py']},
        metadata_CLI=dict(returncode=child.returncode,stdout=json.loads(child.stdout),stderr=child.stderr),
        counts=dict(Torch_imports=0,Actor_Critic_WM_constructors=0,NN_native_FK_steps=0,checkpoint_deserializations=0,NPZ_value_reads=0,optimizer_updates=0),
        scope='Synthetic small .bin tags and JSON health headers exercise routing/lineage only. No actual checkpoint health, weights/norms, Actor forward, native collection or online training is qualified.')
    dest=H/'source_contract_v3.json';need(not dest.exists(),'Fresh source report');dest.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(dict(report=O.pin(dest),passed=len(checks),total=len(checks))))

if __name__=='__main__':run()
