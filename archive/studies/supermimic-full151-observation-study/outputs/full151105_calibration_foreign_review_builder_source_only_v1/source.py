"""DRAFT saved/source-only exact new-label review; no native/framework imports.

Default inspects headers/source identities only. Root must separately authorize
--run after the new discovery and its independent saved review are complete.
No old runtime index, label or qualification is substituted for a new inventory.
"""
from pathlib import Path
from collections import Counter,deque
import argparse,ast,copy,difflib,hashlib,json,re,sys

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'src/full151_gradient/serial_probe105_calibration_v1.py'
WORKER_SHA='1418b7d61bea7274da51be23593807472e7edd7a1720a8bbd25c98624cff133a'
TRANSACTION=ROOT/'src/full151_gradient/transaction105.py'
TRANSACTION_SHA='f1bd0eafed58783a6efbb78a0ba40da086b36851119eddd00e29fcddb972a55c'
PARENT=ROOT/'src/full151_gradient/serial_probe105_v2.py'
PARENT_SHA='a24c89da47f4aad7acfc9d9dfd1f053518cdc878230716bb2c857127ec13e592'
PROPOSAL=ROOT/'protocols/full151105_fresh_calibration_response_proposal_v1.json'
PROPOSAL_SHA='e179d3deb39532b279fbc736645075847aa09399465d45522f68ad3fcb9174fc'
EVIDENCE=ROOT/'outputs/full151105_eight_snapshot_foreign_source_review_v1/foreign_review_v2.json'
PHASES=[24,74,124,148]
LABELS=[f'{p}/{k}' for p in ('A0','A2') for k in PHASES]
EXEMPLARS={label:label.split('/')[0]+'/49' for label in LABELS}
METADATA_KEYS={'role','snapshot_phases','method_proposal','parent_worker','intended_response_release_scope','response_sealing_enforced_by_worker','data_scope','independent_test_trajectory_qualified'}

def require(v,message):
    if not v:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve(strict=True);return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def checked(value):
    require(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str and Path(value['path']).is_absolute(),'exact absolute pin')
    require(type(value['sha256']) is str and re.fullmatch('[a-f0-9]{64}',value['sha256']) is not None,'SHA256')
    path=Path(value['path']).resolve(strict=True);require(pin(path)['sha256']==value['sha256'],'Changed pinned bytes: '+str(path));return path
def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def rooted_names(top):
    entries=top['native_entries'];index={v['identity']:v for v in entries};require(len(index)==len(entries),'unique native owner identities')
    names={}
    for typename,rootname in (('mjlab.envs.manager_based_rl_env.ManagerBasedRlEnv','env'),('dex_supermimic.rl.dex_supermimic.pen.PenSimReader','reader')):
        roots=[v for v in entries if v['type']==typename];require(len(roots)==1,'exact native env/reader source owner')
        queue=deque([(roots[0]['identity'],rootname)])
        while queue:
            identity,name=queue.popleft()
            if identity in names:continue
            names[identity]=name
            if identity not in index:continue
            owner=index[identity]
            for m in owner['members']:
                key=m['key'];suffix='.'+key if owner['kind']=='object' else '['+key+']' if owner['kind']=='list' else '['+repr(key)+']'
                queue.append((m['identity'],name+suffix))
    return names

def semantic(path,top,names):
    match=re.fullmatch(r'native_entry\[(\d+)\]\.value\[(\d+)\](.*)',path)
    if not match:
        require(path.startswith('execution_hook['),'unresolved opaque path: '+path);return path
    i,j,tail=int(match[1]),int(match[2]),match[3];owner=top['native_entries'][i]
    require(owner['entry']==i and owner['identity'] in names and j<len(owner['members']),'actual owner must resolve')
    key=owner['members'][j]['key'];suffix='.'+key if owner['kind']=='object' else '['+key+']' if owner['kind']=='list' else '['+repr(key)+']'
    return names[owner['identity']]+suffix+tail

def function_nodes(path):return {n.name:n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def source_delta():
    require(pin(WORKER)['sha256']==WORKER_SHA and pin(PARENT)['sha256']==PARENT_SHA and pin(TRANSACTION)['sha256']==TRANSACTION_SHA,'exact new worker/old source/transaction')
    old=function_nodes(PARENT);new=function_nodes(WORKER)
    unchanged=[name for name in old if name in new and ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False)]
    required=('controlled_object_score_FP32','object_terms','object_terms_f64','baseline_errors','backend_call_trace','allocate_artifacts','dump_inventory','instrument_transaction_calls','completion_scope','_allowed_set','reviewed_foreign_inventory','validate_review_manifest')
    # Some human-readable messages changed in review helpers; they are checked
    # separately by the source delta. Numerical/transaction helpers remain exact.
    require(all(name in unchanged for name in required if name not in ('reviewed_foreign_inventory','validate_review_manifest')),'unchanged numerical/transaction source helpers')
    run=copy.deepcopy(new['run_probe']);removed=set()
    for node in ast.walk(run):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='schema' and isinstance(k.value,ast.Constant) and k.value.value=='full151105_calibration_response_worker_report:1' for k in node.keywords):
            removed={k.arg for k in node.keywords if k.arg in METADATA_KEYS};node.keywords=[k for k in node.keywords if k.arg not in METADATA_KEYS]
            for k in node.keywords:
                if k.arg=='schema':k.value=ast.Constant('full151105_serial_physical_probe_worker_report:3')
    require(removed==METADATA_KEYS and ast.dump(run,include_attributes=False)==ast.dump(old['run_probe'],include_attributes=False),'run_probe must differ only by declared report metadata')
    return {'unchanged_source_helpers':unchanged,'run_probe_numeric_control_flow_AST_exact_after_report_metadata_removal':True,
       'new_globals_source_scope':'PHASES selects new capture points; parent/method paths and hashes are metadata-only preflight. Release/data-scope strings are report metadata only. No new numerical mutable global or closure state is introduced.',
       'remaining_limits':'Active keyword-default wrappers/external bookkeeping and opaque internals retain the conditional old source limitations; new exact inventory/closure sets must still match per label.'}

def metadata(saved_review=None):
    delta=source_delta();require(pin(PROPOSAL)['sha256']==PROPOSAL_SHA,'current fresh proposal');q=read(PROPOSAL)
    require(q['schema']=='full151105_fresh_calibration_response_physical_proposal:1' and q['role']=='fresh_response_calibration' and q['snapshot_phases']==PHASES,'new proposal namespace/phases')
    require(same(q['parent_worker'],pin(PARENT)),'parent worker source provenance')
    if saved_review is None:return None,delta
    health=read(saved_review);checks=health.get('checks')
    require(type(checks) is dict and checks and all(type(v) is bool and v for v in checks.values()) and type(health.get('passed')) is int and type(health.get('total')) is int and health['passed']==health['total']==len(checks),'strict positive independent saved review')
    require(health.get('engineering_qualified') is True and health.get('probe_mode')=='inventory_discovery_all' and health.get('completed_inventory_snapshots')==8 and health.get('episode_step_returned')==1200 and health.get('restored_branches')==0,'actual new donor-only discovery counts')
    for flag in ('physical_gradient_qualified','whole_hidden_state_qualified','restored_baseline_qualified'):require(health.get(flag) is False,'no inherited restore/gradient/hidden qualification')
    require(list(health['inventory_snapshot_pins'])==LABELS,'eight actual new ordered labels; no old/union/fallback')
    cfg=read(checked(health['config']));report=read(checked(health['producer_report']))
    require(cfg.get('schema')=='full151105_fresh_calibration_response_physical_run:1' and cfg.get('role')=='fresh_response_calibration' and cfg.get('probe_mode')=='inventory_discovery_all','new discovery config required')
    require(same(cfg['proposal'],pin(PROPOSAL)) and same(health['input_bundle'],cfg['input_bundle']),'actual proposal/bundle joins')
    require(report.get('schema')=='full151105_calibration_response_worker_report:1' and report.get('role')=='fresh_response_calibration' and report.get('snapshot_phases')==PHASES,'new producer report identity required')
    require(any(same(pin(WORKER),v) for v in cfg['source_pins'].values()),'actual discovery used exact new worker')
    require(Path(health['run_dir']).samefile(checked(health['producer_report']).parent),'actual new run directory')
    # Intentionally no description/topology or new values file is opened here.
    return health,delta

def evidence_table(old,label):
    exemplar=EXEMPLARS[label];item=old['per_snapshot'][exemplar];desc=read(checked(item['inventory']['description']));top=read(checked(item['inventory']['topology']));names=rooted_names(top)
    actual={(v['path'],v['type']) for v in desc['opaque_identity_only']};allowed=item['allowed_identity_only']
    require(actual=={(v['path'],v['type']) for v in allowed},'previous evidence exact source table')
    roles={}
    for v in allowed:
        key=(v['type'],semantic(v['path'],top,names));require(key not in roles,'unique semantic source owner/type');roles[key]=v
    closures={(semantic(v['path'],top,names),v['value_type']) for v in top['closure_cells']}
    np_roles=Counter(semantic(v['path'],top,names) for v in top['array_values'] if 'path' in v)
    return exemplar,roles,closures,np_roles

def build(saved_review,output):
    health,delta=metadata(saved_review);require(health is not None,'actual independently reviewed discovery required')
    output=Path(output).resolve();require(output.parent==(ROOT/'outputs').resolve() and not output.exists(),'fresh study output')
    output.mkdir();(output/'builder_source.py').write_bytes(Path(__file__).read_bytes())
    try:
        old=read(EVIDENCE);require(old['schema']=='full151105_foreign_inventory_per_snapshot_source_review:2','old conditional source evidence schema')
        require(same(old['worker'],pin(PARENT)),'old evidence pertains to frozen parent source, not new runtime')
        sources={'review_builder':pin(__file__),'new_worker':pin(WORKER),'transaction':pin(TRANSACTION),'parent_worker_source_only':pin(PARENT),'proposal':pin(PROPOSAL),
         'saved_discovery_review':pin(saved_review),'source_inventory_config':health['config'],'source_inventory_report':health['producer_report'],'source_inventory_table':health['source_inventory'],
         'conditional_old_source_evidence_not_new_qualification':pin(EVIDENCE),'input_bundle':health['input_bundle']}
        for k,v in old['source_pins'].items():checked(v);sources['source_basis__'+k]=v
        for key in ('auditor','pure_saved_helper'):
            if key in health:sources['new_saved_'+key]=health[key]
        tables={};details={};occurrences=0
        for label in LABELS:
            item=health['inventory_snapshot_pins'][label];require(set(item)=={'description','topology'},'exact new inventory artifact pair')
            desc=read(checked(item['description']));top=read(checked(item['topology']));names=rooted_names(top);phase=int(label.split('/')[1]);profile=desc['profile']
            require(desc.get('schema')=='full151105_probe_transaction_inventory:1' and desc['label']==label and desc['strict_native_types_checked'] is True,'actual new native inventory label')
            require(type(profile.get('phase')) is int and profile['phase']==phase and profile.get('boundary')=='pre_action' and profile.get('format_version')==105 and profile.get('mode')=='rough_observation_reference_full151_v1' and profile.get('retarget_variant_id')=='legacy_H123_full151_SCP8_knownrough_v1' and profile.get('strict_native_types_checked') is True,'new actual profile105/preaction/phase')
            require(same(top['worker_source'],pin(WORKER)) and same(top['transaction_source'],pin(TRANSACTION)),'new actual topology source identity')
            exemplar,roles,expected_closures,expected_np=evidence_table(old,label)
            observed=desc['opaque_identity_only'];require(type(observed) is list and len(observed)==len({v['path'] for v in observed}),'unique new exact opaque paths')
            observed_roles=[(v['type'],semantic(v['path'],top,names)) for v in observed]
            require(len(set(observed_roles))==len(observed_roles) and set(observed_roles)==set(roles),'new/unreviewed or missing semantic opaque owner/type; stop for new source review, no union fallback')
            closures={(semantic(v['path'],top,names),v['value_type']) for v in top['closure_cells']}
            np_roles=Counter(semantic(v['path'],top,names) for v in top['array_values'] if 'path' in v)
            require(closures==expected_closures and len(closures)==desc['closure_cells'],'new closure path/type set requires explicit review if changed')
            require(np_roles==expected_np and sum(np_roles.values())==desc['explicit_arrays'],'new explicit array role requires review if changed')
            allowed=[];items=[]
            for actual,key in zip(observed,observed_roles):
                typ,role=key;source=roles[key]
                require('*' not in actual['path'] and '*' not in typ,'no wildcard inventory permit')
                if typ=='warp._src.context.Graph':detail='Mutable graph_exec lazy cache is NOT restored bitexact; hidden graph scratch remains UNKNOWN, including non-inventoried solver/SAP/implicit workspaces. Fixed captured pointers, no graph recapture/update and full synchronization plus new per-label3baselines are necessary empirical gates.'
                elif typ=='torch.cuda.streams.ExternalStream':detail='Exact named non-owning TorchArray stream handle only; queue/runtime hidden state is not serialized or qualified.'
                elif typ=='warp._src.context.Device':detail='Fixed device-context handle only; allocator/runtime internals are not serialized or qualified.'
                elif typ in ('mujoco._structs.MjModel','mujoco._structs.MjData'):detail='Exact Simulation CPU setup template only; branch uses WP integration, separate host readback and disabled NanGuard. Its C bytes are not claimed restored.'
                elif typ.startswith('mujoco._specs.'):detail='Exact owner/type from reviewed compile-time spec table, not an exemption for a MuJoCo class. No edit_spec/compile or parameter writes in the source-identical branch.'
                else:require(typ=='torch.autograd.grad_mode.no_grad','unreviewed foreign type');detail='Exact unused original collect decorator factory; no Runner/collect executes and no general thread/grad-mode hidden-state restore is claimed.'
                reason=f'{label}: {role}. {detail} Source basis is the fixed same-donor {exemplar} semantic owner/type review, not its runtime indices, labels, baseline results or qualification. Historical conditional source reason: {source["reason"]} Current allowance is restricted to worker1418/transactionf1bd and this new actual description/topology; current exact runtime paths/types and three fresh per-label baselines remain mandatory.'
                allowed.append({'path':actual['path'],'type':typ,'reason':reason});items.append({'path':actual['path'],'type':typ,'semantic_owner':role,'source_evidence_exemplar':exemplar})
            tables[label]={'inventory':item,'allowed_identity_only':allowed,'source_reviewed_closure_globals':True}
            details[label]={'source_basis_exemplar':exemplar,'semantic_items':items,'closure_roles':sorted(closures),'explicit_NP_roles':dict(np_roles),'opaque_count':len(allowed),'type_counts':dict(Counter(v['type'] for v in allowed))}
            for field,value in item.items():sources['new_inventory_'+label.replace('/','_')+'_'+field]=value
            occurrences+=len(allowed)
        detail={'schema':'full151105_calibration_exact8_source_review_details:1','status':'NEW_EXACT_LABEL_SOURCE_REVIEW_CONDITIONAL_HIDDEN_STATE_UNKNOWN','source_delta':delta,'per_label':details,'exemplar_policy':EXEMPLARS,'no_union_or_runtime_index_fallback':True,'source_evidence_is_not_inherited_native_qualification':True,'whole_hidden_state':'UNKNOWN','execution':{'production_framework_imports':0,'Env_model_WM_Actor_geometry_physics_GPU':0}}
        save(output/'source_review_details.json',detail);sources['new_source_review_details']=pin(output/'source_review_details.json')
        for v in sources.values():checked(v)
        foreign={'schema':'full151105_foreign_inventory_per_snapshot_source_review:2','worker':pin(WORKER),'transaction':pin(TRANSACTION),'input_bundle':health['input_bundle'],
         'source_reviewed_closure_globals':True,'approved_probe_modes':['serial_full'],'allowed_snapshot_labels':LABELS,'phase0_only':False,'per_snapshot':tables,'source_pins':sources,
         'source_inventory_saved_review':pin(saved_review),'source_inventory_config':health['config'],'source_inventory_report':health['producer_report'],'source_inventory_run':health['run_dir'],
         'conditional_engineering_only':True,'whole_graph_hidden_state_qualified':False,'graph_hidden_scratch_dependence':'UNKNOWN_NOT_EXHAUSTIVELY_PROVED_OR_RESTORED',
         'native_restore_qualified':False,'physical_gradient_qualified':False,'GPU_run_authorized':False,'scope':'New calibration phases exact8 source allowance only. No old-label/runtime union, whole-hidden certificate, true Jacobian, independent-state claim or execution authorization.'}
        save(output/'foreign_review_v1.json',foreign)
        report={'schema':'full151105_calibration_exact8_foreign_source_review_enrollment:1','status':'NEW_EXACT8_SOURCE_REVIEW_CONDITIONAL_HIDDEN_STATE_UNKNOWN','foreign_review':pin(output/'foreign_review_v1.json'),'source_inventory_saved_review':pin(saved_review),'source_pins':sources,'inventory_labels':LABELS,'per_label_opaque_counts':{k:len(v['allowed_identity_only']) for k,v in tables.items()},'reviewed_occurrences_not_independent_objects':occurrences,'physical_gradient_qualified':False,'whole_graph_hidden_state_qualified':False,'native_restore_qualified':False,'runtime_validation_not_executed':True,'execution':detail['execution']}
        save(output/'report.json',report);return report
    except BaseException as exc:
        save(output/'failure.json',{'status':'BLOCKED_EXACT_NEW_INVENTORY_SOURCE_REVIEW_NO_REPAIR','exception':repr(exc),'no_new_native_qualification':True,'automatic_retry':False});raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--saved-review',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:
        if args.saved_review is None or args.output is None:parser.error('--run requires actual --saved-review and fresh --output')
        report=build(args.saved_review,args.output);print(json.dumps({'status':report['status'],'foreign_review':report['foreign_review'],'labels':len(LABELS),'reviewed_occurrences':report['reviewed_occurrences_not_independent_objects']}))
    else:
        _,delta=metadata(args.saved_review);print(json.dumps({'status':'SOURCE_HEADER_METADATA_ONLY_NO_NEW_INVENTORY_READ','builder':pin(__file__),'worker':pin(WORKER),'labels':LABELS,'source_delta':delta,'new_description_topology_reads':0,'new_manifest_writes':0}))
    require(not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic','full151_gradient','full151_native') for k in sys.modules),'no production/framework imports')

if __name__=='__main__':main()
