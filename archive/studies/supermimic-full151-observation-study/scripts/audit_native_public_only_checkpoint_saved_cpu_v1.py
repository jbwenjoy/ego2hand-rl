"""SOURCE DRAFT: native52+self15-only serialized checkpoint health.

Default uses JSON/hash metadata only. Actual Root --run reads exactly original
ea051 + this-run initial + fixed-I checkpoint on CPU, plus one cached-label
JSON. No model/optimizer construction, native Log, FW, AD, NPZ or physics.
Old paired readers/namespace and their sixteen-gradient guards are unchanged.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,struct,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE_PATH=ROOT/'scripts/audit_boundary_public_self15_checkpoint_saved_cpu_v1.py'
BASE_SHA='11a38a6ededd06e8734dd98b61bf8e3162d339e6a1ca369485c8d8d90163303b'
SCHEMA='full151105_native_public_only_WM_component:1'
INITIAL_SCHEMA='full151105_native_public_only_initial:1'
MANIFEST_SCHEMA='full151105_native_public_only_checkpoint_manifest:1'
PROTOCOL_SCHEMA='full151105_native_public_only_training_CPU_protocol:1'
ROLE='native52_self15_only_component_training'
ARM='native52_self15_only';INTERNAL_ARM='shared_public_losses'
INPUTS={'public15_eval_protocol','public15_eval_report','public15_eval_saved12'}
LOSS_NAMES=('state','self15','native_pair')
FIT_KEYS=('fit_input_digest','fit_label_digest','fit_self15_label_digest','fit_combined_label_digest')
RECIPE=ROOT/'protocols/full151105_native_public_only_training_recipe_v1.json'
RECIPE_SHA='ffb89e90aaa03099d2cb444d5c660a61fd8fa44b73b9673bd9651f3ce641b4ee'
CONTRACT=ROOT/'docs/public_native_only_training_contract_v1.md'
CONTRACT_SHA='1766b3fe88ee74015df2907762dd70bd0cd241aefa94b8765c008701324196e1'
PREP=ROOT/'outputs/full151105_native_public_only_checkpoint_auditor_preparation_CPU_v1'

def require(v,s):
    if not v:raise ValueError(s)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(1024*1024),b''):h.update(value)
    return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve(strict=True)),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def module(path,expected):
    require(sha(path)==expected,'Frozen saved-value source')
    spec=importlib.util.spec_from_file_location('_native_only_saved_'+Path(path).stem,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

B=module(BASE_PATH,BASE_SHA)  # Pure definitions only, never old metadata/audit/pair validator.
checked=B.checked;same=B.same;map_same=B._map_same
SPECS=B.ACTUAL_PARAMETER_SPECS;CONTEXT_SPECS=B.CONTEXT_SPECS;COUNTS=B.COUNTS
FIXED_RECIPE={k:v for k,v in B.FIXED_RECIPE.items() if k not in ('lambda_mode','lambda_imp','lambda_cons_by_arm','compute_consistency_both_arms')}
FIXED_RECIPE.update(compute_auxiliary_losses=False,optimized_loss='L_native_pair_FP32',named_parameter_count=16,
    active_gradient_parameter_count=14,inactive_head_gradient_None=True,optimizer_state_count=14)
OWN={'auditor':Path(__file__),'checker':ROOT/'scripts/check_native_public_only_checkpoint_auditor_cpu_v1.py',
    'paired_saved_value_definitions':BASE_PATH,'raw_value_helpers':B.HELPER,'component':B.COMPONENT,'frozen_recipe':RECIPE,'source_contract':CONTRACT}

def exact_tree(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(exact_tree(a[k],v) for k,v in b.items())
    if isinstance(a,list):return len(a)==len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
    return a==b

def allchecks(r):
    require(type(r.get('checks')) is dict and r['checks'] and all(type(v) is bool and v for v in r['checks'].values())
        and type(r['passed']) is int and type(r['total']) is int and r['passed']==r['total']==len(r['checks']),'Positive actual saved/source proof')

def metadata_preflight(protocol_path,run_dir,iteration):
    require(type(iteration) is int and iteration in (100,1000),'Only fixed100/1000')
    require(not any(k in sys.modules for k in ('torch','mujoco','warp','dex_supermimic')),'Default metadata has no backend')
    p=read(protocol_path);require(p['schema']==PROTOCOL_SCHEMA and p['role']==ROLE and p['arm']==ARM and set(p['inputs'])==INPUTS
        and p['CPU_run_authorized'] is True and p['GPU_run_authorized'] is p['automatic_retry'] is False
        and p['background_continue_to_final'] is True,'New native-only own3 protocol and authorized fixed1000 background continuation')
    require(sha(RECIPE)==RECIPE_SHA and sha(CONTRACT)==CONTRACT_SHA and same(p['recipe'],pin(RECIPE)) and same(p['source_contract'],pin(CONTRACT)),'Frozen new35 recipe and source contract')
    recipe=read(RECIPE);require(recipe['schema']=='full151105_native_public_only_training_recipe:1' and exact_tree(recipe['recipe'],FIXED_RECIPE),'Exact35 numerical recipe/type, no old loss weights')
    require(Path(p['output']).resolve(strict=True).samefile(Path(run_dir).resolve(strict=True)),'Actual native-only owned output')
    for value in [*p['inputs'].values(),*p['source_pins'].values()]:checked(value)
    i=p['inputs'];evp=read(checked(i['public15_eval_protocol']));ev=read(checked(i['public15_eval_report']));saved=read(checked(i['public15_eval_saved12']));allchecks(saved)
    require(evp['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_protocol:1' and ev['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1'
        and ev['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION' and same(ev['protocol'],i['public15_eval_protocol']),'Qualified parent native evaluation, not prediction targets')
    require(saved['schema']=='full151105_boundary_public_self15_final_prediction_saved_independent:1' and saved['status']=='PASS_SAVED_ALL3_ALL152_PUBLIC_SELF15_FIXED_FINAL_PREDICTIONS'
        and saved['engineering_qualified'] is True and same(saved['producer_report'],i['public15_eval_report']) and same(saved['protocol'],i['public15_eval_protocol']),'Actual parent saved12')
    parent_protocol=evp['inputs']['training_protocol'];parent=read(checked(parent_protocol));parent_inputs=parent['inputs']
    require(parent['schema']==B.PROTOCOL_SCHEMA and len(parent_inputs)==16 and map_same(parent_inputs,read(checked(evp['inputs']['training_result']))['input_pins']),'Separate derived parent_training_inputs16')
    require(parent_inputs['initial_component']['sha256']==B.PARENT_SHA and same(parent_inputs['initial_component'],evp['inputs']['original_initial_component']),'Trusted original ea051')
    expected_fit={key:ev[key] for key in FIT_KEYS};result=read(checked(evp['inputs']['training_result']))
    require(all(expected_fit[key]==result[key] for key in FIT_KEYS),'Actual fixed parent four fit digests')
    for key in ('cached_sensor_labels','cached_sensor_report','cached_sensor_saved7','target_identities'):
        require(same(parent_inputs[key],evp['inputs'][key]),'Original152 labels/source order, not metrics or donor300')
    cached=read(checked(parent_inputs['cached_sensor_report']));cached_review=read(checked(parent_inputs['cached_sensor_saved7']));allchecks(cached_review)
    require(cached_review['schema']=='full151105_cached_self15_saved_independent:1' and cached_review['status']=='PASS_ALL152_ALL3ROLES_CACHED15_METRICS_NOT_GEOMETRY'
        and cached_review['engineering_qualified'] is True and same(cached_review['producer_report'],parent_inputs['cached_sensor_report']),'Only qualified cached15 label source')
    manifest=read(checked(parent_inputs['cached_sensor_report']).parent/'output_pins.json')
    require(same(manifest['joined152_cached_sensor_labels.json'],parent_inputs['cached_sensor_labels']) and same(recipe['cached15_label_only_JSON'],parent_inputs['cached_sensor_labels'])
        and same(recipe['source_targets'],parent_inputs['target_arrays']),'Frozen recipe exact label-only JSON/target provenance')
    for path in OWN.values():require(any(same(pin(path),value) for value in p['source_pins'].values()),'Own native-only reader source enrollment')
    source_review=read(checked(p['source_review']));allchecks(source_review)
    require(source_review['schema']=='full151105_native_public_only_training_source_independent:1' and source_review['source_review_qualified'] is True,'Own new trainer source proof')
    return p,{'parent_inputs':parent_inputs,'parent_protocol':parent_protocol,'expected_fit':expected_fit,'parent_eval':ev}

def gradient_readback(g,specs,iteration,torch,helper):
    names=[n for n,_ in specs];active=[n for n in names if n.startswith('wm.')];inactive=[n for n in names if n.startswith('aux_head.')]
    require(type(g['iteration']) is int and g['iteration']==iteration and g['tag']=='PRE_OPTIMIZER_I' and g['weights_after_updates']==iteration-1
        and g['parameter_names']==names and g['max_norm']==1.,'PRE-I raw grads, AFTER-I weight distinction')
    require(set(g['pre_clip'])==set(g['post_clip'])==set(names),'Ordered16 raw gradient maps, no omitted None')
    for name,shape in specs:
        if name in inactive:require(g['pre_clip'][name] is None and g['post_clip'][name] is None,'Unused head gradients must remain None')
        else:
            B._tensor(g['pre_clip'][name],shape,torch.float32,torch,name+' raw pre')
            B._tensor(g['post_clip'][name],shape,torch.float32,torch,name+' raw post')
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g['pre_clip'][name],2) for name in active]),2)
    postnorm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g['post_clip'][name],2) for name in active]),2)
    B._tensor(g['pre_clip_returned_norm_FP32'],(),torch.float32,torch,'saved clip norm')
    B._tensor(g['post_clip_total_norm_FP32'],(),torch.float32,torch,'saved post clip norm')
    require(helper.tensor_equal(norm,g['pre_clip_returned_norm_FP32'],torch) and helper.tensor_equal(postnorm,g['post_clip_total_norm_FP32'],torch),'Only14 present gradients contribute to clip norm')
    coefficient=torch.clamp(1./(norm+1e-6),max=1.)
    require(all(helper.tensor_equal(g['post_clip'][name],g['pre_clip'][name]*coefficient,torch) for name in active),'foreach=False raw clipping with head None skipped')
    return {'active_gradient_tensors':len(active),'inactive_head_None':len(inactive),'pre_clip_norm':float(norm),'post_clip_norm':float(postnorm),'phase':'PRE_OPTIMIZER_I'}

def optimizer_readback(state,values,specs,iteration,torch,helper):
    require(len(state['param_groups'])==1,'One unchanged ordered AdamW group')
    group=state['param_groups'][0];ids=group['params'];active_count=sum(name.startswith('wm.') for name,_ in specs)
    require(len(ids)==len(specs) and all(type(v) is int for v in ids) and len(set(ids))==len(ids) and set(state['state'])==set(ids[:active_count])
        and all(v not in state['state'] for v in ids[active_count:]),'Group16, only active14 states; no fake head step0/state')
    require(group.get('foreach') is False and helper.finite_tree(state,torch),'Finite foreach=False raw Adam')
    # A read-only projection to the existing generic raw14 validator; original
    # serialized group16 and head absence were checked above, never mutated.
    active_view={'state':state['state'],'param_groups':[{**group,'params':ids[:active_count]}]}
    result=helper.optimizer_readback(active_view,iteration,torch,values[:active_count])
    return {**result,'serialized_group_parameters':len(ids),'inactive_head_state_count':0,'read_only_active_view':True}

def loss_snapshot_readback(payload,initial,iteration,torch,helper):
    e=payload['loss_snapshot'];shapes={'native_D52_FP32':(152,52),'effective_s52_FP32':(52,),'point_squared52_FP32':(152,52),'point_per_row_FP32':(152,),
        'self15_logits_FP32':(152,15),'self15_target_FP32':(152,15),'self15_BCE_elements_FP32':(152,15),'self15_sum_per_row_FP32':(152,),'native_pair_per_row_FP32':(152,)}
    require(set(e)==set(shapes)|{'iteration','tag','weights_after_updates','scalars'} and type(e['iteration']) is int and e['iteration']==iteration
        and e['tag']=='PRE_OPTIMIZER_I' and e['weights_after_updates']==iteration-1,'Only PRE-I nine operands, no mode or aux loss')
    for name,shape in shapes.items():B._tensor(e[name],shape,torch.float32,torch,name)
    require(helper.tensor_equal(e['effective_s52_FP32'],initial['physical_context_state']['effective_s52_FP32'],torch)
        and helper.tensor_equal(e['self15_target_FP32'],initial['cached_self15_target_FP32'],torch),'Unchanged effective52 and cached15')
    square=(e['native_D52_FP32']/e['effective_s52_FP32']).square();point=square.mean(-1)
    ell=e['self15_logits_FP32'].double();target=e['self15_target_FP32'].double();reference=torch.clamp(ell,min=0)-ell*target+torch.log1p(torch.exp(-ell.abs()))
    bound=16*torch.finfo(torch.float32).eps*(1+ell.abs()+reference.abs());error=(e['self15_BCE_elements_FP32'].double()-reference).abs()
    require(bool((error<=bound).all()) and bool((e['self15_BCE_elements_FP32']>=0).all()),'Fixed16eps32 F64 analytic BCE envelope; not stride/kernel bit replay')
    selfrow=e['self15_BCE_elements_FP32'].sum(-1);pair=point+torch.tensor(.1,dtype=torch.float32)*selfrow
    for key,value in [('point_squared52_FP32',square),('point_per_row_FP32',point),('self15_sum_per_row_FP32',selfrow),('native_pair_per_row_FP32',pair)]:require(helper.tensor_equal(e[key],value,torch),'Exact saved FP32 row operand '+key)
    scalar=e['scalars'];require(set(scalar)==set(LOSS_NAMES),'Exactly three scalars; no total/public alias')
    expected={'state':point.mean(),'self15':selfrow.mean(),'native_pair':pair.mean()}
    for key,value in expected.items():B._tensor(scalar[key],(),torch.float32,torch,key);require(helper.tensor_equal(scalar[key],value,torch),'Same-row FP32 sum15/nativepair reduction')
    return {'scalars':{k:float(v) for k,v in scalar.items()},'BCE_max_ratio_to_fixed16eps':float((error/bound).max()),'native_Log_or_model_replayed':False}

def validate_payload(original,initial,payload,logs,iteration,protocol_pin,recipe_pin,inputs,sources,parent_inputs,parent_protocol,expected_fit,torch,helper,*,fixture_specs=None):
    specs=SPECS if fixture_specs is None else tuple(fixture_specs);names=[n for n,_ in specs];active=[n for n in names if n.startswith('wm.')];inactive=[n for n in names if n.startswith('aux_head.')]
    require(type(iteration) is int and iteration in (100,1000),'Fixed checkpoint only')
    if fixture_specs is None:require(len(names)==16 and len(active)==14 and len(inactive)==2,'Actual source-native14 plus untouched head2')
    require(initial['schema']==INITIAL_SCHEMA and initial['arm']==ARM and initial['internal_wrapper_arm']==INTERNAL_ARM and 'iteration' not in initial and 'pair_schema' not in initial,'Own single initial, not old pair')
    require(payload['schema']==SCHEMA and payload['manifest_schema']==MANIFEST_SCHEMA and payload['arm']==ARM and payload['internal_wrapper_arm']==INTERNAL_ARM
        and type(payload['iteration']) is int and payload['iteration']==iteration and payload['checkpoint_tag']=='AFTER_OPTIMIZER_I' and 'pair_schema' not in payload,'Own native-only AFTER-I checkpoint')
    for value in (initial,payload):
        require(same(value['protocol'],protocol_pin) and same(value['recipe'],recipe_pin) and map_same(value['input_pins'],inputs) and map_same(value['source_pins'],sources)
            and map_same(value['parent_training_inputs'],parent_inputs) and same(value['parent_training_protocol'],parent_protocol),'Own3/derived16/source/protocol all distinct and bound')
        require(value['ordered_parameter_names']==names and value['active_gradient_parameter_names']==active and value['inactive_gradient_parameter_names']==inactive
            and all(value[k]==expected_fit[k] for k in FIT_KEYS),'Ordered16/active14/headNone2 and four parent fit digests')
    require(original['schema']=='full151105_WM_common_pre_component_calibration:1' and 'WM' in original and 'WM_state' not in original,'Trusted original initial raw namespace/WM key')
    require(set(original['WM'])==set(initial['WM_state']) and all(helper.tensor_equal(v,initial['WM_state'][k],torch) for k,v in original['WM'].items())
        and original['normalizer_host_configuration']==initial['normalizer_host_configuration'] and same(initial['source_parent'],parent_inputs['initial_component']),'This-run raw initial WM/normhost equals actual original ea051 bytes')
    initial_values=B._state_tensors(initial,specs,torch,helper);values=B._state_tensors(payload,specs,torch,helper)
    require(set(initial['training_counts'])==set(COUNTS) and all(type(n) is int and n==0 for n in initial['training_counts'].values()),'Initial before first new update')
    for state in (initial['aux_head_state'],payload['aux_head_state']):require(all(helper.tensor_equal(v,torch.zeros_like(v),torch) for v in state.values()),'Head raw positive zero unchanged')
    require(all(helper.tensor_equal(v,payload['aux_head_state'][k],torch) for k,v in initial['aux_head_state'].items()),'Head not optimized')
    require(initial['zero_head_state_sha256']==payload['zero_head_state_sha256'],'Producer head source-digest unchanged; raw positive-zero independently checked')
    construction=initial['construction_receipt']
    require(all(construction[k] is True for k in ('raw_WM_unchanged','disjoint_model_head_buffer_storage','constructor_Python_TorchCPU_RNG_unchanged','feature_hooks_independent'))
        and set(construction['raw_clone_state_identity'])==set(B.ARMS) and all(v is True for v in construction['raw_clone_state_identity'].values()),'Source-backed two raw copies before hooks, not a single-copy fiction')
    ctx=initial['physical_context_state'];require(set(ctx)==set(CONTEXT_SPECS),'Full frozen physical buffer set')
    for key,(shape,dtype) in CONTEXT_SPECS.items():B._tensor(ctx[key],shape,getattr(torch,dtype),torch,key)
    require(helper.tensor_equal(ctx['effective_s52_FP32'],original['frozen_s52'],torch) and helper.tensor_equal(ctx['physical_xi46_FP32'],original['physical_xi46'],torch),'Original captured s52/xi unchanged, no NumPy sqrt truth')
    norm=helper.norm_state(initial['WM_state']);require(set(payload['WM_state'])==set(initial['WM_state']) and all(helper.tensor_equal(v,payload['WM_state'][key],torch) for key,v in norm.items())
        and initial['normalizer_host_configuration']==payload['normalizer_host_configuration'] and set(payload['physical_context_state'])==set(ctx)
        and all(helper.tensor_equal(v,payload['physical_context_state'][key],torch) for key,v in ctx.items()),'Raw mixed-F32/F64 normalizers/host/context frozen')
    B._tensor(initial['cached_self15_target_FP32'],(152,15),torch.float32,torch,'cached label target')
    require(bool(((initial['cached_self15_target_FP32']==0)|(initial['cached_self15_target_FP32']==1)).all())
        and helper.tensor_digest({'cached_self15_target_FP32':initial['cached_self15_target_FP32']},torch)==initial['fit_self15_label_digest'],'Raw THIS-run cached label bits/digest')
    require(set(payload['counts'])==set(COUNTS) and all(type(n) is int and n==iteration for n in payload['counts'].values())
        and type(payload['native_difference_calls']) is int and payload['native_difference_calls']==iteration,'One actual native FW/hook/head/D/backward/optimizer per update')
    pending=initial['first_training_forward_parity'];passed=payload['first_training_forward_parity']
    require(pending['status']=='PENDING_ITERATION1_NO_EXTRA_FW' and passed['status']=='PASS_ITERATION1_NATIVE58_SELF15_AND_POSITIVE_ZERO14_BITS'
        and passed['training_iteration']==1 and passed['extra_validation_FW']==0 and same(pending['source_baseline'],parent_inputs['initial_parity_baseline'])
        and same(passed['source_baseline'],pending['source_baseline']) and same(pending['source_parity_report'],parent_inputs['initial_parity_report']),'Initial pending vs first actual training FW parity source')
    checks={'new_initial_raw_parent_and_context_head_zero_own3_parent16':True}
    details={'fixture_only':fixture_specs is not None,'actual16_14_contract':fixture_specs is None,'optimizer':optimizer_readback(payload['optimizer_state'],values,specs,iteration,torch,helper),
        'gradients':gradient_readback(payload['gradient_snapshot'],specs,iteration,torch,helper),'loss_snapshot':loss_snapshot_readback(payload,initial,iteration,torch,helper)}
    checks['saved_WM14_gradients_Adam_steps_head2_None_no_state']=True
    rng=helper.raw_rng_comparison(initial['RNG_this_run'],payload['RNG'],torch);require(rng['all_raw_values_equal'],'THIS-run raw RNG, not historical original/pickle equality');details['raw_RNG']=rng
    require(len(logs)==iteration,'All exact log prefix updates')
    for index,row in enumerate(logs,1):
        require(type(row['iteration']) is int and row['iteration']==index and set(row['arms'])=={ARM},'Single external arm all update rows')
        entry=row['arms'][ARM];require(set(entry['counts'])==set(COUNTS) and all(type(v) is int and v==index for v in entry['counts'].values())
            and type(entry['update_seconds']) in (int,float) and math.isfinite(entry['update_seconds']) and entry['update_seconds']>=0,'Full per-update forward/backward/step budget')
        require(set(entry['losses'])==set(LOSS_NAMES) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in entry['losses'].values()),'Only finite captured three native scalars; do not rebuild rowpair from reduced scalars')
    require(logs[-1]['arms'][ARM]['losses']==details['loss_snapshot']['scalars'],'Checkpoint PRE-I row operands/scalars bind final prefix row')
    checks['all_prefix_rows_counts_native_three_losses_raw_RNG']=True
    details.update(initial_WM_sha256=helper.tensor_digest(initial['WM_state'],torch),fit_digests={k:initial[k] for k in FIT_KEYS},
        loss_scope='Native D52/firstFW parity are source captures, no native Log/FW replay. FP32 BCE elements use fixed16eps envelope, exact sums start from saved elements.',
        construction_scope='Two-RAW-copy/unused wrapper/16 requires_grad and no extra FW facts are producer source/inmemory evidence; saved arrays establish initialized equality, raw grads and Adam partition only.')
    return checks,details

def ready_for_run():
    ready=read(PREP/'READY.json');proof=read(checked(ready['cpu_report']));allchecks(proof)
    require(ready['schema']=='full151105_native_public_only_checkpoint_auditor_READY:1' and proof['schema']=='full151105_native_public_only_checkpoint_auditor_contract:1'
        and proof['actual_real_loads_model_FW_AD_NPZ_physics']==0 and set(ready['source_pins'])==set(proof['source_pins'])==set(OWN),'Executed new reader synthetic proof, never old paired qualification')
    for key,path in OWN.items():require(same(pin(path),ready['source_pins'][key]) and same(ready['source_pins'][key],proof['source_pins'][key]),'Current reader/source proof')
    return pin(PREP/'READY.json'),ready['cpu_report']

def declaration():
    return {'schema':'full151105_native_public_only_checkpoint_auditor_declaration:1','protocol_schema':PROTOCOL_SCHEMA,'role':ROLE,'artifact_schema':SCHEMA,'initial_schema':INITIAL_SCHEMA,
        'manifest_schema':MANIFEST_SCHEMA,'arm':ARM,'own_input_aliases':sorted(INPUTS),'actual_loads':3,'WM_gradient_and_Adam_states':14,'head_grad_None_no_Adam':2,'native_FWs_AD_NPZ_physics':0}

def audit(protocol_path,run_dir,iteration,output):
    out=Path(output);out.mkdir(exist_ok=False);loads=[]
    try:
        p,meta=metadata_preflight(protocol_path,run_dir,iteration);ready,proof=ready_for_run();run=Path(run_dir).resolve(strict=True);parent=meta['parent_inputs']
        manifest_pin=pin(run/f'checkpoint_{iteration:04d}.json');manifest=read(checked(manifest_pin))
        require(manifest['schema']==MANIFEST_SCHEMA and manifest['arm']==ARM and type(manifest['iteration']) is int and manifest['iteration']==iteration
            and same(manifest['protocol'],pin(protocol_path)) and same(manifest['recipe'],p['recipe']) and 'checkpoints' not in manifest,'Fixed single checkpoint manifest, no pair or selection')
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import torch,numpy as np,random
        torch.set_num_threads(4);require(not torch.cuda.is_initialized(),'Fresh CPU-only saved reader');helper=B.load_helpers()
        def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_CPU':torch.get_rng_state().clone(),'CUDA_initialized':False}
        before=rng()
        def trusted_load(expected,name=None):
            path=checked(expected)
            if name is not None:require(path.parent.samefile(run) and path.name==name,'Exact own native-only artifact route')
            first=pin(path);loads.append({'checkpoint':first,'returned':False});value=torch.load(path,map_location='cpu',weights_only=False)
            require(same(first,pin(path)),'Checkpoint changed during load');loads[-1]['returned']=True;return value
        original=trusted_load(parent['initial_component']);initial=trusted_load(manifest['initial_state'],'initial_component_state.pt')
        payload=trusted_load(manifest['single_checkpoint'],f'{ARM}_update_{iteration:04d}.pt')
        label_json=read(checked(parent['cached_sensor_labels']));plan=initial['source_identity_fields']
        require(type(label_json) is list and type(plan) is list and len(label_json)==len(plan)==152,'All original old152 cached labels')
        labels=[];middle=(1,2,6,7,11,12,16,17,21,22);distal=(3,8,13,18,23);tips=(4,9,14,19,24)
        for identity,row in zip(plan,label_json):
            require(all(type(row.get(k)) is type(v) and row[k]==v for k,v in identity.items()) and row['source_frame']==200+identity['k']
                and row['label_time']=='lastsolver_preintegration~kdt+.015' and row['state_time']=='postintegration_kdt+.020' and row['no_u150'] is True,'Identity/phase exact cached15 versus state20')
            raw=row['raw_found25'];target=row['contact15'];require(len(raw)==25 and all(type(v) in (bool,int,float) and math.isfinite(v) and v>=0 and int(v)==v for v in raw)
                and len(target)==15 and all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in target),'Raw25/binary15')
            found=[v>0 for v in raw];decoded=[float(found[j]) for j in middle]+[float(found[d] or found[t]) for d,t in zip(distal,tips)]
            require(all(struct.pack('<f',float(a))==struct.pack('<f',b) for a,b in zip(target,decoded)),'Raw25 middle10/distalORtip5 byte mapping');labels.append(target)
        require(helper.tensor_equal(torch.tensor(labels,dtype=torch.float32),initial['cached_self15_target_FP32'],torch),'One label JSON equals raw initial target tensor')
        prefix=hashlib.sha256();prefix_bytes=0;logs=[]
        with (run/'updates.jsonl').open('rb') as f:
            for _ in range(iteration):
                line=f.readline();require(line.endswith(b'\n'),'Complete requested update prefix');prefix.update(line);prefix_bytes+=len(line);logs.append(json.loads(line))
        lp=payload['log_prefix'];require(Path(lp['path']).samefile(run/'updates.jsonl') and lp['records']==iteration and lp['bytes']==prefix_bytes and lp['sha256']==prefix.hexdigest()
            and exact_tree(lp,manifest['log_prefix']),'Immutable I-line prefix while worker may append')
        checks,details=validate_payload(original,initial,payload,logs,iteration,pin(protocol_path),p['recipe'],p['inputs'],p['source_pins'],parent,meta['parent_protocol'],meta['expected_fit'],torch,helper)
        health=read(checked(manifest['producer_health']));allchecks(health)
        require(health['schema']=='full151105_native_public_only_inmemory_health:1' and health['arm']==ARM and health['iteration']==iteration
            and health['engineering_inmemory_pass'] is True and health['serialized_readback_performed'] is False,'Producer inmemory evidence is not this serialized audit')
        final_scope=False
        if iteration==1000:
            terminal=read(run/'result.json');require(terminal['schema']=='full151105_native_public_only_training_completion:1' and terminal['status']=='COMPLETED_FIXED1000_NATIVE_PUBLIC_ONLY_TRAINING'
                and terminal['iteration']==1000 and same(terminal['protocol'],pin(protocol_path)) and all(terminal['training_counts'][k]==1000 for k in COUNTS)
                and terminal['process_counts']['native_difference_calls']==1000 and terminal['auxiliary_losses_computed'] is terminal['auxiliary_head_optimized'] is False,'Actual fixed1000 final budget/marker, not checkpoint alone')
            final_scope=True
        own_rng=helper.raw_rng_comparison(before,rng(),torch);require(own_rng['all_raw_values_equal'] and len(loads)==3 and all(v['returned'] for v in loads)
            and not torch.cuda.is_initialized() and not any(k in sys.modules for k in ('warp','mujoco','dex_supermimic','mjlab')),'Exactly3 CPU reads/no runtime RNG repair or model/native backend')
        checks['trusted_ea051_and_label_JSON_runtime_readback_scope']=True;details.update(runtime_raw_RNG=own_rng,log_prefix_rows=iteration,log_prefix_bytes=prefix_bytes,log_prefix_sha256=prefix.hexdigest(),
            source_archive_scope='Current pinned protocol/reader/source-review identities checked; execution ZIP/OS lifecycle not independently re-audited by this health reader.',
            label_scope='Old152 cached lastsolver~15ms native self15, not POST20 geometry or policy/force accuracy. Old5/combined digest contents and firstFW execution delegated to source/parent proofs.')
        for row in loads:checked(row['checkpoint'])
        checked(manifest_pin)
        report={'schema':'full151105_native_public_only_checkpoint_saved_review:1','status':'PASS_SAVED_NATIVE52_SELF15_ONLY_COMPONENT_HEALTH_NOT_ACCURACY','engineering_qualified':True,
            'arm':ARM,'iteration':iteration,'final_iteration_scope':final_scope,'checks':checks,'passed':len(checks),'total':len(checks),'auditor':pin(__file__),
            'protocol':pin(protocol_path),'recipe':p['recipe'],'run_dir':str(run),'producer_manifest':manifest_pin,'producer_inmemory_health':manifest['producer_health'],
            'original_initial':loads[0]['checkpoint'],'initial_state':loads[1]['checkpoint'],'single_checkpoint':loads[2]['checkpoint'],'own_CPU_checkpoint_loads':3,
            'cached_label_JSON_value_reads':1,'load_trace':loads,'cpu_ready':ready,'cpu_report':proof,'source_pins':{k:pin(v) for k,v in OWN.items()},'details':details,
            'models_FW_AD_NPZ_MJB_Env_physics_GPU':0,'accuracy_physical_gradient_Actor_resume_or_ego_qualified':False}
        write(out/'report.json',report);print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(out/'report.json')}));return 0
    except BaseException as error:
        if not (out/'failure.json').exists():write(out/'failure.json',{'error':repr(error),'auditor':pin(__file__),'actual_load_trace':loads,'actual_saved_review_completed':False,'automatic_retry':False})
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol');ap.add_argument('--run-dir');ap.add_argument('--iteration',type=int,default=100);ap.add_argument('--output');ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if not args.protocol:print(json.dumps(declaration()));return 0
    require(args.run_dir,'Training run directory required')
    if not args.run:metadata_preflight(Path(args.protocol).resolve(strict=True),args.run_dir,args.iteration);print(json.dumps({'status':'PASS_METADATA_ONLY_NO_TORCH_CHECKPOINT_LABEL_VALUES','actual_loads_models':0}));return 0
    require(args.output,'Fresh audit output required');return audit(Path(args.protocol).resolve(strict=True),args.run_dir,args.iteration,args.output)

if __name__=='__main__':raise SystemExit(main())
