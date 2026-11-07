"""Static source receipt only; never import trainer, reader, Torch or payloads."""
import ast,difflib,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'trainer_core_v2':('src/full151_gradient/boundary_impulse_training105_v2.py','2dd2d3c90f5de47736e4ea4ca87b090c3458659f4982c4ecdf102908c216dbb1'),
 'trainer_CLI_v2':('scripts/train_full151_boundary_impulse_components_cpu_v2.py','6d024309839072973506ae34bd364498631b2534542c65325d53687d1bb84862'),
 'trainer_checker_v2':('scripts/check_full151_boundary_impulse_training_cpu_v2.py','bfe9c3728c53eba9c7b52dc8de9d424b9941f00d3528515000c1821a93e24c55'),
 'checkpoint_health_reader':('scripts/audit_boundary_impulse_checkpoint_saved_cpu_v1.py','0ea09088c77644ce5e2bf8b8e0c9ccd621cdad77481b4ea4ee7c8717767f650c'),
 'component_loss_wrapper':('src/full151_gradient/boundary_impulse_component105_v1.py','003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317'),
 'frozen_recipe':('protocols/full151105_boundary_impulse_matched_training_recipe_v1.json','ed5e6cc378f80d94e1771607b8fd4ed1198ce31eec80e17c0465d9933ae4f855'),
 'actual_initial_parity_saved_review':('outputs/full151105_boundary_component_initial_parity_saved_independent_v3/report.json','da71d009aa7cb08e0caccf0d1d4f45337410d303dfba3fcbb1f3bf5aea5202fe'),
 'trainer_READY':('outputs/full151105_boundary_impulse_training_preparation_CPU_v2/READY.json','fb380af386f858999fc6287af8e6157121191153ec176b6d6302af8982829994'),
 'checkpoint_reader_READY':('outputs/full151105_boundary_impulse_checkpoint_auditor_preparation_CPU_v1/READY.json','ff9bc00517c2ab1eca98b3799aa9d927d4234c72a00cd3d91761311bec1feb19'),
 'preserved_trainer_core_v1':('src/full151_gradient/boundary_impulse_training105_v1.py','8393152ab92a58ee78cc3d8b0e7aef71de0e878d0a0a1912009c328c794f6a7c')}

def require(x,s):
    if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def functions(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(p).read_text()).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
def allchecks(p):
    x=read(p);require(x['passed']==x['total']==len(x['checks']) and all(v is True for v in x['checks'].values()),'All true delegated proof');return x

def main():
    out=ROOT/'outputs/full151105_boundary_impulse_training_source_independent_v1';out.mkdir(exist_ok=False)
    pins={}
    for k,(name,h) in FILES.items():
        p=ROOT/name;require(sha(p)==h,'Current exact source/proof '+k);pins[k]=pin(p)
        (out/(k+('.py' if p.suffix=='.py' else '.json'))).write_bytes(p.read_bytes())
    pins['reviewer']=pin(__file__);(out/'executed_reviewer.py').write_bytes(Path(__file__).read_bytes())
    core=(ROOT/FILES['trainer_core_v2'][0]).read_text();old=(ROOT/FILES['preserved_trainer_core_v1'][0]).read_text()
    reader=(ROOT/FILES['checkpoint_health_reader'][0]).read_text();loss=(ROOT/FILES['component_loss_wrapper'][0]).read_text();checker=(ROOT/FILES['trainer_checker_v2'][0]).read_text()
    for filename in ('src/full151_gradient/boundary_impulse_training105_v2.py','scripts/train_full151_boundary_impulse_components_cpu_v2.py','scripts/check_full151_boundary_impulse_training_cpu_v2.py'):
        v2=ROOT/filename;v1=ROOT/filename.replace('_v2.py','_v1.py');(out/(v2.stem+'.diff')).write_text('\n'.join(difflib.unified_diff(v1.read_text().splitlines(),v2.read_text().splitlines(),fromfile=str(v1),tofile=str(v2))),encoding='utf-8')
    tr=read(ROOT/FILES['trainer_READY'][0]);tp=Path(tr['cpu_report']['path']);require(sha(tp)==tr['cpu_report']['sha256'],'Trainer proof pin');proof=allchecks(tp)
    hr=read(ROOT/FILES['checkpoint_reader_READY'][0]);hp=Path(hr['cpu_report']['path']);require(sha(hp)==hr['cpu_report']['sha256'],'Health proof pin');hproof=allchecks(hp)
    for ready,report in ((tr,proof),(hr,hproof)):
        require(ready['source_pins']==report['source_pins'],'Same proof source map')
        for v in ready['source_pins'].values():require(sha(v['path'])==v['sha256'],'Current proof dependency')
    recipe=read(ROOT/FILES['frozen_recipe'][0])['recipe'];pr=allchecks(ROOT/FILES['actual_initial_parity_saved_review'][0])
    checks={
      'final_v2_core_all13_function_AST_equal_v1_only_OWNroutes':functions(ROOT/FILES['trainer_core_v2'][0])==functions(ROOT/FILES['preserved_trainer_core_v1'][0]) and len(functions(ROOT/FILES['trainer_core_v2'][0]))==13,
      'positive_actual_target8_and_initial_parity10_are_front_prerequisites':'c.allchecks(saved)' in core and "saved['function_identity_qualified'] is True" in core and pr['function_identity_qualified'] is True,
      'shared_native_initial16_parameters_F32_inputs_no_labels_in_forward':"prediction,self15,z,mode=module(x,u,dynamics)" in core and "expected_names=[f'wm.net.{j}.{field}'" in core and "+['aux_head.weight','aux_head.bias']" in core and "torch.zeros((152,174),dtype=torch.float32)" in core,
      'matched_public_point_mode_impulse_and_only_consistency0vs1_nonzero_saved_defects':recipe['lambda_state']==recipe['lambda_mode']==recipe['lambda_imp']==1. and list(recipe['lambda_cons_by_arm'].values())==[0.,1.] and 'public=native_point_loss+lambda_mode*L_mode+lambda_imp*L_imp' in loss and '(rv-rv_saved)/sv' in loss and '(rw-rw_saved)/sw' in loss and 'I1=R1@Ib@R1.transpose' in loss,
      'first_training_forward_reuses_saved_initial_parity_no_extra_validation_FW':core.index('if current==1:')<core.index("losses['L_total'].backward()") and "baseline['prediction58_FP32']" in core and "'validation_FW':0" in core,
      'fixed_fullbatch_lockstep_FW_backward_Adam_clip_and_actual_step_counters':recipe['iterations']==1000 and recipe['fullbatch_rows']==152 and recipe['checkpoint_iterations']==[100,1000] and "losses['L_total'].backward();extra[arm]['backward_calls']+=1" in core and 'optimizer_steps_exact(opt,params[arm],current,torch)' in core and 'torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True,foreach=False)' in core,
      'saved_gradient_PRE_I_weights_AFTER_I_raw_THIS_RUN_RNG_frozen_context_norm':"'weights_after_updates':iteration-1" in core and "'checkpoint_tag':'AFTER_OPTIMIZER_I'" in core and 'module.assert_frozen()' in core and "'RNG_this_run':snapshot_state(rng_this_run,torch)" in core and 'raw_rng_comparison(reference_rng' in reader,
      'health_reads_three_owned_checkpoints_actual16_Adam_gradients_log_prefix_and_no_model':"torch.load(path, map_location='cpu', weights_only=False)" in reader and 'len(loads) == 3' in reader and 'ACTUAL_PARAMETER_SPECS' in reader and 'gradient_readback(payload' in reader and "helper.optimizer_readback(state, iteration, torch, values)" in reader and "'native105_policy_or_Actor_qualified': False" in reader,
      'final_focused_proofs_source_bound_no_reexecution_CPU_mask_before_Torch':proof['passed']==proof['total']==3 and hproof['passed']==hproof['total']==6 and checker.index("os.environ['CUDA_VISIBLE_DEVICES']='-1'")<checker.index('import torch') and len(proof['CUDA_state_observations'])==16 and all(v['CUDA_initialized'] is False for v in proof['CUDA_state_observations']),
      'reviewer_source_JSON_AST_only_no_production_imports_or_numeric_payload':not any(k.split('.')[0] in ('torch','numpy','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules)}
    require(all(checks.values()),'Static review predicate failed')
    for v in pins.values():require(sha(v['path'])==v['sha256'],'No reviewed source changed')
    limits=['SOURCE review and delegated synthetic3/6 evidence only; no actual new training run or checkpoint qualification.',
      'Future health reader checks current protocol source pins, but does not independently read training execution_sources.zip/source_inventory.json; Root may perform a separate saved lifecycle/archive audit.',
      'Training scalar terms and normalizer/input/label/caches are source runtime evidence until actual artifacts; checkpoint loss-log arithmetic is not nativeLog or NN replay.',
      'Raw gradient snapshot is PRE optimizer I using weights after I-1; serialized weights and Adam are after I. No independent backward replay.',
      'No native105 resume, Actor, physical force/gradient, hidden-state, accuracy or ego qualification; no self15 supervision.',
      'Old trainer v1 Toy failure and source remain; v2 changes only route and checker pre-import CUDA masking/observations, with original loss recipe retained.']
    report={'schema':'full151105_boundary_impulse_training_source_independent:1','status':'NO_KNOWN_SOURCE_BLOCKER_FOR_FIXED_MATCHED_COMPONENT_TRAINING','source_review_qualified':True,'checks':checks,'passed':len(checks),'total':len(checks),'reviewed_source_pins':pins,'delegated_proofs':{'trainer_READY':pin(ROOT/FILES['trainer_READY'][0]),'trainer_proof':pin(tp),'health_READY':pin(ROOT/FILES['checkpoint_reader_READY'][0]),'health_proof':pin(hp)},'limits':limits,'actual_execution_authorized_by_this_review':False,'actual_training_checkpoint_health_qualified':False,'reviewer_actual_arrays_checkpoints_models_FW_AD_optimizer_MJB_Env_physics_GPU':0,'known_source_blockers':[]}
    write(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'passed':len(checks),'total':len(checks),'source_review_qualified':True},indent=2))
if __name__=='__main__':main()
