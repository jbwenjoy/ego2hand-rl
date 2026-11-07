"""Final public15 trainer/reader SOURCE receipt; no production imports."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'trainer':('src/full151_gradient/boundary_public_self15_training105_v1.py','796c360539961143fe1f588fd18eb08e4b168d141b34daee23da802a72194e7e'),
 'CLI':('scripts/train_full151_boundary_public_self15_components_cpu_v1.py','be7447a2ab13f7dde5b061455b60ed3cf3d75d5edaecfe72ba817a7ac526ec36'),
 'checker':('scripts/check_full151_boundary_public_self15_training_cpu_v1.py','96cda438a19fb6fd454b8dda43c1ea897f292b718798bd58bc067d30bc937ec7'),
 'reader':('scripts/audit_boundary_public_self15_checkpoint_saved_cpu_v1.py','11a38a6ededd06e8734dd98b61bf8e3162d339e6a1ca369485c8d8d90163303b'),
 'reader_checker':('scripts/check_boundary_public_self15_checkpoint_auditor_cpu_v1.py','093cf7ec263d9ca973a3aa2a6ebf9f9b7f488678972b6977744314f6ad4b309f'),
 'trainer_READY':('outputs/full151105_boundary_public_self15_training_preparation_CPU_v1/READY.json','c2f852a7530e93db0a312e6fbd0cb241d21c4da94fbfbf3f4bf89cdd9e434e05'),
 'reader_READY':('outputs/full151105_boundary_public_self15_checkpoint_auditor_preparation_CPU_v1/READY.json','b0709fe11e39aca835abf6cd26ceaf59b62669ebfe70b77924c99143f5040fe0'),
 'recipe':('protocols/full151105_boundary_public_self15_matched_training_recipe_v1.json','9e0765fd562c3e24bdb09d52f78219204af4228bb6e8b1d8ffd67104ce094bce'),
 'source_contract':('docs/full151_boundary_public_self15_matched_training_contract_v1.md','5a12bb913b21a45f6cfc3bd6df39a8bbe4074e0588ac47f1791f9561ec3f4ed7'),
 'frozen_003':('src/full151_gradient/boundary_impulse_component105_v1.py','003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317'),
 'cached_label_source':('src/full151_gradient/boundary_self15_cached_saved105_v1.py','97b74e0173bb6d95b55c59d924db7c5502c83cf610907c6c3aa357d54808da5e')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(x,s):
    if not x:raise ValueError(s)
def dump(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def assigned_literal(tree,name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
def proof(ready):
    r=read(ready);q=r['cpu_report'];require(sha(q['path'])==q['sha256'],'Delegated proof pin');p=read(q['path']);require(p['passed']==p['total']==len(p['checks']) and all(v is True for v in p['checks'].values()),'Delegated all-true proof')
    require(r['source_pins']==p['source_pins'],'Same proof source maps')
    for x in r['source_pins'].values():require(sha(x['path'])==x['sha256'],'Current delegated source')
    return r,p

def main():
    out=ROOT/'outputs/full151105_boundary_public_self15_training_source_independent_v1';out.mkdir(exist_ok=False);pins={}
    for k,(name,h) in FILES.items():
        p=ROOT/name;require(sha(p)==h,'Final frozen source '+k);pins[k]=pin(p);(out/(k+p.suffix)).write_bytes(p.read_bytes())
    pins['reviewer']=pin(__file__);(out/'executed_reviewer.py').write_bytes(Path(__file__).read_bytes())
    core=(ROOT/FILES['trainer'][0]).read_text();reader=(ROOT/FILES['reader'][0]).read_text();checker=(ROOT/FILES['checker'][0]).read_text();tree=ast.parse(core)
    functions={f.name:f for f in tree.body if isinstance(f,ast.FunctionDef)};r=read(ROOT/FILES['recipe'][0])['recipe'];tr,tp=proof(ROOT/FILES['trainer_READY'][0]);hr,hp=proof(ROOT/FILES['reader_READY'][0])
    actual16={'target_report','target_arrays','target_saved8','target_source_inventory','initial_component','initial_validation','initial_saved_health29','initial_parity_report','initial_parity_saved_review','initial_parity_baseline','component_ready','component_source_review','cached_sensor_labels','cached_sensor_report','cached_sensor_saved7','target_identities'}
    namespaces=assigned_literal(tree,'SCHEMA')=='full151105_boundary_public_self15_matched_training_CPU_protocol:1' and assigned_literal(tree,'ARTIFACT_SCHEMA')=='full151105_boundary_public_self15_WM_component:1' and assigned_literal(tree,'INITIAL_SCHEMA')=='full151105_boundary_public_self15_matched_initial:1' and assigned_literal(tree,'PAIR_SCHEMA')=='full151105_boundary_public_self15_matched_pair:1'
    input_args=[a.arg for a in functions['public_self15_losses'].args.args]
    checks={
      'new_namespaces_exact16_protocol_inputs_payload_input_pins_source_review_interface':namespaces and assigned_literal(tree,'INPUTS')==actual16 and "source_review['source_pins']" in core and "return protocol['inputs'], protocol['source_pins']" in reader and "_map_same(payload['input_pins'], input_pins)" in reader,
      'label_only_original152_plan_rawfound15_phase_and_target_NPZ_row_join':all(k in core for k in ('cached_label_rows(plan,joined)','Original middle10+distalORtip5','lastsolver_preintegration~kdt+.015','validate_target_identity_arrays(plan,raw_data[','source_group_id152','source_substep_rows152x4')) and "joined152_cached_sensor_labels.json" in core and "native15_label_only_JSON" in core,
      'native_defaults_recipe_point_mean52_self_sum15_FP32_rowpair_before_mean':r['lambda_state']==r['lambda_mode']==r['lambda_imp']==1 and r['lambda_self15']==.1 and r['window_length']==1 and r['all_rows_valid'] is True and 'point_row=squared.mean(-1);point=point_row.mean()' in core and "reduction='none');self_row=bce.sum(-1)" in core and "native_row=point_row+torch.tensor(r['lambda_self15'],dtype=torch.float32" in core and 'native_pair=native_row.mean()' in core,
      'same_public_supervision_cons0vs1_old003_five_labels_no_target_forward_input':r['lambda_cons_by_arm']=={'shared_public_losses':0.,'shared_public_plus_cross_target_consistency':1.} and 'prediction,self15,z,mode=module(x,u,dynamics)' in core and "loss=component.component_losses(module,x,pred,z,mode,point,boundary_labels" in core and input_args==['module','x','pred','self15_logits','z','mode','delta52','scale52','target15','boundary_labels','r','torch'],
      'eight_named_loss_state_not_pair_new_mode_saved_element_mean_and_F32_then_F64':len(assigned_literal(tree,'LOSS_FIELDS'))==8 and "'state':'L_state_FP32'" in core and "loss['L_mode_FP32']=mode_elements.mean()" in core and "public=(native_pair+r['lambda_mode']*loss['L_mode_FP32'])+r['lambda_imp']*loss['L_imp_F64']" in core and 'loss.update(L_self15_sum_FP32=self_scalar,L_native_pair_FP32=native_pair' in core,
      'shared_raw_initial16_frozen_norm_context_xi_first_training_FW_no_extra_validation':"expected_names=[f'wm.net.{j}.{field}'" in core and "+['aux_head.weight','aux_head.bias']" in core and "'validation_FW':0" in core and "First TRAINING forward actual-saved initial parity" in core and 'module.assert_frozen()' in core,
      'checkpoint_PRE_I_full_loss_operands_gradients_AFTER_I_Adam_log_prefix':core.index('loss_snapshots[arm]=loss_snapshot_before_update')<core.index("losses['L_total'].backward()")<core.index('opt.step()') and "'weights_after_updates':iteration-1" in core and "'checkpoint_tag':'AFTER_OPTIMIZER_I'" in core and "'native_pair_per_row_FP32':native_row" in core and 'loss_snapshot_readback(payload' in reader and "'PRE_OPTIMIZER_I'" in reader,
      'reader_fixed_BCE_eps_envelope_no_cross_stride_bit_claim_saved_FP32_reductions':"BCE_FP32_ROUNDOFF_FACTOR = 16.0" in reader and "torch.finfo(torch.float32).eps*(1+ell.abs()+bce_reference.abs())" in reader and "self_row = e['self15_BCE_elements_FP32'].sum(-1);pair_row = point_row+FIXED_RECIPE['lambda_self15']*self_row" in reader and "'mode': e['mode_BCE_elements_FP32'].mean()" in reader and "Never reconstruct it from the two" in reader,
      'raw_RNG_THIS_RUN_16_Adam_steps_owned_three_readback_no_native_health_upgrade':"helper.raw_rng_comparison(reference_rng, payload['RNG']" in reader and 'helper.optimizer_readback(state, iteration, torch, values)' in reader and "len(loads) == 3" in reader and "'final_iteration_scope': iteration == 1000" in reader and "'native105_policy_or_Actor_qualified': False" in reader,
      'matched1000_fullbatch_budget_background_explicit_and_paired100_ETA900':r['iterations']==1000 and r['checkpoint_iterations']==[100,1000] and r['fullbatch_rows']==152 and "type(p.get('background_continue_to_final')) is bool" in core and "if current==100 and not p['background_continue_to_final']:break" in core and 'remaining_pairs=900' in core,
      'frozen_Toy3_and_Tiny8_source_bound_not_reexecuted_masked_CUDA_actual0':tp['passed']==3 and hp['passed']==8 and all(v['CUDA_initialized'] is False for v in tp['CUDA_state_observations']) and tp['actual_dataset_labels_weights_reads_native_WM_constructor_FW_AD_optimizer_MJB_Env_physics_GPU']==0 and all(v==0 for v in hp['actual_counts'].values()) and checker.index("os.environ['CUDA_VISIBLE_DEVICES']='-1'")<checker.index('import torch'),
      'reviewer_source_metadata_only_no_real_labels_arrays_or_neural_imports':not any(k.split('.')[0] in ('numpy','torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules)}
    require(all(checks.values()),'Source review predicate failed')
    for v in pins.values():require(sha(v['path'])==v['sha256'],'Reviewed bytes unchanged')
    report={'schema':'full151105_boundary_public_self15_training_source_independent:1','status':'NO_KNOWN_SOURCE_BLOCKER_FOR_MATCHED_PUBLIC_NATIVE15_TRAINING','source_review_qualified':True,'checks':checks,'passed':len(checks),'total':len(checks),'source_pins':pins,'delegated_proofs':{'trainer_READY':pins['trainer_READY'],'trainer_report':tr['cpu_report'],'reader_READY':pins['reader_READY'],'reader_report':hr['cpu_report']},'known_source_blockers':[],
      'limits':['SOURCE review only. Toy3 and Tiny8 are independently hash-bound delegated proof, not rerun; actual new checkpoints/training/labels were not read here.',
       'Native D52/Log and original impulse/consistency tensors remain captured/source evidence; reader does not reload models, replay native Log or physical solve.',
       'BCE bound16*machine-eps32 is a predeclared arithmetic diagnostic accounting for saved clone stride changes, not a universal kernel/libm or scientific accuracy certificate. Exact sums/row means start from saved FP32 BCE elements.',
       'Other prefix rows use captured native_pair scalar. Only checkpoint PRE-I has full per-row operands; scalar state+.1*selfscalar is never imposed as bit truth.',
       '003 mode BCE is still computed once and discarded before new captured-mode element reduction. Same NN/backward/optimizer budget does not imply equal kernel operations, walltime or speedup.',
       'Reader validates current protocol pins; a separate lifecycle/archive review is required to claim training ZIP bytes independently checked. Saved digests do not independently prove live process storage identity.',
       'Cached native15 labels are lastsolver~15ms while state target is POST20. This is known old152 W1 exploration, not geometry/force/human/Actor/ego/hidden-state/generalization or accuracy qualification.'],
      'actual_training_checkpoint_health_or_execution_authorized':False,'reviewer_actual_new_labels_arrays_checkpoints_models_FW_AD_optimizer_MJB_Env_physics_GPU':0}
    dump(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'passed':len(checks),'total':len(checks),'source_review_qualified':True},indent=2))
if __name__=='__main__':main()
