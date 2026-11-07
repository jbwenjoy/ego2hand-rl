"""Publish the completed native-only SOURCE review; no training imports."""
import ast,hashlib,json,sys
from pathlib import Path
R=Path('D:/workspace/supermimic-full151-observation-study')
OUT=R/'outputs/full151105_native_public_only_training_source_independent_v1'
def need(x,s):
    if not x:raise ValueError(s)
def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def checked(p):need(pin(p['path'])['sha256']==p['sha256'],'Current pin');return Path(p['path'])
def allchecks(p):return p['passed']==p['total']==len(p['checks']) and bool(p['checks']) and all(v is True for v in p['checks'].values())
def main():
    need(not OUT.exists(),'Fresh SOURCE receipt only')
    ready_path=R/'outputs/full151105_native_public_only_training_preparation_CPU_v1/READY.json'
    health_path=R/'outputs/full151105_native_public_only_checkpoint_auditor_preparation_CPU_v1/READY.json'
    ready,health=read(ready_path),read(health_path)
    proof,hproof=read(checked(ready['cpu_report'])),read(checked(health['cpu_report']))
    need(allchecks(proof) and proof['passed']==3 and allchecks(hproof),'Executed delegated pure proofs')
    sources={}
    for k,v in ready['source_pins'].items():checked(v);sources[k]=v
    for k,v in ready['dependency_pins'].items():checked(v);sources['training_dependency_'+k]=v
    for k,v in health['source_pins'].items():checked(v);sources['health_'+k]=v
    for k,v in health.get('dependency_pins',{}).items():checked(v);sources['health_dependency_'+k]=v
    sources.update(recipe=pin(R/'protocols/full151105_native_public_only_training_recipe_v1.json'),contract=pin(R/'docs/public_native_only_training_contract_v1.md'),health_READY=pin(health_path),source_reviewer=pin(__file__))
    need(sources['recipe']['sha256']=='ffb89e90aaa03099d2cb444d5c660a61fd8fa44b73b9673bd9651f3ce641b4ee' and sources['contract']['sha256']=='1766b3fe88ee74015df2907762dd70bd0cd241aefa94b8765c008701324196e1','Exact frozen recipe/contract')
    text=Path(sources['consumer']['path']).read_text(encoding='utf-8');tree=ast.parse(text)
    fs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)};src=lambda n:ast.get_source_segment(text,fs[n])
    loss,clip,adam,run,meta,validate=map(src,('native_only_losses','clip_native_gradients','optimizer_steps_exact','run_training','metadata_preflight','validate_protocol'))
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='HEALTH_READY_SHA'}
    need(constants.get('HEALTH_READY_SHA')==pin(health_path)['sha256'],'Current reader READY literal inserted')
    checks={
      'final_consumer_CLI_checker_and_recipe_contract_reader_delegated_proofs_current':True,
      'own3_inputs_and_parent16_maps_four_fit_digests_are_distinct':"set(p['inputs'])==INPUTS" in validate and "parent_inputs=family['training']['inputs']" in meta and "'parent_training_inputs':data_inputs" in run and "'input_pins':i" in run,
      'native_FP32_rowmean52_sum15_rowpair_only_nine_operands_three_scalars':"point_row=squared.mean(-1)" in loss and "self_row=bce.sum(-1)" in loss and "native_row=point_row+torch.tensor(r['lambda_self15']" in loss and "'L_native_pair_FP32':native_row.mean()" in loss and 'component_losses' not in loss and 'public_self15_losses' not in loss,
      'one_pair_backward_no_zero_aux_mode_impulse_consistency_loss':"losses['L_native_pair_FP32'].backward()" in run and 'component_losses(' not in run and 'public_self15_losses(' not in run,
      'sixteen_named_group_WM14_finite_grad_head2_None_Adam14_no_head_state':"params[:14]" in clip and "all(p.grad is None for p in params[14:])" in clip and 'len(optimizer.state)==14' in adam and "all(p not in optimizer.state and p.grad is None for p in params[14:])" in adam,
      'factory_two_copies_selected_wrapper_only_unused_zero_calls_and_state':"modules,construction=component.make_matched_components(raw,ctx)" in run and "module=modules[INTERNAL_ARM]" in run and "all(value==0 for value in unused.counts.values())" in run and 'component.state_digest(unused)==unused_before' in run,
      'first_training_FW_parity_no_extra_validation_and_one_native_difference':"if current==1:" in run and 'First training forward saved initial/native58+self15/+0head parity' in run and "delta=difference(prediction,target);process_counts['native_difference_calls']+=1" in run,
      'PRE_I_loss_gradients_vs_AFTER_I_checkpoint_single_manifest_and_prefix':"checkpoint_payload(module,optimizer,counts,gradients,current" in run and "'single_checkpoint':pin(checkpoint)" in run and "'initial_state':pin(out/'initial_component_state.pt')" in run and "'log_prefix':prefix" in run,
      'single_arm_100_ETA_and_required_background_true_pause_scope':"900*mean" in src('estimate_eta_100') and "'remaining_updates':900" in src('estimate_eta_100') and "p.get('background_continue_to_final') is True" in validate,
      'raw_rng_fit_inputs_head_positive_zero_and_context_unchanged_assertions':"parity.raw_rng_equality(rng_this_run,rng,np,torch)" in run and 'fit_digests()==fit' in run and 'Aux head exact positive-zero state unchanged' in run and 'module.assert_frozen();unused.assert_frozen()' in run,
    }
    need(all(checks.values()),'SOURCE check failed '+str(checks));need(not any(k in sys.modules for k in ('numpy','torch','mujoco','warp')),'Stdlib-only source publisher')
    OUT.mkdir();report={'schema':'full151105_native_public_only_training_source_independent:1','status':'PASS_SOURCE_AND_BOUND_SYNTHETIC_PROOFS_NOT_ACTUAL_TRAINING','source_review_qualified':True,'checks':checks,'passed':len(checks),'total':len(checks),'source_pins':sources,'trainer_READY':pin(ready_path),'trainer_cpu_report':ready['cpu_report'],'health_READY':pin(health_path),'health_cpu_report':health['cpu_report'],'actual_pt_NPZ_Torch_models_Toy_FW_AD_training_physics':0,'scope':['SOURCE/AST plus current source/pure-proof metadata binding; delegated proofs not rerun.','Actual parent JSON preflight, original152 data assembly, actual14 gradient/Adam trajectory and saved checkpoint health remain Root-authorized future work.','No native105/Actor/resume, accuracy, physical-gradient, task or independent-trajectory qualification.'],'reviewed_function_lines':{k:{'line':v.lineno,'end':v.end_lineno} for k,v in fs.items()}}
    with (OUT/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'report':pin(OUT/'report.json'),'passed':len(checks),'total':len(checks)}))
if __name__=='__main__':main()
