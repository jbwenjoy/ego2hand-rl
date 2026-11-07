"""CPU unit and saved-identity proof for the frozen primary pair checker."""
from pathlib import Path
import copy,importlib.util,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'scripts/compare_primary_initial_cpu.py';FROZEN='f5004d41c1a2606ac915143f355ff8ae7ae91843ca6ff7546fde4893b7fd764b'
spec=importlib.util.spec_from_file_location('primary_pair',SOURCE);pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)
def rejected(fn):
    try:fn()
    except (ValueError,KeyError):return True
    return False
def modules():
    return {name:{'parameter_names':['net.0.weight','net.0.bias'],'parameters':[torch.tensor([[.5]],dtype=torch.float32),torch.tensor([.1],dtype=torch.float32)],'shapes':[[1,1],[1]],'dtypes':['torch.float32','torch.float32']} for name in ('actor','wm','fail_head','critic')}
def main():
    out=ROOT/'outputs/primary_initial_pair_cpu_proof_v1';out.mkdir(parents=True,exist_ok=False);checks={};cfgpath=ROOT/'configs/a1l-primary_matched-authorized_v1.json';cfg=pair.read(cfgpath);hpath=ROOT/'outputs/health_primary_a1l_final_v1/report.json';health=pair.read(hpath);run=Path(cfg['output']);controller=pair.read(run/'controller_descriptor.json')
    checks['source_f500_preserved_before_proof']=pair.pin(SOURCE)['sha256']==FROZEN
    x=modules();checks['four_small_CPU_modules_positive']=all(pair.module_checks(x,copy.deepcopy(x))[0].values())
    y=copy.deepcopy(x);y['actor']['parameters'][0][0,0]+=.001;checks['mutated_actor_tensor_rejected']=not all(pair.module_checks(x,y)[0].values())
    y=copy.deepcopy(x);y['wm']['parameter_names'][0]='other.weight';checks['named_parameter_mapping_mutation_rejected']=not all(pair.module_checks(x,y)[0].values())
    y=copy.deepcopy(x);y['critic']['parameters'][0]=y['critic']['parameters'][0].double();checks['tensor_dtype_mutation_rejected']=not all(pair.module_checks(x,y)[0].values())
    y=copy.deepcopy(x);y['fail_head']['parameters'][0][0,0]=float('nan');checks['nonfinite_raw_parameter_rejected']=not all(pair.module_checks(x,y)[0].values())
    y=copy.deepcopy(x);y.pop('critic');checks['missing_module_rejected']=not all(pair.module_checks(x,y)[0].values())
    checks['negative_zero_is_not_bitexact_positive_zero']=not pair.same_tensors(torch.tensor([-0.],dtype=torch.float32),torch.tensor([0.],dtype=torch.float32))
    checks['common_primary_config_positive']=pair.common_config(cfg,copy.deepcopy(cfg))
    changed=copy.deepcopy(cfg);changed['training']['wm_only_iters']+=1;checks['budget_mutation_rejected']=not pair.common_config(cfg,changed)
    changed=copy.deepcopy(cfg);changed['input_bundle']['sha256']='0'*64;checks['reference_input_pin_mutation_rejected']=not pair.common_config(cfg,changed)
    changed=copy.deepcopy(cfg);changed['GPU_run_authorized']=not changed['GPU_run_authorized'];changed['output']='separate-authorized-output';checks['authorization_and_output_are_not_common_budget_fields']=pair.common_config(cfg,changed)
    checks['controller_contract_positive']=pair.same_controls(controller,copy.deepcopy(controller))
    changed=copy.deepcopy(controller);changed['margin_rad']=.021;checks['C2_control_mutation_rejected']=not pair.same_controls(controller,changed)
    changed=copy.deepcopy(controller);changed['native_reference_signature']='0'*64;checks['native_reference_signature_mutation_rejected']=not pair.same_controls(controller,changed)
    baseline=np.zeros((15,20),np.float32);changed=baseline.copy();changed[1,0]=.001;checks['baseline_control_bytes_mutation_rejected']=not pair.same_array(baseline,changed)
    actual=np.zeros((1,3,3),np.float32);changed=actual.copy();changed[0,0,0]=1e-9;checks['actual8_no_tolerance_can_hide_mutation']=not pair.same_array(actual,changed)
    checks['actual_final_health2000_identity_accepted']=pair.health_identity(cfgpath,cfg,run,health)
    pcfgpath=ROOT/'configs/p_only-primary_matched-authorized_v1.json';pcfg=pair.read(pcfgpath);p100=pair.read(ROOT/'outputs/health_primary_p_only_100_v1/report.json');checks['actual_prefix100_health_identity_accepted_without_terminal_requirement']=p100['iteration']==100 and p100['final_iteration_scope'] is False and pair.health_identity(pcfgpath,pcfg,Path(pcfg['output']),p100)
    bad=copy.deepcopy(health);bad['method']='P_ONLY';checks['wrong_arm_health_rejected']=not pair.health_identity(cfgpath,cfg,run,bad)
    bad=copy.deepcopy(health);bad['config']['sha256']='0'*64;checks['changed_config_hash_health_rejected']=not pair.health_identity(cfgpath,cfg,run,bad)
    runtime=pair.read(run/'runtime.json');core,coverage=pair.core_sources(cfg,runtime);checks['actual_core_named_source_closure_readable']=coverage['Dex_declared_pins']>0 and coverage['mjlab_declared_pins']>0 and len(core)>=len(pair.CORE_REL)+2
    bad=copy.deepcopy(cfg);first=next(k for k,v in bad['source_pins'].items() if Path(v['path']).resolve()==(ROOT/'src/rough_prefix/native_v2.py').resolve());bad['source_pins'][first]['sha256']='0'*64;checks['actual_core_source_SHA_mutation_rejected']=rejected(lambda:pair.core_sources(bad,runtime))
    bad_runtime=copy.deepcopy(runtime);bad_runtime['source_sha256']='0'*64;checks['unbound_actual_entry_source_rejected']=rejected(lambda:pair.core_sources(cfg,bad_runtime))
    # Single-participant IO validation is not the actual two-run comparison.
    participant=pair.participant(cfgpath,run,hpath);checks['actual_A1L_final_single_participant_IO_readable']=participant['health']['iteration']==2000 and participant['health']['final_iteration_scope'] is True
    checks['CPU_no_GPU_no_Env']=not torch.cuda.is_initialized() and not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
    checks['source_unchanged_after_proof']=pair.pin(SOURCE)['sha256']==FROZEN
    report={'schema':'primary_initial_pair_CPU_proof:1','status':'PASS_CPU_UNIT_AND_SAVED_IDENTITY_ONLY' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source':pair.pin(SOURCE),'checker':pair.pin(__file__),'actual_A1L_final_health':pair.pin(hpath),'actual_P100_identity_only':pair.pin(ROOT/'outputs/health_primary_p_only_100_v1/report.json'),'core_coverage':coverage,'actual_pair_executed':False,'GPU_used':False,'physics_steps':0,'scope':'Small mutated tensor/config/control/source tests plus saved identity and single-participant IO. No two-run rawinitial comparison, new training or physics is represented by this proof.'}
    path=out/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pair.pin(path)},indent=2));return int(not all(checks.values()))
if __name__=='__main__':raise SystemExit(main())
