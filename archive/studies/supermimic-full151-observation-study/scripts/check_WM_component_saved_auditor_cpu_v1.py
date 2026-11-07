"""Five tiny trusted Tensor serialization/loss fixtures; no real checkpoint/data."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'scripts/audit_WM_component_checkpoint_cpu_v1.py'
OUT=ROOT/'outputs/full151105_component_checkpoint_auditor_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);spec=importlib.util.spec_from_file_location('component_reader',SRC);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    pins={'auditor':a.pin(SRC),'checker':a.pin(__file__),'frozen_recipe':a.pin(a.RECIPE_FILE)}
    for k,p in [('auditor',SRC),('checker',Path(__file__))]:ast.parse(p.read_bytes());(OUT/(k+'_source_before_execution.py')).write_bytes(p.read_bytes())
    a.write(OUT/'source_before_synthetic_IO.json',{'source_pins':pins,'real_components_or_fit_inputs_read':False,'only_new_synthetic_Tensor_files':True})
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4)
    state={'net.0.weight':torch.arange(6,dtype=torch.float32).reshape(2,3),'net.0.bias':torch.zeros(2),
       'norm.mean':torch.zeros(298),'norm.var':torch.ones(298),'norm.count':torch.tensor(7.,dtype=torch.float64),
       'delta_norm.mean':torch.zeros(52),'delta_norm.var':torch.ones(52),'delta_norm.count':torch.tensor(7.,dtype=torch.float64),'error_floor':torch.full((52,),.05)}
    rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch_CPU':torch.random.get_rng_state().clone(),'CUDA_initialized':False}
    opt={'param_groups':[{'params':[0,1],'lr':1e-4,'weight_decay':0.,'betas':(.9,.999),'eps':1e-8}],
       'state':{i:{'step':torch.tensor(100.),'exp_avg':torch.zeros_like(v),'exp_avg_sq':torch.ones_like(v)} for i,v in enumerate([state['net.0.weight'],state['net.0.bias']])}}
    initial={'schema':a.SCHEMA,'WM':state,'frozen_s52':torch.full((52,),1.),'physical_xi46':torch.zeros(46),'RNG':rng}
    torch.save(initial,OUT/'initial_component_state.pt');loaded=[a.load_component_cpu(OUT/'initial_component_state.pt',torch)]
    for arm in a.ARMS:
        torch.save({'schema':a.SCHEMA,'WM_state':copy.deepcopy(state),'optimizer_state':copy.deepcopy(opt),'arm':arm,'iteration':100,'RNG':rng},OUT/(arm+'_update_0100.pt'))
        loaded.append(a.load_component_cpu(OUT/(arm+'_update_0100.pt'),torch))
    ad=[a.optimizer_readback(x['optimizer_state'],100,torch,[x['WM_state']['net.0.weight'],x['WM_state']['net.0.bias']]) for x in loaded[1:]]
    checks={'three_own_trusted_CPU_roundtrips_digest_rng_and_all_Adam100':all(a.tensor_digest(x['WM_state'],torch)==a.tensor_digest(loaded[0]['WM'],torch) and a.rng_digest(x['RNG'],torch)==a.rng_digest(loaded[0]['RNG'],torch) and z['actual_step_values']==[100.] for x,z in zip(loaded[1:],ad))}
    bad=copy.deepcopy(opt);bad['state'][1]['step']=torch.tensor(99.)
    try:a.optimizer_readback(bad,100,torch);rejected=False
    except ValueError:rejected=True
    checks['one_serialized_step99_rejects_even_if_counter100']=rejected
    mutated=copy.deepcopy(loaded[1]['WM_state']);mutated['norm.var'][0]=torch.nextafter(mutated['norm.var'][0],torch.tensor(2.))
    torch.save({'schema':104,'WM_state':state},OUT/'not_component.pt')
    try:a.load_component_cpu(OUT/'not_component.pt',torch);oldaccepted=True
    except ValueError:oldaccepted=False
    checks['raw_norm_ulp_change_and_old104_artifact_rejected']=not a.tensor_equal(mutated['norm.var'],loaded[0]['WM']['norm.var'],torch) and not oldaccepted
    # Captured native52 is synthetic here. No nativeLog or WM is called.
    scale=np.arange(2,54,dtype='f4');pc=np.zeros((152,52),dtype='f4');pc[:,0]=4.;mi=np.arange(0,128,2,dtype='i8');pl=mi+1;pc[pl,0]=5.
    pred=np.zeros((152,58),dtype='f4');pred[:,43:52]=np.eye(3,dtype='f4').reshape(9);pred[:,0]=pc[:,0]
    e={'pred152_FP32':pred,'point_delta52_FP32':-pc,'pred_common52_FP32':pc,'target_common52_FP32':np.zeros_like(pc),
       'scale52_FP32':scale,'minus_indices':mi,'plus_indices':pl,'actual_du_rad_F64':np.full(64,.5,dtype='f8')}
    loss=a.postprocess_saved_losses(e,torch);expected_point=(64*2.5**2+88*2.**2)/(152*52)
    checks['native52_only_postprocessing_pointF32_pairF64_closedform']=abs(loss['pointwise_loss']-expected_point)<1e-7 and abs(loss['common_pre_pair_loss']-1/52)<1e-15 and loss['native_rotvec_norm_max']==0.
    e0={**e,'actual_du_rad_F64':e['actual_du_rad_F64'].copy()};e0['actual_du_rad_F64'][40]=0.
    try:a.postprocess_saved_losses(e0,torch);zeroaccepted=True
    except ValueError:zeroaccepted=False
    checks['one_zero_denom_blocks_complete_replay_no_native_or_GPU']=not zeroaccepted and not torch.cuda.is_initialized() and not any(k in sys.modules for k in ('mujoco','warp','dex_supermimic','mjlab'))
    checks={k:bool(v) for k,v in checks.items()};report={'schema':'full151105_component_checkpoint_saved_auditor_CPU_proof:1','source_pins':pins,
       'passed':sum(checks.values()),'total':len(checks),'checks':checks,'synthetic_only':True,'synthetic_trusted_Tensor_files_saved':4,'synthetic_Torch_CPU_loads':4,
       'actual_existing_component_or_model_loads':0,'actual_fit_or_test_rows_read':0,'models_WM_Actor_Env_nativeLog_FW_AD_physics_GPU':0,
       'loss_fixture_tolerance_scope':'Closed-form F32 mean check1e-7 and F64 scalar1e-15 only; no actual loss audit tolerance is set or changed.'}
    a.write(OUT/'report.json',report);ready={'source_pins':pins,'cpu_report':a.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},
       'api':{'entry':str(SRC),'CLI':'--protocol AUTHORIZED_JSON --run-dir OWN_COMPONENT_RUN --iteration 100 --output FRESH --run'},'actual_saved_audit_authorized':False}
    a.write(OUT/'READY.json',ready);print(json.dumps({'READY':a.pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
