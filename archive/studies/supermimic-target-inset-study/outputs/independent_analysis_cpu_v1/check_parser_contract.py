"""Synthetic parser-contract tests only; never a C2 physical result."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'scripts/summarize_target_inset.py'
spec=importlib.util.spec_from_file_location('target_report',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
results={}

def fixture(margin_mrad):
    lo=np.full(20,-1,dtype=np.float32);hi=-lo;p=np.tile(np.linspace(-1,1,20,dtype=np.float32),(150,1));margin=np.float32(margin_mrad/1000)
    il,iu=lo+margin,hi-margin;e=np.clip(p,il,iu)
    prev=np.arange(20,dtype=np.float32)*.01
    a={'time_start_s':np.arange(150)*.02,'time_end_s':np.arange(1,151)*.02,'u_proposed_original_bounded_rad':p,'u_executed_filtered_rad':e,'u_cached_after_step_rad':e.copy(),'u_actual_ctrl_readback_rad':e.copy(),'ema_residual_before_filter_rad':np.zeros((150,20),np.float32),'ema_residual_after_filter_rad':np.zeros((150,20),np.float32),'lower_bound_rad':lo,'upper_bound_rad':hi,'inset_lower_rad':il,'inset_upper_rad':iu,'u_prev_rad':prev,'initial_ctrl':prev.copy(),'initial_qpos':np.zeros(27),'initial_qvel':np.zeros(26)}
    a['position_ctrl_ids']=np.arange(19,-1,-1);a['initial_ctrl']=prev[::-1].copy();a['initial_cached_target_rad']=prev.copy()
    run={'arrays':{'joint_lower_rad':lo,'joint_upper_rad':hi,'joint_target_rad':np.vstack([prev,e])},'metadata':{'joint_names':[f'joint{i}' for i in range(20)]}}
    job={'margin_rad':margin_mrad/1000,'margin_mrad':margin_mrad}
    return a,run,job

with tempfile.TemporaryDirectory(prefix='C2_parser_') as temporary:
    path=Path(temporary)
    def check(a,run,job):
        np.savez(path/'control_targets_50hz.npz',**a)
        return m.target_diagnostic(path,run,job,None)
    for margin in (0,5,20):
        a,r,j=fixture(margin);_,checks,metrics,_=check(a,r,j)
        results[f'valid_native_FP32_margin_{margin}_mrad']=m.all_checks(checks)
        if margin==0:results['zero_no_numeric_clamp_trigger']=metrics['clamp_component_count']==metrics['clamp_tick_count']==0
        else:results[f'clamp_denominator_{margin}_mrad']=metrics['clamp_component_count']==300 and metrics['clamp_tick_count']==150 and metrics['clamp_component_fraction']==.1 and metrics['clamp_tick_fraction']==1
    a,r,j=fixture(5);a['u_cached_after_step_rad'][5,3]+=.001;results['wrong_cache_rejected']=check(a,r,j)[1]['cached_actual_equals_executed'] is False
    a,r,j=fixture(20);a['u_actual_ctrl_readback_rad'][8,4]+=.001;results['wrong_actual_ctrl_rejected']=check(a,r,j)[1]['cached_actual_equals_executed'] is False
    a,r,j=fixture(5);a['u_executed_filtered_rad'][2,1]+=.01;results['wrong_filter_formula_rejected']=check(a,r,j)[1]['executed_native_FP32_clip_formula'] is False
    a,r,j=fixture(0);a['ema_residual_after_filter_rad'][3,1]=1e-4;results['zero_EMA_mutation_rejected']=check(a,r,j)[1]['zero_margin_EMA_identity_bitwise'] is False
    a,r,j=fixture(20);a['ema_residual_after_filter_rad'][3,1]=1e-4;results['nonzero_EMA_mutation_rejected']=check(a,r,j)[1]['EMA_residual_unchanged_bitwise'] is False
    a,r,j=fixture(0);a['time_start_s'][0]=.02;results['off_by_one_interval_rejected']=check(a,r,j)[1]['canonical_interval_starts'] is False
    a,r,j=fixture(5);a['u_prev_rad'][0]=1e-3;results['wrong_previous_target_rejected']=check(a,r,j)[1]['u_prev_matches_initial_ctrl'] is False
results['no_torch_import']='torch' not in sys.modules
results['no_physics_library_import']=not any(k.split('.')[0] in ('mujoco','mujoco_warp','warp') for k in sys.modules)
record={'scope':'Synthetic target-log mathematical/parser-contract checks only; no measured C2 physical trajectory or performance claim. CPU NumPy only.','status':'PASS' if all(results.values()) else 'FAIL','checks':results,'check_count':len(results),'analysis_source_sha256':m.sha(SOURCE),'analysis_protocol_sha256':m.sha(ROOT/'analysis_protocol_v1.json'),'check_source_sha256':m.sha(Path(__file__))}
m.write(Path(__file__).with_name('checks.json'),record)
print(json.dumps(record,indent=2))
raise SystemExit(0 if all(results.values()) else 1)
