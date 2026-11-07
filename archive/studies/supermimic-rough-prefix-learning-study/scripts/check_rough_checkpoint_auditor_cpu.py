"""Small CPU auditor fixtures; never substitute these for real smoke audits."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import copy,hashlib,importlib.util,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('rough_auditor',ROOT/'scripts/audit_rough_checkpoint_cpu.py')
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)

def cap_record(seq,event):
    return {'sequence':seq,'event':event,'stage':'synthetic_CPU_fixture','backend_field':'env.sim.wp_data.overflow','read_attempts_for_record':1,'raw_flags_dtype':'int32','raw_flags_shape':[1],'raw_flags_values':[0],'raw_bitmask':0,'read_status':'OK','flag_shape_is_world_count':False,'flags_cleared_by_observer':False,'cumulative_OR_bitmask':0,'cumulative_unknown_reads':0,'overall_status':'OK'}
def capacity(kind):
    records=[cap_record(1,'before_reset'),cap_record(2,'checkpoint')]
    final=kind=='finalrun'
    if final:records.append(cap_record(3,'final'))
    return {'schema':'rough_prefix_reset_aware_capacity_observation:1','backend_field':'env.sim.wp_data.overflow','records':records,'record_count':len(records),'before_reset_read_count':1,'final_read_count':int(final),'configured_fixed_capacities':{'nconmax':192,'njmax':4096},'flag_shape_is_world_count':False,'cumulative_OR_bitmask':0,'unknown_read_count':0,'status':'OK','finalized':final,'engineering_capacity_qualified':final,'evidence_kind':'entire_run_final' if final else 'checkpoint_prefix','prefix_at_checkpoint_capacity_qualified':not final,'entire_run_finalized':final}

def actual_fixture(folder):
    folder.mkdir();cfg={'warmup_steps':16,'wm_window':8,'utd':.4,'num_envs':128};steps=A.expected_collect_steps(cfg,10);baseline=np.zeros((15,20),np.float32)+np.arange(15,dtype=np.float32)[:,None]*np.float32(.01);rows=[];previous=0
    for call,n in enumerate(steps,1):
        phase=np.broadcast_to(((np.arange(n)+previous)%15)[:,None],(n,128)).copy();previous+=n;nominal=baseline[phase];residual=np.full_like(nominal,.005);proposed=nominal+residual
        row={'collect_call':call,'iteration':max(0,call-2),'steps':n,'worlds':128,'real_transitions':n*128,'all_batch_exact':True,'all_finite':True,'max_abs_errors':dict.fromkeys(A.ERROR_KEYS,0.),'phase_min':int(phase.min()),'phase_max':int(phase.max()),'phase_histogram':np.bincount(phase.flatten(),minlength=15).tolist(),'clamped_components':0,'inactive_real_actions':0};rows.append(row)
        if call in (1,2,3):
            crop=lambda x:x[-min(4,n):,:4]
            np.savez_compressed(folder/f'rough_actualu_trace_{call:05d}.npz',phase=crop(phase),baseline=crop(nominal),proposed=crop(proposed),residual_before_filter=crop(residual),cached=crop(proposed),actual=crop(proposed),residual=crop(residual),replay_u=crop(proposed),replay_c=crop(residual))
    path=folder/'rough_actualu_evidence.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    saved={'iteration':10,'config':cfg,'sim_steps':5248,'rough_controller':{'latest_actualu_evidence':rows[-1],'target_lower_rad':[-1.]*20,'target_upper_rad':[1.]*20}}
    return saved,baseline,rows

def initial_fixture(folder):
    folder.mkdir();sources={'training_adapter':A.pin(__file__),'capacity_observer':A.pin(__file__),'native_v2':A.pin(__file__)};raw={'schema':'rough_prefix_initial_parameters:1','capture_point':'after_native_DexRunner_init_before_first_collect_or_optimizer','source_pins':sources,'modules':{}};saved={'iteration':10,'config':{'wm_only_iters':2},'modules':{},'rough_capacity_evidence':{'source_pins':sources}};fp={};result={'parameter_delta_l2':{}}
    for name in ('actor','wm','fail_head','critic'):
        values=[torch.tensor([[.5]],dtype=torch.float32),torch.tensor([.25],dtype=torch.float32)];names=['net.0.weight','net.0.bias'];h=hashlib.sha256()
        for v in values:h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
        fp[name]={'sha256':h.hexdigest(),'tensors':2,'parameters':2};raw['modules'][name]={'parameter_names':names,'parameters':values,'shapes':[[1,1],[1]],'dtypes':['torch.float32','torch.float32']};saved['modules'][name]={k:v+.125 for k,v in zip(names,values)};result['parameter_delta_l2'][name]=(.125**2*2)**.5
    saved['rough_initial_parameter_fingerprints']=fp;torch.save(raw,folder/'initial_parameter_tensors.pt');A.json_write(folder/'initial_parameters.json',fp)
    return saved,result

def main():
    out=ROOT/'outputs/rough_checkpoint_auditor_cpu_v1';out.mkdir(parents=True,exist_ok=False);checks={}
    cfg={'warmup_steps':16,'wm_window':8,'utd':.4,'num_envs':128}
    checks['10iter_11collect_41steps_5248transitions']=A.expected_collect_steps(cfg,10)==[16,2,3,2,3,2,3,2,3,2,3] and sum(A.expected_collect_steps(cfg,10))*128==5248
    checks['prefix2_not_full10_budget']=sum(A.expected_collect_steps(cfg,2))*128==2688
    checks['small_CPU_tensors_finite']=A.finite({'x':[torch.zeros(1),np.ones(2)]})
    checks['nonfinite_tensor_rejected']=not A.finite({'x':torch.tensor([float('nan')])})
    checks['Inf_optimizer_tensor_rejected']=not A.finite({'optimizer':{'exp_avg':torch.tensor([float('inf')])}})
    checks['nonfinite_diagnostic_report_serializable']=A.pure({'failed_delta':float('nan')})=={'failed_delta':'nan'}
    shapes=A.expected_shapes();checks['actual_native_feature_architecture_contract']=shapes['wm']['net.0.weight']==[512,298] and shapes['wm']['net.9.weight']==[67,512] and shapes['actor']['net.0.weight']==[512,155] and shapes['critic']['net.0.weight']==[512,301] and shapes['fail_head']['net.0.weight']==[256,426]
    for kind in ('checkpoint_prefix','finalrun'):
        c,d=A.capacity_checks(capacity(kind),kind);checks[kind+'_raw_capacity_positive']=all(c.values())
    altered=capacity('checkpoint_prefix');altered['records'][-1]['raw_flags_values']=[1]
    checks['capacity_raw_overflow_cannot_be_hidden_by_OK']=not all(A.capacity_checks(altered,'checkpoint_prefix')[0].values())
    altered=capacity('checkpoint_prefix');altered['records'][-1]['read_status']='UNKNOWN';altered['unknown_read_count']=1
    checks['capacity_UNKNOWN_cannot_pass']=not all(A.capacity_checks(altered,'checkpoint_prefix')[0].values())
    altered=capacity('checkpoint_prefix');altered['finalized']=True
    checks['checkpoint_not_misrepresented_as_final']=not all(A.capacity_checks(altered,'checkpoint_prefix')[0].values())
    altered=capacity('finalrun');altered['final_read_count']=2
    checks['final_read_twice_rejected']=not all(A.capacity_checks(altered,'finalrun')[0].values())
    altered=capacity('finalrun');altered['records'][-1]['flags_cleared_by_observer']=True
    checks['cleared_flag_claim_rejected']=not all(A.capacity_checks(altered,'finalrun')[0].values())
    s,b,rows=actual_fixture(out/'synthetic_actualu');c,d,prefix=A.actualu_checks(out/'synthetic_actualu',s,b);checks['allbatch11prefix_and_three_trace_positive']=all(c.values())
    with (out/'synthetic_actualu/rough_actualu_evidence.jsonl').open('a') as f:f.write(json.dumps({'collect_call':12,'later_append_outside_scope':True})+'\n{')
    c,d,prefix2=A.actualu_checks(out/'synthetic_actualu',s,b);checks['later_append_and_partial_suffix_not_prefix_drift']=all(c.values()) and prefix==prefix2 and d['later_complete_rows_ignored']==1
    bad=copy.deepcopy(s);bad['rough_controller']['latest_actualu_evidence']['max_abs_errors']['cache_actual']=.1
    checks['marker_prefix_mismatch_rejected']=not all(A.actualu_checks(out/'synthetic_actualu',bad,b)[0].values())
    path=out/'synthetic_actualu/rough_actualu_trace_00001.npz';data=A.loadnp(path);data['replay_c'][0,0,0]+=.01;np.savez_compressed(path,**data)
    checks['raw_replayc_mismatch_rejected']=not all(A.actualu_checks(out/'synthetic_actualu',s,b)[0].values())
    pcfg={'horizon':8,'pi_batch':128,'critic_epochs':2,'critic_minibatches':2};start=(np.arange(128)%15)[:,None];mask=start.reshape(1,-1)+np.arange(8)[:,None]<15;counts=[int(mask[:4].sum()),int(mask[4:].sum())]*2
    e={'policy':{'start_phases':start.tolist(),'active_mask_H_B':mask.tolist(),'critic_norm_eligible':int(mask.sum()+(start[:,0]+8<15).sum()),'critic_loss_minibatch_active_counts':counts,'optimizer_actor_calls':1,'optimizer_critic_calls':4,'critic_total_candidate_rows':1024,'hard_mask_rule':'phase<15 only; physical soft termination retains native survival semantics'},'latest_window':{'candidate_window_entries':2048,'valid_window_entries':1500,'terminal_valid_entries':48,'reset_suffix_entries_masked':548}}
    checks['H8_phase_terminal_norm_and_loss_mask_positive']=all(A.policy_checks(e,pcfg).values())
    bad=copy.deepcopy(e);bad['policy']['active_mask_H_B'][7][14]=True
    checks['terminal15_active_mask_rejected']=not all(A.policy_checks(bad,pcfg).values())
    bad=copy.deepcopy(e);bad['policy']['critic_norm_eligible']+=1
    checks['critic_deadrow_norm_count_rejected']=not all(A.policy_checks(bad,pcfg).values())
    bad=copy.deepcopy(e);bad['latest_window']['reset_suffix_entries_masked']-=1
    checks['reset_suffix_summary_accounting_rejected']=not all(A.policy_checks(bad,pcfg).values())
    initial,result=initial_fixture(out/'synthetic_initial');c,d=A.initial_checks(initial,out/'synthetic_initial',result);checks['raw_initial_fingerprint_and_final_delta_positive']=all(c.values())
    bad=copy.deepcopy(initial);bad['rough_initial_parameter_fingerprints']['actor']['sha256']='0'*64
    checks['raw_initial_fingerprint_mismatch_rejected']=not all(A.initial_checks(bad,out/'synthetic_initial',result)[0].values())
    bad=copy.deepcopy(initial);bad['modules']['actor']['net.0.weight']+=1
    checks['raw_initial_final_delta_mismatch_rejected']=not all(A.initial_checks(bad,out/'synthetic_initial',result)[0].values())
    checks['no_simulator_environment_imports']=not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
    checks['CUDA_masked_and_uninitialized']=os.environ['CUDA_VISIBLE_DEVICES']=='-1' and not torch.cuda.is_initialized()
    report={'schema':'rough104_auditor_CPU_fixture_checks:1','status':'PASS_HELPER_FIXTURES_NOT_REAL_TRAINING' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':A.pin(ROOT/'scripts/audit_rough_checkpoint_cpu.py'),'checker':A.pin(__file__),'scope':'Small synthetic tensor/metadata/control/capacity/phase fixtures; no actual training checkpoint was graded and no Env/physics was constructed. Full real audit remains pending actualsmokes.','GPU_used':False,'environment_created':False,'physics_steps':0}
    A.json_write(out/'report.json',report);print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':A.pin(out/'report.json')},indent=2));return int(not all(checks.values()))

if __name__=='__main__':raise SystemExit(main())
