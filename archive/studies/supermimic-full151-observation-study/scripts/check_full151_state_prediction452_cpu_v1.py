"""Narrow synthetic group/metric/single-forward contracts, no actual WM/data."""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,pickle,random,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import state_prediction452105_v1 as a


def fixture():
    identities=a.expected_schedule();n=1808
    q=np.zeros((n,27),np.float32);q[:,3]=1;v=np.zeros((n,26),np.float32)
    x=np.zeros((2260,58),np.float32);x[:,43:52]=np.eye(3,dtype=np.float32).reshape(1,9)
    pp=np.zeros((n,3),np.float32);R=np.broadcast_to(np.eye(3,dtype=np.float32),(n,3,3)).copy()
    clock=np.empty(601,np.float32);clock[0]=0
    for i in range(600):clock[i+1]=np.float32(clock[i]+np.float32(.005))
    samples=[];rewards=[];labels=[]
    for row in identities:
        i=row['group_id'];k=row['k'];base=4*i;value=np.float32(k/1024)
        q[base:base+4,7]=value;v[base:base+4,6]=value
        x[base:base+4,0]=value;x[base:base+4,20]=value;x[1808+i,0]=value;x[1808+i,20]=value
        u=np.zeros(20,np.float32);u[0]=np.float32(i/1024)
        rewards.append({'path':row['path'],'k':k,'branch_id':row['branch_id'],'physics':x[1808+i].tolist()})
        labels.append({'path':row['path'],'k':k,'branch_id':row['branch_id'],'last_sample_index':base+3})
        for sub in range(1,5):
            j=base+sub-1;samples.append({'path':row['path'],'phase':k,'mode':row['mode'],'branch_id':row['branch_id'],'substep':sub,
                'source_frame':200+k,'canonical_pre_tick':4*k+sub-1,'canonical_post_tick':4*k+sub,'pre_time':float(clock[4*k+sub-1]),'post_time':float(clock[4*k+sub]),
                'pre_qpos':q[j].tolist(),'pre_qvel':v[j].tolist(),'post_qpos':q[j].tolist(),'post_qvel':v[j].tolist(),'ctrl':u.tolist(),'external_force_max':0.,
                'phase_contract':'solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward'})
    mapping={'hand_qpos':np.arange(7,27,dtype=np.int32),'hand_dof':np.arange(6,26,dtype=np.int32),'object_qpos_start':np.array(0,np.int32),'object_dof_start':np.array(0,np.int32)}
    packet={'native_GPU_prestate58':x[:1808].copy(),'completed_mask':np.ones(1808,np.bool_),'source_pre_row':np.arange(1808,dtype=np.int64),'palm_pos':pp,'palm_rot':R,**mapping}
    raw_gpu={'native_GPU_state58_raw':x,'completed_mask':np.ones(2260,np.bool_)}
    forensic={'pre_qpos':q,'pre_qvel':v,'palm_pos_all2260':pp.copy(),'palm_rot_all2260':R.copy(),**mapping}
    return samples,rewards,packet,raw_gpu,labels,forensic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).resolve();a.require(not out.exists(),'Fresh narrow CPU proof required');out.mkdir(parents=True);(out/'executed_sources').mkdir()
    pins={k:a.pin(v) for k,v in a.OWN.items()};deps={'qualified_metadata':a.pin(a.qualified.__file__),'shared_saved_helpers':a.pin(a.common.__file__)}
    for k,v in {**pins,**deps}.items():(out/'executed_sources'/(k+'_'+Path(v['path']).name)).write_bytes(Path(v['path']).read_bytes())
    a.write(out,'MODULE_FREEZE.json',{'source_pins':pins,'dependency_pins':deps,'protocol':a.pin(a.PREREG),'scope':'Frozen before synthetic tests; no actual data/loader/model/physics execution.'})
    checks={};old_load=np.load;actual_npz={'calls':0}
    def check(k,v):
        a.require(k not in checks,'Duplicate check');checks[k]=bool(v)
        if not checks[k]:raise AssertionError(k)
    def reject(k,fn):
        try:fn()
        except (ValueError,KeyError,TypeError):check(k,True)
        else:check(k,False)
    def deny(*args,**kwargs):actual_npz['calls']+=1;raise AssertionError('No saved NPZ inputs in synthetic proof')
    np.load=deny
    try:
        for v in pins.values():ast.parse(Path(v['path']).read_text(encoding='utf-8-sig'))
        check('AST_and_default_no_backend',all(m not in sys.modules for m in ('torch','warp','mujoco')))
        check('frozen_prereg3765_correct_source_passive_MJB3',a.pin(a.PREREG)['sha256']==a.PREREG_SHA)
        h={'schema':a.SCHEMA,'role':a.ROLE,'protocol':{},'output':str(ROOT/'outputs/full151105_state_prediction452_CPU_fixture_NEVER_RUN'),
            'CPU_run_authorized':False,'cpu_threads':4,'source_pins':{'fixture':{}},'cpu_ready':{}}
        a.validate_header(h);check('new_header_fixed_CPU4',True)
        reject('reject_nonbool_authorization',lambda:a.validate_header({**h,'CPU_run_authorized':1}))
        reject('reject_other_thread_mode',lambda:a.validate_header({**h,'cpu_threads':1}))
        # File identity and bytes are deliberately different concepts.
        f=out/'canonical_fixture';f.mkdir();(f/'aliases').mkdir();one=f/'one.py';two=f/'different_copy.py';one.write_bytes(b'# identical fixture\n');two.write_bytes(one.read_bytes())
        v1=a.pin(one);alias={**v1,'path':str(f/'aliases'/'..'/'one.py')};v2=a.pin(two)
        canon=a.canonical_sources([v1,alias,v2]);check('samefile_alias_dedup_but_equal_bytes_other_copy_retained',len(canon)==2)
        check('first_recorded_pin_spelling_preserved',list(a.canonical_sources([alias,v1]).values())[0]==alias)
        reject('samefile_conflicting_SHA_rejected',lambda:a.canonical_sources([v1,{**alias,'sha256':'0'*64}]))
        data=fixture();arrays,ids,rule=a.assemble_groups(*data)
        check('all452_exact_firstPRE_lastPOST_source_order',len(ids)==452 and np.array_equal(arrays['group_first_pre_indices'],np.arange(0,1808,4)) and np.array_equal(arrays['post20_GPU_row_indices'],np.arange(1808,2260)))
        check('all300donor24baseline128direction_kept',[(m,sum(r['mode']==m for r in ids)) for m in ('donor','baseline','direction')]==[('donor',300),('baseline',24),('direction',128)])
        check('phase149_terminal150_keeps_its_last_u149',sum(r['terminal_observation']==150 for r in ids)==40 and all(not r['valid_next_policy_decision'] and r['no_u150'] for r in ids if r['k']==149))
        changed=list(data);changed[0]=copy.deepcopy(data[0]);changed[0][3]['ctrl'][0]=float(np.float32(.25))
        reject('four_substep_absolute_control_mismatch_rejected',lambda:a.assemble_groups(*changed))
        changed=list(data);changed[4]=copy.deepcopy(data[4]);changed[4][0]['last_sample_index']=2
        reject('wrong_endpoint_mapping_rejected',lambda:a.assemble_groups(*changed))
        changed=list(data);changed[0]=copy.deepcopy(data[0]);changed[0][0]['pre_qpos'][7]=float(np.float32(.25))
        reject('physical_vs_GPU_pre_q_row_mismatch_rejected',lambda:a.assemble_groups(*changed))
        changed=list(data);changed[0]=copy.deepcopy(data[0]);changed[0][3]['post_qvel'][6]=float(np.float32(.25))
        reject('physical_last_qd_target_mismatch_rejected',lambda:a.assemble_groups(*changed))
        changed=list(data);changed[3]=copy.deepcopy(data[3]);changed[3]['native_GPU_state58_raw'][1808,55]=np.float32(.001)
        reject('endpoint_full58_original_8eps_retained',lambda:a.assemble_groups(*changed))
        predicted=np.zeros((452,58),np.float32);truth=predicted.copy();predicted[:,0]=7;predicted[:,20]=2;predicted[:,40]=.001;predicted[:,43]=2
        delta,absolute,stats=a.error_statistics(predicted,truth,ids)
        check('F64_raw_errors_no_q_wrap_or_R_projection',delta.dtype==np.float64 and np.all(delta[:,0]==7) and np.all(delta[:,43]==2))
        check('per_unit_closedform_RMSE_no_mixed_unit_score',np.isclose(stats['all452']['blocks']['q']['component_RMSE'],7/np.sqrt(20)) and np.isclose(stats['all452']['blocks']['R']['component_RMSE'],2/3) and 'overall58' not in stats)
        reject('partial_prediction_batch_not_selected',lambda:a.error_statistics(predicted[:451],truth[:451],ids[:451]))
        xi=np.arange(46,dtype=np.float32)/64;a.verify_fixed_xi46(xi,xi.copy());check('fixed_physical_xi46_bitexact_positive',True)
        altered=xi.copy();altered[0]=np.float32(.25)
        reject('changed_xi46_rejected',lambda:a.verify_fixed_xi46(altered,xi))
        reject('context368_cannot_replace_physical_xi46',lambda:a.verify_fixed_xi46(np.zeros(368,np.float32),xi))
        # Check two allowed internal read attempts on a CLOSED stub, no torch.load.
        fake=SimpleNamespace(load=lambda path,**kw:{'fixture_only':True});original=fake.load;counts={'checkpoint_torch_loads':{'attempted':0,'returned':0}};trace=[]
        with a.counted_checkpoint_reads(fake,v1,counts,trace):fake.load(one,map_location='cpu',weights_only=False);fake.load(one,map_location='cpu',weights_only=False)
        check('two_read_budget_stub_and_exception_safe_restore',counts['checkpoint_torch_loads']=={'attempted':2,'returned':2} and len(trace)==2 and fake.load is original)
        counts={'checkpoint_torch_loads':{'attempted':0,'returned':0}};trace=[]
        def excessive():
            with a.counted_checkpoint_reads(fake,v1,counts,trace):
                for _ in range(3):fake.load(one,map_location='cpu')
        reject('third_checkpoint_read_rejected_without_retry',excessive)
        proc=subprocess.Popen([sys.executable,'-B',str(a.OWN['CLI'])],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
        try:stdout,stderr=proc.communicate(timeout=20)
        except BaseException:proc.kill();proc.communicate(timeout=2);raise
        (out/'default_stdout.json').write_text(stdout,encoding='utf-8');(out/'default_stderr.txt').write_text(stderr,encoding='utf-8');a.require(proc.returncode==0,stderr)
        default=json.loads(stdout);check('actual_default_noTorch_noNPZ_no_model',default['Torch_imported'] is False and default['NPZ_opened']==default['model_loads']==default['WM_forwards']==0)
        import torch
        torch.set_num_threads(4)
        class ClosedModel(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0;self.register_buffer('norm_count',torch.zeros((),dtype=torch.float64))
            def forward(self,x,u):self.calls+=1;return x+.5*u[:,:1],x.new_zeros((len(x),15))
        model=ClosedModel().eval();x=torch.from_numpy(arrays['pre_x452_FP32'].copy());u=torch.from_numpy(arrays['actual_u452_FP32'].copy())
        def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
        before_rng=rng();before=a.common._tensor_digest(model.state_dict());before_x=x.numpy().tobytes();y,aux=a.predict_once(model,x,u)
        check('closed_model_exact_one_B452_forward',model.calls==1 and tuple(y.shape)==(452,58) and tuple(aux.shape)==(452,15))
        check('closed_model_RNG_normalizer_and_inputs_unchanged',rng()==before_rng and a.common._tensor_digest(model.state_dict())==before and x.numpy().tobytes()==before_x and not y.requires_grad)
        reject('wrong_batch_rejected_before_second_forward',lambda:a.predict_once(model,x[:451],u[:451]))
        check('rejected_shape_did_not_retry_or_forward',model.calls==1)
        reject('requires_grad_input_rejected',lambda:a.predict_once(model,x.clone().requires_grad_(True),u))
        check('no_actual_WM_loader_objective_native_or_CUDA',all(m not in sys.modules for m in ('full151_native.world_model_checkpoint','full151_intent.binding_v3','mujoco','warp')) and not torch.cuda.is_initialized() and actual_npz['calls']==0)
        text=Path(a.__file__).read_text(encoding='utf-8')
        check('source_no_score_goal_objective_or_native_state_call',all(s not in text for s in ('object_terms_f64(', 'build_penalty_only_objective(', 'quat_to_mat(', 'SimReader.state(')))
        check('source_one_strict_loader_and_single_predict_call',text.count("call('strict_WM_loader'")==1 and text.count("call('WM_forward'")==1 and 'if not passed:retry' not in text)
        for k,v in {**pins,**deps}.items():check('source_frozen_'+k,a.pin(v['path'])==v)
        proof={'schema':'full151105_state_prediction452_CPU_contract:1','status':'PASS_SYNTHETIC_GROUP_ERROR_AND_SINGLE_FORWARD_CONTRACT','source_pins':pins,'dependency_pins':deps,'protocol':a.pin(a.PREREG),
            'checks':checks,'passed':sum(checks.values()),'total':len(checks),'scope':{'actual_saved_input_assembly':0,'actual_WM_loads':0,'actual_WM_forwards':0,'actual_model_or_MJB_or_Env_or_GPU':0,
                'synthetic_groups':452,'synthetic_closed_Torch_forward':1,'new_AD':0}}
        a.write(out,'report.json',proof)
        ready={'schema':'full151105_state_prediction452_CPU_READY:1','status':proof['status'],'source_pins':pins,'dependency_pins':deps,'cpu_report':a.pin(out/'report.json'),
            'cpu_checks':{'passed':proof['passed'],'total':proof['total']},'protocol':a.pin(a.PREREG),'CPU_scientific_run_authorized':False,'GPU_run_authorized':False,
            'api':{'entry':str(a.OWN['CLI']),'module':str(a.OWN['consumer']),'schema':a.SCHEMA,'role':a.ROLE,'keys':sorted(a.KEYS),'required_sources':'required_sources(protocol_dict)','header':'validate_header(config_dict)'}}
        a.write(out,'READY.json',ready);print(json.dumps({'passed':proof['passed'],'total':proof['total'],'report':a.pin(out/'report.json'),'READY':a.pin(out/'READY.json'),'source_pins':pins},indent=2))
    except BaseException as error:
        a.write(out,'failure.json',{'status':'FAILED_CPU_FIXTURE_PRESERVED','error':repr(error),'checks_completed':checks,'source_pins':pins,'actual_model_or_data_or_GPU':False});raise
    finally:np.load=old_load


if __name__=='__main__':main()
