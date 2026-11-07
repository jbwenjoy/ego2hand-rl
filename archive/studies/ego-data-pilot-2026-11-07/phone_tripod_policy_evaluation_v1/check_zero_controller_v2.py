"""Qualify scheduled task control against saved constant-GN original native gates."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,time,traceback
H=Path(__file__).resolve().parent;P=H.parent;T=P/'phone_geometry_actor_integration_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(v):assert pin(v['path'])==v;return Path(v['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def mod(name,path):
    import sys
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

class TaskController:
    """Injected deterministic Actor or zero action, no native observer forwards.

    This class loads no NN. A trained Actor must be explicitly supplied later.
    Memory advances when the command is issued; next callback is at the next H.
    """
    def __init__(self,task,geometry,state_codec,dt,phone,*,actor=None,device='cpu'):
        self.task=task;self.geometry=geometry;self.codec=state_codec;self.dt=dt;self.phone=phone;self.actor=actor;self.device=device;self.memory=None;self.records=[];self.NN_FW=0
    def __call__(self,tick,dt,m,d):
        import numpy as np
        import torch
        assert dt==self.dt
        x=torch.tensor(self.codec.actual_state(d.qpos,d.qvel,m.body_ipos[self.phone]),dtype=torch.float64)
        if self.memory is None:self.memory=self.task.initial_memory(x,native_h_s=dt,n_substeps=round(.02/dt))
        assert int(self.memory['step'])==tick//round(.02/dt)
        with torch.no_grad():
            obs=self.task.observation(x,self.memory,self.geometry)
            if self.actor is None:action=torch.zeros(20,dtype=torch.float32)
            else:
                action=self.actor(obs.to(device=self.device,dtype=torch.float32).unsqueeze(0)).squeeze(0).to('cpu');self.NN_FW+=1
                if action.shape!=(20,):raise ValueError('Explicit singleton Actor action20')
            residual,u=self.task.control_target(action,self.memory)
        self.records.append(dict(pre_tick=int(tick),time_s=tick*dt,source_state58=x.numpy().tolist(),memory83=self.task.flatten_memory(self.memory).numpy().tolist(),observation222_FP32=obs.numpy().tolist(),action20_FP32=action.numpy().tolist(),ctrl20_F64=u.numpy().tolist()))
        self.memory=self.task.advance_memory(self.memory,action,residual,u)
        return u.numpy()

def prepare():
    b=read(T/'task_bundle_v2.json');ip=read(T/'protocol_CPUFalse_v2.json');prior=P/'functional_tripod_loaded_dynamics_v1/completed_run1/report.json';old=read(prior)
    originals=[v for v in old['cases'] if v['name']=='exact__GN_best_final_fresh']
    assert len(originals)==2
    p=dict(schema='phone_scheduled_zero_action_original_native_loop_CPU_protocol:1',CPU_run_authorized=False,learned_policy_or_NN_training_authorized=False,
        source=pin(__file__),sources=dict(kernel=pin(H/'controlled_native_kernel_v1.py'),kernel_derivation=pin(H/'kernel_derivation_v1.json'),task=ip['sources']['task'],directional=ip['sources']['directional'],state_codec=pin(P/'phone_transition_dataset_v1/transition_codec_v1.py')),
        inputs=dict(bundle=pin(T/'task_bundle_v2.json'),catalog=pin(P/'phone_directional_gap_features_v1/geometry_catalog_v1.json'),model=b['model'],binding=pin(P/'hand2_holding_v1/compiled_nominal_v2/binding.json'),original_dynamic_report=pin(prior)),
        original_cases=[dict(dt_s=v['dt_s'],trace=v['trace'],command=v['command'],termination=v['termination'],steps=v['finite_steps_saved'],metrics=v['metrics']) for v in originals],
        dt_s=[.0003125,.00015625],budget=dict(MJB=1,initial_forward=2,maximum_native_steps=19200,maximum_callback_calls=200,NN_calls=0,optimizer_steps=0,GPU=0),
        output=str(H/'zero_qualified_run2'),scope='New50Hz control hook with zero normalized action must reproduce the frozen constant GN route; no learned policy or holding success. Original2s gates, pernative loss abort and tail nulls remain. Newtracking errors are descriptors of the commanded reference, not new success gates.')
    write(H/'zero_CPUFalse_v2.json',p);print(json.dumps(dict(protocol=pin(H/'zero_CPUFalse_v2.json'))))
def preflight(path):
    p=read(path);assert p['schema']=='phone_scheduled_zero_action_original_native_loop_CPU_protocol:1' and p['source']==pin(__file__) and p['learned_policy_or_NN_training_authorized'] is False
    for v in [*p['sources'].values(),*p['inputs'].values()]:checked(v)
    for c in p['original_cases']:checked(c['trace'])
    assert p['dt_s']==[.0003125,.00015625] and not Path(p['output']).exists();return p
def run(path):
    p=preflight(path);assert p['CPU_run_authorized'] is True
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import torch
    import mujoco
    torch.set_num_threads(4);out=Path(p['output']);out.mkdir();started=time.perf_counter();phase='model'
    try:
        kernel=mod('zero_qualified_kernel',checked(p['sources']['kernel']));taskmod=mod('zero_qualified_task',checked(p['sources']['task']));gap=mod('zero_qualified_geometry',checked(p['sources']['directional']));codec=mod('zero_qualified_state',checked(p['sources']['state_codec']))
        m=mujoco.MjModel.from_binary_path(str(checked(p['inputs']['model'])));phone,pg,table,pads=kernel.frozen.assert_model(m,mujoco,np,read(checked(p['inputs']['binding'])))
        initialdigest=kernel.frozen.model_digest_except_dt(m,np);counts=dict(MJB_reads=1,MjData=0,initial_forward=0,integration_steps=0,contact_force_reads=0,controller_calls=0,controller_receipt_reads=0,NN_calls=0)
        b=read(checked(p['inputs']['bundle']));task=taskmod.PhoneTask(checked(p['inputs']['bundle']),mode='A1_tracking');geometry=gap.DirectionalGapProvider(checked(p['inputs']['catalog']))
        results=[];comparisons=[]
        for h in p['dt_s']:
            phase='dt_'+str(h);old=next(v for v in p['original_cases'] if v['dt_s']==h);variant=dict(old['command'])
            assert np.array(variant['qpos27']).tobytes()==np.array(b['initial']['qpos27_F64']).tobytes() and np.array(variant['ctrl20']).tobytes()==np.array(b['initial']['ctrl20_F64']).tobytes()
            case_dir=out/('dt_'+str(h));case_dir.mkdir();callback=TaskController(task,geometry,codec,h,phone)
            summary,endpoint=kernel.evaluate_case_controlled(m,variant,h,case_dir,phone,pg,table,pads,mujoco,np,counts,callback)
            write(case_dir/'task_controller_records.json',callback.records)
            equal={}
            with np.load(checked(old['trace']),allow_pickle=False) as ref,np.load(case_dir/'trace.npz',allow_pickle=False) as actual:
                for key in ref.files:
                    a,z=ref[key],actual[key];equal[key]=(a.dtype==z.dtype and a.shape==z.shape and a.tobytes()==z.tobytes())
                ctr=actual['pre_executed_ctrl20'];constant=(ctr==np.array(b['initial']['ctrl20_F64'])).all()
            checks=dict(all_prior_trace_arrays_rawbytes_equal=all(equal.values()),planned_control_only_constant=bool(constant),callback_NN_zero=callback.NN_FW==0,exact_original_abort_and_steps=summary['termination']==old['termination'] and summary['finite_steps_saved']==old['steps'],metrics_exact=summary['metrics']==old['metrics'])
            comparisons.append(dict(dt_s=h,checks=checks,per_prior_array_raw_equality=equal,controller_calls=len(callback.records),original_trace=old['trace'],new_trace=summary['trace']))
            results.append(summary)
        checks={f'dt{i}_{k}':v for i,c in enumerate(comparisons) for k,v in c['checks'].items()};checks['model_unchanged_except_dt']=initialdigest==kernel.frozen.model_digest_except_dt(m,np);checks['CPU_only']=not torch.cuda.is_initialized()
        report=dict(schema='scheduled_task_controller_native_loop_zero_equivalence:1',status='PASS_CALLBACK_MATCHES_FROZEN_NATIVE_LOOP' if all(checks.values()) else 'FAIL_PRESERVED_NO_RETRY',protocol=pin(path),source=pin(__file__),checks=checks,passed=sum(checks.values()),total=len(checks),
            cases=results,comparisons=comparisons,counts=counts,elapsed_s=time.perf_counter()-started,scope=p['scope'],learned_Actor_executed=False,nominal_holding_benefit_proven=False)
        write(out/'report.json',report);print(json.dumps(dict(passed=report['passed'],total=report['total'],report=pin(out/'report.json'))));assert all(checks.values())
    except BaseException as e:write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--protocol',type=Path);a.add_argument('--run',action='store_true');arg=a.parse_args()
    if arg.prepare:prepare()
    elif arg.run:run(arg.protocol)
    else:preflight(arg.protocol);print('METADATA_ONLY_NO_NATIVE_NN')
