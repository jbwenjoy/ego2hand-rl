"""New prior scope/RNG/save contracts; no Env/WM/optimizer/training execution."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,hashlib,importlib.util,json,os,random,subprocess,sys,time,traceback
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'src/full151_native/learning_v2.py',ROOT/'src/full151_native/prior_hook_v1.py',ROOT/'src/full151_native/training_capacity.py',ROOT/'src/full151_native/controller.py',ROOT/'src/full151_native/runtime.py']
    pins={};src=out/'source';src.mkdir()
    for p in files:(src/p.name).write_bytes(p.read_bytes());pins[p.name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    checks={};errors={}
    def check(k,v):checks[k]=bool(v)
    def folder(n):p=out/n;p.mkdir();return p
    def case(n,fn):
        try:fn()
        except BaseException as exc:check(n,False);errors[n]={'error':repr(exc),'traceback':traceback.format_exc()}
    # Actual new metadata consumer import, in a separate no-Torch process.
    command="import sys;sys.path.insert(0,"+repr(str(ROOT/'src'))+");import full151_native.learning_v2;assert all(k not in sys.modules for k in ('torch','warp','mujoco_warp','full151_intent.binding_v3','full151_native.prior_hook_v1'));print('NO_DEVICE_IMPORT')"
    c=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True,timeout=15)
    (out/'metadata_import_stdout.txt').write_text(c.stdout);(out/'metadata_import_stderr.txt').write_text(c.stderr)
    check('actual_learning_v2_metadata_import_no_Torch_Warp_factory_hook',c.returncode==0 and c.stdout.strip()=='NO_DEVICE_IMPORT')
    import numpy as np
    import torch
    from full151_native import learning_v2 as L
    from full151_native.prior_hook_v1 import ImaginationPrior,COUNTERS
    from full151_native.io import pin,read
    from full151_native.bundle import load_bundle
    torch.set_num_threads(4)

    class Objective:
        metadata={'CPU_fixture_only':True}
        def __init__(self,method):self.method=method;self.calls=0
        def __call__(self,x,m):
            self.calls+=1;v=x[:,0] if self.method=='A2' else -.1*x[:,0].square()
            return v,{'fixture_bonus':v,'fixture_constant':torch.ones_like(v)}
    def native(x,m,*args,**kwargs):return x[:,1].square(),m,{'native_unchanged':x[:,2]}
    def runner():return SimpleNamespace(reward_step=native,iteration=20)
    def call(r,x,t,soft=True):return r.reward_step(x,{'time':t[:,None]},None,None,None,None,soft=soft)
    def hook_math(method):
        p=folder(method+'_hook');obj=Objective(method);prior=ImaginationPrior(method,obj,{'path':'CPU_fixture','sha256':'CPU_fixture'},p);r=runner();original=r.reward_step
        x=torch.zeros((4,58));x[:,0]=torch.tensor([1.,2.,3.,4.]);x[:,1]=2.;x.requires_grad_();start=torch.tensor([0.,148.,149.,150.]);before=torch.get_rng_state().clone()
        with prior.policy_scope(r):
            for k in range(8):
                time_=start+k;reward,cage,terms=call(r,x,time_)
                bonus=x[:,0] if method=='A2' else -.1*x[:,0].square()
                check(method+'_exact_dt_once_'+str(k),torch.equal(reward,x[:,1].square()+.02*bonus))
                check(method+'_native_terms_and_cage_unchanged_'+str(k),cage['time'].shape==(4,1) and torch.equal(terms['native_unchanged'],x[:,2]))
                if k==0:torch.where(time_<150,reward,torch.zeros_like(reward)).sum().backward()
        expected0=.02 if method=='A2' else -.004
        check(method+'_bonus_input_gradient_retained_and_dead_row_masked',abs(float(x.grad[0,0])-expected0)<1e-8 and float(x.grad[3].abs().max())==0)
        e=prior.evidence();check(method+'_H8_counts_active11_inactive21',e['policy_update_calls']==1 and e['objective_forward_calls']==8 and e['dt_applications']==8 and e['active_rows']==11 and e['inactive_rows']==21)
        check(method+'_scope_restore_no_RNG_change',r.reward_step is original and not hasattr(r,'_full151_prior_scope_active') and torch.equal(before,torch.get_rng_state()))
        rows=[json.loads(v) for v in (p/'full151_prior_evidence.jsonl').read_text().splitlines()]
        check(method+'_JSONL_complete_prefix_marker',len(rows)==1 and rows[0]['complete'] is True and rows[0]['cumulative']==e and rows[0]['active_rows']==11)
        # Real/outside calls retain the exact native path after scope exit.
        check(method+'_outside_scope_unshaped',torch.equal(call(r,x,start,False)[0],x[:,1].square()) and obj.calls==8)
        with prior.policy_scope(r):
            reward,_,_=call(r,x,torch.full((4,),150.));torch.where(torch.zeros(4,dtype=torch.bool),reward,torch.zeros_like(reward)).sum().backward()
        check(method+'_allinactive_logs_finite_zero',all(torch.isfinite(v) and float(v)==0 for v in prior.last_metrics.values()))
        try:
            with prior.policy_scope(r):call(r,x,start,False)
        except ValueError:check(method+'_softFalse_guard_restore',r.reward_step is original and not hasattr(r,'_full151_prior_scope_active'))
        else:check(method+'_softFalse_guard_restore',False)
    case('A2_hook_fixture',lambda:hook_math('A2'));case('P_ONLY_hook_fixture',lambda:hook_math('P_ONLY'))

    def baseline():
        p=folder('A1L_NONE');prior=L.construct_prior({'method':'A1L'},p,'cpu');r=runner();old=r.reward_step
        with prior.policy_scope(r):check('A1L_reward_never_wrapped',r.reward_step is old)
        check('A1L_NONE_allfive_counters0',prior.evidence()['status']=='NONE' and all(prior.evidence()[k]==0 for k in COUNTERS))
        check('A1L_no_factory_import_no_priorrows', 'full151_intent.binding_v3' not in sys.modules and not (p/'full151_prior_evidence.jsonl').exists() and read(p/'objective_build_rng.json')['status']=='NOT_CONSTRUCTED')
    case('A1L_baseline_fixture',baseline)

    # Actual new CPU factories only, no geometry forward. Tests the learner's
    # entire import/build range once each, not the binding owner's forward proof.
    specpath=ROOT/'outputs/full151_intent_binding_CPU_v3/objective_spec.json';specpin=pin(specpath)
    actual={}
    for method in ('P_ONLY','A2'):
        def factory(method=method):
            p=folder(method+'_actual_factory');before=L.objective_rng_snapshot();actual[method]=L.construct_prior({'method':method,'contact_binding':specpin},p,'cpu');report=read(p/'objective_build_rng.json')
            check(method+'_actual_import_build_rng_exact',report['status']=='PRESERVED' and all(report['checks'].values()) and L.objective_rng_comparison(before,L.objective_rng_snapshot())['status']=='PRESERVED')
            check(method+'_actual_factory_no_forward_counts',all(actual[method].evidence()[k]==0 for k in COUNTERS))
        case(method+'_actual_CPU_factory',factory)
    def rng_negative():
        import full151_intent.binding_v3 as B
        before=L.objective_rng_snapshot();old=B.build_contact_objective
        def bad(*args,**kwargs):random.random();np.random.random();torch.rand(1);return Objective('A2')
        B.build_contact_objective=bad
        try:
            try:L.construct_prior({'method':'A2','contact_binding':specpin},folder('RNG_changed_negative'),'cpu')
            except RuntimeError:
                report=read(out/'RNG_changed_negative/objective_build_rng.json')
                check('RNG_changed_factory_aborts_without_learner_repair',report['status']=='RNG_CHANGED_ABORT' and not report['checks']['Python_exact'] and not report['checks']['NumPy_exact'] and not report['checks']['TorchCPU_exact'])
            else:check('RNG_changed_factory_aborts_without_learner_repair',False)
        finally:
            B.build_contact_objective=old
            # Test harness restores its deliberate negative-fixture draws only;
            # production construct_prior contains no RNG setter/repair.
            random.setstate(before['python']);np.random.set_state(before['numpy']);torch.set_rng_state(before['torch_cpu'])
    case('RNG_negative_fixture',rng_negative)

    b=load_bundle(ROOT/'outputs/full151_native_input_v1/bundle.json');oldcfg=read(ROOT/'configs/full151-a1l-smoke40-authorized-v1.json')
    cfg={**oldcfg,'schema':'full151_knownrough_native_training_run:2','GPU_run_authorized':False}
    for method in ('A1L','P_ONLY','A2'):
        v=copy.deepcopy(cfg);v['method']=method;v['contact_binding']=None if method=='A1L' else specpin
        check(method+'_actual_metadata_budget_spec_validation',L.validate_training_inputs(v,b)['method']==method)
    for label,key,value in [('oldschema','schema','full151_knownrough_native_training_run:1'),('oldmethod','method','A0'),('wrongrole','role','native_a0')]:
        v=copy.deepcopy(cfg);v[key]=value
        try:L.validate_training_inputs(v,b)
        except ValueError:check('reject_'+label,True)
        else:check('reject_'+label,False)
    v=copy.deepcopy(cfg);v['method']='P_ONLY';v['contact_binding']=specpin;v['GPU_run_authorized']=True
    try:L.validate_training_inputs(v,b)
    except ValueError:check('CPUfalse_spec_rejects_actual_GPU_training_before_import',True)
    else:check('CPUfalse_spec_rejects_actual_GPU_training_before_import',False)

    # Actual native atomic save method, controller3->105, new runtime/prior
    # layer, then unchanged capacity observer. No native Runner is constructed.
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from full151_native.controller import full151_controller
    from full151_native.training_capacity import TrainingCapacityScope
    class Env:
        def __init__(self,cfg):self.sim=SimpleNamespace(wp_data=SimpleNamespace(overflow=SimpleNamespace(numpy=lambda:np.zeros(1,np.int32))));self.common_step_counter=164
        def reset(self,**kwargs):return None
        def close(self):pass
        def step(self,*args):raise AssertionError('No Env step')
    class MDP:
        def __init__(self):pass
        def control_target(self,*args):pass
    class NativeSaveFixture:
        save=DexSuperMimicRunner.save
        def __init__(self,env,prior):
            self.env=env;self.device='cpu';self.iteration=39;self.sim_steps=20992;self.full151_prior=prior
            self.cfg=SimpleNamespace(algorithm='world_model',to_dict=lambda:{'CPU_fixture':True})
            self.mdp=SimpleNamespace(reference_signature='CPU_fixture',full151_controller_metadata={'CPU_fixture':True})
            self.torque_norm=SimpleNamespace(state_dict=lambda:{})
            self.context_dim=368;self.contact_dim=15;self.optimizers={}
            self.reader=SimpleNamespace(context_names=[],robot=SimpleNamespace(joint_names=[]));self.wall_offset=0.;self.started_at=time.perf_counter()
            self._modules={}
            for name in L.MODULE_KEYS:
                m=torch.nn.Module();m.register_parameter('weight',torch.nn.Parameter(torch.arange(3,dtype=torch.float32)));self._modules[name]=m
            _,self.full151_initial_parameter_fingerprints=L.parameter_fingerprints(self._modules)
        def modules(self):return self._modules
        def format_version(self):return 3
        def collect(self,*args):raise AssertionError('No collect')
        def load(self,*args):pass
        def load_model_only(self,*args):pass
    class Recorded(NativeSaveFixture):
        def save(self,path,next_iteration=None):
            with L.checkpoint_evidence_scope(self,{'policy_evidence':lambda r:{'status':'NOT_RUN'}},torch):return super().save(path,next_iteration)
    def save_fixture():
        p=folder('actual_native_save');original=torch.save;oldsave=NativeSaveFixture.save
        with TrainingCapacityScope(Env,torch,p,stage='CPU_fixture',source_pins=L.source_pins()):
            env=Env(SimpleNamespace(sim=SimpleNamespace(nconmax=192,njmax=4096)))
            with full151_controller(Env,MDP,NativeSaveFixture,b['baseline_u150'],b['descriptor']):
                r=Recorded(env,actual['P_ONLY']);r.save(p/'model_40.pt',40);saved=torch.load(p/'model_40.pt',map_location='cpu',weights_only=False)
                check('actual_native_atomic105_allfive_evidence_coexist',saved['format_version']==105 and all(k in saved for k in ('full151_controller','full151_runtime_evidence','full151_initial_parameter_fingerprints','full151_prior_evidence','full151_capacity_evidence')))
                check('actual_native_save_prior_exact_and_cap_nonfinal',saved['full151_prior_evidence']==actual['P_ONLY'].evidence() and saved['full151_capacity_evidence']['finalized'] is False and not (p/'model_40.tmp').exists())
        check('actual_native_save_scopes_restored',torch.save is original and NativeSaveFixture.save is oldsave)
    case('actual_native_save_fixture',save_fixture)
    check('capacity_source_pins_fivekeys_learning_is_v2',set(L.source_pins())=={'learning','training_capacity','controller','runtime','native'} and Path(L.source_pins()['learning']['path']).name=='learning_v2.py')
    check('CUDA_never_initialized_no_training',not torch.cuda.is_initialized())
    check('all_execution_source_snapshots_unchanged',all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in pins.values()))
    report={'schema':'full151_prior_learning_CPU_contract:1','status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'errors':errors,'source_pins':pins,'actual_Env_Runner_WM_optimizer_training_created':False,'GPU_initialized':torch.cuda.is_initialized(),'scope':'Synthetic per-row reward/autograd/H8mask logging and actual new CPU objective construction without objective forward; production RNG comparison includes negative fixture; actual native save method and tiny parameter modules, not native Env/WM/optimizer/training.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'errors':errors},indent=2));raise SystemExit(0 if all(checks.values()) else 1)

if __name__=='__main__':main()
