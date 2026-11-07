"""CPU native-method and ABI104 checks; never a physical/training qualification."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--bundle',type=Path,default=ROOT/'outputs/input_bundle_v1/bundle.json');a=ap.parse_args()
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);snap=output/'source';snap.mkdir()
    controller_path=ROOT/'src/rough_prefix/controller.py'
    for p in (controller_path,Path(__file__)): (snap/p.name).write_bytes(p.read_bytes())
    frozen={'controller':sha(controller_path),'checker':sha(__file__)}
    checks=[];cases={};report={'status':'RUNNING','source_sha256':frozen,'GPU_requested':False}
    def test(name,fn):
        try:
            if not bool(fn()):raise AssertionError('False contract')
            checks.append({'name':name,'passed':True})
        except BaseException as exc:checks.append({'name':name,'passed':False,'error':repr(exc),'traceback':traceback.format_exc()})
    def raises(fn):
        try:fn()
        except (ValueError,AssertionError,RuntimeError):return True
        return False
    try:
        import torch
        import numpy as np
        import mujoco
        from rough_prefix.bundle import load_bundle
        from rough_prefix.controller import rough_controller
        from dex_supermimic.rl.dex_supermimic.pen import PenMDP,PEN_MEMORY_DIMS
        from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
        from dex_supermimic.rl.dex_supermimic.networks import WorldModel
        bundle=load_bundle(a.bundle)
        baseline=torch.tensor(bundle['baseline_u15']); descriptor=bundle['descriptor']
        model_path=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb')
        fixture_path=Path('D:/workspace/supermimic-chord-study/scripts/check_pen_target_limits.py')
        spec=importlib.util.spec_from_file_location('_rough_native_fixture',fixture_path);fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        # Fixture construction is before patching the real PenMDP constructor.
        prepared=[fixture.fixture(model_path,torch.full((2,20),6. if k%2 else -6.),reverse_ctrl_order=k%2==1) for k in range(19)]
        class CPUEnv:
            """Method fixture; four apply calls, no dynamics, no envreset implementation."""
            def __init__(self, parts):
                self.parts=parts;_,entity,term,oldmdp,_=parts;native=term._env
                self.__dict__.update(vars(native));term._env=self
                self.cfg.auto_reset=False
                self.motion=self.command_manager.get_term('motion')
                self.motion.time_steps=torch.zeros(2,dtype=torch.long)
                self.motion._frames={'joint_pos':self.motion.joint_pos[:1].repeat(16,1),'object_pose':torch.tensor([[0.,0,0,1,0,0,0]]).repeat(16,1),'link_pos':torch.zeros(16,21,3)}
                self.motion.group=SimpleNamespace(trajectory_end=torch.tensor([16]))
                self.sim.data=SimpleNamespace(ctrl=torch.zeros(2,parts[0].nu))
                self.reader=SimpleNamespace(env=self,robot=entity,kin=oldmdp.kin)
            def step(self,action):
                term=self.action_manager.get_term('joint_pos');term.process_actions(action)
                for _ in range(4):term.apply_actions();self.sim.data.ctrl[:,term._position_ctrl_ids]=self.scene['robot'].last_target
                self.motion.time_steps+=1
                return {'CPU_fixture_only':True}
        class CPURunner:
            def __init__(self,env,mdp):
                self.env=env;self.mdp=mdp;self.device='cpu';self.iteration=0
                self.log_dir=output/'fixture_train'/'train';self.log_dir.mkdir(parents=True,exist_ok=True)
                self.buffer=SimpleNamespace(ptr=0,capacity=32,data={'u':torch.zeros(32,2,20),'c':torch.zeros(32,2,20)})
            def collect(self,steps):
                term=self.env.action_manager.get_term('joint_pos')
                for _ in range(steps):
                    self.env.step(torch.zeros(2,20));j=self.buffer.ptr
                    self.buffer.data['u'][j]=term._targets;self.buffer.data['c'][j]=term.processed_action
                    self.buffer.ptr=(j+1)%self.buffer.capacity
                return {'CPU_method_fixture':True}
            def save(self,path,next_iteration=None):
                p=Path(path);tmp=p.with_suffix('.tmp');torch.save({'format_version':3,'reference_signature':self.mdp.reference_signature,'modules':{'dummy':torch.ones(1)},'iteration':next_iteration or 0},tmp);tmp.replace(p)
            def load(self,*args,**kwargs):return 'native_load'
            def load_model_only(self,*args,**kwargs):return 'native_model_load'
        def memory(phase):
            m={k:torch.zeros(2,n) for k,n in PEN_MEMORY_DIMS.items()};m['end'][:]=16;m['time'][:]=phase;m['age'][:]=1;m['c1'][:]=.1;m['c2'][:]=-.2
            return m
        originals=(CPUEnv.__init__,CPUEnv.step,PenMDP.__init__,PenMDP.control_target,CPURunner.collect,CPURunner.save,CPURunner.load,CPURunner.load_model_only)
        rng=torch.get_rng_state().clone();envs=[];mdps=[];saved_metadata=None
        with rough_controller(CPUEnv,PenMDP,CPURunner,baseline,descriptor):
            for k in range(15):
                env=CPUEnv(prepared[k]);mdp=PenMDP(env.reader);envs.append(env);mdps.append(mdp);term=env.action_manager.get_term('joint_pos');entity=env.scene['robot']
                env.motion.time_steps[:]=k;term._processed_actions[:]=.1;m=memory(k);oldc1=m['c1'].clone();oldc2=m['c2'].clone()
                action=torch.full((2,20),8. if k%2 else -8.)
                residual,u=mdp.control_target(action,m,entity.data.encoder_bias)
                env.step(action)
                low,high=term.target_position_limits
                expected=(baseline[k]+residual).clamp(min=low,max=high).clamp(min=low+.02,max=high-.02)
                check=(torch.equal(u,expected) and torch.equal(term._targets,u) and torch.equal(self_actual:=env.sim.data.ctrl[:,term._position_ctrl_ids],u) and torch.equal(residual,term.processed_action) and torch.equal(m['c1'],oldc1) and torch.equal(m['c2'],oldc2) and torch.equal(m['frozen_target'],u))
                test(f'phase{k:02d}_native_real_imag_residual_reconstruction',lambda check=check:check)
                advance=DexSuperMimicRunner.advance_memory(SimpleNamespace(mdp=mdp),torch.zeros(2,58),m,action,residual,torch.zeros(2,20),torch.eye(3)[None].repeat(2,1,1))
                test(f'phase{k:02d}_memory_keeps_residual_not_u',lambda:torch.equal(advance['c1'],residual) and torch.equal(advance['c2'],oldc1))
                wm_in=WorldModel.inputs(None,torch.zeros(2,278),u);mdp.kp_default=torch.ones(20);mdp.kd_default=-torch.ones(20);mdp.force_limit=torch.ones(20)*3
                x=torch.zeros(2,58);context=torch.ones(2,46);torque=mdp.pd_torque(x,u,context)
                test(f'phase{k:02d}_WM_input_and_PD_actualu',lambda:torch.equal(wm_in[:,-20:],u) and torch.equal(torque,u.clamp(-3,3)))
                cases[f'phase{k:02d}_actual_u']=self_actual.detach().numpy();cases[f'phase{k:02d}_imagined_u']=u.detach().numpy()
            env=CPUEnv(prepared[15]);mdp=PenMDP(env.reader);envs.append(env);mdps.append(mdp);runner=CPURunner(env,mdp)
            runner.collect(3)
            test('allbatch_collect_cache_actual_replay_evidence',lambda:runner.rough_actualu_evidence['all_batch_exact'] and runner.rough_actualu_evidence['phase_histogram']==[2,2,2]+[0]*12 and all(v==0 for v in runner.rough_actualu_evidence['max_abs_errors'].values()))
            test('real_terminal_obs15_reject_before_action',lambda:raises(lambda:(env.motion.time_steps.fill_(15),env.step(torch.zeros(2,20)))))
            test('real_bias_nonzero_rejected',lambda:raises(lambda:(env.motion.time_steps.zero_(),env.scene['robot'].data.encoder_bias.fill_(.001),env.step(torch.zeros(2,20)))))
            env.scene['robot'].data.encoder_bias.zero_();env.motion.time_steps.zero_()
            test('imag_nonzero_bias_rejected',lambda:raises(lambda:mdp.control_target(torch.zeros(2,20),memory(0),torch.ones(2,20)*.001)))
            for bad in (-1,.5,float('nan')):
                test('imag_invalid_phase_'+str(bad),lambda bad=bad:raises(lambda:mdp.control_target(torch.zeros(2,20),memory(bad),torch.zeros(2,20))))
            test('nested_controller_rejected',lambda:raises(lambda:rough_controller(CPUEnv,PenMDP,CPURunner,baseline,descriptor).__enter__()))
            test('all_old_and_new_resume_paths_rejected',lambda:all(raises(lambda fn=fn:fn(output/'absent_old103.pt')) for fn in (runner.load,runner.load_model_only)))
            checkpoint=output/'fixture_model104.pt';native_torch_save=torch.save;runner.save(checkpoint,3);state=torch.load(checkpoint,map_location='cpu',weights_only=False);saved_metadata=state['rough_controller']
            test('ABI104_atomic_save_has_newsignature_and_actualu',lambda:state['format_version']==104 and 'c2_controller' not in state and state['reference_signature']==saved_metadata['reference_controller_signature'] and saved_metadata['latest_actualu_evidence']['all_batch_exact'] and not checkpoint.with_suffix('.tmp').exists() and torch.save is native_torch_save)
            test('descriptor_is_frozen_and_combinedsignature_not_native',lambda:saved_metadata['root_descriptor']==descriptor and saved_metadata['reference_controller_signature']!=saved_metadata['native_reference_signature'])
        test('all_classes_and_instance_process_methods_restored',lambda:(CPUEnv.__init__,CPUEnv.step,PenMDP.__init__,PenMDP.control_target,CPURunner.collect,CPURunner.save,CPURunner.load,CPURunner.load_model_only)==originals and all(not hasattr(env,'_rough_controller_active') for env in envs) and all(not hasattr(mdp,'rough_controller_metadata') for mdp in mdps))
        # Gradient case deliberately makes native qref hard-clamp every component;
        # only reconstruction from residual can preserve the interior derivative.
        lo,hi=prepared[16][2].target_position_limits;special=((lo[0]+hi[0])/2)[None].repeat(15,1);special[:,0]=hi[0,0]-.025
        with rough_controller(CPUEnv,PenMDP,CPURunner,special,descriptor):
            env=CPUEnv(prepared[16]);mdp=PenMDP(env.reader);m=memory(0);m['c1'].zero_();m['time'][1]=15
            action=torch.zeros(2,20,requires_grad=True);action.data[:,0]=.2
            with torch.autocast('cpu',dtype=torch.bfloat16):residual,u=mdp.control_target(action,m,torch.zeros(2,20))
            u.sum().backward();expected=torch.full((2,20),.25);expected[:,0]=0;expected[1]=0
            test('outerAMP_FP32_interior_saturated_inactive_gradients',lambda:u.dtype==torch.float32 and torch.equal(action.grad,expected))
            test('inactive_phase15_exact_finite_u14_not_extra_target',lambda:torch.equal(u[1],special[14]) and torch.isfinite(u).all())
            test('inactive_phase99_same_placeholder_and_no_live_state',lambda:torch.equal(mdp.control_target(torch.ones(2,20)*5,memory(99),torch.zeros(2,20))[1],special[14].expand(2,-1)))
            cases['gradient_actual']=action.grad.numpy();cases['gradient_expected']=expected.numpy()
        # Snapshot independence and exception restoration, no scientific solver.
        mutable_baseline=baseline.clone();mutable_descriptor=copy.deepcopy(descriptor)
        try:
            with rough_controller(CPUEnv,PenMDP,CPURunner,mutable_baseline,mutable_descriptor):
                mutable_baseline.fill_(999);mutable_descriptor['caller_mutation_after_enter']=True
                env=CPUEnv(prepared[17]);mdp=PenMDP(env.reader)
                test('caller_cannot_mutate_frozen_baseline_descriptor',lambda:torch.equal(mdp._rough_controller_baseline,baseline) and 'caller_mutation_after_enter' not in mdp.rough_controller_metadata['root_descriptor'])
                raise RuntimeError('intentional restoration fixture')
        except RuntimeError as exc:
            if str(exc)!='intentional restoration fixture':raise
        test('exception_restores_classes_and_global_torchsave',lambda:PenMDP.control_target==originals[3] and torch.save is native_torch_save)
        for bad in (torch.zeros(14,20),torch.full((15,20),float('nan'))):
            test('bad_baseline_shape_or_finite_rejected_'+str(tuple(bad.shape)),lambda bad=bad:raises(lambda:rough_controller(CPUEnv,PenMDP,CPURunner,bad,descriptor).__enter__()))
        test('descriptor_noncanonical_rejected',lambda:raises(lambda:rough_controller(CPUEnv,PenMDP,CPURunner,baseline,{'tensor':torch.ones(1)}).__enter__()))
        test('context_rng_unchanged',lambda:torch.equal(rng,torch.get_rng_state()))
        test('cuda_not_initialized_or_available',lambda:not torch.cuda.is_initialized() and not torch.cuda.is_available() and torch.cuda.device_count()==0)
        test('frozen_source_snapshots_unchanged',lambda:sha(controller_path)==frozen['controller']==sha(snap/'controller.py') and sha(__file__)==frozen['checker']==sha(snap/Path(__file__).name))
        np.savez_compressed(output/'native_method_cases.npz',**cases)
        report.update(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,check_count=len(checks),passed=sum(c['passed'] for c in checks),metadata=saved_metadata,bundle_file={'path':str(a.bundle.resolve()),'sha256':sha(a.bundle)},native_fixture={'path':str(fixture_path),'sha256':sha(fixture_path)},cases_sha256=sha(output/'native_method_cases.npz'),scope='Actual native ActionTerm/PenMDP/advance_memory/WM.inputs/PD methods on CPU fixtures; fake envstep/replay/save prove glue contracts only. No native realenv step, physics, learned WM forward, CUDA/AMP-CUDA or training qualification.',execution={'physics_steps':0,'solver_calls':0,'geometry_queries':0,'GPU':0,'training_updates':0,'cuda_initialized':torch.cuda.is_initialized()})
    except BaseException as exc:
        report.update(status='HARNESS_FAILURE',error=repr(exc),traceback=traceback.format_exc(),checks=checks)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report.get(k) for k in ('status','check_count','passed','error')},indent=2))
    if report['status']!='PASS':print(json.dumps([c for c in checks if not c['passed']],indent=2))
    raise SystemExit(report['status']!='PASS')
if __name__=='__main__':main()
