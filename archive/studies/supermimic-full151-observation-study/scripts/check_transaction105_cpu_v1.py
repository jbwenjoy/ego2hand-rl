"""Isolated105 transaction contracts on NumPy/Python surrogates only."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import importlib.util
import json
import random
import sys
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
SOURCE=ROOT/'src/full151_gradient/transaction105.py'
OUT=ROOT/'outputs/full151105_transaction_preparation_CPU_v1'
ATOMIC=Path('D:/workspace/supermimic-rough-prefix-learning-study/vendor/runtime_io.py')
ATOMIC_SHA='159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1'

def pin(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

class CPUObject:
    def __init__(self,**values):self.__dict__.update(values)
    def step(self,*args,**kwargs):raise AssertionError('No physics/step allowed')
    def forward(self,*args,**kwargs):raise AssertionError('No forward allowed')
    def compute(self,*args,**kwargs):raise AssertionError('No reward computation allowed')
    def process_actions(self,*args,**kwargs):raise AssertionError('No action processing allowed')
    def get_term(self,name):return self.terms[name]

class CPUReader(CPUObject):
    def memory(self):
        out={k:np.zeros((1,w),np.float32) for k,w in self.widths.items()}
        term=self.env.action_manager.get_term('joint_pos');motion=self.env.command_manager.get_term('motion')
        for k in ('c1','c2'):out[k]=term._processed_actions.copy()
        out['time']=motion.time_steps.astype(np.float32).reshape(1,1)
        out['end'][:]=151;out['frozen_target']=term._targets.copy()
        return out

class FakeSnapshot:
    """Deliberately no closures: exercises the production explicit sidecars."""
    constructors=0
    def __init__(self,root):
        type(self).constructors+=1;self.entries=[];self.arrays=[];self.tensors=[];self.restore_calls=0;seen=set()
        def walk(value):
            if id(value) in seen:return
            seen.add(id(value))
            if isinstance(value,np.ndarray):self.arrays.append((value,value.copy()));return
            if isinstance(value,dict):
                self.entries.append((value,dict(value),'dict'))
                for v in value.values():walk(v)
            elif isinstance(value,list):
                self.entries.append((value,list(value),'list'))
                for v in value:walk(v)
            elif isinstance(value,tuple):
                for v in value:walk(v)
            elif isinstance(value,CPUObject):
                self.entries.append((value,dict(vars(value)),'object'))
                for v in vars(value).values():walk(v)
        walk(root.env);walk(root.reader)
    def restore(self):
        self.restore_calls+=1
        for obj,saved,kind in self.entries:
            if kind=='dict':obj.clear();obj.update(saved)
            elif kind=='list':obj[:]=saved
            else:vars(obj).clear();vars(obj).update(saved)
        for target,saved in self.arrays:np.copyto(target,saved)

class FakeObserver:
    def __init__(self,env):
        self.env=env;self.stage='fixture';self.records=[];self.cumulative_bitmask=0;self.unknown_reads=0;self.finalized=False
    def set_stage(self,stage):self.stage=stage
    def _read(self,event):
        raw=int(self.env.sim.wp_data.overflow[0]);self.cumulative_bitmask|=max(0,raw)
        if raw<0:self.unknown_reads+=1
        record={'event':event,'stage':self.stage,'raw':raw,'OR':self.cumulative_bitmask,'unknown':self.unknown_reads}
        self.records.append(record);return record
    def _raise_if_unqualified(self,record):
        if self.cumulative_bitmask or self.unknown_reads:raise RuntimeError('Fake capacity unqualified')

def fixture(m,phase=149):
    z=lambda *shape:np.zeros(shape,np.float32)
    term=CPUObject(_raw_actions=z(1,20),_processed_actions=np.full((1,20),.01,np.float32),_targets=np.full((1,20),.02,np.float32),
        _targets_valid=np.ones(1,bool),_mask_active=np.zeros(1,bool),_mask_remaining=np.zeros(1,np.int64),_mask_dofs=np.zeros((1,20),bool))
    motion=CPUObject(time_steps=np.array([phase],np.int64),trajectory_ids=np.zeros(1,np.int64),_episode_start_steps=np.zeros(1,np.int64),_reset_prepared=np.ones(1,bool),
        _frames={'joint_vel':z(151,20),'object_linvel_w':z(151,3),'object_angvel_w':z(151,3)},
        _full151_runtime_native_velocity_diagnostics={'joint_vel':z(151,20)},_full151_runtime_velocity_authority={'frames':151})
    model=CPUObject(**{k:z(1,2,3) for k in m.ACTUAL8})
    data=CPUObject(**{k:z(1,2) for k in m.DATA_FIELDS if k not in ('contact','efc')})
    data.qpos=z(1,27);data.qvel=z(1,26);data.ctrl=z(1,20);data.time=np.array([phase*.02],np.float32)
    data.overflow=np.zeros(1,np.int32);data.nacon=np.zeros(1,np.int32);data.nefc=np.zeros(1,np.int32)
    data.contact=CPUObject(geom=np.zeros((2,2),np.int32));data.efc=CPUObject(force=z(1,3))
    history=CPUObject(_buffer=z(4,1,155),_pointer=np.zeros(1,np.int64))
    descriptor={'schema':'full151_knownrough_controller_input_descriptor:1','mode':m.MODE,'retarget_variant_id':m.VARIANT,'physical_reference_qualified':False}
    env=CPUObject(num_envs=1,sim=CPUObject(wp_model=model,wp_data=data),scene={'finger_collision':CPUObject(_cached_data=CPUObject(found=np.zeros((1,32),bool)),_cache_valid=True)},
        command_manager=CPUObject(terms={'motion':motion}),action_manager=CPUObject(terms={'joint_pos':term},_action=z(1,20),_prev_action=z(1,20),_prev_prev_action=z(1,20)),
        observation_manager=CPUObject(_obs_buffer={'policy':z(1,155)},_group_obs_term_delay_buffer={'policy':history},_group_obs_term_history_buffer={'policy':history},_group_obs_class_instances={}),
        termination_manager=CPUObject(_term_dones={'reference_end':np.array([phase==150])},_truncated_buf=np.array([phase==150]),_terminated_buf=np.zeros(1,bool)),reward_manager=CPUObject(),
        episode_length_buf=np.array([phase]),common_step_counter=phase,_sim_step_counter=4*phase,_manual_reset_pending=False,_command_dt=.02,obs_buf={'policy':z(1,155)},extras={},
        _full151_controller_pending=[{'phase':k} for k in range(phase)],_full151_controller_active=False,_full151_controller_process_calls=phase,
        _full151_controller_baseline=z(150,20),_full151_controller_lower=np.full(20,-1,np.float32),_full151_controller_upper=np.ones(20,np.float32),
        _full151_runtime_episode_generation=np.ones(1,np.int64),_full151_runtime_descriptor=descriptor,_full151_runtime_last_reset={'phase':np.zeros(1,np.int64)})
    if phase:env._full151_controller_last={'phase':phase-1,'residual_before_filter':term._processed_actions.copy()}
    metadata={'schema':'full151_knownrough_baseline_EMA_C2:1','format_version':105,'mode':m.MODE,'retarget_variant_id':m.VARIANT,
        'baseline_shape':[150,20],'active_intervals':[0,150],'terminal_observation':150,'root_descriptor':copy.deepcopy(descriptor),'reference_controller_signature':'fixture_signature'}
    reader=CPUReader(env=env,robot=CPUObject(),obj=CPUObject(),kin=CPUObject(),actuator_defaults={'kp':np.ones(20,np.float32)},middle=list(range(10)),distal=list(range(5)),tips=list(range(5)),
        palm_pos=z(1,3),palm_rot=np.eye(3,dtype=np.float32)[None],xi=z(1,46),reward_snapshot=None,widths=m.PEN_MEMORY_WIDTHS,
        mdp=CPUObject(motion=motion,full151_controller_metadata=metadata,_full151_controller_baseline=z(150,20),reference_signature='fixture_signature'))
    return env,reader,FakeObserver(env)

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();snap=OUT/'source_before';snap.mkdir()
    for p in (SOURCE,Path(__file__)): (snap/p.name).write_bytes(p.read_bytes())
    m=load('_transaction105_frozen_CPU_contract',SOURCE)
    checks={};details={}
    def check(name,value):
        if not value:raise AssertionError(name)
        checks[name]=True
    def rejects(name,fn):
        try:fn()
        except (m.TransactionError,m.UncoveredSidecarError,ValueError,RuntimeError) as exc:checks[name]=True;details[name]=repr(exc);return
        raise AssertionError('Expected rejection: '+name)
    def tx(phase=149,**kwargs):
        e,r,o=fixture(m,phase)
        t=m.Transaction105(e,r,capacity_observer=o,source_action=np.zeros((1,20),np.float32),label='CPU/'+str(phase),expected_phase=phase,snapshot_factory=FakeSnapshot,strict_native=False,**kwargs)
        return e,r,o,t
    e,r,o,t=tx()
    check('phase149_preaction_and_no_native_qualification',t.describe()['profile']['action_allowed'] and all(v=='PENDING' for v in t.describe()['qualification'].values()))
    e0,r0,o0,t0=tx(0);e0._full151_controller_last={'new_after_first_action':True};t0.begin('zero');check('phase0_absent_last_restored',not hasattr(e0,'_full151_controller_last'));t0.finish('zero')
    e150,r150,_=fixture(m,150);terminal=m.profile105(e150,r150,expected_phase=150,boundary='terminal_observation',strict_native=False)
    check('terminal150_inspection_not_action_bootstrap',not terminal['action_allowed'] and terminal['terminal_policy_bootstrap_allowed'] is False)
    rejects('terminal150_cannot_branch',lambda:m.Transaction105(e150,r150,capacity_observer=FakeObserver(e150),source_action=np.zeros((1,20),np.float32),label='terminal',expected_phase=150,snapshot_factory=FakeSnapshot,strict_native=False))
    rejects('strict_native_rejects_surrogates',lambda:m.profile105(e,r,expected_phase=149))
    for phase in (-1,151,True):rejects('invalid_phase_'+str(phase),lambda p=phase:m.profile105(e,r,expected_phase=p,strict_native=False))
    for field in ('_full151_runtime_episode_generation','_full151_controller_pending'):
        ee,rr,_=fixture(m);delattr(ee,field);rejects('missing_'+field,lambda:m.profile105(ee,rr,expected_phase=149,strict_native=False))
    for root,field in (('term','_targets_valid'),('data','sensordata'),('model','actuator_biasprm')):
        ee,rr,_=fixture(m);obj=ee.action_manager.get_term('joint_pos') if root=='term' else ee.sim.wp_data if root=='data' else ee.sim.wp_model
        delattr(obj,field);rejects('missing_'+root+'_'+field,lambda:m.profile105(ee,rr,expected_phase=149,strict_native=False))
    for mutation in ('old104','oldbaseline15','wrongcontext368','invalidtarget','activecapture','wrongphase','wrongsignature'):
        ee,rr,_=fixture(m)
        if mutation=='old104':rr.mdp.full151_controller_metadata['format_version']=104
        elif mutation=='oldbaseline15':ee._full151_controller_baseline=np.zeros((15,20),np.float32)
        elif mutation=='wrongcontext368':rr.xi=np.zeros((1,368),np.float32)
        elif mutation=='invalidtarget':ee.action_manager.get_term('joint_pos')._targets_valid[:]=False
        elif mutation=='activecapture':ee._full151_controller_active=True
        elif mutation=='wrongphase':ee.command_manager.get_term('motion').time_steps[:]=148
        else:rr.mdp.reference_signature='wrong'
        rejects(mutation,lambda:m.profile105(ee,rr,expected_phase=149,strict_native=False))
    term=e.action_manager.get_term('joint_pos');original=t.verify_current()['value_inventory_sha256']
    random_before=random.getstate();numpy_before=np.random.get_state()
    term._raw_actions[:]=7;term._processed_actions[:]=8;term._targets[:]=9;term._targets_valid[:]=False
    e.sim.wp_data.qacc_warmstart[:]=3;e.sim.wp_model.actuator_gainprm[:]=4
    e.observation_manager._group_obs_term_delay_buffer['policy']._buffer[:]=5
    e.scene['finger_collision']._cached_data.found[:]=True;e._full151_runtime_episode_generation[:]=99
    e._full151_controller_pending.append({'new':True});random.random();np.random.random()
    receipt=t.begin('restore_all');check('rawprocessed_cache_model_warmstart_obs_sensor_generation_restore',receipt['inventory']['bitexact'] and t.verify_current()['value_inventory_sha256']==original)
    check('Python_NumPy_RNG_restored',random.getstate()==random_before and np.array_equal(np.random.get_state()[1],numpy_before[1]) and np.random.get_state()[2:]==numpy_before[2:])
    t.finish('restore_all');check('external_ledger_records_survive_restore',len(o.records)==2 and o.records[0]['event']=='before_restore' and o.records[1]['event']=='after_branch')
    rejects('duplicate_branch_no_retry',lambda:t.begin('restore_all'))
    check('source_action_readonly',not t.source_action.flags.writeable)
    for name,alias in (('observer','observer'),('records_alias','records')):
        ee,rr,oo=fixture(m);ee.extras['bad']=oo if alias=='observer' else oo.records;before=FakeSnapshot.constructors
        rejects('external_'+name+'_rejected_before_snapshot',lambda:m.Transaction105(ee,rr,capacity_observer=oo,source_action=np.zeros((1,20),np.float32),label='bad',expected_phase=149,snapshot_factory=FakeSnapshot,strict_native=False))
        check('external_'+name+'_factory_not_called',FakeSnapshot.constructors==before)
    for raw in (2,-1):
        ee,rr,oo,tt=tx();ee.sim.wp_data.overflow[0]=raw;rejects('capacity_pre_restore_'+str(raw),lambda:tt.begin('cap'))
        check('capacity_'+str(raw)+'_no_restore_no_clear',tt.native.restore_calls==0 and int(ee.sim.wp_data.overflow[0])==raw and tt.failed)
    ee,rr,oo,tt=tx()
    try:
        with tt.branch('body_error'):
            ee.sim.wp_data.overflow[0]=-1
            raise ValueError('original body failure')
    except ValueError as exc:check('body_error_preserved_with_secondary_capacity_unknown',str(exc)=='original body failure' and bool(exc.__notes__) and tt.active is None and tt.failed)
    rejects('failed_body_cannot_restart',lambda:tt.begin('new'))
    # The fake snapshot intentionally omits closure data; explicit sidecars own it.
    ee,rr,oo=fixture(m);box={'counter':np.array([1],np.int64)}
    def closure_step():return box['counter']
    ee.sim.step=closure_step
    tt=m.Transaction105(ee,rr,capacity_observer=oo,source_action=np.zeros((1,20),np.float32),label='closure',expected_phase=149,snapshot_factory=FakeSnapshot,strict_native=False)
    box['counter'][0]=9;tt.begin('closure');check('explicit_closure_array_restored',box['counter'][0]==1 and tt.describe()['closure_cells']>0);tt.finish('closure')
    class Foreign:
        def __init__(self):self.value=1
    ee,rr,oo=fixture(m);foreign=Foreign()
    def bad_hook():return foreign.value
    ee.sim.step=bad_hook
    rejects('unregistered_foreign_closure_rejects',lambda:m.Transaction105(ee,rr,capacity_observer=oo,source_action=np.zeros((1,20),np.float32),label='foreign',expected_phase=149,snapshot_factory=FakeSnapshot,strict_native=False))
    ee,rr,oo=fixture(m);extra=np.array([object()],object)
    rejects('object_dtype_sidecar_rejects',lambda:m.Transaction105(ee,rr,capacity_observer=oo,source_action=np.zeros((1,20),np.float32),label='object',expected_phase=149,snapshot_factory=FakeSnapshot,strict_native=False,extra_sidecars={'bad':extra}))
    ee,rr,oo,tt=tx();ee.sim.wp_data.qpos[:]=2;tt.native.restore=lambda:None
    rejects('incomplete_restore_detected',lambda:tt.begin('partial'))
    # Parse native width declarations without importing Torch or native classes.
    native=NATIVE=Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic')
    state_ast=ast.parse((native/'state.py').read_text(encoding='utf-8-sig'));pen_ast=ast.parse((native/'pen.py').read_text(encoding='utf-8-sig'))
    widths=next(ast.literal_eval(n.value) for n in state_ast.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='MEMORY_DIMS' for v in n.targets))
    call=next(n.value for n in pen_ast.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='PEN_MEMORY_DIMS' for v in n.targets))
    widths.update({k.arg:ast.literal_eval(k.value) for k in call.keywords})
    check('PEN_widths_match_installed_source_AST',widths==m.PEN_MEMORY_WIDTHS)
    check('no_Torch_Warp_MuJoCo_Env_import',not any(name in sys.modules for name in ('torch','warp','mujoco','mjlab')))
    source_pins={'transaction105':pin(SOURCE),'checker':pin(__file__),'predecessor':pin(m.PREDECESSOR),'capacity':pin(m.CAPACITY_SOURCE),
        'native_snapshot':pin(native/'snapshot.py'),'native_pen':pin(native/'pen.py'),'native_state':pin(native/'state.py'),
        'controller105':pin(ROOT/'src/full151_native/controller.py'),'runtime105':pin(ROOT/'src/full151_native/runtime.py'),
        'native105':pin(ROOT/'src/full151_native/native.py'),'atomic_writer159c':pin(ATOMIC)}
    check('frozen_capacity_and_writer_exact',source_pins['capacity']['sha256']==m.CAPACITY_SHA256 and source_pins['atomic_writer159c']['sha256']==ATOMIC_SHA)
    writer=load('_transaction105_qualified_writer',ATOMIC)
    report={'schema':'full151105_transaction_preparation_CPU_contract:1','status':'PASS_CPU_SURROGATE_INVENTORY_ONLY','passed':len(checks),'total':len(checks),'checks':checks,'details':details,
        'source_pins':source_pins,'API':{'profile105':'profile105(env,reader,expected_phase,boundary=pre_action|terminal_observation,strict_native=True)',
        'Transaction105':'Transaction105(env,reader,capacity_observer,source_action FP32[1,20],label,expected_phase,external_roots,extra_sidecars)',
        'methods':['begin(branch_id)','finish(branch_id,failure=None)','branch(branch_id)','verify_current()','describe()']},
        'profile149':t.describe(),'terminal_profile150':terminal,'required_future_qualification':{k:'PENDING' for k in m.NATIVE_PENDING},
        'framework_imports':[],'actual_Env_instances':0,'actual_model_loads':0,'actual_physics_steps':0,'WM_calls':0,'GPU_calls':0,
        'surrogate_scope':'Python/NumPy fake graph, fake snapshot and fake capacity backend only. No native restore/full-state baseline/long-clock/GPU qualification.'}
    writer.write_json(OUT/'report.json',report)
    ready={'status':'CPU_SURROGATE_PREPARATION_ONLY','source_pins':source_pins,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':len(checks),'total':len(checks)},
        'api':report['API'],'inventory_roots':list(t.describe()['profile']['required_fields']),'pending':report['required_future_qualification'],
        'integration_warning':'Frozen observed_capacity attaches ledger to env; future serial worker must use standalone host observer, never monkey-patch old wrapper.'}
    writer.write_json(OUT/'READY.json',ready)
    print(json.dumps({'status':report['status'],'passed':len(checks),'total':len(checks),'report':pin(OUT/'report.json'),'READY':pin(OUT/'READY.json')}))

if __name__=='__main__':main()
