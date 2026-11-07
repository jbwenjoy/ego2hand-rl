"""Predeclared no-learning control-authority audit under the exact task map.

All65 schedules and both dt retain the original initial state/physics/gates.
The two zero cases must reproduce every qualified trace byte before any new
schedule executes. A finite failed search is not proof of impossibility.
"""
from pathlib import Path
import argparse,datetime,hashlib,itertools,json,os,sys,time,traceback
import numpy as np
from evaluate_online import preflight

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.learning import online_contract as C,runner as R
from supermimic_research.runtime import native_evaluation as E


def schedules():
    result=[('zero',np.zeros((100,20),dtype=np.float64))]
    def constant(name,vector):result.append((name,np.broadcast_to(np.asarray(vector,dtype=np.float64),(100,20)).copy()))
    for axis in range(20):
        for sign in (-1,1):
            vector=np.zeros(20);vector[axis]=sign;constant(f'axis{axis:02d}_{sign:+d}',vector)
    for block in range(5):
        for sign in (-1,1):
            vector=np.zeros(20);vector[4*block:4*block+4]=sign;constant(f'block{block}_{sign:+d}',vector)
    for signs in itertools.product((-1,1),repeat=3):
        vector=np.zeros(20)
        for block,sign in enumerate(signs):vector[4*block:4*block+4]=sign
        constant('first3blocks_'+'_'.join(f'{s:+d}' for s in signs),vector)
    for sign in (-1,1):constant(f'all20_{sign:+d}',np.full(20,sign))
    t=(np.arange(100,dtype=np.float64)+.5)*.02;j=np.arange(20,dtype=np.float64)
    sin=np.sin(2*np.pi*(.5+j/20)[None,:]*t[:,None]+j[None,:]*.37)
    cos=np.cos(2*np.pi*(.75+j/40)[None,:]*t[:,None]+j[None,:]*.19)
    result.extend((name,array) for name,array in zip(('sin_plus','sin_minus','cos_plus','cos_minus'),(sin,-sin,cos,-cos)))
    C.need(len(result)==65 and len({n for n,_ in result})==65,'Fixed65 distinct names')
    C.need(all(a.shape==(100,20) and a.dtype==np.float64 and np.isfinite(a).all() and (abs(a)<=1).all() for _,a in result),'Finite legal100x20 action arrays')
    return result


class ScheduleController:
    def __init__(self,task,actions,dt):
        self.task=task;self.actions=actions;self.dt=dt;self.memory=None;self.records=[]
    def __call__(self,tick,dt,model,data):
        import torch
        C.need(dt==self.dt and tick%round(.02/dt)==0,'Explicit native control tick')
        k=tick//round(.02/dt)
        if self.memory is None:
            like=torch.tensor(self.task.b['initial']['native_PRE_state58_F64'],dtype=torch.float64)
            self.memory=self.task.initial_memory(like,native_h_s=dt,n_substeps=round(.02/dt))
        C.need(int(self.memory['step'])==k and k<100,'Exact schedule and memory clock')
        action=torch.tensor(self.actions[k],dtype=torch.float64)
        with torch.no_grad():residual,u=self.task.control_target(action,self.memory)
        self.records.append(dict(pre_tick=tick,command_index=k,memory83_before=self.task.flatten_memory(self.memory).numpy().tolist(),
            action20_F64=action.numpy().tolist(),residual20_F64=residual.numpy().tolist(),executed_u20_F64=u.numpy().tolist()))
        self.memory=self.task.advance_memory(self.memory,action,residual,u)
        return u.numpy()


def compare_trace(reference_pin,actual_path,registry):
    with np.load(registry.resolve_pin(reference_pin),allow_pickle=False) as reference,np.load(actual_path,allow_pickle=False) as actual:
        C.need(set(reference.files)==set(actual.files),'Exact baseline trace fields')
        return {key:reference[key].dtype==actual[key].dtype and reference[key].shape==actual[key].shape and reference[key].tobytes()==actual[key].tobytes() for key in reference.files}


def static_force_balance(rows,COM,mass=.2,mu=.7,radius=.1,rays=4):
    """Unbounded nonnegative point-force LP; explicit ideal-cone approximation.

    This omits PD/force limits, evolving geometry and native compliance. It is
    not the MuJoCo cone algorithm or a dynamic grasp/holding certificate.
    """
    from scipy.optimize import linprog
    columns=[];contacts=[];observed=np.zeros(6)
    for row in np.asarray(rows,dtype=np.float64).reshape(-1,24):
        g1,g2=map(int,row[1:3])
        if 71 not in (g1,g2):continue
        C.need(int(row[3])==3,'This point-force diagnostic supports actual condim3 only')
        sign=1 if g2==71 else -1;frame=row[9:18].reshape(3,3);position=row[6:9];lever=position-COM
        f=sign*frame.T@row[18:21];direct=sign*frame.T@row[21:24];observed+=np.r_[f,np.cross(lever,f)+direct]
        normal=sign*frame[0]
        contacts.append(dict(other_geom=g1 if g2==71 else g2,position_world_m=position.tolist(),normal_on_phone_world=normal.tolist(),
            actual_Fn_N=float(row[18]),actual_force_on_phone_world_N=f.tolist(),
            ideal_mu_max_upward_per_normal_force=float(normal[2]+mu*np.sqrt(frame[1,2]**2+frame[2,2]**2))))
        for theta in np.arange(rays)*2*np.pi/rays:
            direction=sign*(frame[0]+mu*(np.cos(theta)*frame[1]+np.sin(theta)*frame[2]))
            columns.append(np.r_[direction,np.cross(lever,direction)])
    C.need(bool(columns) and radius>0,'Observed contacts and declared torque scaling')
    matrix=np.array(columns).T;scaled=matrix.copy();scaled[3:]/=radius;target=np.array([0.,0.,mass*9.81,0.,0.,0.])
    result=linprog(np.ones(matrix.shape[1]),A_eq=scaled,b_eq=target,bounds=(0,None),method='highs')
    residual=matrix@result.x-target if result.success else None
    return dict(rays_per_contact=rays,mu_assumption=mu,moment_row_scale_m=radius,observed_contacts=contacts,
        observed_contact_wrench_world=observed.tolist(),gravity_force_world_N=[0.,0.,-mass*9.81],
        initial_net_force_world_N=(observed[:3]+[0,0,-mass*9.81]).tolist(),LP_success=bool(result.success),LP_status=int(result.status),LP_message=result.message,
        minimum_sum_normal_coefficients_N=float(result.fun) if result.success else None,
        force_residual_world_N=residual[:3].tolist() if residual is not None else None,moment_residual_world_Nm=residual[3:].tolist() if residual is not None else None,
        scope='Sampled ideal isotropic Coulomb point-force cone, nonnegative/unbounded forces. Not the exact native pyramidal cone, actuator-feasible solution, state-derivative or dynamic certificate.')


def run(args):
    path,c,registry,root,final,health,zero,binding,out=preflight(args)
    plan=schedules();out.mkdir(parents=True);tick=time.perf_counter();phase='plan';cases=[];barrier=[]
    actions=np.stack([a for _,a in plan])
    with (out/'planned_actions.npz').open('xb') as stream:np.savez_compressed(stream,normalized_action65x100x20_F64=actions)
    intent=dict(schema='supermimic_research.control_authority_intent:1',tool=file_pin(__file__),configuration=file_pin(path),
        data_role='NO_FITTING_CONTROL_AUTHORITY_DIAGNOSTIC',training_authorized=False,source_pins=C.source_pins(),
        task_bundle=registry.pin(c['artifacts']['task']),model=registry.pin(c['artifacts']['model']),binding=file_pin(binding),
        qualified_zero_report=file_pin(args.zero_report),plan=file_pin(out/'planned_actions.npz'),
        schedules=[dict(index=i,name=name,action_bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest()) for i,(name,a) in enumerate(plan)],
        dt_s=E.DTS,native_gates=E.GATES,dt_pair_gates=E.PAIR_GATES,maximum_cases=130,maximum_native_steps=65*(6400+12800),maximum_control_calls=13000,
        zero_barrier='All qualified trace arrays at both dt must match typed raw bytes before any nonzero action case',
        automatic_retry=False,profile_selection_after_execution=False,
        scope='Closed action-interface box with original0.05rad residual/EMA. Exogenous sequences, no policy/checkpoint load or reference tracking claim. Finite failure is not impossibility.')
    C.write(out/'intent.json',intent)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch,mujoco
    import scipy.optimize  # Initialize the post-analysis dependency before the RNG snapshot.
    from supermimic_research.learning import world_model as W
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
    counts=dict(MJB_reads=1,MjData=0,initial_forward=0,integration_steps=0,contact_force_reads=0,controller_calls=0,controller_receipt_reads=0,NN_FW=0,observer_forward=0,learning_optimizer_steps=0,GPU_calls=0,static_LP_calls=0)
    try:
        model=mujoco.MjModel.from_binary_path(str(registry.resolve(c['artifacts']['model'])));phone,pg,table,pads=E.assert_model(model,mujoco,np,C.read(binding))
        digest=E.model_digest_except_dt(model,np);rng=W.capture_rng();bundle=C.read(registry.resolve(c['artifacts']['task']));variants=[];endpoints={};zero_contacts=[]
        for index,(name,action) in enumerate(plan):
            if index:C.need(len(barrier)==2 and all(all(v['arrays'].values()) for v in barrier),'All-zero global barrier before new physical controls')
            variant=E.make_variant(name=name,qpos27=bundle['initial']['qpos27_F64'],ctrl20=bundle['initial']['ctrl20_F64'],control_bounds20x2=model.actuator_ctrlrange,
                initial_state_provenance=dict(task_bundle=intent['task_bundle'],qualified_zero_report=intent['qualified_zero_report']))
            variants.append(variant);endpoints[name]=[]
            for dt in E.DTS:
                phase=f'{index:02d}_{name}_dt{dt}';folder=out/phase;folder.mkdir()
                task=R._task(registry,c,'A1_tracking',None);controller=ScheduleController(task,action,dt)
                summary,endpoint=E.evaluate_case_controlled(model,variant,dt,folder,phone,pg,table,pads,mujoco,np,counts,controller)
                C.write(folder/'controller_records.json',controller.records);endpoints[name].append(endpoint)
                if index==0:
                    reference=next(case for case in zero['cases'] if case['dt_s']==dt)
                    comparison=compare_trace(reference['trace'],folder/'trace.npz',registry)
                    C.need(all(comparison.values()),'Zero baseline raw parity failure; stop before any new nonzero schedule')
                    barrier.append(dict(dt_s=dt,reference=reference['trace'],trace=summary['trace'],arrays=comparison))
                    with np.load(folder/'trace.npz',allow_pickle=False) as z:COM=z['initial_COM_world3'].copy()
                    zero_contacts.append((dt,summary['initial_native']['contact_rows24'],COM))
                    if len(barrier)==2:C.write(out/'zero_barrier.json',dict(all_passed=True,cases=barrier,nonzero_cases_executed_before_barrier=0))
                C.need(counts['integration_steps']<=intent['maximum_native_steps'] and counts['controller_calls']<=intent['maximum_control_calls'],'Actual finite budget')
                cases.append(dict(schedule_index=index,schedule=name,dt_s=dt,native_summary=summary,controller_records=file_pin(folder/'controller_records.json')))
                C.write(out/f'completed_case{len(cases):03d}.json',dict(schedule=name,dt_s=dt,summary=file_pin(folder/'summary.json')))
            print(json.dumps(dict(index=index,schedule=name,steps=[case['native_summary']['finite_steps_saved'] for case in cases[-2:]],passes=[case['native_summary']['nominal_holding_dynamics_pass'] for case in cases[-2:]])),flush=True)
        pairs=E.dt_pair_summary(variants,[case['native_summary'] for case in cases],endpoints,np);static=[]
        for dt,rows,COM in zero_contacts:
            for rays in (4,16):static.append(dict(dt_s=dt,analysis=static_force_balance(rows,COM,radius=bundle['support']['radius_m'],rays=rays)));counts['static_LP_calls']+=1
        C.write(out/'initial_contact_force_balance.json',static)
        checks=dict(all_fixed_cases_recorded=len(cases)==130,zero_global_barrier=len(barrier)==2 and all(all(v['arrays'].values()) for v in barrier),
            original_model_unchanged_except_dt=digest==E.model_digest_except_dt(model,np),all_schedule_bytes_unchanged=all(hashlib.sha256(actions[i].tobytes()).hexdigest()==item['action_bytes_sha256'] for i,item in enumerate(intent['schedules'])),
            RNG_unchanged=W.raw_tree_equal(rng,W.capture_rng()),CPU_only=not torch.cuda.is_initialized(),
            original_sources_unchanged=C.fingerprint(intent['source_pins'])==C.fingerprint(C.source_pins()),finite_declared_budget=counts['integration_steps']<=intent['maximum_native_steps'])
        C.need(all(checks.values()),'Control-authority integrity failure')
        report=dict(schema='supermimic_research.control_authority_result:1',status='ALL_FIXED_SCHEDULES_RECORDED_NOT_PROOF_OF_IMPOSSIBILITY',intent=file_pin(out/'intent.json'),
            checks=checks,passed=sum(checks.values()),total=len(checks),cases=cases,dt_pairs=pairs,counts=counts,
            individual_cases_passing=sum(case['native_summary']['nominal_holding_dynamics_pass'] for case in cases),
            paired_schedules_passing=[p['name'] for p in pairs if p['both_tripod_cases_pass'] and p['dt_pair_consistency_pass']],
            initial_contact_analysis=file_pin(out/'initial_contact_force_balance.json'),elapsed_s=time.perf_counter()-tick,
            NN_constructors=0,own_checkpoint_loads=0,learning_updates=0,policy_benefit_qualified=False,data_role=intent['data_role'],scope=intent['scope'])
        C.write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),passing_schedules=report['paired_schedules_passing'],counts=counts)),flush=True);return report
    except BaseException as error:C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),counts=counts,completed_cases=len(cases),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--zero-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:
        started=datetime.datetime.now(datetime.timezone.utc).isoformat();code=1
        try:run(a);code=0
        finally:
            if a.output.exists():C.write(a.output/'terminal.json',dict(pid=os.getpid(),returncode=code,command=sys.argv,tool=file_pin(__file__),started_UTC=started,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False))
    else:
        *_,out=preflight(a);print(json.dumps(dict(status='METADATA_ONLY_CONTROL_AUTHORITY_READY',schedules=len(schedules()),cases=130,maximum_native_steps=65*19200,output=str(out),Torch_imported='torch' in sys.modules,MuJoCo_imported='mujoco' in sys.modules)))
