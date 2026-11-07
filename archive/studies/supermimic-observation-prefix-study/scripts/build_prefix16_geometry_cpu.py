"""CPU prefix16 producer. Default --check never solves; --run needs a Root plan."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json
STARTED=time.perf_counter()

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path)
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--run',action='store_true');modes.add_argument('--check',action='store_true');args=ap.parse_args()
    planpath=args.plan.resolve(strict=True);rawplan=json.loads(planpath.read_text(encoding='utf-8-sig'))
    if args.run and rawplan.get('optimization_authorized') is not True:raise ValueError('Root optimization authorization absent; no output/solve')
    if args.run and args.output is None:raise ValueError('--output required for --run')
    out=args.output.resolve() if args.output else None
    if out is not None:out.mkdir(parents=True,exist_ok=False)
    runtime={'schema':'prefix16_geometry_worker_runtime:1','status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'started_perf_counter':STARTED,'plan_sha256':sha(planpath),'entry_sha256':sha(__file__),'optimization_started':False,'optimization_runs':0,'physics_steps':0,'GPU_used':False,'jac_calls':0,'fun_calls':0,'callback_calls':0,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'OMP_NUM_THREADS':os.environ['OMP_NUM_THREADS'],'MKL_NUM_THREADS':os.environ['MKL_NUM_THREADS']}
    if out is not None:write_json(out/'runtime.json',runtime)
    # PID and parent metadata already committed before NumPy/SciPy imports so
    # the reviewed redirector-aware supervisor can identify the actual child.
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import mujoco
    from prefix_study.inputs import checked_plan
    from prefix_study.problem import Prefix16GeometryProblem
    from prefix_study.io import atomic_npz,atomic_jsonl
    forbidden={name:0 for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce')}
    original_geometry_guard={name:getattr(mujoco,name) for name in forbidden}
    def deny(name):
        def fail(*a,**kw):forbidden[name]+=1;raise AssertionError('Geometry-only worker forbids '+name)
        return fail
    for name in forbidden:setattr(mujoco,name,deny(name))
    runtime['forbidden_calls']=forbidden
    def deadline():
        if time.perf_counter()-STARTED>110:raise TimeoutError('110s whole-worker softdeadline; preserve iterate, no retry')
    def event_sink(event):
        if out is not None:write_json(out/'collision_events'/f"{event['query_index']:07d}.json",event)
    problem=None;history=[];error=None;result=None;first_formal_x=None
    def checkpoint(x,source,*,accepted=False):
        nonlocal first_formal_x
        deadline()
        if out is None:return
        arrays=problem.diagnostic_arrays(x)
        if first_formal_x is not None:arrays['solver_first_fun_x']=first_formal_x
        wall=time.perf_counter()-STARTED
        arrays.update(checkpoint_plan_sha256=np.array(sha(planpath)),checkpoint_protocol_sha256=np.array(p['protocol']['sha256']),checkpoint_pid=np.array(os.getpid()),
                      checkpoint_parent_pid=np.array(os.getppid()),checkpoint_wall_seconds=np.array(wall),checkpoint_source=np.array(source),checkpoint_accepted=np.array(accepted))
        stem='latest_accepted_iterate' if accepted else 'latest_iterate'
        atomic_npz(out/(stem+'.npz'),**arrays)
        header={'source':source,'accepted_iterate':accepted,'pid':os.getpid(),'parent_pid':os.getppid(),'plan_sha256':sha(planpath),'protocol_sha256':p['protocol']['sha256'],'wall_seconds':wall,'checkpoint_npz_sha256':sha(out/(stem+'.npz')),'cost':float(.5*arrays['residual']@arrays['residual']),'counts':problem.counters(),'jac_calls':runtime['jac_calls'],'fun_calls':runtime['fun_calls'],'formal_optimization_started':runtime['optimization_started'],'hardq0_bitexact':bool(np.array_equal(arrays['q_goal_rad'][0],problem.q0))}
        write_json(out/(stem+'.json'),header)
        runtime.update(status='optimizing' if runtime['optimization_started'] else 'checking',wall_seconds=time.perf_counter()-STARTED,latest_checkpoint=header)
        write_json(out/'runtime.json',runtime)
    try:
        p,protocol=checked_plan(planpath,run=args.run)
        if p.get('output') is not None and out is not None and Path(p['output']).resolve()!=out:raise ValueError('Output differs from plan')
        if out is not None:
            write_json(out/'plan_snapshot.json',p);write_json(out/'protocol_snapshot.json',protocol)
        deadline();problem=Prefix16GeometryProblem(protocol,event_sink=event_sink,deadline=deadline)
        with problem.stage('preflight_initial16'):
            r,J,_=problem.evaluate(problem.x0)
            if out is not None:checkpoint(problem.x0,'preflight_x0')
        initial=problem.diagnostic_summary(problem.x0)
        if out is not None:write_json(out/'input_and_initial16_checks.json',initial)
        if not args.run:
            runtime.update(status='checked_no_optimization',wall_seconds=time.perf_counter()-STARTED,counts=problem.counters())
            if out is not None:write_json(out/'runtime.json',runtime)
            print(json.dumps({'status':'CPU_CHECK_ONLY_NO_OPTIMIZATION','checks':problem.data['checks'],'q_shape':[16,20],'free_variables':300,'residual_shape':list(r.shape),'jacobian_shape':list(J.shape),'jacobian_format':J.format,'hardq0_exact':initial['hardq0_bitexact'],'requested_x0_all_repeated_q0':True,'counts':problem.counters(),'forbidden_calls':forbidden,'Torch_imported':'torch'in sys.modules,'Warp_imported':'warp'in sys.modules or 'mujoco_warp'in sys.modules,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'optimization_runs':0,'physics_steps':0,'GPU_used':False},indent=2));return
        from scipy.optimize import least_squares
        runtime.update(status='optimizing',optimization_started=True,optimization_runs=1)
        write_json(out/'runtime.json',runtime)
        def fun(x):
            nonlocal first_formal_x
            deadline();runtime['fun_calls']+=1
            if first_formal_x is None:
                first_formal_x=np.array(x,copy=True)
                atomic_npz(out/'solver_initialization.npz',requested_x0=problem.x0,solver_first_fun_x=first_formal_x,free_initialization_shift=first_formal_x-problem.x0,q0_source64=problem.q0)
            with problem.stage('formal_fun'):value=problem.evaluate(x)[0]
            history.append({'kind':'fun','call':runtime['fun_calls'],'wall_s':time.perf_counter()-STARTED,'cost':float(.5*value@value),'x_max_abs_from_requested_x0':float(np.max(np.abs(x-problem.x0)))})
            return value.copy()
        def jac(x):
            deadline();runtime['jac_calls']+=1
            with problem.stage('formal_jac'):
                value=problem.evaluate(x)[1];checkpoint(x,'jac')
            atomic_jsonl(out/'optimizer_iterations.jsonl',history)
            return value.copy()
        def callback(intermediate_result):
            deadline();runtime['callback_calls']+=1
            # Do not trigger uncounted collision work just to log an accepted
            # callback. The latest jac checkpoint already contains diagnostics.
            atomic_npz(out/'accepted_solver_x.npz',x=intermediate_result.x,q_goal_rad=problem.q_from_x(intermediate_result.x),residual=intermediate_result.fun)
            write_json(out/'accepted_solver_x.json',{'source':'scipy_callback','callback_calls':runtime['callback_calls'],'plan_sha256':sha(planpath),'pid':os.getpid(),'wall_seconds':time.perf_counter()-STARTED,'scope':'Accepted x/residual only; full latest_iterate remains jac-labelled.'})
        result=least_squares(fun,problem.x0.copy(),jac=jac,bounds=(problem.lower,problem.upper),method='trf',tr_solver='lsmr',max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8,callback=callback)
        with problem.stage('formal_final_diagnostics'):
            problem.evaluate(result.x);checkpoint(result.x,'final',accepted=True)
            atomic_npz(out/'candidate.npz',**problem.diagnostic_arrays(result.x))
            write_json(out/'native_contacts.json',problem.native_records(result.x))
            report=problem.diagnostic_summary(result.x)
        report.update(status='FORMAL_CPU_GEOMETRY_SOLVE_COMPLETED_ONLY',optimizer_success=bool(result.success),optimizer_status=int(result.status),optimizer_message=str(result.message),nfev=int(result.nfev),njev=int(result.njev),optimality=float(result.optimality),candidate_sha256=sha(out/'candidate.npz'),solver_first_fun_x_max_delta_from_requested=float(np.max(np.abs(first_formal_x-problem.x0))),solver_free_x0_interior_shift_scope='SciPy TRF may move only the300 freevars inward; source64 q0 is outside solvervector and never clipped.',plan_sha256=sha(planpath),protocol_sha256=p['protocol']['sha256'],entry_sha256=sha(__file__),optimization_runs=1,physics_steps=0,GPU_used=False,curve_controller_physical_qualification='PENDING')
        write_json(out/'report.json',report);runtime['status']='completed_formal_CPU_geometry_only'
        write_json(out/'manifest.json',{'schema':'prefix16_geometry_candidate:1','status':'FORMAL_CPU_GEOMETRY_ONLY_NOT_PHYSICS_QUALIFIED','protocol':p['protocol'],
                   'candidate':{'path':str(out/'candidate.npz'),'sha256':sha(out/'candidate.npz')},'report':{'path':str(out/'report.json'),'sha256':sha(out/'report.json')},
                   'source_identity':protocol['source_identity'],'q0_source':protocol['inputs']['canonical_H123_q0'],'observation':protocol['inputs']['observation_fixture'],
                   'mapped_pose_parent':protocol['inputs']['frozen_full151_geometry'],'frozen_material_anchors':protocol['inputs']['frozen_material_anchors'],
                   'input_array_read_log':problem.data['read_log'],'optimizer_success':bool(result.success),'hardq0_bitexact':report['hardq0_bitexact'],
                   'curve_controller_qualification':'PENDING','physical_reference_qualified':False,'qd_or_u_exported':False})
        print(json.dumps({k:report[k] for k in ('status','optimizer_success','nfev','njev','cost','hardq0_bitexact','q_goal_hard_excess_max_rad','counts')},indent=2))
    except BaseException as exc:
        error=exc;runtime.update(status='failed_engineering_partial',error_type=type(exc).__name__,error=str(exc))
        if out is not None:write_json(out/'failure.json',{**runtime,'latest_iterate_may_exist':(out/'latest_iterate.npz').exists(),'no_retry':True})
        raise
    finally:
        for name,method in original_geometry_guard.items():setattr(mujoco,name,method)
        runtime.update(forbidden_calls=forbidden,Torch_imported='torch'in sys.modules,Warp_imported='warp'in sys.modules or 'mujoco_warp'in sys.modules)
        if out is not None:
            if problem is not None:
                runtime['counts']=problem.counters();atomic_jsonl(out/'collision_events.jsonl',problem.events)
            atomic_jsonl(out/'optimizer_iterations.jsonl',history)
            runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write_json(out/'runtime.json',runtime)
if __name__=='__main__':main()
