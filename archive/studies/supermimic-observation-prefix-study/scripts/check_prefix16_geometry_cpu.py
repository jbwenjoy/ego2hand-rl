"""Small CPU semantic/CSR/FD checks, no least_squares or entrypoint execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import argparse,ast,copy,inspect,json,sys,time
import numpy as np
import mujoco
import scipy.optimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prefix_study.inputs import checked_protocol,checked_plan,PROTOCOL,PROTOCOL_SHA
from prefix_study.problem import Prefix16GeometryProblem
from prefix_study.io import sha,write_json,atomic_npz


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/producer_CPU_checks_v1');args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    forbidden={n:0 for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce')}
    def deny(n):
        def f(*a,**kw):forbidden[n]+=1;raise AssertionError('No physics/force API '+n)
        return f
    for n in forbidden:setattr(mujoco,n,deny(n))
    optimizer_calls=[]
    def no_solve(*a,**kw):optimizer_calls.append(True);raise AssertionError('This selfcheck must not optimize')
    callback_supported='callback' in inspect.signature(scipy.optimize.least_squares).parameters
    scipy.optimize.least_squares=no_solve
    read_requests=[]
    def poison_oldq(role,key,data):
        read_requests.append([role,key])
        if role=='frozen_full151_geometry' and key=='joint_pos_rad':raise AssertionError('Forbidden oldfullq read')
        if role=='observation_fixture' and key not in ('time_s','hand_landmarks_source_w','object_pose_source_w'):raise AssertionError('Forbidden observationfield')
        return data[key]
    p=checked_protocol();problem=Prefix16GeometryProblem(p,read_array=poison_oldq)
    with problem.stage('selfcheck_base16'):r,J,base=problem.evaluate(problem.x0)
    arrays=problem.diagnostic_arrays(problem.x0);checks={};fd_reports=[];FDarrays={}
    checks['input_mapping_unit64_and_pose_checks']=all(problem.data['checks'].values())
    checks['three_observation_fields_only']=set(key for role,key in read_requests if role=='observation_fixture')=={'time_s','hand_landmarks_source_w','object_pose_source_w'}
    checks['full539d_joint_q_never_requested']=not any(role=='frozen_full151_geometry' and key=='joint_pos_rad' for role,key in read_requests)
    checks['targets_rebuilt_explicit_float64']=problem.data['desired_bones'].dtype==np.float64 and problem.data['normalized_target_points_w'].dtype==np.float64
    checks['x0_exact_repeated_canonical_q0']=np.array_equal(arrays['q_goal_rad'],np.broadcast_to(problem.q0,(16,20)))
    checks['300freevars_6596rows_sparseCSR']=problem.x0.shape==(300,) and r.shape==(6596,) and J.shape==(6596,300) and J.format=='csr'
    checks['frame0_has_no_columns']=J[:problem.local_rows].nnz==0
    checks['each_local_frame_only_own_freeblock']=all(set(J[f*problem.local_rows:(f+1)*problem.local_rows].nonzero()[1]).issubset(set(range((f-1)*20,f*20))) for f in range(1,16))
    Q=arrays['q_goal_rad'];first=16*problem.local_rows;second=first+300
    checks['adjacent_regularizer_exact_no_dt_division']=np.array_equal(r[first:second],(.03*(Q[1:]-Q[:-1])).ravel())
    checks['curvature_regularizer_exact_no_dt_division']=np.array_equal(r[second:],(.03*(Q[2:]-2*Q[1:-1]+Q[:-2])).ravel())
    checks['q0_never_clipped_at_lower']=np.array_equal(Q[0],problem.q0) and abs(Q[0,3]-problem.robot.lower[3])<1e-14
    checks['frozen_material_not_reselected']=len(problem.anchors)==3 and [a.geom_id for a in problem.anchors]==[a['geom_id'] for a in problem.bindings['frozen_material_anchors']]
    checks['anchor_BODYlocal_not_cylinder_geomlocal']=np.max(np.abs(problem.object_points_body-np.array([h['object_surface_closest_o'] for h in problem.bindings['observation_hypotheses']])))<1e-12
    checks['frame0_parent_position_normal_targets']=np.max(np.abs(problem.anchor_goals[0]-problem.bindings['target_material_points_w']))<1e-12 and np.max(np.abs(problem.normal_goals[0]-problem.bindings['target_hand_normals_w']))<1e-12
    checks['all16_object_transport_outward_unit']=np.max(np.abs(np.linalg.norm(problem.surface_normals,axis=-1)-1))<1e-12
    checks['absent_pair_gap_UNKNOWN_not_zero']=np.isnan(arrays['native_selected_contact_distance_m'][~arrays['native_generated_pair_mask']]).all()
    checks['native_allowance_scale_exact']=problem.g.metadata['allowance_m']==.0005 and problem.g.metadata['normalization_m']==.001
    for label,bad in [('wrongshape',np.zeros(299)),('nonfinite',np.r_[np.full(299,0.),np.nan])]:
        try:problem.q_from_x(bad);rejected=False
        except ValueError:rejected=True
        checks[label+'_freevars_rejected']=rejected
    native_rows=np.concatenate([np.arange(f*problem.local_rows+45,f*problem.local_rows+45+len(problem.g.pairs)) for f in range(16)])
    smooth=np.ones(problem.rows,bool);smooth[native_rows]=False
    directions=[np.sin(np.arange(300)+.3),np.eye(1,300,23).ravel(),np.eye(1,300,280).ravel()]
    for k,direction in enumerate(directions):
        direction=direction/max(1.,np.max(np.abs(direction)));eps=1e-6
        with problem.stage(f'selfcheck_FD{k}_base'):rb,Jb,lb=problem.evaluate(problem.x0)
        with problem.stage(f'selfcheck_FD{k}_plus'):rp,_,lp=problem.evaluate(problem.x0+eps*direction)
        with problem.stage(f'selfcheck_FD{k}_minus'):rm,_,lm=problem.evaluate(problem.x0-eps*direction)
        finite=(rp-rm)/(2*eps);analytic=Jb@direction;error=np.abs(finite-analytic)
        stable=[]
        for b,a,c in zip(lb,lp,lm):
            b,a,c=b['native'],a['native'],c['native']
            mask=b.generated_mask&a.generated_mask&c.generated_mask&(b.tie_count==1)&(a.tie_count==1)&(c.tie_count==1)&(b.active_mask==a.active_mask)&(b.active_mask==c.active_mask)
            mask &= (np.sum(b.selected_normal_a_to_b_w*a.selected_normal_a_to_b_w,axis=1)>.99999)&(np.sum(b.selected_normal_a_to_b_w*c.selected_normal_a_to_b_w,axis=1)>.99999)
            stable.append(mask)
        stable=np.array(stable).reshape(-1);native_error=error[native_rows];local_max=float(native_error[stable].max()) if stable.any() else None
        checks[f'FD{k}_noncollision_rows']=float(error[smooth].max())<1e-5
        checks[f'FD{k}_raw_temporal_blocks']=float(error[first:].max())<1e-9
        fd_reports.append({'direction':k,'eps':eps,'noncollision_max_abs':float(error[smooth].max()),'temporal_max_abs':float(error[first:].max()),'native_allrows_max_abs_scaled_residual':float(native_error.max()),'native_stable_unique_selected_rows':int(stable.sum()),'native_stable_unique_max_abs_scaled_residual':local_max,'native_gradient_global_qualified':False,'scope':'Native first-argmin ties/feature/contact switching and collision numeric tolerance are retained; no all-contact derivative PASS. Central FD extends free-boundarycoordinates geometrically, not as a physical initialization.'})
        FDarrays[f'FD{k}__finite']=finite;FDarrays[f'FD{k}__analytic']=analytic;FDarrays[f'FD{k}__native_stable_mask']=stable
    with problem.stage('selfcheck_restore_base'):problem.evaluate(problem.x0)
    counts=problem.counters();checks['every_actual_collision_stage_recorded']=sum(counts['collision_calls_by_stage'].values())==counts['actual_collision_calls']==len(problem.events)
    checks['no_formal_stage_or_optimizer']=not optimizer_calls and not any(s.startswith('formal') for s in counts['collision_calls_by_stage'])
    checks['no_forbidden_physics_force_APIs']=not any(forbidden.values())
    checks['no_Torch_Warp_import']=not any(k.split('.')[0] in ('torch','warp','mujoco_warp') for k in sys.modules)
    checks['SciPy_callback_available']=callback_supported
    dependencies=[ROOT/'scripts/build_prefix16_geometry_cpu.py',*sorted((ROOT/'src/prefix_study').glob('*.py')),ROOT/'vendor/runtime_io.py']+[Path(p['inputs'][k]['path']) for k in ('kinematics','native_filtered_geometry','material_geometry')]
    fake={'schema':'prefix16_geometry_cpu_plan:1','optimization_authorized':False,'protocol':{'path':str(PROTOCOL),'sha256':PROTOCOL_SHA},'inputs':p['inputs'],'source_pins':{str(i):{'path':str(f),'sha256':sha(f)} for i,f in enumerate(dependencies)},'frames':16,'joints':20,'max_nfev':80,'soft_wall_seconds':110,'owned_process_seconds':120,'automatic_retry':False}
    fakepath=out/'selfcheck_plan_not_formal.json';write_json(fakepath,fake);checked_plan(fakepath,run=False)
    try:checked_plan(fakepath,run=True);rejected=False
    except ValueError:rejected=True
    checks['optimization_false_rejected']=rejected
    tree=ast.parse((ROOT/'scripts/build_prefix16_geometry_cpu.py').read_text());source=(ROOT/'scripts/build_prefix16_geometry_cpu.py').read_text()
    checks['entry_guard_lists_all_forbidden_APIs']=all(repr(name) in source for name in forbidden)
    checks['entry_runtime_before_numpy_import']=source.index("write_json(out/'runtime.json',runtime)")<source.index('import numpy as np')
    checks['entry_fixed_solver_budget_callback']=all(s in source for s in ("method='trf'","tr_solver='lsmr'","max_nfev=80","ftol=1e-8","110s whole-worker","callback=callback"))
    atomic_npz(out/'selfcheck_arrays.npz',**arrays,**FDarrays)
    write_json(out/'collision_events.json',problem.events)
    report={'status':'SMALL_CPU_INPUT_SPARSE_SMOOTH_CHECKS_PASS' if all(checks.values()) else 'CPU_SELF_CHECK_FAIL','checks':{k:bool(v) for k,v in checks.items()},'FD_reports':fd_reports,'counts':counts,'read_log':problem.data['read_log'],'checker_source_sha256':sha(__file__),'problem_source_sha256':sha(ROOT/'src/prefix_study/problem.py'),'protocol_sha256':PROTOCOL_SHA,'source64_q0_lower_margin_rad':float(problem.q0[3]-problem.robot.lower[3]),'q0_FP32_cast_excess_vs_original_double_rad':float(problem.robot.lower[3]-np.float32(problem.q0[3])),'entrypoint_check_executed':False,'optimization_calls':len(optimizer_calls),'physics_steps':0,'GPU_used':False,'forbidden_calls':forbidden,'wall_seconds':time.perf_counter()-start,'scope':'Smooth bone/material/regularization/CSR and strict input contracts only. Local native collision differences/ties remain explicit; no global contact smoothness, continuouscurve, actualu, dynamics or physical qualification.'}
    write_json(out/'report.json',report);print(json.dumps({'status':report['status'],'checks_passed':sum(report['checks'].values()),'checks_total':len(checks),'failed':[k for k,v in checks.items() if not v],'counts':counts,'FD_reports':fd_reports},indent=2));raise SystemExit(not all(checks.values()))

if __name__=='__main__':main()
