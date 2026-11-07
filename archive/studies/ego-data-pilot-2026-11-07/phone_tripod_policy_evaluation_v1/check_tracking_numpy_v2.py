"""Pure reference/index tests for future evaluation, no NN/native/checkpoint load."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,sys
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent;O=H/'tracking_numpy_check_v2'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def mod(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
evaluator=H/'evaluate_learned_actors_v2.py';assert pin(evaluator)['sha256']=='bf154e3810bfc4a587c03e70281ff250a5079fb5ff382c54a733662663fa2c43'
E=mod('numpy_tracking_fixed_evaluator',evaluator);codec_path=P/'phone_transition_dataset_v1/transition_codec_v1.py';C=mod('numpy_tracking_fixed_codec',codec_path)
bpath=P/'phone_geometry_actor_integration_v1/task_bundle_v2.json';b=read(bpath);physical_path=P/'phone_object_wrench_labels_v1/completed_v1/physical_parameters.json';physical=read(physical_path)
with np.load(b['reference_arrays']['path'],allow_pickle=False) as z:task=SimpleNamespace(reference_pose=z['control_T_world_from_phone_F64'].copy(),reference_COM=z['control_COM_world_m'].copy())
q0=np.array(b['initial']['qpos27_F64']);ipos=np.array(physical['body_ipos_body_m']);x0=C.actual_state(q0,np.zeros(26),ipos)
O.mkdir();cases=[]
for dt in [.0003125,.00015625]:
    n=round(.02/dt)
    for label,N in [('one_H_plus_one_tick',n+1),('complete_2s',100*n)]:
        p=O/(str(dt)+'_'+label+'.npz');np.savez_compressed(p,post_qpos=np.tile(q0,(N,1)),post_qvel=np.zeros((N,26)),initial_qpos27=q0,post_time=np.full(N,-777.))
        arrays,stats=E.tracking_from_saved(p,task,C,ipos,dt)
        wanted=101 if label=='complete_2s' else 2
        expected_p=np.linalg.norm(x0[40:43]-task.reference_COM,axis=1)
        expected_R=np.array([np.linalg.norm(C.log_world(x0[43:52].reshape(3,3)@R.T)) for R in task.reference_pose[:,:3,:3]])
        k=arrays['known101'];checks=dict(integer_indices_not_poison_clock=int(k.sum())==wanted and int(arrays['native_POST_tick101'][1])==n,
            missing_tail_unfilled=bool(np.isnan(arrays['state58_F64'][~k]).all() and np.isnan(arrays['COM_error_m101'][~k]).all()),
            static_COM_matches_analytic=bool(np.allclose(arrays['COM_error_m101'][k],expected_p[k],rtol=0,atol=1e-14)),
            SO3_matches_independent_log=bool(np.allclose(arrays['SO3_geodesic_error_rad101'][k],expected_R[k],rtol=0,atol=1e-12)),
            initialized_frame_excluded=stats['available_POST_command_frames']==wanted-1,
            no_tracking_success_gate=stats['tracking_success_gate'] is None)
        cases.append(dict(dt_s=dt,fixture=label,checks=checks,stats=stats,input=pin(p)))
axis=np.array([0.,0.,1.]);quarter=C.exp_world(axis*np.pi/4);half=C.exp_world(axis*np.pi)
checks={str(i)+'_'+k:v for i,c in enumerate(cases) for k,v in c['checks'].items()}
checks['full_SO3_angles_45_and_180']=abs(E.angle_rad(quarter,np.eye(3))-np.pi/4)<1e-12 and abs(E.angle_rad(half,np.eye(3))-np.pi)<1e-12
checks['no_NN_or_backend_import']=not any(name in sys.modules for name in ['torch','mujoco','warp'])
r=dict(schema='phone_policy_tracking_numpy_command_grid_check:1',checks=checks,passed=sum(checks.values()),total=len(checks),cases=cases,source=pin(__file__),evaluator=pin(evaluator),codec=pin(codec_path),task_bundle=pin(bpath),physical=pin(physical_path),
    counts=dict(synthetic_traces=4,tracking_helper_calls=4,NN_native_GPU_calls=0,checkpoint_loads=0),scope='Synthetic static-object integer command-grid/correct COM/full-SO3/missing-tail semantics only. No learned Actor or physical task/holding qualification.')
with (O/'report.json').open('x',encoding='utf8') as f:json.dump(r,f,indent=2,allow_nan=False)
print(json.dumps(dict(passed=r['passed'],total=r['total'],report=pin(O/'report.json'))));assert all(checks.values())
