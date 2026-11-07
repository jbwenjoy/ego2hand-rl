"""Sampled native geometry diagnostic of an explicitly FAILED CPU candidate.

This cannot issue READY or run dynamics. Absent native contacts retain NaN gap.
"""
from pathlib import Path
import argparse,importlib.util,json,os,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151.io import checked,pin,read,write,all_pins,load_selected


def main(protocol_path):
    start=time.perf_counter();p=read(protocol_path)
    if p['schema']!='full151_failed_candidate_geometry_diagnostic:1' or p['queries']!=601 or p['physics'] or p['training']:raise ValueError('Wrong diagnostic contract')
    for value in all_pins(p):checked(value)
    if checked(p['sources']['producer'])!=Path(__file__).resolve():raise ValueError('Wrong source')
    old=read(checked(p['inputs']['failed_receipt']))
    if old['status']!='CPU_CANDIDATE_GATES_FAIL_NO_QUALIFICATION' or (Path(p['inputs']['failed_receipt']['path']).parent/'READY.json').exists():raise ValueError('This diagnostic requires retained FAILED input, never READY')
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    report={'schema':'full151_sampled_geometry_diagnostic_receipt:1','status':'STARTED','protocol':pin(protocol_path),
      'parent_candidate_status':old['status'],'input_checks':{},'checks':{},'limitations':p['limitations'],
      'physics_qualified':False,'contact_target_ready':False,'training_authorized':False,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
    forbidden={};allowed={};saved={};queried=np.zeros(601,bool)
    distances=np.full((601,273),np.nan);mask=np.zeros((601,273),bool);ties=np.zeros((601,273),np.int32);counts=np.zeros((601,273),np.int32);points=np.full((601,21,3),np.nan)
    try:
        import mujoco
        import scipy.optimize as opt
        from scipy.spatial.transform import Rotation
        ref=load_selected(p['inputs']['reference'],['planned_time601_s','planned_q601_rad','object_COM_world_m','object_rotation_w','mapped_object_pose_w'])
        times=ref['planned_time601_s'];q=ref['planned_q601_rad'];R=ref['object_rotation_w']
        if not np.array_equal(times,np.arange(601)*.005) or q.shape!=(601,20) or R.shape!=(601,3,3) or not np.isfinite(q).all():raise ValueError('Invalid601 diagnostic samples')
        xyzw=Rotation.from_matrix(R).as_quat();poses=np.c_[ref['object_COM_world_m'],xyzw[:,[3,0,1,2]]]
        # Knot poses are the exact source representation, not a quaternion roundtrip.
        poses[::4]=ref['mapped_object_pose_w']
        Rcheck=Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix()
        report['input_checks']['rotation_representation_error_le1e12']=bool(np.max(np.abs(Rcheck-R))<=1e-12)
        if not all(report['input_checks'].values()):raise ValueError('Quaternion representation drift')
        def deny(counter,name):
            def f(*a,**kw):counter[name]+=1;raise AssertionError('Diagnostic forbids '+name)
            return f
        def wrap(counter,name,f):
            def call(*a,**kw):counter[name]+=1;return f(*a,**kw)
            return call
        for namespace,names in [(mujoco,['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce']),(opt,['linprog','minimize','least_squares'])]:
            for name in names:
                saved[(namespace,name)]=getattr(namespace,name);forbidden[name]=0;setattr(namespace,name,deny(forbidden,name))
        for name in ['mj_kinematics','mj_comPos','mj_collision']:
            saved[(mujoco,name)]=getattr(mujoco,name);allowed[name]=0;setattr(mujoco,name,wrap(allowed,name,getattr(mujoco,name)))
        sys.path.insert(0,str(checked(p['sources']['kinematics']).parent))
        spec=importlib.util.spec_from_file_location('_full151_frozen_native_geometry',checked(p['sources']['native_geometry']));g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)
        query=g.NativeFilteredContactGeometry(checked(p['inputs']['model']),checked(p['inputs']['actual_context']))
        actual_kin=Path(sys.modules[g.RobotKinematics.__module__].__file__).resolve()
        if actual_kin!=checked(p['sources']['kinematics']) or len(query.pairs)!=273:raise ValueError('Native query source/catalog changed')
        write(out/'query_metadata.json',query.metadata)
        (out/'source_before').mkdir()
        for name,s in p['sources'].items():(out/'source_before'/(name+Path(s['path']).suffix)).write_bytes(checked(s).read_bytes())
        np.savez_compressed(out/'catalog.npz',pairs=query.pairs,pair_kind=query.pair_kind,hand_ids=query.hand_ids,object_ids=query.object_ids,geom_names=np.asarray(query.names))
        with (out/'native_contacts.jsonl').open('x',encoding='utf-8') as stream:
            for i in range(601):
                query.set_object_pose(poses[i]);v=query.evaluate(q[i])
                distances[i]=v.signed_distance_m;mask[i]=v.generated_mask;ties[i]=v.tie_count;counts[i]=v.contact_count
                points[i]=query.robot.cached[0];queried[i]=True
                stream.write(json.dumps({'sample':i,'time_s':float(times[i]),'native_ncon':v.native_contact_count,'uncatalogued':v.uncatalogued_native_contact_count,'records':v.native_contact_records},allow_nan=False)+'\n')
        kind=query.pair_kind;depth=np.where(mask,np.maximum(-distances,0),0)
        max_depth=np.stack([depth[:,kind==k].max(axis=1) for k in ('hand_object','hand_self')],axis=-1)
        report['summary']={'maximum_generated_hand_object_depth_m':float(max_depth[:,0].max()),'maximum_generated_hand_self_depth_m':float(max_depth[:,1].max()),
          'sample_counts_over_original3mm':{'hand_object':int(np.count_nonzero(max_depth[:,0]>.003)),'hand_self':int(np.count_nonzero(max_depth[:,1]>.003))},
          'sample_counts_without_generated_hand_object':int(np.count_nonzero(~mask[:,kind=='hand_object'].any(axis=1))),
          'sample_counts_with_tied_pairs':int(np.count_nonzero(np.any(ties>1,axis=1))),
          'sampled_depth3mm_pass':bool(np.all(max_depth<=.003)),
          'q_control_input_max_abs_rad':float(np.max(np.abs(q)))}
        worst=np.unravel_index(int(np.argmax(depth)),depth.shape)
        report['worst_generated_overlap']={'sample':int(worst[0]),'time_s':float(times[worst[0]]),'pair_index':int(worst[1]),'pair_kind':str(kind[worst[1]]),'geom_names':[query.names[k] for k in query.pairs[worst[1]]],'depth_m':float(depth[worst]),'ties':int(ties[worst])}
        report['checks']={'all601_samples_completed':bool(queried.all()),'exact601_collision_queries':query.collision_query_count==601 and allowed['mj_collision']==601,
          '602_kinematics_including_constructor':allowed['mj_kinematics']==602 and allowed['mj_comPos']==602,
          'absent_native_pair_gaps_remain_NaN':bool(np.isnan(distances[~mask]).all()),
          'no_dynamics_optimization_or_force_query':not any(forbidden.values()),
          'clock_velocity_unchanged':query.data.time==0 and not np.any(query.data.qvel),
          'all_inputs_sources_unchanged':all(checked(s) for s in all_pins(p))}
        np.savez_compressed(out/'generated_depth_summary.npz',time_s=times,hand_object_depth_m=max_depth[:,0],hand_self_depth_m=max_depth[:,1])
        report['status']='COMPLETED_SAMPLED_GEOMETRY_DIAGNOSTIC' if all(report['checks'].values()) else 'GEOMETRY_DIAGNOSTIC_ENGINEERING_FAIL'
    except Exception as e:
        report['status']='GEOMETRY_DIAGNOSTIC_EXCEPTION_PRESERVED';report['exception']=repr(e);report['traceback']=traceback.format_exc()
    finally:
        for (namespace,name),value in saved.items():setattr(namespace,name,value)
    np.savez_compressed(out/'sampled_native_geometry.npz',queried=queried,selected_distance_m=distances,generated_mask=mask,tie_count=ties,contact_count=counts,fk_points_w=points)
    report['actual_calls']={'allowed':allowed,'forbidden':forbidden,'sample_count':int(queried.sum())}
    report['wall_s']=time.perf_counter()-start;report['artifacts']=[pin(f) for f in sorted(out.rglob('*')) if f.is_file()]
    write(out/'receipt.json',report)
    print(json.dumps({'status':report['status'],'summary':report.get('summary'),'actual_calls':report['actual_calls'],'receipt':pin(out/'receipt.json')}),flush=True)
    return 0 if report['status']=='COMPLETED_SAMPLED_GEOMETRY_DIAGNOSTIC' else 1


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--protocol',required=True);args=a.parse_args();raise SystemExit(main(Path(args.protocol)))
