"""Predeclared long-axis patch migration, geometry restoration and native tests."""
from pathlib import Path
import argparse,datetime,json,os,sys,time,traceback
import numpy as np
from evaluate_online import preflight
from control_authority import compare_trace
from search_initialization import jsonable
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.learning import online_contract as C
from supermimic_research.runtime import native_evaluation as E


def source_pins():
    paths=[Path(__file__),ROOT/'tools/search_initialization.py',ROOT/'tools/control_authority.py',ROOT/'tools/evaluate_online.py']
    paths.extend(ROOT/'src/supermimic_research/mechanics'/name for name in ('__init__.py','initialization.py','collision.py','convex.py','sliding.py','cached_collision.py'))
    return {str(p.relative_to(ROOT)):file_pin(p) for p in paths}


def run(args):
    path,c,registry,root,training,health,zero,binding,out=preflight(args)
    prior=C.read(args.initialization_report);C.need(C.positive(prior) and prior['qualified_candidates']==[],'Completed negative local initialization prerequisite')
    from supermimic_research.mechanics.sliding import OFFSETS_M
    out.mkdir(parents=True);tick=time.perf_counter();phase='intent'
    intent=dict(schema='supermimic_research.patch_migration_intent:2',configuration=file_pin(path),sources=source_pins(),runtime_sources=C.source_pins(),
        model=registry.pin(c['artifacts']['model']),parent_task=registry.pin(c['artifacts']['task']),qualified_zero=file_pin(args.zero_report),binding=file_pin(binding),prior=file_pin(args.initialization_report),
        offsets_m=list(OFFSETS_M),axis='Largest covariance eigenvector of compiled phone convex vertices, sign canonicalized in geom-local coordinates, transformed by original geom pose',
        restoration=dict(variable_joints=args.restore_joints,joint_delta_rad=.35,joint_margin_rad=.005,maximum_calls_per_seed=args.restore_calls,maximum_seconds_per_seed=args.restore_seconds,
            query_cache=args.query_cache,secant_step=.001,target_pad_gap_m=-.000075,geometry_phone_pairs=22,geometry_self_pairs=189),
        force_refinement=dict(maximum_calls_per_seed=900,maximum_seconds_per_seed=10.,implementation='Unchanged NativeInitialization, local q12±.2rad/pose±15mm,±.25rotvec/PD±.3rad',
            eligibility='Restored full geometry passes and fresh native solve contains every designated pad-phone contact',selection='Minimum declared native-force score over all logged proposals within that seed, before dynamic tests'),
        seed_target='Original u0 plus restored q-q0, explicitly projected to real control limits with1mrad interior margin; seed construction only, no execution clamp',
        final_evaluation='Both original dt for every constructed force candidate, even if final geometry fails; qualification requires fresh full geometry plus both native passes and pair consistency',
        budgets=dict(maximum_restore_kinematics=8*(args.restore_calls+1),maximum_force_forwards=7216,maximum_screen_forwards=8,maximum_native_cases=18,maximum_native_steps=172800),
        native_gates=E.GATES,pair_gates=E.PAIR_GATES,role='NO_FITTING_PATCH_MIGRATION_DEVELOPMENT',automatic_retry=False,NN_training_authorized=False,
        scope='New broader initialization development experiment. Same MJB/fixed wrist/materials/gravity/actuator limits/no-aid/holding gates. All seed failures retained. No task bundle or old-policy compatibility claim; source pose deviations are algorithmic, not measured ego uncertainty.')
    C.write(out/'intent.json',intent)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import mujoco
    from supermimic_research.mechanics.collision import CollisionAudit
    from supermimic_research.mechanics.initialization import NativeInitialization
    from supermimic_research.mechanics.sliding import collider_axes,GeometryRestoration,compose_deviation
    counts=dict(MJB_reads=1,MjData=0,initial_forward=0,integration_steps=0,contact_force_reads=0,controller_calls=0,controller_receipt_reads=0,NN_FW=0,learning_updates=0,GPU_calls=0)
    screen_counts=dict(forward=0,contact_force_reads=0);axis_counts=dict(MjData=0,kinematics=0)
    records=[];cases=[];variants=[];endpoints={};barrier=[];restore_counts=[];force_counts=[]
    try:
        model=mujoco.MjModel.from_binary_path(str(registry.resolve(c['artifacts']['model'])));phone,pg,table,pads=E.assert_model(model,mujoco,np,C.read(binding));digest=E.model_digest_except_dt(model,np)
        b=C.read(registry.resolve(c['artifacts']['task']));q0=np.array(b['initial']['qpos27_F64']);u0=np.array(b['initial']['ctrl20_F64'])
        variant=E.make_variant(name='original_zero',qpos27=q0.tolist(),ctrl20=u0.tolist(),control_bounds20x2=model.actuator_ctrlrange,initial_state_provenance=dict(parent=intent['parent_task']))
        for dt in E.DTS:
            phase='zero_'+str(dt);folder=out/phase;folder.mkdir()
            summary,_=E.evaluate_case_controlled(model,variant,dt,folder,phone,pg,table,pads,mujoco,np,counts,lambda *unused:u0.copy())
            ref=next(case for case in zero['cases'] if case['dt_s']==dt);comparison=compare_trace(ref['trace'],folder/'trace.npz',registry)
            C.need(all(comparison.values()),'Full original zero barrier before migration')
            barrier.append(dict(dt_s=dt,arrays=comparison));cases.append(dict(role='original_zero',native_summary=summary))
        C.write(out/'zero_barrier.json',dict(passed=True,cases=barrier))
        if args.query_cache:
            from supermimic_research.mechanics.cached_collision import CachedCollisionAudit
            audit=CachedCollisionAudit(model)
        else:audit=CollisionAudit(model)
        model.opt.timestep=E.DTS[0];axis_data=mujoco.MjData(model);axis_counts['MjData']+=1;axis_data.qpos[:]=q0
        mujoco.mj_kinematics(model,axis_data);axis_counts['kinematics']+=1
        descriptor=collider_axes(audit.locals[pg],axis_data.geom_xmat[pg].reshape(3,3));axis=np.array(descriptor['axes']['long']['world'])
        C.write(out/'compiled_axes.json',descriptor)
        for index,offset in enumerate(OFFSETS_M):
            phase=f'seed{index:02d}';folder=out/phase;folder.mkdir();model.opt.timestep=E.DTS[0]
            shift=offset*axis;qseed=q0.copy();qseed[20:23]+=shift
            restorer=GeometryRestoration(model,audit,qseed,variable_joints=args.restore_joints);answer=restorer.restore(folder/'restore_proposals.jsonl',maximum_calls=args.restore_calls,maximum_seconds=args.restore_seconds);restore_counts.append(answer['counts'])
            record=dict(index=index,offset_m=offset,shift_world_m=shift.tolist(),restoration_proposals=file_pin(folder/'restore_proposals.jsonl'),restoration=jsonable({k:v for k,v in answer.items() if k not in ('qpos27','geometry')}),
                status='RESTORATION_UNAVAILABLE',native_cases=0,qualified=False)
            if answer['best'] is None:
                C.write(folder/'seed.json',record);records.append(record);print(json.dumps(dict(seed=index,offset_m=offset,status=record['status'])),flush=True);continue
            restored_q=answer['qpos27'];C.write(folder/'restored_geometry.json',answer['geometry']);record['restored_geometry']=file_pin(folder/'restored_geometry.json')
            with (folder/'restored_state.npz').open('xb') as f:np.savez_compressed(f,qpos27=restored_q,z_joint=np.array(answer['best']['z_joint']))
            record['restored_state']=file_pin(folder/'restored_state.npz')
            if not answer['qualified']:
                record['status']='RESTORED_GEOMETRY_FAILED';C.write(folder/'seed.json',record);records.append(record)
                print(json.dumps(dict(seed=index,offset_m=offset,status=record['status'])),flush=True);continue
            raw_u=u0+restored_q[:20]-q0[:20];u_seed=np.minimum(model.actuator_ctrlrange[:,1]-.001,np.maximum(model.actuator_ctrlrange[:,0]+.001,raw_u))
            d=restorer.data;d.ctrl[:]=u_seed;d.qvel[:]=0;d.qacc_warmstart[:]=0;mujoco.mj_forward(model,d);screen_counts['forward']+=1
            C.need(not np.any(d.warning.number) and not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied),'Native restoration screen warning/aid')
            snapshot=E.contact_snapshot(model,d,pg,table,pads,mujoco,np,screen_counts,-1);rows=np.array(snapshot[-1]).reshape(-1,24)
            seen={int(row[1] if row[2]==pg else row[2]) for row in rows if pg in (int(row[1]),int(row[2]))}
            C.write(folder/'screen.json',dict(contacts24=rows.tolist(),pad_force_N=snapshot[0].tolist(),raw_seed_u20=raw_u.tolist(),legal_seed_u20=u_seed.tolist(),seed_projection20=(raw_u!=u_seed).tolist()))
            record['native_screen']=file_pin(folder/'screen.json')
            if not set(pads).issubset(seen):
                record['status']='NATIVE_PAD_CONTACT_MISSING';C.write(folder/'seed.json',record);records.append(record)
                print(json.dumps(dict(seed=index,offset_m=offset,status=record['status'])),flush=True);continue
            phase=f'seed{index:02d}_force_refinement';engine=NativeInitialization(model,restored_q,u_seed)
            C.need(engine.base['contacts24'].tobytes()==rows.tobytes(),'Fresh force initializer equals screened native reset')
            force=engine.search(np.zeros(38),folder/'force_proposals.jsonl',maximum_calls=900,maximum_seconds=10.);final=force.pop('final');force_counts.append(engine.counts.copy())
            geometry=audit.evaluate(engine.data);C.write(folder/'final_geometry.json',geometry)
            deviation=compose_deviation(engine.deviation(final['qpos27']),restored_q,q0,shift);C.write(folder/'reference_deviation.json',deviation)
            with (folder/'final_state.npz').open('xb') as f:np.savez_compressed(f,**final,z38=force['best']['z'])
            record.update(status='FORCE_CANDIDATE_EVALUATED',force_search=jsonable(force),force_proposals=file_pin(folder/'force_proposals.jsonl'),final_state=file_pin(folder/'final_state.npz'),
                final_geometry=file_pin(folder/'final_geometry.json'),final_geometry_pass=geometry['passed'],reference_deviation=file_pin(folder/'reference_deviation.json'),native_cases=2,qualified=None)
            name=f'seed{index:02d}_slide_{offset:+.3f}m';variant=E.make_variant(name=name,qpos27=final['qpos27'].tolist(),ctrl20=final['ctrl20'].tolist(),control_bounds20x2=model.actuator_ctrlrange,
                initial_state_provenance=dict(new_initialization=True,parent=intent['parent_task'],raw=record['final_state'],deviation=record['reference_deviation']))
            variants.append(variant);endpoints[name]=[];record['variant_name']=name
            for dt in E.DTS:
                phase=name+'_dt'+str(dt);case_dir=folder/('dt'+str(dt));case_dir.mkdir()
                summary,endpoint=E.evaluate_case_controlled(model,variant,dt,case_dir,phone,pg,table,pads,mujoco,np,counts,lambda *unused,u=final['ctrl20']:u.copy())
                cases.append(dict(role='patch_migration',seed=index,native_summary=summary));endpoints[name].append(endpoint)
                C.need(counts['integration_steps']<=172800,'Declared native budget')
            C.write(folder/'seed.json',record);records.append(record)
            print(json.dumps(dict(seed=index,offset_m=offset,status=record['status'],final_geometry=geometry['passed'],passes=[x['native_summary']['nominal_holding_dynamics_pass'] for x in cases[-2:]])),flush=True)
        pairs=E.dt_pair_summary(variants,[x['native_summary'] for x in cases if x['role']=='patch_migration'],endpoints,np)
        for record in records:
            if record['status']=='FORCE_CANDIDATE_EVALUATED':
                pair=next(x for x in pairs if x['name']==record['variant_name']);record['qualified']=bool(record['final_geometry_pass'] and pair['both_tripod_cases_pass'] and pair['dt_pair_consistency_pass'])
            qualification=out/f"seed{record['index']:02d}"/'qualification.json'
            C.write(qualification,dict(status=record['status'],qualified=record['qualified'],native_cases=record['native_cases']))
            record['qualification']=file_pin(qualification)
        checks=dict(all_eight_seed_attempts=len(records)==8 and [r['offset_m'] for r in records]==list(OFFSETS_M),zero_barrier=all(all(x['arrays'].values()) for x in barrier),
            fixed_model=digest==E.model_digest_except_dt(model,np),fixed_runtime=C.fingerprint(intent['runtime_sources'])==C.fingerprint(C.source_pins()),fixed_proposal_sources=intent['sources']==source_pins(),
            native_budget=len(cases)<=18 and counts['integration_steps']<=172800,restoration_budget=sum(x['kinematics'] for x in restore_counts)<=intent['budgets']['maximum_restore_kinematics'],
            force_budget=sum(x['static_forward'] for x in force_counts)<=7216,screen_budget=screen_counts['forward']<=8,no_Torch_NN_or_learning='torch' not in sys.modules and counts['learning_updates']==0)
        C.need(all(checks.values()),'Migration experiment integrity failure')
        report=dict(schema='supermimic_research.patch_migration_result:1',intent=file_pin(out/'intent.json'),checks=checks,passed=sum(checks.values()),total=len(checks),records=records,cases=cases,dt_pairs=pairs,
            qualified=[x['index'] for x in records if x['qualified']],counts=counts,axis_counts=axis_counts,screen_counts=screen_counts,restoration_counts=restore_counts,force_counts=force_counts,
            geometry_query_cache=getattr(audit,'query_counts',None),
            role=intent['role'],new_task_bundles=0,learning_updates=0,elapsed_s=time.perf_counter()-tick,scope=intent['scope'])
        C.write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),qualified=report['qualified'],counts=counts)),flush=True)
    except BaseException as error:
        C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),counts=counts,completed_seeds=len(records),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('config','result','terminal','zero-report','initialization-report','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--restore-joints',type=int,choices=(12,20),default=12);p.add_argument('--restore-calls',type=int,default=180);p.add_argument('--restore-seconds',type=float,default=12.);p.add_argument('--query-cache',action='store_true')
    p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:
        *_,out=preflight(args);print(json.dumps(dict(status='METADATA_ONLY_PATCH_MIGRATION_READY',output=str(out),seed_count=8,Torch_imported='torch' in sys.modules,MuJoCo_imported='mujoco' in sys.modules)))
    else:
        started=datetime.datetime.now(datetime.timezone.utc).isoformat();code=1
        try:run(args);code=0
        finally:
            if args.output.exists():C.write(args.output/'terminal.json',dict(returncode=code,started_UTC=started,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=sys.argv,tool=file_pin(__file__),automatic_retry=False))
