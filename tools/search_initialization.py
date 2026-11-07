"""New nominal-phone initialization proposals and fixed native qualification.

Default metadata only. All search seeds/budgets/scales are fixed before native
execution. No NN/checkpoint is loaded, no legacy task/checkpoint is modified.
"""
from pathlib import Path
import argparse,datetime,json,os,sys,time,traceback
import numpy as np
from evaluate_online import preflight
from control_authority import compare_trace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning import online_contract as C
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime import native_evaluation as E


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.floating,np.integer,np.bool_)):return x.item()
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    return x


def sources():
    return {str(p.relative_to(ROOT)):file_pin(p) for p in [Path(__file__),*(ROOT/'src/supermimic_research/mechanics').glob('*.py')]}


def run(args):
    path,c,registry,root,result,health,zero,binding,out=preflight(args)
    authority=C.read(args.authority_report)
    C.need(authority['individual_cases_passing']==0 and C.positive(authority),'Completed negative authority prerequisite')
    out.mkdir(parents=True);tick=time.perf_counter();phase='intent'
    intent=dict(schema='supermimic_research.initialization_search_intent:1',sources=sources(),frozen_runtime_sources=C.source_pins(),
        model=registry.pin(c['artifacts']['model']),parent_task=registry.pin(c['artifacts']['task']),authority=file_pin(args.authority_report),binding=file_pin(binding),
        seeds=['original_q_pose_u','native_hand_net_PD_correction_bounded'],selection='Lowest total declared score over all recorded proposals within each seed, before dynamic qualification',
        variables='First12 q,3 root-world translations,3 world left-rotvec,20 absolute PD targets; other8 q fixed',
        trust=dict(q12_delta_rad=.2,root_translation_axis_m=.015,world_rotvec_axis_rad=.25,absolute_PD_target_axis_delta_rad=.3),
        search=dict(maximum_calls_per_seed=1800,maximum_seconds_per_seed=35.,maximum_optimizer_nfev=150,scaled_secant_step=.001,maximum_static_forward=3603),
        fixed_residual_scales=dict(hand_net_Nm=.01,phone_force_N=.1,phone_body_torque_Nm=.005,missing_pad_force_N=.1,nonpad_table_force_N=.01,
            excess_pad_or_self_penetration_m=.00005,material_plane_gap_m=.001,parameter_regularizer=.003),
        geometry=dict(phone_pairs=22,self_pairs=189,pad_gap_m=[-.0001,.0002],nonpad_self_gap_min_m=0.,unknown_cannot_pass=True),
        native=dict(dt_s=E.DTS,gates=E.GATES,pair_gates=E.PAIR_GATES,maximum_cases=6,maximum_steps=57600,zero_both_dt_raw_barrier_before_search=True),
        role='NO_FITTING_INITIALIZATION_DEVELOPMENT',NN_training_authorized=False,automatic_retry=False,
        unchanged=['MJB bytes','physics','gravity','fixed wrist','force/target saturations','no-aid','native success gates'],
        scope='Distinct initialization development experiment. Original task/online300 lineage retained. Native secants and old material-plane anchors guide proposals only. Fresh full geometry and both dt qualify each candidate; no static-only holding claim.')
    C.write(out/'intent.json',intent)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import mujoco
    from supermimic_research.mechanics.initialization import NativeInitialization
    from supermimic_research.mechanics.collision import CollisionAudit
    counts=dict(MJB_reads=1,MjData=0,initial_forward=0,integration_steps=0,contact_force_reads=0,controller_calls=0,controller_receipt_reads=0,NN_FW=0,learning_updates=0,GPU_calls=0)
    cases=[];barrier=[];candidates=[]
    try:
        model=mujoco.MjModel.from_binary_path(str(registry.resolve(c['artifacts']['model'])));phone,pg,table,pads=E.assert_model(model,mujoco,np,C.read(binding));digest=E.model_digest_except_dt(model,np)
        bundle=C.read(registry.resolve(c['artifacts']['task']));q0=np.array(bundle['initial']['qpos27_F64']);u0=np.array(bundle['initial']['ctrl20_F64'])
        variant=E.make_variant(name='original_zero',qpos27=q0.tolist(),ctrl20=u0.tolist(),control_bounds20x2=model.actuator_ctrlrange,initial_state_provenance=dict(parent=intent['parent_task']))
        for dt in E.DTS:
            phase='zero_'+str(dt);folder=out/phase;folder.mkdir()
            summary,endpoint=E.evaluate_case_controlled(model,variant,dt,folder,phone,pg,table,pads,mujoco,np,counts,lambda *unused:u0.copy())
            reference=next(case for case in zero['cases'] if case['dt_s']==dt)
            comparison=compare_trace(reference['trace'],folder/'trace.npz',registry)
            C.need(all(comparison.values()),'Both-dt zero trace barrier before new search')
            barrier.append(dict(dt_s=dt,arrays=comparison));cases.append(dict(role='original_zero',native_summary=summary))
        C.write(out/'zero_barrier.json',dict(cases=barrier,passed=all(all(row['arrays'].values()) for row in barrier)))
        model.opt.timestep=E.DTS[0];engine=NativeInitialization(model,q0,u0);audit=CollisionAudit(model)
        expected=next(case for case in authority['cases'] if case['schedule']=='zero' and case['dt_s']==E.DTS[0])['native_summary']['initial_native']['contact_rows24']
        C.need(engine.base['contacts24'].tobytes()==np.array(expected).tobytes(),'Static original reset matches authority bytes')
        corrected=np.zeros(38);raw_correction=-engine.base['net_generalized26'][:20]/model.actuator_gainprm[:,0]/.3
        corrected[18:]=np.minimum(engine.hi[18:],np.maximum(engine.lo[18:],raw_correction))
        # Explicit bounded proposal seed, never a clamp of an executed action.
        starts=[np.zeros(38),corrected]
        C.write(out/'search_seeds.json',dict(parameters=jsonable(starts),raw_hand_PD_correction20=jsonable(raw_correction),seed_bound_projection20=jsonable(raw_correction!=corrected[18:]),
            scale38=jsonable(engine.scale),lower38=jsonable(engine.lo),upper38=jsonable(engine.hi),original_reset=jsonable(engine.base)))
        variants=[];endpoints={}
        for i,start in enumerate(starts):
            phase=f'search_seed{i}';model.opt.timestep=E.DTS[0];folder=out/phase;folder.mkdir()
            answer=engine.search(start,folder/'proposals.jsonl');final=answer.pop('final');proposal_pin=file_pin(folder/'proposals.jsonl')
            geometry=audit.evaluate(engine.data)
            C.write(folder/'geometry.json',geometry)
            with (folder/'reset_state.npz').open('xb') as f:np.savez_compressed(f,**final,z38=answer['best']['z'])
            deviation=engine.deviation(final['qpos27']);C.write(folder/'reference_deviation.json',deviation)
            name=f'seed{i}_force_aware_initialization';variant=E.make_variant(name=name,qpos27=final['qpos27'].tolist(),ctrl20=final['ctrl20'].tolist(),control_bounds20x2=model.actuator_ctrlrange,
                initial_state_provenance=dict(new_initialization=True,parent=intent['parent_task'],raw=file_pin(folder/'reset_state.npz'),deviation=file_pin(folder/'reference_deviation.json')))
            variants.append(variant);endpoints[name]=[]
            record=dict(name=name,seed=i,search=jsonable(answer),proposal_log=proposal_pin,reset_state=file_pin(folder/'reset_state.npz'),geometry=file_pin(folder/'geometry.json'),
                geometry_pass=geometry['passed'],reference_deviation=file_pin(folder/'reference_deviation.json'),initial_net_generalized26=jsonable(final['net_generalized26']),initial_pad_force_N=jsonable(final['pad_force_N']))
            # Execute both predeclared cases even if static geometry fails, but
            # keep the failed geometry gate in the final qualification.
            for dt in E.DTS:
                phase=name+'_dt'+str(dt);case_dir=folder/('dt'+str(dt));case_dir.mkdir()
                summary,endpoint=E.evaluate_case_controlled(model,variant,dt,case_dir,phone,pg,table,pads,mujoco,np,counts,lambda *unused,u=final['ctrl20']:u.copy())
                endpoints[name].append(endpoint);cases.append(dict(role='new_initialization',candidate=name,native_summary=summary))
                C.need(counts['integration_steps']<=57600,'Declared native budget')
            C.write(folder/'candidate.json',record);candidates.append(record)
            print(json.dumps(dict(seed=i,calls=answer['calls'],cost=answer['fresh_final_cost'],geometry_pass=geometry['passed'],native_passes=[case['native_summary']['nominal_holding_dynamics_pass'] for case in cases[-2:]])),flush=True)
        pairs=E.dt_pair_summary(variants,[case['native_summary'] for case in cases if case['role']=='new_initialization'],endpoints,np)
        for candidate,pair in zip(candidates,pairs):candidate['qualified']=bool(candidate['geometry_pass'] and pair['both_tripod_cases_pass'] and pair['dt_pair_consistency_pass'])
        checks=dict(original_zero_both_dt_raw=all(all(row['arrays'].values()) for row in barrier),all_predeclared_cases=len(cases)==6 and len(candidates)==2,
            fixed_model=digest==E.model_digest_except_dt(model,np),fixed_runtime=C.fingerprint(intent['frozen_runtime_sources'])==C.fingerprint(C.source_pins()),fixed_search_source=intent['sources']==sources(),
            declared_native_budget=counts['integration_steps']<=57600,declared_static_budget=engine.counts['static_forward']<=3603,
            no_Torch_or_NN='torch' not in sys.modules and counts['learning_updates']==0)
        C.need(all(checks.values()),'Initialization run integrity checks')
        report=dict(schema='supermimic_research.initialization_search_result:1',intent=file_pin(out/'intent.json'),checks=checks,passed=sum(checks.values()),total=len(checks),
            candidates=candidates,cases=cases,dt_pairs=pairs,counts=counts,static_counts=engine.counts,qualified_candidates=[x['name'] for x in candidates if x['qualified']],
            data_role=intent['role'],new_task_bundles=0,NN_training_updates=0,elapsed_s=time.perf_counter()-tick,scope=intent['scope'])
        C.write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),qualified=report['qualified_candidates'],counts=counts,static_counts=engine.counts)),flush=True)
    except BaseException as error:
        C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),counts=counts,completed_cases=len(cases),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('config','result','terminal','zero-report','authority-report','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:
        *_,out=preflight(args);print(json.dumps(dict(status='METADATA_ONLY_INITIALIZATION_READY',output=str(out),maximum_static_forward=3603,maximum_native_steps=57600,Torch_imported='torch' in sys.modules,MuJoCo_imported='mujoco' in sys.modules)))
    else:
        started=datetime.datetime.now(datetime.timezone.utc).isoformat();code=1
        try:run(args);code=0
        finally:
            if args.output.exists():C.write(args.output/'terminal.json',dict(returncode=code,started_UTC=started,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tool=file_pin(__file__),automatic_retry=False))
