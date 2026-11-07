"""Fixed final-online300 Actors in the shared original two-dt native loop.

Default is metadata-only. No model selection, training, old Python imports,
physics changes, aid, missing-tail completion or retry. All four cases remain.
"""
from pathlib import Path
import argparse,datetime,json,os,sys,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.learning import online_contract as C,runner as R


def preflight(args):
    path,c,registry=C.configuration(args.config);root=C.root_metadata(c,registry)
    result=C.read(args.result);terminal=C.read(args.terminal)
    C.need(result['status']=='COMPLETED_THREE_COLLECTION_UPDATE_CYCLES_NOT_POLICY_QUALIFICATION' and result['completed_updates']==300,'Fixed complete online300 only')
    health=C.read(registry.resolve_pin(result['health']))
    C.validate_own_header(health,result['checkpoint'],c,C.source_pins(),root,registry)
    C.need(health['update']==300 and health['checkpoint']==result['checkpoint'],'Final300 health/checkpoint join')
    registry.resolve_pin(result['checkpoint']);C.verify_collections(result['collections'],registry)
    C.need(len(result['collections'])==3 and result['collections']==health['collections'],'All three completed collections')
    C.need(terminal['returncode']==0 and terminal['configuration']==file_pin(path) and C.fingerprint(terminal['sources'])==C.fingerprint(C.source_pins()),'Successful actual final source/config process')
    executed=C.read(registry.resolve_pin(health['protocol']))
    C.need(executed['schema']==C.PROTOCOL and executed['configuration']==c and C.fingerprint(executed['sources'])==C.fingerprint(C.source_pins()),'Executed final protocol')
    zero=C.read(args.zero_report)
    C.need(zero['status']=='PASS_CALLBACK_MATCHES_FROZEN_NATIVE_LOOP' and C.positive(zero),'Existing actual zero qualification')
    zp=C.read(registry.resolve_pin(zero['protocol']));bundle=C.read(registry.resolve(c['artifacts']['task']))
    C.need([case['dt_s'] for case in zero['cases']]==[.0003125,.00015625],'Both predetermined native dt')
    C.need(C.content(zp['inputs']['model'])==C.content(registry.pin(c['artifacts']['model'])) and C.content(zp['inputs']['bundle'])==C.content(registry.pin(c['artifacts']['task'])),'Same qualified physical model/task')
    for case in zero['cases']:
        C.need(case['command']['qpos27']==bundle['initial']['qpos27_F64'] and case['command']['ctrl20']==bundle['initial']['ctrl20_F64'],'Exact qualified initial q/u')
    binding=registry.resolve_pin(zp['inputs']['binding'])
    out=args.output.resolve();C.need(out.is_relative_to(ROOT/'runs') and not out.exists(),'Fresh evaluation under unified runs')
    return path,c,registry,root,result,health,zero,binding,out


def angle_rad(R,Rref):
    rel=R@Rref.T;v=np.array([rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1]])*.5
    return float(np.arctan2(np.linalg.norm(v),(np.trace(rel)-1)*.5))


def tracking_from_saved(trace_path,task,codec,ipos,dt):
    """Frozen original integer-POST tracking math; no missing native tail fill."""
    with np.load(trace_path,allow_pickle=False) as z:q=z['post_qpos'].copy();v=z['post_qvel'].copy();q0=z['initial_qpos27'].copy()
    n=int(round(.02/dt));C.need(n in [64,128] and n*dt==.02,'Fixed integer H ticks')
    ticks=np.arange(101,dtype=np.int64)*n;known=ticks<=len(q);x0=codec.actual_state(q0,np.zeros(26),ipos)
    x=np.full((101,58),np.nan);x[0]=x0
    for k in np.flatnonzero(known)[1:]:x[k]=codec.actual_state(q[ticks[k]-1],v[ticks[k]-1],ipos)
    refCOM=np.asarray(task.reference_COM);refR=np.asarray(task.reference_pose)[:,:3,:3]
    p_error=np.full(101,np.nan);R_error=np.full(101,np.nan)
    fixedp=np.linalg.norm(x0[40:43]-refCOM,axis=1);fixedR=np.array([angle_rad(x0[43:52].reshape(3,3),r) for r in refR])
    for k in np.flatnonzero(known):p_error[k]=np.linalg.norm(x[k,40:43]-refCOM[k]);R_error[k]=angle_rad(x[k,43:52].reshape(3,3),refR[k])
    def stats(a,mask):
        z=a[mask];return dict(count=int(len(z)),RMS=float(np.sqrt(np.mean(z*z))) if len(z) else None,mean=float(np.mean(z)) if len(z) else None,max=float(np.max(z)) if len(z) else None,final_available=float(z[-1]) if len(z) else None)
    post=known.copy();post[0]=False
    arrays=dict(command_index101=np.arange(101,dtype=np.int64),nominal_time_s101=np.arange(101)*.02,native_POST_tick101=ticks,known101=known,state58_F64=x,reference_COM_world_m101=refCOM,reference_R_BODY_world101=refR,COM_error_m101=p_error,SO3_geodesic_error_rad101=R_error,fixed_object_COM_error_m101=fixedp,fixed_object_SO3_geodesic_error_rad101=fixedR)
    stats_out=dict(available_POST_command_frames=int(post.sum()),missing_POST_command_frames=int((~known).sum()),COM_error_m=stats(p_error,post),SO3_geodesic_error_rad=stats(R_error,post),fixed_object_on_same_available_POST_frames=dict(COM_error_m=stats(fixedp,post),SO3_geodesic_error_rad=stats(fixedR,post)),fixed_object_full101_command_descriptor=dict(COM_error_m=stats(fixedp,np.ones(101,dtype=bool)),SO3_geodesic_error_rad=stats(fixedR,np.ones(101,dtype=bool))),tracking_success_gate=None,scope='Full SO3 angle and COM tracking descriptors, not holding5mm success; no missing-tail fill, temporal stretch or accumulated-clock tick index')
    return arrays,stats_out


def run(args):
    path,c,registry,root,result,health,zero,binding,out=preflight(args)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch,mujoco
    from supermimic_research.learning import imagination as IM,world_model as WM
    from supermimic_research.runtime import native_evaluation as E,state_codec as codec
    from supermimic_research.runtime.controller import TaskController
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    C.need(not torch.cuda.is_initialized(),'CPU evaluation')
    out.mkdir(parents=True);tick=time.perf_counter();phase='load';cases=[]
    counts=dict(MJB_reads=0,MjData=0,initial_forward=0,integration_steps=0,contact_force_reads=0,controller_calls=0,controller_receipt_reads=0,Actor_FW=0,WM_FW=0,Critic_FW=0,observer_forward=0,optimizer_steps=0,backward_calls=0,GPU_calls=0)
    intent=dict(schema='supermimic_research.online300_evaluation_intent:1',tool=file_pin(__file__),configuration=file_pin(path),
        result=file_pin(args.result),health=result['health'],checkpoint=result['checkpoint'],training_terminal=file_pin(args.terminal),
        sources=C.source_pins(),qualified_zero_report=file_pin(args.zero_report),binding=file_pin(binding),
        modes=R.MODES,dt_s=E.DTS,physical_gates=E.GATES,dt_pair_gates=E.PAIR_GATES,maximum_native_steps=38400,
        automatic_retry=False,scope='All final300 cases; no training, auxiliary forces, source-state selection or physical gate change')
    C.write(out/'intent.json',intent)
    try:
        ck=torch.load(registry.resolve_pin(result['checkpoint']),map_location='cpu',weights_only=False)
        actor_input=dict(kind='own_online',stage_id=3,checkpoint=result['checkpoint'],health=result['health'],training_protocol=health['protocol'])
        C.validate_actor_payload(ck,actor_input,health,root,c,C.source_pins());actors={}
        for mode in R.MODES:
            pair=IM.make_fresh_actor_critic(seed=c['training']['seed'],device='cpu');actor=pair['actor']
            actor.load_state_dict(ck['arms'][mode]['stateactor']);actor.eval();actor.requires_grad_(False)
            C.need(WM.raw_tree_equal(actor.state_dict(),ck['arms'][mode]['stateactor']) and WM.raw_tree_equal(actor.norm.state_dict(),ck['arms'][mode]['norm']['actor']),'Raw Actor and normalizer load')
            C.need(all(torch.isfinite(v).all() for v in actor.state_dict().values()),'Finite serialized Actor')
            actors[mode]=actor
        before={m:copy_state(a,torch) for m,a in actors.items()};host_before={m:norm_host(a) for m,a in actors.items()};rng=WM.capture_rng()
        model=mujoco.MjModel.from_binary_path(str(registry.resolve(c['artifacts']['model'])));counts['MJB_reads']=1
        phone,pg,table,pads=E.assert_model(model,mujoco,np,C.read(binding));digest=E.model_digest_except_dt(model,np)
        geometry=R._geometry(registry,c);bundle=C.read(registry.resolve(c['artifacts']['task']));variants=[];endpoints={}
        for mode in R.MODES:
            task=R._task(registry,c,'A1_tracking',geometry)
            variant=E.make_variant(name=mode+'__online300',qpos27=bundle['initial']['qpos27_F64'],ctrl20=bundle['initial']['ctrl20_F64'],
                control_bounds20x2=model.actuator_ctrlrange,initial_state_provenance=dict(task_bundle=registry.pin(c['artifacts']['task']),qualified_zero_report=file_pin(args.zero_report),checkpoint=result['checkpoint']))
            variants.append(variant);endpoints[variant['name']]=[]
            for dt in E.DTS:
                phase=mode+'_dt_'+str(dt);folder=out/phase;folder.mkdir()
                callback=TaskController(task,geometry,dt=dt,phone=phone,actor=actors[mode],device='cpu')
                summary,endpoint=E.evaluate_case_controlled(model,variant,dt,folder,phone,pg,table,pads,mujoco,np,counts,callback)
                endpoints[variant['name']].append(endpoint);counts['Actor_FW']+=callback.NN_FW
                C.need(callback.NN_FW==len(callback.records) and all(callback.memory[k].dtype==torch.float64 for k in ('a1','a2','c1','last_u')),'One Actor call and canonical F64 history per command')
                C.write(folder/'controller_records.json',callback.records)
                arrays,tracking=tracking_from_saved(folder/'trace.npz',task,codec,model.body_ipos[phone].copy(),dt)
                with (folder/'tracking101.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
                C.write(folder/'tracking_statistics.json',tracking)
                cases.append(dict(mode=mode,dt_s=dt,native_summary=summary,endpoint=endpoint,tracking=tracking,
                    controller_records=file_pin(folder/'controller_records.json'),tracking_arrays=file_pin(folder/'tracking101.npz')))
                C.write(out/f'completed_case{len(cases)}.json',cases[-1])
                print(json.dumps(dict(mode=mode,dt_s=dt,steps=summary['finite_steps_saved'],termination=summary['termination'],holding_pass=summary['nominal_holding_dynamics_pass'])),flush=True)
        pairs=E.dt_pair_summary(variants,[case['native_summary'] for case in cases],endpoints,np)
        checks=dict(all_four_cases=len(cases)==4,exact_Actor_calls=counts['Actor_FW']==counts['controller_calls'],
            bounded_native=counts['integration_steps']<=38400 and counts['Actor_FW']<=400,
            read_only_weights_norm=all(WM.raw_tree_equal(before[m],actors[m].state_dict()) for m in R.MODES) and host_before=={m:norm_host(a) for m,a in actors.items()},
            read_only_RNG=WM.raw_tree_equal(rng,WM.capture_rng()),
            no_learning_or_observer=all(counts[k]==0 for k in ('WM_FW','Critic_FW','optimizer_steps','backward_calls','observer_forward','GPU_calls')),
            model_unchanged_except_dt=digest==E.model_digest_except_dt(model,np),CPU_only=not torch.cuda.is_initialized(),
            source_unchanged=C.fingerprint(intent['sources'])==C.fingerprint(C.source_pins()))
        C.need(all(checks.values()),'Evaluation integrity failure; no retry')
        report=dict(schema='supermimic_research.online300_native_evaluation:1',status='ALL_FOUR_CASES_RECORDED_PHYSICAL_GATES_NOT_ASSUMED',intent=file_pin(out/'intent.json'),
            checkpoint=result['checkpoint'],checks=checks,passed=sum(checks.values()),total=len(checks),cases=cases,dt_pairs=pairs,counts=counts,
            own_CPU_checkpoint_reads=1,Actor_constructors=2,unused_Critic_constructors=2,WM_constructors=0,
            tripod_cases_passing=sum(case['native_summary']['nominal_holding_dynamics_pass'] for case in cases),
            physical_pair_passes=sum(pair['both_tripod_cases_pass'] and pair['dt_pair_consistency_pass'] for pair in pairs),
            elapsed_s=time.perf_counter()-tick,scope=intent['scope'])
        C.write(out/'report.json',report);return report
    except BaseException as error:
        C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),counts=counts,completed_cases=len(cases),automatic_retry=False));raise


def copy_state(actor,torch):return {k:v.detach().clone() for k,v in actor.state_dict().items()}
def norm_host(actor):return {k:v for k,v in vars(actor.norm).items() if isinstance(v,(bool,int,float,str))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--terminal',type=Path,required=True);parser.add_argument('--zero-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        *_,out=preflight(args);print(json.dumps(dict(status='METADATA_ONLY_FINAL300_READY',output=str(out),Torch_imported='torch' in sys.modules,MuJoCo_imported='mujoco' in sys.modules)));return 0
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();code=1
    try:
        result=run(args);code=0;print(json.dumps(dict(report=file_pin(args.output/'report.json'),passing_cases=result['tripod_cases_passing'],counts=result['counts'])),flush=True)
    finally:
        if args.output.exists():C.write(args.output/'terminal.json',dict(pid=os.getpid(),returncode=code,command=sys.argv,tool=file_pin(__file__),started_UTC=started,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False))
    return code


if __name__=='__main__':raise SystemExit(main())
