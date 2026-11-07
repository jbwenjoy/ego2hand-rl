"""Fixed paired1000 Actor CPU inference in the frozen original native loop.

Default is JSON/hash metadata only. No top-level Torch/MuJoCo import, model,
Actor or arrays. Four declared cases only; aborts are results, never retried.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
SCHEMA='phone_tripod_fixed1000_Actor_native_evaluation_protocol:2'
PAIR_GATES={'final_COM_difference_m':.002,'final_orientation_difference_rad':.03490658503988659,'require_both_duration_complete':True}
ARTIFACT='phone_tripod_sharedWM_A1A2_training_checkpoint:1'
MODES=['A1_tracking','A2_engineered_material_proxy'];DTS=[.0003125,.00015625]
FINAL_INPUTS=['training_final_protocol','final_result','final_health','final_terminal']
def need(ok,msg):
    if not ok:raise ValueError(msg)
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def checked(v):
    need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Current absolute source/input pin');return Path(v['path'])
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def metadata_preflight(path,*,check_output_fresh=True):
    p=read(path);need(p['schema']==SCHEMA and p['source']==pin(__file__),'New fixed1000 evaluation source')
    need(p['fixed_paired_round']==1000 and p['modes']==MODES and p['dt_s']==DTS and p['device']=='cpu','Fixed final two arms/four CPU cases')
    need(p['native_dt_pair_gates']==PAIR_GATES and p['native_dt_pair_rule']['source']==p['sources']['kernel_frozen_loop'] and p['native_dt_pair_rule']['function']=='dt_pair_summary','Original full double-dt endpoint gates')
    need(p['GPU_run_authorized'] is False and p['training_authorized'] is False,'No GPU/training')
    for k in ['CPU_run_authorized','NN_forward_authorized','native_execution_authorized']:need(type(p[k]) is bool,'Typed execution permission')
    for v in p['sources'].values():checked(v)
    for v in p['inputs'].values():
        if v is not None:checked(v)
    need(p['sources']['trainer']['sha256']=='a1bcf9f820e00d7da4ac515dd81d45d39a2c59abd07c85319d1346db384b2439','Frozen canonical-memory trainer')
    b=read(checked(p['inputs']['task_bundle']));seed=read(checked(p['inputs']['training_seed_protocol']))
    need(b['schema']=='phone_fixed_wrist_ego61_tripod_task:1' and b['task_designated_geom_ids']==[15,28,41],'New tripod role remains explicit')
    need(seed['inputs']['task_bundle']==p['inputs']['task_bundle'] and seed['inputs']['geometry_catalog']==p['inputs']['geometry_catalog'] and b['model']==p['inputs']['model'],'Same training reference/initial/action/model/current geometry')
    zero=read(checked(p['inputs']['qualified_zero_report']))
    need(zero['status']=='PASS_CALLBACK_MATCHES_FROZEN_NATIVE_LOOP' and zero['checks'] and all(v is True for v in zero['checks'].values()),'Qualified original native scheduled controller')
    need(len(p['original_cases'])==2 and [c['dt_s'] for c in p['original_cases']]==DTS,'Two fixed actual initial variants')
    for c in p['original_cases']:
        z=next(q for q in zero['cases'] if q['dt_s']==c['dt_s'])
        need(c['command']==z['command'],'Actual qualified initial native variant, not PRE-optimizer log')
        need(c['command']['qpos27']==b['initial']['qpos27_F64'] and c['command']['ctrl20']==b['initial']['ctrl20_F64'],'Exact task initial qpos/ctrl')
    pending=any(p['inputs'][k] is None for k in FINAL_INPUTS)
    if pending:need(not any(p[k] for k in ['CPU_run_authorized','NN_forward_authorized','native_execution_authorized']),'Final file pending: metadata only')
    else:
        final=read(checked(p['inputs']['final_result']));health=read(checked(p['inputs']['final_health']));terminal=read(checked(p['inputs']['final_terminal']));tp=read(checked(p['inputs']['training_final_protocol']))
        need(final['schema']=='phone_tripod_matched_supermimic_training_result:1' and final['status']=='COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION','Completed exact target, not checkpoint100')
        need(final['paired_round']==1000 and final['shared_WM_updates']==1100,'Final AFTER paired1000/WM1100')
        need(health['schema']=='phone_tripod_matched_learning_checkpoint_health:1' and health['paired_round']==1000 and health['shared_WM_updates']==1100 and health['checkpoint']==final['checkpoint'],'Final checkpoint health join')
        need(health['checks'] and all(v is True for v in health['checks'].values()),'Actual final serialized health')
        checked(final['checkpoint'])
        need(terminal['returncode']==0 and terminal['protocol']==p['inputs']['training_final_protocol'] and terminal['source']==p['sources']['trainer'],'Final training terminal/current source')
        need(tp['sources']['trainer']==p['sources']['trainer'] and tp['inputs']==seed['inputs'] and tp['recipe']==seed['recipe'],'Same fixed recipe/data family lineage')
        for k,v in p['sources'].items():
            if k in tp['sources']:need(v==tp['sources'][k],'Same training direct source: '+k)
        for k,n in [('WM_optimizer_steps',1100),('Actor_optimizer_steps',2000),('Critic_optimizer_steps',2000)]:need(final['counts'][k]==n,'Complete fixed training count '+k)
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==H,'Local absolute evaluation output')
    if check_output_fresh:need(not out.exists(),'Fresh evaluation output')
    return p,b,pending

def raw_equal(a,b,torch):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(raw_equal(v,b[k],torch) for k,v in a.items())
    return a==b

def load_final_actors(protocol,torch):
    """One trusted own CPU load; two frozen factory calls, zero forward.

    Returns actors, receipt. Factory makes Actor2+unused Critic2; Critic and
    shared-WM weights are never constructed or forwarded from the checkpoint.
    The returned Actors include their serialized EmpiricalNorm buffers.
    """
    p,b,pending=metadata_preflight(protocol,check_output_fresh=False);need(not pending,'Actual final1000 must be bound')
    final=read(checked(p['inputs']['final_result']));ck=torch.load(checked(final['checkpoint']),map_location='cpu',weights_only=False)
    need(ck['schema']==ARTIFACT and ck['paired_round']==1000 and ck['shared_WM_updates']==1100,'Own final artifact phase')
    tp=read(checked(p['inputs']['training_final_protocol']))
    need(ck['protocol']==p['inputs']['training_final_protocol'] and ck['recipe']==tp['recipe'] and ck['inputs']==tp['inputs'] and ck['sources']==tp['sources'],'Payload source/input/recipe/protocol binding')
    need(ck['task_bundle']==p['inputs']['task_bundle'] and set(ck['arms'])==set(MODES),'Paired Actor task/interface')
    need(ck['policy_interface']==dict(obs_dim=222,memory_dim=83,action_dim=20,arm_fields=['stateactor','statecritic','statecritic_target','norm']),'Explicit current Actor interface')
    actor_module=mod('fixed1000_actor_factory',checked(p['sources']['imagination']))
    root=Path(p['sources']['Actor_Critic']['path']).parents[3];sys.path.insert(0,str(root))
    actors={}
    for mode in MODES:
        modules=actor_module.make_fresh_actor_critic(seed=tp['recipe']['seed'],source_pins=p['sources'],device='cpu')
        actor=modules['actor'];actor.load_state_dict(ck['arms'][mode]['stateactor'],strict=True);actor.eval();actor.requires_grad_(False)
        need(raw_equal(actor.state_dict(),ck['arms'][mode]['stateactor'],torch) and raw_equal(actor.norm.state_dict(),ck['arms'][mode]['norm']['actor'],torch),'Exact serialized Actor + norm')
        need(all(torch.isfinite(v).all().item() for v in actor.state_dict().values()),'Finite Actor weights/norm')
        actors[mode]=actor
        del modules
    return actors,dict(checkpoint=final['checkpoint'],own_CPU_checkpoint_loads=1,factory_calls=2,Actor_constructors=2,unused_Critic_constructors=2,WM_constructors=0,WM_FW=0,Critic_FW=0,loaded_raw_Actor_and_norm_equal=True)

def angle_rad(R,Rref):
    rel=R@Rref.T;v=np.array([rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1]])*.5
    return float(np.arctan2(np.linalg.norm(v),(np.trace(rel)-1)*.5))
def tracking_from_saved(trace_path,task,codec,ipos,dt):
    """101 command ticks, native integer POST indexing, no clock rounding.

    Missing native tail stays NaN/masked; fixed-object comparator is not a
    new physical rollout. No geometry/force/holding success is inferred.
    """
    with np.load(trace_path,allow_pickle=False) as z:q=z['post_qpos'].copy();v=z['post_qvel'].copy();q0=z['initial_qpos27'].copy()
    n=int(round(.02/dt));need(n in [64,128] and n*dt==.02,'Fixed integer H ticks')
    ticks=np.arange(101,dtype=np.int64)*n;known=ticks<=len(q);x0=codec.actual_state(q0,np.zeros(26),ipos)
    x=np.full((101,58),np.nan);x[0]=x0
    for k in np.flatnonzero(known)[1:]:x[k]=codec.actual_state(q[ticks[k]-1],v[ticks[k]-1],ipos)
    refCOM=np.asarray(task.reference_COM);refR=np.asarray(task.reference_pose)[:,:3,:3]
    p_error=np.full(101,np.nan);R_error=np.full(101,np.nan)
    fixedp=np.linalg.norm(x0[40:43]-refCOM,axis=1);fixedR=np.array([angle_rad(x0[43:52].reshape(3,3),r) for r in refR])
    for k in np.flatnonzero(known):p_error[k]=np.linalg.norm(x[k,40:43]-refCOM[k]);R_error[k]=angle_rad(x[k,43:52].reshape(3,3),refR[k])
    def stats(a,mask):
        z=a[mask];return dict(count=int(len(z)),RMS=float(np.sqrt(np.mean(z*z))) if len(z) else None,mean=float(np.mean(z)) if len(z) else None,max=float(np.max(z)) if len(z) else None,final_available=float(z[-1]) if len(z) else None)
    # k0 is disclosed; headline post statistics exclude the initialized pose.
    post=known.copy();post[0]=False
    arrays=dict(command_index101=np.arange(101,dtype=np.int64),nominal_time_s101=np.arange(101)*.02,native_POST_tick101=ticks,known101=known,state58_F64=x,reference_COM_world_m101=refCOM,reference_R_BODY_world101=refR,COM_error_m101=p_error,SO3_geodesic_error_rad101=R_error,fixed_object_COM_error_m101=fixedp,fixed_object_SO3_geodesic_error_rad101=fixedR)
    stats_out=dict(available_POST_command_frames=int(post.sum()),missing_POST_command_frames=int((~known).sum()),COM_error_m=stats(p_error,post),SO3_geodesic_error_rad=stats(R_error,post),fixed_object_on_same_available_POST_frames=dict(COM_error_m=stats(fixedp,post),SO3_geodesic_error_rad=stats(fixedR,post)),fixed_object_full101_command_descriptor=dict(COM_error_m=stats(fixedp,np.ones(101,dtype=bool)),SO3_geodesic_error_rad=stats(fixedR,np.ones(101,dtype=bool))),tracking_success_gate=None,scope='Full SO3 angle and COM tracking descriptors, not holding5mm success; no missing-tail fill, temporal stretch or accumulated-clock tick index')
    return arrays,stats_out

def run(protocol):
    p,b,pending=metadata_preflight(protocol);need(not pending and all(p[k] is True for k in ['CPU_run_authorized','NN_forward_authorized','native_execution_authorized']),'Root actual authorization required')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch,mujoco
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(p['output']);out.mkdir();write(out/'protocol.json',p);start=time.perf_counter();phase='load';cases=[]
    counts=dict(MJB_reads=0,MjData=0,initial_forward=0,integration_steps=0,contact_force_reads=0,controller_calls=0,controller_receipt_reads=0,Actor_FW=0,WM_FW=0,Critic_FW=0,observer_forward=0,optimizer_steps=0,backward_calls=0,GPU_calls=0)
    try:
        actors,load_receipt=load_final_actors(protocol,torch)
        trainer=mod('fixed1000_canonical_memory',checked(p['sources']['trainer']));controller_module=mod('fixed1000_control_callback',checked(p['sources']['controller']))
        kernel=mod('fixed1000_original_native_kernel',checked(p['sources']['kernel']));taskmod=mod('fixed1000_phone_task',checked(p['sources']['task']));gap=mod('fixed1000_current_geometry',checked(p['sources']['directional']));codec=mod('fixed1000_COM_world_codec',checked(p['sources']['scalar_codec']))
        model=mujoco.MjModel.from_binary_path(str(checked(p['inputs']['model'])));counts['MJB_reads']=1
        phone,pg,table,pads=kernel.frozen.assert_model(model,mujoco,np,read(checked(p['inputs']['binding'])))
        digest=kernel.frozen.model_digest_except_dt(model,np);geometry=gap.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']))
        need(kernel.frozen.PAIR_GATES==p['native_dt_pair_gates'],'Actual frozen native pair gates')
        variants=[];endpoints_by_variant={}
        before={mode:{k:v.detach().clone() for k,v in actor.state_dict().items()} for mode,actor in actors.items()};CPU_rng=torch.get_rng_state().clone()
        for mode in MODES:
            # Both modes share observation/control semantics; rewards are not
            # evaluated here, so A2's engineered proxy need not be constructed.
            task=trainer.CanonicalMemoryTask(taskmod.PhoneTask(checked(p['inputs']['task_bundle']),mode='A1_tracking',geometry=geometry))
            for dt in DTS:
                phase=mode+'_dt_'+str(dt);base=next(c for c in p['original_cases'] if c['dt_s']==dt);variant=dict(base['command']);variant['name']=mode+'__fixed1000'
                if variant['name'] not in endpoints_by_variant:variants.append(variant);endpoints_by_variant[variant['name']]=[]
                folder=out/(mode+'__dt'+str(dt));folder.mkdir()
                callback=controller_module.TaskController(task,geometry,codec,dt,phone,actor=actors[mode],device='cpu')
                summary,endpoint=kernel.evaluate_case_controlled(model,variant,dt,folder,phone,pg,table,pads,mujoco,np,counts,callback)
                endpoints_by_variant[variant['name']].append(endpoint)
                need(callback.NN_FW==len(callback.records),'One Actor forward per issued50Hz command')
                need(all(callback.memory[k].dtype==torch.float64 for k in ['a1','a2','c1','last_u']),'Canonical F64 executed history')
                write(folder/'controller_records.json',callback.records);counts['Actor_FW']+=callback.NN_FW
                arrays,tracking=tracking_from_saved(folder/'trace.npz',task,codec,model.body_ipos[phone].copy(),dt)
                np.savez_compressed(folder/'tracking101.npz',**arrays);write(folder/'tracking_statistics.json',tracking)
                cases.append(dict(mode=mode,dt_s=dt,native_summary=summary,endpoint=endpoint,controller_calls=len(callback.records),controller_records=pin(folder/'controller_records.json'),tracking=tracking,tracking_arrays=pin(folder/'tracking101.npz'),tracking_statistics=pin(folder/'tracking_statistics.json')))
                write(out/'completed_case_manifest.json',dict(cases=cases,planned_cases=4,all_cases_complete=len(cases)==4))
        pairs=kernel.frozen.dt_pair_summary(variants,[c['native_summary'] for c in cases],endpoints_by_variant,np)
        need(len(pairs)==2 and {v['name'] for v in pairs}=={m+'__fixed1000' for m in MODES},'Both full frozen dt-pair records retained')
        write(out/'native_dt_pair_consistency.json',pairs)
        pair_by_name={v['name']:v for v in pairs}
        checks=dict(all4_fixed_cases_recorded=len(cases)==4,exact_Actor_FW_callback_count=counts['Actor_FW']==counts['controller_calls'],finite_declared_budget=counts['integration_steps']<=38400 and counts['Actor_FW']<=400,Actor_weights_norm_unchanged=all(raw_equal(before[k],actors[k].state_dict(),torch) for k in MODES),CPU_RNG_unchanged=raw_equal(CPU_rng,torch.get_rng_state(),torch),no_WM_training_or_observer_forward=counts['WM_FW']==counts['Critic_FW']==counts['optimizer_steps']==counts['backward_calls']==counts['observer_forward']==0,model_unchanged_except_dt=digest==kernel.frozen.model_digest_except_dt(model,np),CPU_only=not torch.cuda.is_initialized())
        need(all(checks.values()),'Evaluation integrity failed; preserve output, no retry')
        report=dict(schema='phone_tripod_fixed1000_Actor_native_evaluation_result:2',status='COMPLETED_ALL4_CASES_PHYSICAL_GATES_REPORTED_NOT_ASSUMED',protocol=pin(protocol),source=pin(__file__),checkpoint=load_receipt['checkpoint'],load_receipt=load_receipt,checks=checks,passed=sum(checks.values()),total=len(checks),counts=counts,cases=cases,
            physical_results=dict(tripod_cases_passing=sum(v['native_summary']['nominal_holding_dynamics_pass'] for v in cases),old_two_pad_cases_passing=sum(v['native_summary']['old_two_pad_pass'] for v in cases),double_dt_by_mode={mode:dict(comparison_status=pair_by_name[mode+'__fixed1000']['comparison_status'],gates=pair_by_name[mode+'__fixed1000']['gates'],dt_pair_consistency_pass=pair_by_name[mode+'__fixed1000']['dt_pair_consistency_pass'],both_tripod_cases_and_dt_pair_pass=pair_by_name[mode+'__fixed1000']['both_tripod_cases_pass'] and pair_by_name[mode+'__fixed1000']['dt_pair_consistency_pass'],both_old_two_pad_cases_and_dt_pair_pass=pair_by_name[mode+'__fixed1000']['both_old_two_pad_cases_pass'] and pair_by_name[mode+'__fixed1000']['dt_pair_consistency_pass']) for mode in MODES}),
            dt_pair_consistency=pairs,dt_pair_consistency_output=pin(out/'native_dt_pair_consistency.json'),
            elapsed_s=time.perf_counter()-start,scope=p['scope'],tracking_success_qualified=None,full_online_SuperMimic_implemented=False)
        write(out/'report.json',report);print(json.dumps(dict(report=pin(out/'report.json'),case_count=4,counts=counts,physical_results=report['physical_results'])),flush=True);return 0
    except BaseException as e:
        write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),counts=counts,completed_cases=len(cases),automatic_retry=False));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,required=True);a.add_argument('--run',action='store_true');args=a.parse_args()
    if args.run:raise SystemExit(run(args.protocol))
    p,b,pending=metadata_preflight(args.protocol);print(json.dumps(dict(status='METADATA_ONLY_PENDING_FINAL1000' if pending else 'METADATA_ONLY_FIXED_FINAL1000_READY',Torch_imported='torch' in sys.modules,NN_native_calls=0,output=p['output'])))
