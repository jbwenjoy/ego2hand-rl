"""Same-world native104 absolute-target interventions, without learning.

Nothing imports a device framework or constructs an environment on import.
The physical producer never loads or executes an Actor or a world model.
"""
from __future__ import annotations
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import importlib.util,json,random,time,sys
import numpy as np
from .io import checked,pin,read,jsonable
from .metadata import write

PATHS=['A0','A1L','P_ONLY','A2']
PHASES=[0,4,8,12]


def branch_schedule():
    rows=[]
    for s in range(16):
        base=dict(snapshot_id=s,path=PATHS[s//4],k=PHASES[s%4])
        slots=[dict(type='baseline',baseline_repeat=b,j=None,epsilon_index=None,epsilon_rad=None,sign=0) for b in range(3)]
        slots += [dict(type='direction',baseline_repeat=None,j=j,epsilon_index=e,epsilon_rad=eps,sign=sign)
                  for e,eps in enumerate((.002,.001)) for j in range(8) for sign in (-1,1)]
        for slot in slots:
            rows.append(dict(row_id=len(rows),**base,**slot,actualu=None,r_k=None,r_kplus1=None,
                             restorequalified=False,status='NOT_RUN',reason='not_executed'))
    return rows


def baseline_errors(x,truth,reward,truth_reward):
    delta=abs(np.asarray(x,dtype=np.float64)-np.asarray(truth,dtype=np.float64))
    groups=dict(q=(0,20,2e-5),qd=(20,40,2e-4),p=(40,43,2e-5),
                R=(43,52,2e-5),v=(52,55,2e-4),omega=(55,58,2e-4))
    errors={name:float(delta[a:b].max()) for name,(a,b,tol) in groups.items()}
    errors['reward']=abs(float(reward)-float(truth_reward))
    passed=all(errors[name]<=tol for name,(_,_,tol) in groups.items()) and errors['reward']<=2e-5
    return dict(errors=errors,passed=bool(passed))


def object_terms(x,p,R):
    """Original FP32 2+8 formula, evaluated on CPU copies of measured state."""
    import torch
    a=torch.as_tensor(np.asarray(x,np.float32)).reshape(1,58)
    pos=a[:,40:43];rot=a[:,43:52].reshape(-1,3,3)
    gp=torch.as_tensor(np.asarray(p,np.float32)).reshape(1,3)
    gR=torch.as_tensor(np.asarray(R,np.float32)).reshape(1,3,3)
    off=.1*rot[:,:,0];goff=.1*gR[:,:,0]
    endpoint=((pos+off-gp-goff).norm(dim=-1)+(pos-off-gp+goff).norm(dim=-1))*.5
    terms=torch.stack((2*torch.exp(-((pos-gp).norm(dim=-1)/.05).square()),
                       8*torch.exp(-(endpoint/.04).square())),-1)[0]
    return terms.numpy().copy(),float(terms.sum())


@contextmanager
def backend_call_trace(stage,counts):
    """Count public simulation calls and nested construction/capture API calls.

    Nested boundaries are reported separately, never summed as kernel calls.
    Graph replay public step calls represent one native step graph each.
    """
    import mujoco,mujoco_warp as mjwarp
    from mjlab.sim.sim import Simulation
    originals=[]
    for owner,name,label in [(Simulation,'step','Simulation.step'),(Simulation,'forward','Simulation.forward'),
                             (mjwarp,'step','mjwarp.step_API'),(mjwarp,'forward','mjwarp.forward_API'),
                             (mujoco,'mj_forward','CPU_mj_forward_API')]:
        original=getattr(owner,name);originals.append((owner,name,original))
        def counted(*args,_original=original,_label=label,**kwargs):
            counts[stage['name']][_label]+=1
            return _original(*args,**kwargs)
        setattr(owner,name,counted)
    try:yield
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)


def run_physical_probe(config,bundle,output,emit):
    import torch,warp as wp,mujoco,mujoco_warp as mjwarp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP,PenSimReader
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand
    from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat,mm,mv
    from .runtime_v2 import rough_runtime
    from .controller import rough_controller,PURE_PATH,PURE_SHA256,_canonical
    from .evaluation_v2 import configure_full_horizon_terminations
    from .native_v2 import install_actual_context
    from .policy_probe import expected_control
    from .capacity import CapacityObserver
    from .probe_transaction_v3 import ProbeTransaction
    from .bundle import FIELDS

    output=Path(output);deadline=config['_worker_started_perf']+config['soft_worker_seconds']
    seed=config['evaluation_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch._dynamo.config.suppress_errors=False
    if not torch.cuda.is_available():raise RuntimeError('Isolated NVIDIA environment required')
    wp.set_device('cuda:0')
    proposal=read(checked(config['parent_proposal']))
    with np.load(checked(config['directions']['arrays']),allow_pickle=False) as data:
        dirs={k:data[k].copy() for k in data.files}
    if not dirs['valid_pair'].all():raise ValueError('Frozen direction set contains invalid interventions')
    rows=branch_schedule();snapshots=[];donors=[];ledgers=[];restores=[]
    arrays={name:np.full(shape,np.nan,np.float32) for name,shape in {
        'snapshot_x':(16,58),'snapshot_u':(16,20),'donor_next_x':(16,58),'donor_self15':(16,15),
        'goal_p':(16,3),'goal_R':(16,3,3),'goal_p_next':(16,3),'goal_R_next':(16,3,3),
        'branch_x_next':(560,58),'branch_self15':(560,15),'branch_u':(560,20),
        'branch_object_terms_k':(560,2),'branch_object_terms_kplus1':(560,2),
        'donor_object_terms_k':(16,2),'donor_object_terms_kplus1':(16,2)}.items()}
    arrays.update(snapshot_completed=np.zeros(16,bool),completed=np.zeros(560,bool),restore_qualified=np.zeros(560,bool))
    stage={'name':'imports_and_setup'};counts=defaultdict(lambda:defaultdict(int));phase_seconds=defaultdict(float);phase_clock=[time.perf_counter()]
    def mark(name):
        now=time.perf_counter();phase_seconds[stage['name']]+=now-phase_clock[0];phase_clock[0]=now;stage['name']=name
    report=dict(schema='current104_physical_probe_worker_report:1',status='INITIALIZING',
                engineering_qualified=False,baseline_all_passed=False,capacity_finalized=False,
                capacity_qualified=False,source_lineage_N=1,independent_ego_N=0,
                planned_branch_count=560,scheduled_substeps=0,optimizer_updates=0,
                WM_forwards=0,Actor_forwards=0,physical_reference_qualified=False)
    wall=time.perf_counter();error=None
    if pin(PURE_PATH)['sha256']!=PURE_SHA256:raise ValueError('Frozen C2 changed')
    spec=importlib.util.spec_from_file_location('_physical_probe_C2',PURE_PATH)
    pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    def checktime():
        if time.perf_counter()>deadline:raise TimeoutError('Fixed current104 probe deadline; no retry')
    def save():
        np.savez_compressed(output/'physical_arrays.npz',**arrays)
        write(output/'branch_index.json',rows);write(output/'snapshot_records.json',snapshots)
        write(output/'donor_records.json',donors);write(output/'restore_receipts.json',restores)
        write(output/'capacity_ledgers.json',ledgers)
        report.update(completed_branch_count=int(arrays['completed'].sum()),
            completed_snapshot_count=int(arrays['snapshot_completed'].sum()),
            backend_call_counts={k:dict(v) for k,v in counts.items()},
            phase_wall_seconds={**dict(phase_seconds),stage['name']:phase_seconds[stage['name']]+time.perf_counter()-phase_clock[0]},
            call_count_scope='PublicSimulation and nested API boundaries separate; graph capture API calls not summed with public graph replays.',
            native_wall_seconds_excluding_imports=time.perf_counter()-wall)
        write(output/'physical_report.json',report)

    try:
        with torch.no_grad(),backend_call_trace(stage,counts):
            with (output/'physical_samples.jsonl').open('x',encoding='utf-8') as samplefile, (output/'reward_snapshots.jsonl').open('x',encoding='utf-8') as rewardfile:
                for path_i,dataset in enumerate(config['datasets']):
                    checktime();name=dataset['name'];pathdir=output/('donor_'+name);pathdir.mkdir()
                    archived=Path(dataset['directory']);old_controller=read(checked(dataset['input_pins']['controller_descriptor.json']))
                    descriptor=old_controller['root_descriptor'];events=read(checked(dataset['input_pins']['control_events.json']))
                    with np.load(checked(dataset['input_pins']['raw_states.npz']),allow_pickle=False) as old:
                        archive={k:old[k].copy() for k in old.files}
                    targets=np.asarray([e['actualu'] for e in events],np.float32).reshape(15,20)
                    actions=np.zeros((15,1,20),np.float32) if name=='A0' else np.asarray([e['action'] for e in events],np.float32)
                    if actions.shape!=(15,1,20) or not np.isfinite(actions).all() or (abs(actions)>1).any():raise ValueError('Bad archived source action')
                    if name!='A0' and any(not np.array_equal(np.clip(np.asarray(e['actor_raw_action'],np.float32),-1,1),actions[k]) for k,e in enumerate(events)):
                        raise ValueError('Saved action differs from original Actor clamp')
                    cfg=make_env_cfg(1,50,play=True,recipe_path=Path(bundle['native_reference']['recipe_path']),clips_root=Path(bundle['native_reference']['clips_root']))
                    cfg.seed=seed;cfg.auto_reset=False;motion=cfg.commands['motion']
                    motion.first_frame_hold_prob=0.;motion.first_frame_hold_s=(0.,0.)
                    motion.trajectory_sampling_mode='uniform';motion.debug_vis=False
                    cfg.actions['joint_pos'].mask_prob=0.;cfg.sim.nconmax=192;cfg.sim.njmax=4096
                    if cfg.sim.nan_guard.enabled is not False:raise ValueError('Only current disabled NanGuard snapshot ABI is qualified')
                    configure_full_horizon_terminations(cfg)
                    if set(cfg.events)!={'reset_to_reference_state','pen_dimensions'} or any(g.enable_corruption for g in cfg.observations.values()):raise ValueError('Unexpected exogenous event/noise')
                    for key in ('diameter_range_m','total_length_range_m'):
                        a,b=cfg.events['pen_dimensions'].params[key]
                        if a!=b:raise ValueError('Randomized pen dimensions prohibited')
                    env=None;reader=None;observer=None;original_step=None;original_process=None
                    local_txs={};local_states=[];local_events=[]
                    ctx={'expected_u':None,'step_in_interval':0,'mode':'setup','branch_id':None,'phase':0,'override':None,'previous_raw':None,'nominal_prediction':None}
                    try:
                        with rough_runtime(ManagerBasedRlEnv,SingleHandTrackingCommand,PenMDP,DexSuperMimicRunner,bundle['arrays'],descriptor):
                            with rough_controller(ManagerBasedRlEnv,PenMDP,DexSuperMimicRunner,bundle['baseline_u15'],descriptor):
                                mark(name+'/constructor');emit(status='initializing',phase=stage['name'])
                                env=ManagerBasedRlEnv(cfg,device='cuda:0')
                                model=env.sim.mj_model
                                if model.nplugin!=0:raise ValueError('No plugin state ABI is qualified')
                                mujoco.mj_saveModel(model,str(pathdir/'nominal_model.mjb'),None)
                                if pin(pathdir/'nominal_model.mjb')['sha256']!=bundle['model']['sha256']:raise ValueError('Nominal MJB differs')
                                mark(name+'/actual8');install_actual_context(env,bundle,pathdir)
                                observer=CapacityObserver(env,stage=name,fixed_capacities={'nconmax':192,'njmax':4096})
                                mark(name+'/checked_reset');observer.before_reset();obs,_=env.reset(seed=seed)
                                reader=PenSimReader(env);reader.attach_capture()
                                capture_reward=env.reward_manager.compute
                                def reward_capture(dt):
                                    result=capture_reward(dt)
                                    reader.reward_snapshot['raw_found']=env.scene['finger_collision'].data.found.clone()
                                    reader.reward_snapshot['raw_sensordata']=env.sim.wp_data.sensordata.numpy().copy()
                                    return result
                                env.reward_manager.compute=reward_capture
                                if not np.array_equal(reader.xi.cpu().numpy(),np.broadcast_to(bundle['context']['reader_context'][0],reader.xi.shape)):raise ValueError('Actual xi46 differs')
                                if _canonical(reader.mdp.rough_controller_metadata)!=_canonical(old_controller):raise ValueError('Original controller descriptor differs')
                                np.savez(pathdir/'reader_context.npz',reader_context=reader.xi.cpu().numpy().copy())
                                write(pathdir/'controller_descriptor.json',reader.mdp.rough_controller_metadata)
                                term=env.action_manager.get_term('joint_pos');host=mujoco.MjData(model)
                                jids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
                                aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids])
                                obj=model.body('object/cube').id
                                lower,upper=[v[0].cpu().numpy().copy() for v in term.target_position_limits]
                                bias=term._entity.data.encoder_bias[:,term._target_ids].cpu().numpy().copy()
                                write(pathdir/'control_metadata.json',dict(lower=lower,upper=upper,encoder_bias=bias,actuator_ids=aids,
                                    actuator_force_range=model.actuator_forcerange[aids].copy(),joint_ids=jids,
                                    target_projection_on_each_native_apply='Existing hard/ctrl limiter remains identity for every prevalidated absolute intervention; actual ctrl checked all4substeps.'))
                                actual_expected={f:bundle['context'][f][0].copy() for f in FIELDS}
                                sig=int(mujoco.mjtState.mjSTATE_INTEGRATION);size=mujoco.mj_stateSize(model,sig)
                                statebuf=wp.empty((1,size),dtype=wp.float32,device=env.device)
                                def integration():
                                    mjwarp.get_state(env.sim.wp_model,env.sim.wp_data,statebuf,sig);wp.synchronize()
                                    return statebuf.numpy().copy()
                                def raw():
                                    wp.synchronize_device(env.device)
                                    nacon=int(env.sim.wp_data.nacon.numpy().reshape(-1)[0]);nefc=int(env.sim.wp_data.nefc.numpy().reshape(-1)[0])
                                    if env.sim.wp_data.contact.geom.shape[0]!=192 or env.sim.wp_data.efc.force.shape[-1]!=4096 or nacon>=192 or nefc>=4096:raise ValueError('Actual capacity/maxcount unqualified')
                                    mjwarp.get_data_into(host,model,env.sim.wp_data,world_id=0)
                                    out=dict(qpos=host.qpos.astype(np.float32).copy(),qvel=host.qvel.astype(np.float32).copy(),
                                        ctrl=host.ctrl[aids].astype(np.float32).copy(),time=float(host.time),
                                        mocap_pos=host.mocap_pos.astype(np.float32).copy(),mocap_quat=host.mocap_quat.astype(np.float32).copy(),
                                        nacon=nacon,nefc=nefc)
                                    if not all(np.isfinite(out[k]).all() for k in ('qpos','qvel','ctrl','mocap_pos','mocap_quat')):raise FloatingPointError('Nonfinite raw state')
                                    if max(abs(host.xfrc_applied).max(),abs(host.qfrc_applied).max())!=0:raise ValueError('Exogenous object/state force prohibited')
                                    return out
                                initial=raw()
                                write(pathdir/'initial_readback.json',initial)
                                for key,oldkey in [('qpos','full_qpos'),('qvel','full_qvel'),('ctrl','ctrl_rad'),('mocap_pos','mocap_pos_w'),('mocap_quat','mocap_quat_wxyz')]:
                                    if not np.array_equal(initial[key].view(np.uint32),archive[oldkey][0].astype(np.float32).view(np.uint32)):raise ValueError('Archived startup identity differs '+key)
                                if abs(initial['time'])>2e-6:raise ValueError('Initial clock differs')
                                local_states.append(initial)
                                original_step=env.sim.step;original_process=term.process_actions
                                def process(action):
                                    prior=term.processed_action.cpu().numpy().copy()
                                    pred=expected_control(action.cpu().numpy(),prior,bundle['baseline_u15'][ctx['phase']],lower,upper,bias,pure.target_inset)
                                    original_process(action)
                                    if not np.array_equal(term.processed_action.cpu().numpy().view(np.uint32),pred['residual'].view(np.uint32)) or not np.array_equal(term._targets.cpu().numpy().view(np.uint32),pred['actual'].view(np.uint32)):
                                        raise ValueError('Original EMA/C2 processing differs')
                                    ctx['nominal_prediction']=pred
                                    if not np.array_equal(pred['actual'][0].view(np.uint32),targets[ctx['phase']].view(np.uint32)):raise ValueError('Fresh nominal target differs from archived center')
                                    wanted=targets[ctx['phase']] if ctx['override'] is None else ctx['override']
                                    if np.any(wanted<dirs['lower']) or np.any(wanted>dirs['upper']):raise ValueError('Target intervention outside legal inner domain')
                                    if ctx['override'] is not None:term._targets.copy_(torch.from_numpy(wanted.copy())[None].to(env.device))
                                    ctx['expected_u']=wanted.copy()
                                def substep():
                                    checktime();original_step();now=raw();ctx['step_in_interval']+=1
                                    record=observer._read('after_substep');observer._raise_if_unqualified(record)
                                    if not np.array_equal(now['ctrl'].view(np.uint32),ctx['expected_u'].view(np.uint32)):raise ValueError('Actual intervention target changed by native apply')
                                    prev=ctx['previous_raw']
                                    if prev is None or abs(now['time']-prev['time']-.005)>2e-6:raise ValueError('Substep clock mismatch')
                                    contacts=[]
                                    for ci,c in enumerate(host.contact):
                                        force=np.zeros(6);mujoco.mj_contactForce(model,host,ci,force)
                                        if not all(np.isfinite(v).all() for v in (force,c.pos,c.frame,np.asarray(c.dist))):raise FloatingPointError('Nonfinite solved contact')
                                        contacts.append(dict(geom=list(map(int,c.geom)),efc_address=int(c.efc_address),dist=float(c.dist),pos=np.array(c.pos),frame=np.array(c.frame).reshape(3,3),force6=force))
                                    native_contact={f:getattr(env.sim.wp_data.contact,f).numpy()[:now['nacon']].copy() for f in ('geom','type','worldid','efc_address')}
                                    sample=dict(path=name,phase=ctx['phase'],mode=ctx['mode'],branch_id=ctx['branch_id'],substep=ctx['step_in_interval'],
                                        pre_time=prev['time'],post_time=now['time'],pre_qpos=prev['qpos'],pre_qvel=prev['qvel'],post_qpos=now['qpos'],post_qvel=now['qvel'],
                                        ctrl=now['ctrl'],actuator_force=host.actuator_force[aids].copy(),nacon=now['nacon'],nefc=now['nefc'],
                                        object_solver_COM=host.xipos[obj].copy(),object_solver_pose=np.r_[host.xpos[obj],host.xquat[obj]],
                                        external_force_max=0.,raw_wp_contact=native_contact,raw_efc_force=env.sim.wp_data.efc.force.numpy()[0,:now['nefc']].copy(),
                                        raw_efc_type=env.sim.wp_data.efc.type.numpy()[0,:now['nefc']].copy(),contacts=contacts,
                                        phase_contract='solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward')
                                    samplefile.write(json.dumps(jsonable(sample),allow_nan=False)+'\n')
                                    report['scheduled_substeps']+=1;ctx['previous_raw']=now
                                    if ctx['mode']=='donor':local_states.append(now)
                                term.process_actions=process;env.sim.step=substep
                                palm=reader.palm_rot[0].T;pp=reader.palm_pos[0]
                                poses=torch.tensor(bundle['arrays']['reference']['mapped_object_pose_w'],dtype=torch.float32,device=env.device)
                                gp=mv(palm[None],poses[:,:3]-pp).cpu().numpy();gR=mm(palm[None],quat_to_mat(poses[:,3:])).cpu().numpy()
                                def interval(k,override=None,branch_id=None):
                                    checktime();ctx.update(phase=k,override=override,branch_id=branch_id,step_in_interval=0,previous_raw=raw())
                                    phase=env.command_manager.get_term('motion').time_steps
                                    if phase.tolist()!=[k]:raise ValueError('Preaction phase mismatch')
                                    before=counts[stage['name']]['Simulation.forward']
                                    ob,reward,terminated,truncated,_=env.step(torch.from_numpy(actions[k].copy()).to(env.device))
                                    if ctx['step_in_interval']!=4 or counts[stage['name']]['Simulation.forward']-before!=1:raise ValueError('Native4step/oneforward cadence differs')
                                    if env.command_manager.get_term('motion').time_steps.tolist()!=[k+1] or bool(terminated.any()) or bool(truncated.any())!=(k==14):raise ValueError('Native phase/termination differs')
                                    snap=reader.reward_snapshot
                                    if snap is None:raise ValueError('Missing original reward capture')
                                    x=snap['physics'][0].cpu().numpy().copy();contact=snap['contact'][0].cpu().numpy().copy()
                                    if x.shape!=(58,) or contact.shape!=(15,) or not np.isfinite(x).all() or not np.isin(contact,[0,1]).all():raise ValueError('Bad native training labels')
                                    if not np.array_equal(x.view(np.uint32),reader.state()[0].cpu().numpy().view(np.uint32)):raise ValueError('Reward capture/raw state58 differ')
                                    rawstate=raw()
                                    if abs(rawstate['time']-(k+1)*.02)>2e-6:raise ValueError('Interval clock mismatch')
                                    out=dict(x=x,contact=contact,torque=snap['torque'][0].cpu().numpy().copy(),
                                        raw_found=snap['raw_found'].cpu().numpy().copy(),
                                        raw_sensordata=snap['raw_sensordata'].copy(),
                                        rawstate=rawstate,terms_k=object_terms(x,gp[k],gR[k])[0],terms_next=object_terms(x,gp[k+1],gR[k+1])[0])
                                    rewardfile.write(json.dumps(jsonable(dict(path=name,k=k,mode=ctx['mode'],branch_id=branch_id,
                                        physics=x,contact=contact,torque=out['torque'],raw_found=out['raw_found'],
                                        raw_sensordata=out['raw_sensordata'],label_time='lastsolver_preintegration~kdt+.015',
                                        state_time='postintegration_kdt+.020',capture='after_original_reward_compute_before_Manager_forward')),allow_nan=False)+'\n')
                                    return out
                                ctx['mode']='donor';mark(name+'/donor')
                                for k in range(15):
                                    if k in PHASES:
                                        mark(name+'/snapshot_capture')
                                        sid=path_i*4+PHASES.index(k)
                                        if not np.array_equal(targets[k].view(np.uint32),dirs['centers'][sid].view(np.uint32)):raise ValueError('Frozen direction center differs')
                                        arrays['snapshot_x'][sid]=reader.state()[0].cpu().numpy();arrays['snapshot_u'][sid]=targets[k]
                                        arrays['goal_p'][sid]=gp[k];arrays['goal_R'][sid]=gR[k];arrays['goal_p_next'][sid]=gp[k+1];arrays['goal_R_next'][sid]=gR[k+1]
                                        # Populate read-only memory/bridge caches before inventory capture.
                                        # No observe/sense/forward/action processing is invoked here.
                                        reader.memory()
                                        tx=ProbeTransaction(env,reader,capacity_observer=observer,source_action=actions[k].copy(),label=name+'/'+str(k),
                                            external_roots={'ctx':ctx,'stage':stage,'counts':counts,'samplefile':samplefile,'rewardfile':rewardfile,'arrays':arrays,'rows':rows,'records':restores,
                                                            'report':report,'local_states':local_states,'local_events':local_events,'local_txs':local_txs,'bundle':bundle,'dirs':dirs,'archive':archive,
                                                            'proposal':proposal,'config':config,'dataset':dataset,'snapshots':snapshots,'donors':donors,'ledgers':ledgers,
                                                            'Torch_module':torch,'Warp_module':wp,'MuJoCo_module':mujoco,'MJWarp_module':mjwarp,
                                                            'C2_readonly_module':pure,'host_readback_only':host})
                                        integ=integration();mjwarp.set_state(env.sim.wp_model,env.sim.wp_data,statebuf,sig)
                                        if not np.array_equal(integration().view(np.uint32),integ.view(np.uint32)):raise ValueError('INTEGRATION set/get roundtrip differs')
                                        tx.verify_current()
                                        state_archive={'integration':integ.copy(),'source_action':actions[k].copy()}
                                        for key,value in reader.memory().items():state_archive['PEN__'+key]=value.cpu().numpy().copy()
                                        for i,(_,value) in enumerate(tx.native.arrays):state_archive['native_Warp_'+str(i)]=value.numpy().copy()
                                        for i,(_,value) in enumerate(tx.native.tensors):state_archive['native_Torch_'+str(i)]=value.cpu().numpy().copy()
                                        for i,(_,_,value,_) in enumerate(tx.sidecars.arrays):
                                            state_archive['explicit_array_'+str(i)]=value.cpu().numpy().copy() if torch.is_tensor(value) else value.numpy().copy() if hasattr(value,'numpy') else value.copy()
                                        state_archive['RNG_Torch_CPU']=tx._rng['torch_cpu'].numpy().copy()
                                        state_archive['RNG_NumPy_MT']=tx._rng['numpy'][1].copy()
                                        for i,value in enumerate(tx._rng.get('torch_cuda',[])):state_archive['RNG_Torch_CUDA_'+str(i)]=value.cpu().numpy().copy()
                                        archivefile=pathdir/('snapshot_'+str(k)+'.npz');np.savez(archivefile,**state_archive)
                                        rngfile=pathdir/('snapshot_'+str(k)+'_RNG.json')
                                        write(rngfile,dict(python=tx._rng['python'],NumPy_header=[tx._rng['numpy'][0],*tx._rng['numpy'][2:]]))
                                        local_txs[k]=(tx,integ.copy());snapshots.append(dict(snapshot_id=sid,path=name,k=k,integration_sig=sig,integration_size=size,
                                            integration_roundtrip_bitexact=True,inventory=tx.describe(),state_archive=pin(archivefile),rng_archive=pin(rngfile),
                                            archive_scope='Native Warp/Torch arrays, explicit array sidecars, PEN memory, integration and RNG; Python object identities/containers remain in-process receipt evidence.'))
                                        mark(name+'/donor')
                                    outcome=interval(k);local_events.append(dict(phase=k,actualu=ctx['expected_u'],**outcome))
                                    if k in PHASES:
                                        sid=path_i*4+PHASES.index(k);arrays['donor_next_x'][sid]=outcome['x'];arrays['donor_self15'][sid]=outcome['contact']
                                        arrays['donor_object_terms_k'][sid]=outcome['terms_k'];arrays['donor_object_terms_kplus1'][sid]=outcome['terms_next'];arrays['snapshot_completed'][sid]=True
                                    if k%4==3:emit(status='running',phase=stage['name'],scheduled_substeps=report['scheduled_substeps'])
                                observer._raise_if_unqualified(observer._read('donor_terminal'))
                                donor_delta_q=max(float(abs(v['qpos'].astype(float)-archive['full_qpos'][i]).max()) for i,v in enumerate(local_states))
                                donor_delta_v=max(float(abs(v['qvel'].astype(float)-archive['full_qvel'][i]).max()) for i,v in enumerate(local_states))
                                qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids]
                                oj=model.joint('object/cube_freejoint').id;oq=int(model.jnt_qposadr[oj]);ov=int(model.jnt_dofadr[oj])
                                from .bundle_v2 import rotation
                                def drift_at(indices):
                                    q=np.stack([local_states[i]['qpos'] for i in indices]).astype(float);v=np.stack([local_states[i]['qvel'] for i in indices]).astype(float)
                                    qold=archive['full_qpos'][indices];vold=archive['full_qvel'][indices]
                                    return dict(joint_q_maxabs_rad=float(abs(q[:,qa]-qold[:,qa]).max()),joint_qd_maxabs_rad_s=float(abs(v[:,va]-vold[:,va]).max()),
                                        object_p_maxabs_m=float(abs(q[:,oq:oq+3]-qold[:,oq:oq+3]).max()),
                                        object_R_entry_maxabs=float(abs(rotation(q[:,oq+3:oq+7])-rotation(qold[:,oq+3:oq+7])).max()),
                                        object_v_world_maxabs_m_s=float(abs(v[:,ov:ov+3]-vold[:,ov:ov+3]).max()),
                                        object_omega_body_maxabs_rad_s=float(abs(v[:,ov+3:ov+6]-vold[:,ov+3:ov+6]).max()))
                                donors.append(dict(path=name,full61_state_count=len(local_states),actions=actions,events=local_events,
                                    archive_all61_drift=drift_at(list(range(61))),archive_selected_snapshot_drift=drift_at([4*k for k in PHASES]),
                                    archive_selected_next_drift=drift_at([4*k+4 for k in PHASES]),
                                    legacy_mixed_q_maxabs_descriptive_only=donor_delta_q,legacy_mixed_qd_maxabs_descriptive_only=donor_delta_v,
                                    archival_drift_flag=donor_delta_q>2e-5 or donor_delta_v>2e-4,
                                    archive_quality_not_regraded=True))
                                for k in PHASES:
                                    sid=path_i*4+PHASES.index(k);tx,integ=local_txs[k];baseline_pass=[]
                                    for row in rows[sid*35:(sid+1)*35]:
                                        checktime();bid=row['row_id'];ctx['mode']=row['type'];mark(name+'/restore')
                                        restore=None
                                        try:
                                            with tx.branch('branch_'+str(bid)) as restore:
                                                if not np.array_equal(integration().view(np.uint32),integ.view(np.uint32)):raise ValueError('Restored integration is not bitexact')
                                                for f,value in actual_expected.items():
                                                    got=getattr(env.sim.wp_model,f).numpy()
                                                    if not np.array_equal(got,np.broadcast_to(value.astype(got.dtype),got.shape)):raise ValueError('Restored actual8 differs '+f)
                                                arrays['restore_qualified'][bid]=True;row['restorequalified']=True
                                                u=targets[k].copy() if row['type']=='baseline' else dirs['plus' if row['sign']==1 else 'minus'][sid,row['j'],row['epsilon_index']].copy()
                                                mark(name+'/'+row['type'])
                                                outcome=interval(k,u,bid)
                                        except BaseException as exc:
                                            row.update(status='INVALID',reason=repr(exc));raise
                                        finally:
                                            restores.append(dict(branch_id=bid,begin=restore,transaction_events=tx.events[-2:]))
                                        arrays['branch_x_next'][bid]=outcome['x'];arrays['branch_self15'][bid]=outcome['contact'];arrays['branch_u'][bid]=u
                                        arrays['branch_object_terms_k'][bid]=outcome['terms_k'];arrays['branch_object_terms_kplus1'][bid]=outcome['terms_next']
                                        arrays['completed'][bid]=True
                                        row.update(actualu=u,r_k=float(np.sum(outcome['terms_k'],dtype=np.float32)),r_kplus1=float(np.sum(outcome['terms_next'],dtype=np.float32)),status='COMPLETED',reason=None)
                                        if row['type']=='baseline':
                                            gate=baseline_errors(outcome['x'],arrays['donor_next_x'][sid],row['r_k'],float(arrays['donor_object_terms_k'][sid].sum(dtype=np.float32)))
                                            baseline_pass.append(gate['passed']);restores[-1]['baseline_gate']=gate
                                            if row['baseline_repeat']==2 and not all(baseline_pass):raise ValueError('Three fresh baseline replay gates failed; no perturbations/retry')
                                    emit(status='running',phase=stage['name'],completed_branches=int(arrays['completed'].sum()),scheduled_substeps=report['scheduled_substeps']);save()
                    finally:
                        body_exception=sys.exc_info()[0] is not None;cleanup_errors=[]
                        mark(name+'/cleanup')
                        if env is not None:
                            if original_step is not None:env.sim.step=original_step
                            if original_process is not None:env.action_manager.get_term('joint_pos').process_actions=original_process
                        if observer is not None:
                            try:observer.final_read()
                            except BaseException as exc:cleanup_errors.append(dict(stage='capacity_final',error=repr(exc)))
                            finally:ledgers.append(dict(path=name,ledger=observer.snapshot()))
                        if reader is not None:
                            try:reader.detach_capture()
                            except BaseException as exc:cleanup_errors.append(dict(stage='detach_capture',error=repr(exc)))
                        if env is not None:
                            try:env.close()
                            except BaseException as exc:cleanup_errors.append(dict(stage='env.close',error=repr(exc)))
                        if cleanup_errors:
                            report.setdefault('cleanup_errors',[]).append(dict(path=name,errors=cleanup_errors))
                            if not body_exception:raise RuntimeError('Unqualified physical probe cleanup')
                report.update(status='COMPLETED_PHYSICAL_PROBE',engineering_qualified=True,baseline_all_passed=True,
                    capacity_finalized=len(ledgers)==4 and all(v['ledger']['finalized'] for v in ledgers),
                    capacity_qualified=len(ledgers)==4 and all(v['ledger']['engineering_capacity_qualified'] for v in ledgers))
                if not arrays['completed'].all() or not arrays['snapshot_completed'].all() or report['scheduled_substeps']!=2480 or not report['capacity_qualified']:
                    raise ValueError('Incomplete physical probe coverage/capacity')
    except BaseException as exc:
        error=exc;report.update(status='FAILED_ENGINEERING_PARTIAL',engineering_qualified=False,baseline_all_passed=False,error=repr(exc))
        raise
    finally:
        save()
    return report
