"""Full151/105 same-Warp serial executed-u probe; lazy Env/device imports.
No Actor, WM, optimizer or learning. Only Root-authorized run_probe runs physics.
"""
from __future__ import annotations
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path
import importlib.util,json,random,time,sys,re
import numpy as np
from full151_native.io import checked,pin,read,jsonable
from full151_native.metadata import write

PATHS=['A0','A2']
PHASES=[0,49,99,149]
JOINTS=[0,4,8,12]
EPS=(.002,.001)
SCALAR_CONTRACT={'primary':'native_GPU_RewardManager_step_reward_object_pos_and_object_rot_FP32_then_GPU_FP32_sum',
    'goal_phase':'preaction k','units':'unscaled object reward rate/rad','weights':[2.,8.],'sigma_m':[.05,.04],
    'axis_point_radius_m':.1,'dt_s':.02,'native_rounding':'RewardManager value=raw*weight*dt; _step_reward=value/dt; sum on GPU',
    'CPU_FP32_and_F64_state_rescore':'secondary arithmetic diagnostic, never substituted for primary',
    'kplus1':'CPU_FP32_state_rescore diagnostic only; no native command mutation',
    'arithmetic_bound':'8*eps32*max(1,abs(F64_expected_term)) per native rate term'}
DONOR_FILES=('controller_descriptor.json','control_events.json','raw_states.npz',
             'evaluation.json','config.json','fixed_physics_context.npz','native_reward_transitions.json')


class InventoryReviewRequired(RuntimeError):
    """Discovery evidence is retained; no baseline/restore qualification granted."""


def validate_config(config,bundle=None):
    """Metadata-only preflight, with no framework import or donor execution."""
    if config.get('schema')!='full151105_same_warp_physical_probe_run:1' or config.get('role')!='full151105_serial_physical_probe':raise ValueError('Explicit105 physical-probe schema/role required')
    if config.get('probe_mode') not in ('engineering_restore_smoke','serial_full'):raise ValueError('Unknown physical-probe mode')
    if type(config.get('GPU_run_authorized')) is not bool or config.get('training_authorized') is not False or config.get('automatic_retry') is not False or config.get('compile_fallback') is not False:raise ValueError('Explicit GPU flag/no-training/no-retry/no-fallback contract required')
    if type(config.get('evaluation_seed')) is not int or config['evaluation_seed']!=0:raise ValueError('Fixed source seed0 required')
    if type(config.get('soft_worker_seconds')) not in (int,float) or not 0<config['soft_worker_seconds']<=180 or config.get('hard_process_seconds')!=240:raise ValueError('Fixed bounded physical-probe budget required')
    proposal=read(checked(config['proposal']));checked(config['protocol']);checked(config['input_bundle'])
    for field,value in {'schema':'full151105_same_Warp_control_gradient_diagnostic_proposal:1','snapshot_phases':PHASES,'direction_joint_indices':JOINTS,'eps_rad':list(EPS),'baseline_repeats':3,'planned_branches':152,'planned_total_episode_substeps':1808,'physical_reference_qualified':False}.items():
        if proposal.get(field)!=value:raise ValueError('Frozen proposal field differs: '+field)
    if set(config.get('donor_inputs',{}))!=set(PATHS):raise ValueError('Exactly A0/A2 donor action sources required')
    for name in PATHS:
        donor=config['donor_inputs'][name];directory=Path(donor['directory']).resolve(strict=True)
        if set(donor['input_pins'])!=set(DONOR_FILES):raise ValueError('Exact donor saved-artifact keyset required')
        for filename in DONOR_FILES:
            if checked(donor['input_pins'][filename]).resolve()!=directory/filename:raise ValueError('Donor artifact route differs: '+filename)
        c=read(directory/'config.json');e=read(directory/'evaluation.json')
        if c['method']!=name or not e['complete'] or e['physical_steps']!=600 or e['control_intervals']!=150 or e['physical_reference_qualified'] is not False:raise ValueError('Full knownrough150-interval archived donor required')
        if name=='A2' and c['checkpoint']!=proposal['WM_checkpoint']:raise ValueError('A2 final checkpoint/source action identity differs from fixed proposal')
    pins=config.get('source_pins')
    if type(pins) is not dict or not pins:raise ValueError('Frozen source closure is required')
    for value in pins.values():checked(value)
    own=pin(__file__)
    if own not in pins.values():raise ValueError('This new worker source is not in the config closure')
    for alias in ('controller105','runtime105','native105','policy_probe105','native_pen','native_state','native_snapshot','Warp_support','Warp_io','old104_capacity','measured_contacts'):
        expected=proposal['source_pins'][alias];checked(expected)
        if expected not in pins.values():raise ValueError('Frozen proposal execution dependency absent from new closure: '+alias)
    transaction=pin(Path(__file__).with_name('transaction105.py'))
    if transaction not in pins.values():raise ValueError('Executed105 transaction component missing from source closure')
    if config.get('foreign_inventory_review') is not None:checked(config['foreign_inventory_review'])
    if bundle is not None:
        if bundle['descriptor']['input_bundle']!=config['input_bundle'] or np.asarray(bundle['baseline_u150']).shape!=(150,20):raise ValueError('Exact105 bundle/baseline150 required')
    return {'schema':'full151105_serial_probe_metadata_preflight:1','probe_mode':config['probe_mode'],'branch_slots':152,
            'scheduled_branches':152 if config['probe_mode']=='serial_full' else 3,'episode_substeps':1808 if config['probe_mode']=='serial_full' else 16,
            'GPU_started':False,'WM_or_Actor_forward':False,'Env_created':False,'foreign_inventory_review_pending':config.get('foreign_inventory_review') is None}


def dump_inventory(tx,directory,integration_value,source_action):
    """Save all visited array bytes plus object topology, even after init failure."""
    from .transaction105 import _bytes,_meta
    directory=Path(directory);directory.mkdir(exist_ok=False)
    values={'INTEGRATION_FP32':integration_value.copy(),'source_action_FP32':source_action.copy()};meta=[];entries=[];side_entries=[];cells=[];generators=[];pen_meta={}
    def topology(contents,kind):
        view=contents.items() if kind in ('object','dict') else enumerate(contents)
        return [{'key':str(k),'type':type(v).__module__+'.'+type(v).__qualname__,'identity':id(v),
            'scalar_repr':repr(v) if isinstance(v,(str,int,float,bool,type(None))) else None} for k,v in view]
    native=getattr(tx,'native',None)
    if native is not None:
        for kind,items in (('Warp',native.arrays),('Torch',native.tensors)):
            for i,(target,saved) in enumerate(items):
                key=kind+'_'+str(i)+'_bytes';raw=_bytes(saved);values[key]=np.frombuffer(raw,dtype=np.uint8).copy()
                meta.append({'key':key,'target_meta':_meta(target),'value_sha256':__import__('hashlib').sha256(raw).hexdigest()})
        for i,(obj,contents,kind) in enumerate(native.entries):
            entries.append({'entry':i,'kind':kind,'type':type(obj).__module__+'.'+type(obj).__qualname__,'identity':id(obj),
                'members':topology(contents,kind)})
    sidecars=getattr(tx,'sidecars',None)
    if sidecars is not None:
        for i,(path,target,saved,identity) in enumerate(sidecars.arrays):
            key='sidecar_'+str(i)+'_bytes';raw=_bytes(saved);values[key]=np.frombuffer(raw,dtype=np.uint8).copy()
            meta.append({'key':key,'path':path,'target_meta':identity,'value_sha256':__import__('hashlib').sha256(raw).hexdigest()})
        for path,obj,contents,kind in sidecars.entries:
            side_entries.append({'path':path,'kind':kind,'identity':id(obj),'members':topology(contents,kind)})
        for path,cell,saved in sidecars.cells:
            cells.append({'path':path,'identity':id(cell),'value_type':type(saved).__module__+'.'+type(saved).__qualname__,'value_identity':id(saved),
                          'scalar_repr':repr(saved) if isinstance(saved,(str,int,float,bool,type(None))) else None})
        for i,(path,obj,state,kind) in enumerate(sidecars.generators):
            item={'path':path,'kind':kind,'identity':id(obj)}
            if kind=='torch':
                key='local_generator_'+str(i)+'_bytes';values[key]=np.frombuffer(_bytes(state),np.uint8).copy();item['state_bytes_key']=key
            else:item['state']=state
            generators.append(item)
    for name,(identity,raw) in getattr(tx,'_memory',{}).items():
        key='PEN__'+name+'__bytes';values[key]=np.frombuffer(raw,np.uint8).copy();pen_meta[name]={'key':key,'meta':identity}
    rng=getattr(tx,'_rng',None)
    if rng is not None:
        write(directory/'RNG_python_numpy.json',{'python':rng['python'],'numpy_header':[rng['numpy'][0],*rng['numpy'][2:]]})
        values['RNG_numpy_MT']=rng['numpy'][1].copy()
        for label in ('torch_cpu','torch_cuda'):
            for i,value in enumerate(rng.get(label,[]) if label=='torch_cuda' else [rng[label]] if label in rng else []):
                values[label+'_'+str(i)]=np.frombuffer(_bytes(value),dtype=np.uint8).copy()
    np.savez_compressed(directory/'inventory_values.npz',**values)
    write(directory/'inventory_topology.json',{'schema':'full151105_snapshot_inventory_archive:1','native_entries':entries,'array_values':meta,
          'explicit_container_entries':side_entries,'closure_cells':cells,'local_generators':generators,'PEN_memory':pen_meta,
          'worker_source':pin(__file__),'transaction_source':pin(Path(__file__).with_name('transaction105.py')),
          'scope':'Visited in-process bytes/topology; opaque foreign internals/function globals are not serialized as complete native state.'})
    description=tx.describe() if hasattr(tx,'sidecars') and hasattr(tx,'_rng') else {'schema':'full151105_partial_transaction_construction:1','profile':getattr(tx,'_profile',None),'qualification':'PENDING','native_entries':len(native.entries) if native is not None else 0}
    write(directory/'transaction_description.json',description)
    return {'values':pin(directory/'inventory_values.npz'),'topology':pin(directory/'inventory_topology.json'),'description':pin(directory/'transaction_description.json')}


def reviewed_foreign_inventory(config,tx,bundle):
    """An explicit new source-bound review is necessary; None means discovery."""
    from . import transaction105
    review_pin=config.get('foreign_inventory_review');description=tx.describe()
    if review_pin is None:raise InventoryReviewRequired('Initial105 inventory discovery: no foreign/closure review supplied; no branches qualified')
    review=read(checked(review_pin))
    required={'worker':pin(__file__),'transaction':pin(transaction105.__file__),'input_bundle':config['input_bundle']}
    if review.get('schema')!='full151105_foreign_inventory_source_review:1' or any(review.get(k)!=v for k,v in required.items()) or review.get('source_reviewed_closure_globals') is not True:
        raise InventoryReviewRequired('Foreign/closure review is not for this exact105 worker/transaction/bundle')
    allowed=review.get('allowed_identity_only')
    if not isinstance(allowed,list) or any(set(v)!={'path','type','reason'} or not all(isinstance(v[k],str) and v[k] for k in v) for v in allowed):raise InventoryReviewRequired('Explicit per-path/type/reason opaque review required')
    actual={(v['path'],v['type']) for v in description['opaque_identity_only']}
    if actual!={(v['path'],v['type']) for v in allowed}:raise InventoryReviewRequired('Actual opaque inventory differs from reviewed exact set')
    for value in review.get('source_pins',{}).values():checked(value)
    if not review.get('source_pins'):raise InventoryReviewRequired('Independent foreign source review pins missing')
    return {'review':review_pin,'actual_opaque_count':len(actual),'qualification_scope':'Source-reviewed identity-only handles plus later empirical restore/baseline gates; no general serializer guarantee'}


def branch_schedule(probe_mode='serial_full'):
    if probe_mode not in ('serial_full','engineering_restore_smoke'):raise ValueError('Unknown probe mode')
    rows=[]
    for s in range(8):
        base=dict(snapshot_id=s,path=PATHS[s//4],k=PHASES[s%4])
        slots=[dict(type='baseline',baseline_repeat=b,j=None,epsilon_index=None,epsilon_rad=None,sign=0) for b in range(3)]
        slots += [dict(type='direction',baseline_repeat=None,j=j,epsilon_index=e,epsilon_rad=eps,sign=sign)
                  for e,eps in enumerate(EPS) for j in range(4) for sign in (-1,1)]
        for slot in slots:
            scheduled=probe_mode=='serial_full' or (s==0 and slot['type']=='baseline')
            rows.append(dict(row_id=len(rows),**base,**slot,joint_index=None if slot['j'] is None else JOINTS[slot['j']],actualu=None,r_k=None,r_kplus1=None,
                             scheduled=scheduled,restorequalified=False,status='NOT_RUN',reason='not_executed' if scheduled else 'not_scheduled_engineering_restore_smoke'))
    return rows


def execution_plan(probe_mode):
    """Pure mode/index contract; no donor file, framework or model access."""
    rows=branch_schedule(probe_mode);full=probe_mode=='serial_full'
    snapshots=[dict(snapshot_id=s,path=PATHS[s//4],k=PHASES[s%4],source_frame=200+PHASES[s%4],
        scheduled=full or s==0,status='NOT_RUN',reason='not_executed' if full or s==0 else 'not_scheduled_engineering_restore_smoke',
        integration_roundtrip_bitexact=False,inventory_construction_completed=False,foreign_review_qualified=False,
        full_native_restore_qualified=False,baseline3_qualified=False,longclock_qualified=False) for s in range(8)]
    return {'probe_mode':probe_mode,'donors':PATHS if full else PATHS[:1],'capture_phases':PHASES if full else [0],
        'donor_intervals':150 if full else 1,'donor_state_rows':601 if full else 5,
        'snapshot_ticks':[4*k for k in (PHASES if full else [0])],
        'next_ticks':[4*k+4 for k in (PHASES if full else [0])],
        'episode_substeps':1808 if full else 16,'branch_count':152 if full else 3,'snapshot_count':8 if full else 1,
        'branch_slots':rows,'snapshot_slots':snapshots}


def allocate_artifacts():
    """Fixed8/152 ABI, including unexecuted NaN rows and explicit masks."""
    arrays={name:np.full(shape,np.nan,np.float32) for name,shape in {
        'snapshot_x':(8,58),'snapshot_u':(8,20),'donor_next_x':(8,58),'donor_self15':(8,15),
        'goal_p':(8,3),'goal_R':(8,3,3),'goal_p_next':(8,3),'goal_R_next':(8,3,3),
        'branch_x_next':(152,58),'branch_self15':(152,15),'branch_u':(152,20),
        'branch_object_terms_k':(152,2),'branch_object_terms_k_CPU_secondary':(152,2),'branch_object_score_k':(152,),
        'branch_object_terms_kplus1':(152,2),'donor_object_terms_k':(8,2),'donor_object_terms_k_CPU_secondary':(8,2),
        'donor_object_score_k':(8,),'donor_object_terms_kplus1':(8,2)}.items()}
    arrays.update(snapshot_completed=np.zeros(8,bool),completed=np.zeros(152,bool),restore_inventory_bitexact=np.zeros(152,bool),restore_qualified=np.zeros(152,bool))
    directions={key:np.full((8,4,2,20),np.nan,np.float32) for key in ('plus','minus')}
    directions.update(d_eff=np.full((8,4,2,20),np.nan,np.float64),center_offset=np.full((8,4,2,20),np.nan,np.float64),
        centers=np.full((8,20),np.nan,np.float32),clipped=np.zeros((8,4,2),bool))
    return arrays,directions


def baseline_errors(x,truth,reward,truth_reward):
    delta=abs(np.asarray(x,dtype=np.float64)-np.asarray(truth,dtype=np.float64))
    groups=dict(q=(0,20,2e-5),qd=(20,40,2e-4),p=(40,43,2e-5),
                R=(43,52,2e-5),v=(52,55,2e-4),omega=(55,58,2e-4))
    errors={name:float(delta[a:b].max()) for name,(a,b,tol) in groups.items()}
    errors['reward']=abs(float(reward)-float(truth_reward))
    passed=all(errors[name]<=tol for name,(_,_,tol) in groups.items()) and errors['reward']<=2e-5
    return dict(errors=errors,passed=bool(passed))


def object_terms(x,p,R):
    """Secondary FP32CPU state rescore; never primary native GPU evidence."""
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


def object_terms_f64(x,p,R):
    a=np.asarray(x,np.float64);pos=a[40:43];rot=a[43:52].reshape(3,3)
    p=np.asarray(p,np.float64);R=np.asarray(R,np.float64);off=.1*rot[:,0];goff=.1*R[:,0]
    endpoint=(np.linalg.norm(pos+off-p-goff)+np.linalg.norm(pos-off-p+goff))*.5
    return np.array((2*np.exp(-(np.linalg.norm(pos-p)/.05)**2),8*np.exp(-(endpoint/.04)**2)),np.float64)


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
            result=_original(*args,**kwargs)
            counts[stage['name']][_label+'_returned']+=1
            return result
        setattr(owner,name,counted)
    try:yield
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)


def run_probe(bundle,config,output,emit):
    validate_config(config,bundle)
    if config['GPU_run_authorized'] is not True:raise ValueError('This run requires Root explicit GPU authorization')
    import torch,warp as wp,mujoco,mujoco_warp as mjwarp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP,PenSimReader
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingCommand
    from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat,mm,mv
    from full151_native.runtime import full151_runtime
    from full151_native.controller import full151_controller,PURE_PATH,PURE_SHA256,_canonical
    from full151_native.native import configure_a0_env
    from full151_native.native import install_actual_context
    from full151_native.policy_probe import expected_control
    from .transaction105 import CAPACITY_SOURCE,CAPACITY_SHA256
    from .transaction105 import Transaction105,profile105
    from full151_native.bundle import FIELDS
    from full151_native.probe import fp32_clock_trace,CONTACT_SOURCE,CONTACT_SHA256

    output=Path(output);deadline=config['_worker_started_perf']+config['soft_worker_seconds']
    seed=config['evaluation_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch._dynamo.config.suppress_errors=False
    if not torch.cuda.is_available():raise RuntimeError('Isolated NVIDIA environment required')
    wp.set_device('cuda:0')
    proposal=read(checked(config['proposal']))
    capspec=importlib.util.spec_from_file_location('_serial105_external_capacity',CAPACITY_SOURCE)
    capmodule=importlib.util.module_from_spec(capspec);capspec.loader.exec_module(capmodule)
    CapacityObserver=capmodule.CapacityObserver
    contactspec=importlib.util.spec_from_file_location('_serial105_contacts',CONTACT_SOURCE)
    measured=importlib.util.module_from_spec(contactspec);contactspec.loader.exec_module(measured)
    if pin(CAPACITY_SOURCE)['sha256']!=CAPACITY_SHA256 or pin(CONTACT_SOURCE)['sha256']!=CONTACT_SHA256:raise ValueError('Frozen capacity/contact source drift')
    expected_clock=fp32_clock_trace();mode=config['probe_mode']
    plan=execution_plan(mode);planned_steps=plan['episode_substeps'];planned_branches=plan['branch_count'];planned_snapshots=plan['snapshot_count'];planned_donors=len(plan['donors'])
    rows=plan['branch_slots'];snapshots=plan['snapshot_slots'];donors=[];ledgers=[];restores=[];baseline_gates=[]
    arrays,dirs=allocate_artifacts()
    stage={'name':'imports_and_setup'};counts=defaultdict(lambda:defaultdict(int));phase_seconds=defaultdict(float);phase_clock=[time.perf_counter()]
    def mark(name):
        now=time.perf_counter();phase_seconds[stage['name']]+=now-phase_clock[0];phase_clock[0]=now;stage['name']=name
    report=dict(schema='full151105_serial_physical_probe_worker_report:1',status='INITIALIZING',
                engineering_qualified=False,baseline_all_passed=False,capacity_finalized=False,
                capacity_qualified=False,source_lineage_N=1,independent_ego_N=0,
                probe_mode=mode,planned_full_branch_slots=152,planned_branch_count=planned_branches,planned_episode_substeps=planned_steps,scheduled_substeps=0,optimizer_updates=0,
                full_serial_qualified=False,foreign_inventory_qualified=False,restore_inventory_qualified=False,longclock_qualified=False,
                WM_forwards=0,Actor_forwards=0,physical_reference_qualified=False,model_loads=0,Runner_instances=0,
                source_action_semantics='Fresh native replay of fixed action-source arrays; not archived-state or closed-loop Actor reproduction',
                fullstate_scope='Visited native+explicit sidecar inventory, INTEGRATION and empirical3baseline gates; opaque internals are source-reviewed, not a universal serializer proof',
                episode_step_attempted=0,episode_step_returned=0,control_interval_attempted=0,control_interval_returned=0,
                scalar_contract=SCALAR_CONTRACT)
    wall=time.perf_counter();error=None
    if pin(PURE_PATH)['sha256']!=PURE_SHA256:raise ValueError('Frozen C2 changed')
    spec=importlib.util.spec_from_file_location('_physical_probe_C2',PURE_PATH)
    pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    def checktime():
        if time.perf_counter()>deadline:raise TimeoutError('Fixed full151105 probe deadline; no retry')
    def save():
        np.savez_compressed(output/'physical_arrays.npz',**arrays,**{'direction__'+k:v for k,v in dirs.items()})
        write(output/'branch_index.json',rows);write(output/'snapshot_records.json',snapshots)
        write(output/'donor_records.json',donors);write(output/'restore_receipts.json',restores)
        write(output/'capacity_ledgers.json',ledgers);write(output/'baseline_gates.json',baseline_gates)
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
                datasets=[dict(name=name,**config['donor_inputs'][name]) for name in PATHS]
                for path_i,dataset in enumerate(datasets[:planned_donors]):
                    checktime();name=dataset['name'];pathdir=output/('donor_'+name);pathdir.mkdir()
                    archived=Path(dataset['directory']);old_controller=read(checked(dataset['input_pins']['controller_descriptor.json']))
                    descriptor=old_controller['root_descriptor'];events=read(checked(dataset['input_pins']['control_events.json']))
                    with np.load(checked(dataset['input_pins']['raw_states.npz']),allow_pickle=False) as old:
                        archive={k:old[k].copy() for k in old.files}
                    targets=np.asarray([e['actualu'] for e in events],np.float32).reshape(150,20)
                    actions=np.zeros((150,1,20),np.float32) if name=='A0' else np.asarray([e['action'] for e in events],np.float32)
                    if name=='A0' and any('action' in e for e in events):raise ValueError('A0 source must remain explicitly declared zero, not a claimed saved action')
                    if actions.shape!=(150,1,20) or not np.isfinite(actions).all() or (abs(actions)>1).any():raise ValueError('Bad archived source action')
                    if name!='A0' and any(not np.array_equal(np.clip(np.asarray(e['actor_raw_action'],np.float32),-1,1),actions[k]) for k,e in enumerate(events)):
                        raise ValueError('Saved action differs from original Actor clamp')
                    cfg=configure_a0_env(make_env_cfg,bundle,seed)
                    if cfg.sim.nan_guard.enabled is not False:raise ValueError('Only disabled NanGuard inventory is reviewed')
                    env=None;reader=None;observer=None;original_step=None;original_process=None
                    local_txs={};local_states=[];local_events=[]
                    ctx={'expected_u':None,'step_in_interval':0,'mode':'setup','branch_id':None,'phase':0,'override':None,'previous_raw':None,'nominal_prediction':None}
                    try:
                        with full151_runtime(ManagerBasedRlEnv,SingleHandTrackingCommand,PenMDP,DexSuperMimicRunner,bundle['arrays'],descriptor):
                            with full151_controller(ManagerBasedRlEnv,PenMDP,DexSuperMimicRunner,bundle['baseline_u150'],descriptor):
                                mark(name+'/constructor');emit(status='initializing',phase=stage['name'])
                                env=ManagerBasedRlEnv(cfg,device='cuda:0')
                                model=env.sim.mj_model
                                if model.nplugin!=0:raise ValueError('No plugin state ABI is qualified')
                                mujoco.mj_saveModel(model,str(pathdir/'nominal_model.mjb'),None)
                                if pin(pathdir/'nominal_model.mjb')['sha256']!=bundle['model']['sha256']:raise ValueError('Nominal MJB differs')
                                observer=CapacityObserver(env,stage=name,fixed_capacities={'nconmax':192,'njmax':4096})
                                observer._raise_if_unqualified(observer._read('after_constructor_before_actual8'))
                                mark(name+'/actual8');install_actual_context(env,bundle,pathdir)
                                observer._raise_if_unqualified(observer._read('after_actual8_before_reset'))
                                mark(name+'/checked_reset');observer.before_reset();obs,_=env.reset(seed=seed)
                                reader=PenSimReader(env);reader.attach_capture()
                                if reader.mdp.radius!=.1:raise ValueError('Frozen native2+8 scalar requires axis-point radius.1')
                                rm=env.reward_manager
                                if rm._scale_by_dt is not True:raise ValueError('Native RewardManager dt-scaled rate ABI changed')
                                rate_columns=[rm._term_names.index(key) for key in ('object_pos','object_rot')]
                                for key,weight,sigma in (('object_pos',2.,.05),('object_rot',8.,.04)):
                                    termcfg=rm.get_term_cfg(key)
                                    if termcfg.weight!=weight or termcfg.params.get('sigma',sigma)!=sigma:raise ValueError('Native object reward weights/sigma differ')
                                capture_reward=env.reward_manager.compute
                                def reward_capture(dt):
                                    if dt!=.02:raise ValueError('Native reward control dt changed')
                                    result=capture_reward(dt)
                                    reader.reward_snapshot['raw_found']=env.scene['finger_collision'].data.found.clone()
                                    reader.reward_snapshot['raw_sensordata']=env.sim.wp_data.sensordata.numpy().copy()
                                    rate=rm._step_reward[:,rate_columns].clone()
                                    if rate.dtype!=torch.float32 or tuple(rate.shape)!=(1,2):raise ValueError('Native GPU object-rate terms ABI changed')
                                    reader.reward_snapshot['object_terms_native_rate_FP32']=rate
                                    reader.reward_snapshot['object_score_native_rate_FP32']=rate.sum(-1,dtype=torch.float32).clone()
                                    reader.reward_snapshot['object_rate_capture_metadata']={
                                        'compute_dt_s':dt,'scale_by_dt':rm._scale_by_dt,'term_names':['object_pos','object_rot'],
                                        'term_columns':list(rate_columns),'weights':[rm.get_term_cfg(key).weight for key in ('object_pos','object_rot')],
                                        'rate_source':'RewardManager._step_reward after original compute, before Managerforward',
                                        'subset_sum':'capture-time GPU torch.sum over [object_pos,object_rot], dtypeFP32; not original full reward aggregate',
                                        'dt_scaled_rate_if_derived':'derived only; original per-term predivision step-value cache not retained'}
                                    return result
                                env.reward_manager.compute=reward_capture
                                if not np.array_equal(reader.xi.cpu().numpy(),np.broadcast_to(bundle['context']['reader_context'][0],reader.xi.shape)):raise ValueError('Actual xi46 differs')
                                if _canonical(reader.mdp.full151_controller_metadata)!=_canonical(old_controller):raise ValueError('Original controller descriptor differs')
                                np.savez(pathdir/'reader_context.npz',reader_context=reader.xi.cpu().numpy().copy())
                                write(pathdir/'controller_descriptor.json',reader.mdp.full151_controller_metadata)
                                term=env.action_manager.get_term('joint_pos');host=mujoco.MjData(model)
                                jids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
                                aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids])
                                obj=model.body('object/cube').id
                                if np.any(model.body_ipos[obj]):raise ValueError('Bound object COM0 required')
                                hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
                                object_geoms=set(np.flatnonzero(model.geom_bodyid==obj).tolist())
                                lower,upper=[v[0].cpu().numpy().copy() for v in term.target_position_limits]
                                inner_lo=lower+np.float32(.02);inner_hi=upper-np.float32(.02)
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
                                if initial['time']!=0.:raise ValueError('Initial clock differs')
                                write(pathdir/'runtime_reset_readback.json',env._full151_runtime_last_reset)
                                write(pathdir/'sensor_label_mapping.json',dict(middle=reader.middle,distal=reader.distal,tips=reader.tips,
                                    native_sensor='finger_collision',label='10middle plus5(distal OR tip), not hand-object',
                                    state_phase='postintegration20ms',sensor_phase='lastsolver preintegration15ms before Managerforward'))
                                local_states.append(initial)
                                original_step=env.sim.step;original_process=term.process_actions
                                def process(action):
                                    prior=term.processed_action.cpu().numpy().copy()
                                    pred=expected_control(action.cpu().numpy(),prior,bundle['baseline_u150'][ctx['phase']],lower,upper,bias,pure.target_inset)
                                    original_process(action)
                                    if not np.array_equal(term.processed_action.cpu().numpy().view(np.uint32),pred['residual'].view(np.uint32)) or not np.array_equal(term._targets.cpu().numpy().view(np.uint32),pred['actual'].view(np.uint32)):
                                        raise ValueError('Original EMA/C2 processing differs')
                                    ctx['nominal_prediction']=pred
                                    if not np.array_equal(pred['actual'][0].view(np.uint32),targets[ctx['phase']].view(np.uint32)):raise ValueError('Fresh nominal target differs from archived center')
                                    wanted=targets[ctx['phase']] if ctx['override'] is None else ctx['override']
                                    if np.any(wanted<inner_lo) or np.any(wanted>inner_hi):raise ValueError('Target intervention outside legal inner domain')
                                    identity=pure.target_inset(torch.from_numpy(wanted.copy())[None],torch.from_numpy(lower.copy()),torch.from_numpy(upper.copy()),.02).numpy()[0]
                                    if not np.array_equal(identity.view(np.uint32),wanted.view(np.uint32)):raise ValueError('Prevalidated branch target is not C2 identity')
                                    if ctx['override'] is not None:term._targets.copy_(torch.from_numpy(wanted.copy())[None].to(env.device))
                                    ctx['expected_u']=wanted.copy()
                                def substep():
                                    checktime();report['episode_step_attempted']+=1;original_step();report['episode_step_returned']+=1
                                    now=raw();ctx['step_in_interval']+=1
                                    record=observer._read('after_substep');observer._raise_if_unqualified(record)
                                    if not np.array_equal(now['ctrl'].view(np.uint32),ctx['expected_u'].view(np.uint32)):raise ValueError('Actual intervention target changed by native apply')
                                    prev=ctx['previous_raw']
                                    tick=ctx['phase']*4+ctx['step_in_interval']
                                    if prev is None or np.float32(now['time']).view(np.uint32)!=expected_clock[tick].view(np.uint32) or float(np.float32(now['time']))!=now['time'] or abs(now['time']-tick*.005)>5e-5:raise ValueError('Full151 FP32 recurrence/longclock mismatch')
                                    contacts=[]
                                    for ci,c in enumerate(host.contact):
                                        force=np.zeros(6);mujoco.mj_contactForce(model,host,ci,force)
                                        if not all(np.isfinite(v).all() for v in (force,c.pos,c.frame,np.asarray(c.dist))):raise FloatingPointError('Nonfinite solved contact')
                                        contacts.append(dict(geom=list(map(int,c.geom)),efc_address=int(c.efc_address),dist=float(c.dist),pos=np.array(c.pos),frame=np.array(c.frame).reshape(3,3),force6=force))
                                    native_contact={f:getattr(env.sim.wp_data.contact,f).numpy()[:now['nacon']].copy() for f in ('geom','type','worldid','efc_address')}
                                    measurement=measured.capture_contacts(model,host,obj,object_geoms,hand)
                                    F=np.asarray(measurement['force_w_N']);T=np.asarray(measurement['torque_at_contact_w_Nm'])
                                    point_w=host.xpos[obj]+np.asarray(measurement['pos_o_m'])@host.xmat[obj].reshape(3,3).T
                                    totalF=F.sum(axis=0);totalT=(np.cross(point_w-host.xipos[obj],F)+T).sum(axis=0)
                                    sample=dict(path=name,phase=ctx['phase'],mode=ctx['mode'],branch_id=ctx['branch_id'],substep=ctx['step_in_interval'],
                                        pre_time=prev['time'],post_time=now['time'],pre_qpos=prev['qpos'],pre_qvel=prev['qvel'],post_qpos=now['qpos'],post_qvel=now['qvel'],
                                        ctrl=now['ctrl'],actuator_force=host.actuator_force[aids].copy(),nacon=now['nacon'],nefc=now['nefc'],
                                        canonical_pre_tick=tick-1,canonical_post_tick=tick,source_frame=200+ctx['phase'],measured_hand_object=measurement,
                                        hand_object_force_on_object_w_N=totalF,hand_object_moment_about_COM_w_Nm=totalT,direct_torque_included_once=True,
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
                                    report['control_interval_attempted']+=1
                                    ob,reward,terminated,truncated,_=env.step(torch.from_numpy(actions[k].copy()).to(env.device))
                                    report['control_interval_returned']+=1
                                    if ctx['step_in_interval']!=4 or counts[stage['name']]['Simulation.forward']-before!=1:raise ValueError('Native4step/oneforward cadence differs')
                                    if env.command_manager.get_term('motion').time_steps.tolist()!=[k+1] or bool(terminated.any()) or bool(truncated.any())!=(k==149):raise ValueError('Native phase/termination differs')
                                    snap=reader.reward_snapshot
                                    if snap is None:raise ValueError('Missing original reward capture')
                                    x=snap['physics'][0].cpu().numpy().copy();contact=snap['contact'][0].cpu().numpy().copy()
                                    if x.shape!=(58,) or contact.shape!=(15,) or not np.isfinite(x).all() or not np.isin(contact,[0,1]).all():raise ValueError('Bad native training labels')
                                    if not np.array_equal(x.view(np.uint32),reader.state()[0].cpu().numpy().view(np.uint32)):raise ValueError('Reward capture/raw state58 differ')
                                    rawstate=raw()
                                    if np.float32(rawstate['time']).view(np.uint32)!=expected_clock[4*(k+1)].view(np.uint32):raise ValueError('Interval FP32 clock mismatch')
                                    raw_found=snap['raw_found'].cpu().numpy().copy()
                                    if raw_found.shape[0]!=1:raise ValueError('Single-world sensorfound required')
                                    found=raw_found>0;independent=np.concatenate((found[:,reader.middle],found[:,reader.distal]|found[:,reader.tips]),axis=-1).astype(np.float32)
                                    if not np.array_equal(independent[0].view(np.uint32),contact.view(np.uint32)):raise ValueError('Raw sensorfound does not reconstruct native self15')
                                    primary=snap['object_terms_native_rate_FP32'][0].cpu().numpy().copy()
                                    primary_score=snap['object_score_native_rate_FP32'][0].cpu().numpy().copy()
                                    diagnostic=object_terms_f64(x,gp[k],gR[k]);bound=8*np.finfo(np.float32).eps*np.maximum(1.,abs(diagnostic))
                                    if not np.isfinite(primary).all() or not np.isfinite(primary_score) or np.any(abs(primary.astype(float)-diagnostic)>bound):raise ValueError('Native GPU object terms exceed frozen F64 arithmetic diagnostic bound')
                                    out=dict(x=x,contact=contact,torque=snap['torque'][0].cpu().numpy().copy(),
                                        raw_found=raw_found,
                                        raw_sensordata=snap['raw_sensordata'].copy(),
                                        rawstate=rawstate,terms_k=primary,r_native=float(primary_score),terms_k_CPU=object_terms(x,gp[k],gR[k])[0],
                                        terms_F64_diagnostic=diagnostic,term_arithmetic_bound=bound,terms_next=object_terms(x,gp[k+1],gR[k+1])[0])
                                    rewardfile.write(json.dumps(jsonable(dict(path=name,k=k,mode=ctx['mode'],branch_id=branch_id,
                                        physics=x,contact=contact,torque=out['torque'],raw_found=out['raw_found'],
                                        raw_sensordata=out['raw_sensordata'],label_time='lastsolver_preintegration~kdt+.015',
                                        native_GPU_object_rate_terms_FP32=primary,native_GPU_object_rate_score_FP32=out['r_native'],
                                        native_GPU_capture_metadata=snap['object_rate_capture_metadata'],
                                        CPU_FP32_object_rate_terms_secondary=out['terms_k_CPU'],F64_object_rate_terms_diagnostic=diagnostic,
                                        F64_arithmetic_bound=bound,scalar_contract=SCALAR_CONTRACT,
                                        state_time='postintegration_kdt+.020',source_frame=200+k,terminal_observation=k+1,
                                        valid_next_policy_decision=k+1<150,no_u150=True,
                                        capture='after_original_reward_compute_before_Manager_forward')),allow_nan=False)+'\n')
                                    return out
                                ctx['mode']='donor';mark(name+'/donor')
                                capture_phases=plan['capture_phases']
                                for k in range(plan['donor_intervals']):
                                    reader.refresh_frame()
                                    if k in capture_phases:
                                        mark(name+'/snapshot_capture');sid=path_i*4+PHASES.index(k)
                                        pred=expected_control(actions[k],term.processed_action.cpu().numpy().copy(),bundle['baseline_u150'][k],lower,upper,bias,pure.target_inset)
                                        center=pred['actual'][0].copy()
                                        if not np.array_equal(center.view(np.uint32),targets[k].view(np.uint32)):raise ValueError('Fresh fixed-source center differs from archived actual target')
                                        dirs['centers'][sid]=center
                                        for j,joint in enumerate(JOINTS):
                                            direction=np.eye(20,dtype=np.float32)[joint]
                                            for ei,epsilon in enumerate(EPS):
                                                plus=np.minimum(np.maximum(center+np.float32(epsilon)*direction,inner_lo),inner_hi)
                                                minus=np.minimum(np.maximum(center-np.float32(epsilon)*direction,inner_lo),inner_hi)
                                                dirs['plus'][sid,j,ei]=plus;dirs['minus'][sid,j,ei]=minus
                                                dirs['d_eff'][sid,j,ei]=(plus.astype(float)-minus.astype(float))/(2*epsilon)
                                                dirs['center_offset'][sid,j,ei]=(plus.astype(float)+minus.astype(float))*.5-center.astype(float)
                                                dirs['clipped'][sid,j,ei]=not(np.array_equal(plus,center+np.float32(epsilon)*direction) and np.array_equal(minus,center-np.float32(epsilon)*direction))
                                        arrays['snapshot_x'][sid]=reader.state()[0].cpu().numpy();arrays['snapshot_u'][sid]=center
                                        arrays['goal_p'][sid]=gp[k];arrays['goal_R'][sid]=gR[k];arrays['goal_p_next'][sid]=gp[k+1];arrays['goal_R_next'][sid]=gR[k+1]
                                        reader.memory() # Prime lazy read-only views; no sense/observe/forward.
                                        profile=profile105(env,reader,expected_phase=k)
                                        write(pathdir/('profile_'+str(k)+'.json'),profile)
                                        integ=integration();tx=Transaction105.__new__(Transaction105)
                                        record=snapshots[sid];record.update(status='CAPTURING',reason=None,integration_sig=sig,integration_size=size)
                                        external={'ctx':ctx,'stage':stage,'counts':counts,'phase_seconds':phase_seconds,'samplefile':samplefile,'rewardfile':rewardfile,
                                            'arrays':arrays,'rows':rows,'records':restores,'report':report,'local_states':local_states,'local_events':local_events,'local_txs':local_txs,
                                            'bundle':bundle,'dirs':dirs,'archive':archive,'proposal':proposal,'config':config,'dataset':dataset,'snapshots':snapshots,'donors':donors,'ledgers':ledgers,
                                            'baseline_gates':baseline_gates,'Torch_module':torch,'Warp_module':wp,'MuJoCo_module':mujoco,'MJWarp_module':mjwarp,
                                            'C2_readonly_module':pure,'measured_contacts_module':measured,'host_readback_only':host,
                                            'statebuf_readback_only':statebuf,'captured_expected_actual8':actual_expected}
                                        try:
                                            Transaction105.__init__(tx,env,reader,capacity_observer=observer,source_action=actions[k].copy(),label=name+'/'+str(k),
                                                expected_phase=k,external_roots=external)
                                            record['inventory_construction_completed']=True
                                            mjwarp.set_state(env.sim.wp_model,env.sim.wp_data,statebuf,sig)
                                            if not np.array_equal(integration().view(np.uint32),integ.view(np.uint32)):raise ValueError('INTEGRATION set/get roundtrip differs')
                                            tx.verify_current();record['integration_roundtrip_bitexact']=True
                                            record['inventory']=tx.describe()
                                            record['inventory_archive']=dump_inventory(tx,pathdir/('snapshot_'+str(k)+'_inventory'),integ,actions[k])
                                            record['foreign_review']=reviewed_foreign_inventory(config,tx,bundle)
                                            record['foreign_review_qualified']=True
                                        except BaseException as exc:
                                            record.update(status='INVENTORY_REVIEW_REQUIRED' if isinstance(exc,InventoryReviewRequired) else 'INVALID',reason=repr(exc),construction_or_review_error=repr(exc))
                                            if 'inventory_archive' not in record:
                                                record['inventory_archive']=dump_inventory(tx,pathdir/('snapshot_'+str(k)+'_inventory'),integ,actions[k])
                                            try:save()
                                            except BaseException as saving:exc.add_note('Secondary discovery-save failure: '+repr(saving))
                                            raise
                                        local_txs[k]=(tx,integ.copy())
                                        record.update(status='CAPTURED',reason=None)
                                        mark(name+'/donor')
                                    outcome=interval(k);local_events.append(dict(phase=k,actualu=ctx['expected_u'],**outcome))
                                    if k in PHASES:
                                        sid=path_i*4+PHASES.index(k);arrays['donor_next_x'][sid]=outcome['x'];arrays['donor_self15'][sid]=outcome['contact']
                                        arrays['donor_object_terms_k'][sid]=outcome['terms_k'];arrays['donor_object_terms_kplus1'][sid]=outcome['terms_next'];arrays['snapshot_completed'][sid]=True
                                        arrays['donor_object_score_k'][sid]=np.float32(outcome['r_native']);arrays['donor_object_terms_k_CPU_secondary'][sid]=outcome['terms_k_CPU']
                                    if k%4==3:emit(status='running',phase=stage['name'],scheduled_substeps=report['scheduled_substeps'])
                                observer._raise_if_unqualified(observer._read('donor_terminal'))
                                donor_delta_q=max(float(abs(v['qpos'].astype(float)-archive['full_qpos'][i]).max()) for i,v in enumerate(local_states))
                                donor_delta_v=max(float(abs(v['qvel'].astype(float)-archive['full_qvel'][i]).max()) for i,v in enumerate(local_states))
                                qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids]
                                oj=model.joint('object/cube_freejoint').id;oq=int(model.jnt_qposadr[oj]);ov=int(model.jnt_dofadr[oj])
                                from full151_native.bundle import rotation
                                def drift_at(indices):
                                    q=np.stack([local_states[i]['qpos'] for i in indices]).astype(float);v=np.stack([local_states[i]['qvel'] for i in indices]).astype(float)
                                    qold=archive['full_qpos'][indices];vold=archive['full_qvel'][indices]
                                    return dict(joint_q_maxabs_rad=float(abs(q[:,qa]-qold[:,qa]).max()),joint_qd_maxabs_rad_s=float(abs(v[:,va]-vold[:,va]).max()),
                                        object_p_maxabs_m=float(abs(q[:,oq:oq+3]-qold[:,oq:oq+3]).max()),
                                        object_R_entry_maxabs=float(abs(rotation(q[:,oq+3:oq+7])-rotation(qold[:,oq+3:oq+7])).max()),
                                        object_v_world_maxabs_m_s=float(abs(v[:,ov:ov+3]-vold[:,ov:ov+3]).max()),
                                        object_omega_body_maxabs_rad_s=float(abs(v[:,ov+3:ov+6]-vold[:,ov+3:ov+6]).max()))
                                donors.append(dict(path=name,fresh_state_count=len(local_states),actions=actions,events=local_events,
                                    archive_observed_horizon_drift=drift_at(list(range(len(local_states)))),archive_selected_snapshot_drift=drift_at([4*k for k in capture_phases]),
                                    archive_selected_next_drift=drift_at([4*k+4 for k in capture_phases]),
                                    legacy_mixed_q_maxabs_descriptive_only=donor_delta_q,legacy_mixed_qd_maxabs_descriptive_only=donor_delta_v,
                                    archival_drift_flag=donor_delta_q>2e-5 or donor_delta_v>2e-4,
                                    archive_quality_not_regraded=True))
                                for k in (PHASES if mode=='serial_full' else [0]):
                                    sid=path_i*4+PHASES.index(k);tx,integ=local_txs[k];baseline_pass=[]
                                    for row in rows[sid*19:(sid+1)*19]:
                                        if not row['scheduled']:continue
                                        checktime();bid=row['row_id'];ctx['mode']=row['type'];mark(name+'/restore')
                                        restore=None
                                        try:
                                            with tx.branch('branch_'+str(bid)) as restore:
                                                if not np.array_equal(integration().view(np.uint32),integ.view(np.uint32)):raise ValueError('Restored integration is not bitexact')
                                                for f,value in actual_expected.items():
                                                    got=getattr(env.sim.wp_model,f).numpy()
                                                    if not np.array_equal(got,np.broadcast_to(value.astype(got.dtype),got.shape)):raise ValueError('Restored actual8 differs '+f)
                                                arrays['restore_inventory_bitexact'][bid]=True
                                                u=targets[k].copy() if row['type']=='baseline' else dirs['plus' if row['sign']==1 else 'minus'][sid,row['j'],row['epsilon_index']].copy()
                                                mark(name+'/'+row['type'])
                                                outcome=interval(k,u,bid)
                                        except BaseException as exc:
                                            row.update(status='INVALID',reason=repr(exc));raise
                                        finally:
                                            restores.append(dict(branch_id=bid,begin=restore,transaction_events=tx.events[-2:]))
                                        arrays['branch_x_next'][bid]=outcome['x'];arrays['branch_self15'][bid]=outcome['contact'];arrays['branch_u'][bid]=u
                                        arrays['branch_object_terms_k'][bid]=outcome['terms_k'];arrays['branch_object_terms_kplus1'][bid]=outcome['terms_next']
                                        arrays['branch_object_score_k'][bid]=np.float32(outcome['r_native']);arrays['branch_object_terms_k_CPU_secondary'][bid]=outcome['terms_k_CPU']
                                        arrays['completed'][bid]=True
                                        row.update(actualu=u,r_k=outcome['r_native'],r_kplus1=float(np.sum(outcome['terms_next'],dtype=np.float32)),status='COMPLETED',reason=None,
                                            source_frame=200+k,terminal_observation=k+1,valid_next_policy_decision=k+1<150,no_u150=True)
                                        if row['type']=='baseline':
                                            gate=baseline_errors(outcome['x'],arrays['donor_next_x'][sid],row['r_k'],float(arrays['donor_object_score_k'][sid]))
                                            baseline_pass.append(gate['passed']);restores[-1]['baseline_gate']=gate
                                            baseline_gates.append(dict(row_id=bid,snapshot_id=sid,path=name,k=k,**gate))
                                            if row['baseline_repeat']==2 and not all(baseline_pass):raise ValueError('Three fresh baseline replay gates failed; no perturbations/retry')
                                            if row['baseline_repeat']==2:
                                                for b in range(3):arrays['restore_qualified'][sid*19+b]=True;rows[sid*19+b]['restorequalified']=True
                                                snapshots[sid].update(baseline3_qualified=True,full_native_restore_qualified=True,status='BASELINE3_QUALIFIED')
                                        else:
                                            if len(baseline_pass)!=3 or not all(baseline_pass):raise ValueError('Directions cannot precede3 qualified fresh baselines')
                                            arrays['restore_qualified'][bid]=True;row['restorequalified']=True
                                    emit(status='running',phase=stage['name'],completed_branches=int(arrays['completed'].sum()),scheduled_substeps=report['scheduled_substeps']);save()
                    finally:
                        body_exception=sys.exc_info()[0] is not None;cleanup_errors=[]
                        mark(name+'/cleanup')
                        if env is not None:
                            if original_step is not None:env.sim.step=original_step
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
                report.update(status='COMPLETED_PHYSICAL_PROBE' if mode=='serial_full' else 'COMPLETED_ENGINEERING_RESTORE_SMOKE',
                    engineering_qualified=True,baseline_all_passed=len(baseline_gates)==planned_snapshots*3 and all(v['passed'] for v in baseline_gates),
                    capacity_finalized=len(ledgers)==planned_donors and all(v['ledger']['finalized'] for v in ledgers),
                    capacity_qualified=len(ledgers)==planned_donors and all(v['ledger']['engineering_capacity_qualified'] for v in ledgers))
                if int(arrays['completed'].sum())!=planned_branches or int(arrays['snapshot_completed'].sum())!=planned_snapshots or report['scheduled_substeps']!=planned_steps or report['episode_step_returned']!=planned_steps or not report['capacity_qualified'] or not report['baseline_all_passed']:
                    raise ValueError('Incomplete physical probe coverage/capacity')
                report.update(foreign_inventory_qualified=all(v['foreign_review_qualified'] for v in snapshots if v['scheduled']),
                    restore_inventory_qualified=all(arrays['restore_qualified'][v['row_id']] for v in rows if v['scheduled']),
                    full_serial_qualified=mode=='serial_full',longclock_qualified=mode=='serial_full',
                    longclock_scope='Full151 sequentialFP32 clock checked' if mode=='serial_full' else 'PENDING_ENGINEERING_SMOKE_ONLY',
                    baseline_gate_count=len(baseline_gates),no_next_policy_at_terminal150=True)
                for snapshot in snapshots:
                    if snapshot['scheduled']:snapshot['longclock_qualified']=mode=='serial_full'
    except BaseException as exc:
        error=exc;report.update(status='INVENTORY_REVIEW_REQUIRED' if isinstance(exc,InventoryReviewRequired) else 'FAILED_ENGINEERING_PARTIAL',
            engineering_qualified=False,baseline_all_passed=False,full_serial_qualified=False,error=repr(exc))
        raise
    finally:
        try:save()
        except BaseException as saving:
            if error is None:raise
            error.add_note('Secondary final evidence-save failure: '+repr(saving))
    return report
