"""Injected Reader -> new phone task -> generic upstream replay, no native loop.

No source is imported from the native simulator. Calling reset/step later uses
the provided frozen Reader. A raw sink is mandatory before nan_to_num replay.
Original two-pad summaries are retained as original_two_pad_summary; new task
three-pad roles are derived separately from the same raw contact rows.
"""
import hashlib
import numpy as np

PADS=(15,28,41)
PAIRS=(0,2,8,11,13,15,21,24,26,28,34,37,39,41,47,50,52,54,60,63,65,67)

def raw_receipt(q,v,u):
    parts={name:dict(dtype=np.asarray(a).dtype.str,shape=list(np.asarray(a).shape),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()) for name,a in [('qpos27',q),('qvel26',v),('ctrl20',u)]}
    return dict(schema='phone_raw_qpos_qvel_ctrl_receipt:1',parts=parts,full_integration300_receipt=False)

def regroup_interval_contacts(interval):
    """Observed solver-PRE data only; never call for imagined WM features."""
    tr=interval['trace'];ticks=np.asarray(tr['pre_tick']);rows=np.asarray(tr['contact_rows24'])
    fn=np.zeros((len(ticks),22),dtype=np.float64);generated=np.zeros((len(ticks),22),dtype=np.int64)
    lookup={int(t):i for i,t in enumerate(ticks)}
    for row in rows:
        tick,g1,g2=map(int,row[:3])
        if 71 not in [g1,g2]:continue
        gid=g2 if g1==71 else g1
        if gid not in PAIRS:raise ValueError('Unexpected physical phone boundary; no silent remap')
        i=lookup[tick];j=PAIRS.index(gid);generated[i,j]+=1;fn[i,j]+=max(0.,float(row[18]))
    p=[PAIRS.index(g) for g in PADS]
    return dict(schema='phone_explicit_tripod_observed_PRE_contact_summary:1',pair_geom_ids=list(PAIRS),designated_geom_ids=list(PADS),
        per_tick_pair_positive_Fn22_N=fn,per_tick_pair_generated22=generated,
        pair_coverage22=np.mean(fn>.1,axis=0),pair_mean_positive_Fn22_N=np.mean(fn,axis=0),
        triple_simultaneous_coverage=float(np.mean(np.all(fn[:,p]>.1,axis=1))),
        generated_is_force_is_contact_truth_equivalent=False,contact_phase='actual integer solverPRE only; marginal coverage is not simultaneous triple coverage',
        original_two_pad_summary=interval['summary'])

def build_replay_transition(*,interval,x,memory,action,residual,executed_u,observation,reward_packet,task,raw_sink,episode_id):
    """Upstream ReplayBuffer-compatible tensors, with a mandatory raw sidecar.

    A partial native prefix is kept raw but never becomes an H20ms target.
    The upstream buffer's nan_to_num behavior is safe only with valid=False.
    No old PenMDP/Runner contact15 or right-rotation state is implied.
    """
    import torch
    if raw_sink is None:raise ValueError('Persist raw interval, receipts and partial masks before replay insertion')
    if interval['state_schema']!='phone_COM_world_state58:1' or interval['delta_schema']!='phone_physical_world_delta52:2':raise ValueError('Phone world/COM state only')
    if not np.array_equal(np.asarray(interval['actual_control20']),executed_u.detach().cpu().numpy()):raise ValueError('Executed u receipt does not match actual mapped control')
    if not np.array_equal(np.asarray(interval['source_state58']),x.detach().cpu().numpy()):raise ValueError('Source state/memory phase mismatch')
    contacts=regroup_interval_contacts(interval)
    complete=bool(interval['complete_H20ms']);finite=bool(interval['finite']);warnings=bool(np.any(interval['warning_numbers']))
    objective_known=bool(reward_packet['objective_known'].detach().cpu())
    valid=bool(complete and finite and not warnings and objective_known)
    timeout=bool(memory['step'].item()+1>=100 and complete)
    physical_done=interval['termination']!='COMPLETE_INTERVAL'
    raw_sink(dict(schema='phone_actor_raw_transition_sidecar:1',episode_id=int(episode_id),interval=interval,tripod_contacts=contacts,
        memory_before=task.flatten_memory(memory).detach().cpu().numpy(),action_normalized20=action.detach().cpu().numpy(),
        residual20=residual.detach().cpu().numpy(),executed_u20=executed_u.detach().cpu().numpy(),complete_H=complete,valid=valid,
        source_receipt=interval['source_receipt'],after_action_receipt=interval['source_after_action_receipt'],target_receipt=interval['target_receipt'],
        target_H58=np.asarray(interval['target_state58']) if complete else np.full(58,np.nan),observed_prefix_state58=np.asarray(interval['target_state58']),
        actual_contact_known=True,proxy_is_observed_contact=False,full_MjData_or_state58_Markov_qualified=False))
    def T(v,dtype=None):return torch.as_tensor(v,device=x.device,dtype=x.dtype if dtype is None else dtype)
    target=np.asarray(interval['target_state58']) if complete else np.full(58,np.nan)
    torque=np.asarray(interval['trace']['pre_qfrc_actuator20'])[-1]
    fields=dict(x=x,x1=T(target),u=executed_u,c=residual,action=action,m=task.flatten_memory(memory),obs=observation,
        push=torch.zeros_like(x),contact=T(contacts['pair_coverage22']),torque=T(torque),reward=reward_packet['total'],
        done=T(physical_done or timeout or not complete,torch.bool),terminated=T(physical_done or not complete,torch.bool),valid=T(valid,torch.bool),
        complete_H=T(complete,torch.bool),episode_id=T(int(episode_id),torch.int64),target_frame=memory['step']+1)
    # ReplayBuffer.add expects leading environment dimension, even for one Reader.
    return {k:v.unsqueeze(0) for k,v in fields.items()}

def online_actor_action(actor,observation222,*,device='cpu'):
    """Scalar Reader observation -> explicit B1 Actor -> scalar action20."""
    import torch
    if observation222.shape!=(222,):raise ValueError('Online singleton observation222')
    action=actor(observation222.to(device=device,dtype=torch.float32).unsqueeze(0))
    if action.shape!=(1,20) or not torch.isfinite(action).all():raise ValueError('Actor output must be explicit[1,20]')
    return action[0]

class PhoneReaderTaskBridge:
    """Callable online adapter; construction performs no reset or stepping.

    This is not an autonomous trainer/evaluator. Native task truth remains the
    separately pinned two-second/double-dt evaluator; no success is inferred
    from scalar reward, initial loading or this Reader's acquisition safety.
    """
    def __init__(self,reader,task,geometry,raw_sink):
        if reader.model_pin['sha256']!=task.b['model']['sha256'] or reader.n not in [64,128]:raise ValueError('Pinned model and integer20ms tick plan')
        self.reader=reader;self.task=task;self.geometry=geometry;self.raw_sink=raw_sink;self.memory=None;self.x=None;self.episode_id=0;self.closed=True

    def reset(self):
        import torch
        v=self.reader.reset(self.task.b['initial']['qpos27_F64'],self.task.b['initial']['ctrl20_F64'])
        x=np.asarray(v['state58']);expected=np.array(self.task.b['initial']['native_PRE_state58_F64'])
        if not np.allclose(x,expected,rtol=0,atol=1e-12):raise ValueError('Fresh reset state differs from named native PRE source')
        self.raw_sink(dict(schema='phone_actor_native_reset_receipt:1',episode_id=self.episode_id,reset=v,source_initial=self.task.b['initial']['source_probe'],
            receipt300=v['receipt'],static_source_receipt_is_full300=False))
        self.x=torch.tensor(x,dtype=torch.float64)
        self.memory=self.task.initial_memory(self.x,native_h_s=self.reader.h,n_substeps=self.reader.n)
        self.closed=False
        return self.x,self.task.observation(self.x,self.memory,self.geometry)

    def step(self,normalized_action20):
        import torch
        if self.memory is None or self.closed:raise ValueError('reset required')
        x=self.x;m=self.memory;normalized_action20=normalized_action20.to(device=x.device)
        obs=self.task.observation(x,m,self.geometry)
        residual,u=self.task.control_target(normalized_action20,m)
        interval=self.reader.step_control(u.detach().cpu().numpy())
        y=torch.tensor(np.asarray(interval['target_state58']),dtype=x.dtype)
        reward=self.task.rewards(y,m,normalized_action20) if interval['finite'] and interval['complete_H20ms'] else dict(total=x.new_tensor(float('nan')),objective_known=torch.tensor(False))
        replay=build_replay_transition(interval=interval,x=x,memory=m,action=normalized_action20,residual=residual,executed_u=u,
            observation=obs,reward_packet=reward,task=self.task,raw_sink=self.raw_sink,episode_id=self.episode_id)
        self.x=y
        self.memory=self.task.advance_memory(m,normalized_action20,residual,u)
        done=bool(replay['done'][0])
        if done:self.episode_id+=1;self.closed=True
        return dict(state58=y,reward=reward,replay_transition=replay,raw_interval=interval,done=done,task_geometry_success=False)

    def step_actor(self,actor,*,device='cpu'):
        if self.memory is None or self.closed:raise ValueError('reset required')
        obs=self.task.observation(self.x,self.memory,self.geometry)
        return self.step(online_actor_action(actor,obs,device=device))
