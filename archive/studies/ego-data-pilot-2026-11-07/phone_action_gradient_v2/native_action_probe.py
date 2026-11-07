"""Hypothetical native branches from pinned integration states; never an actor run.

No reset/forward is used for restoration. Direct INTEGRATION restoration needs
an empirical matched-action baseline contract before new perturbations are used.
"""
from pathlib import Path
import hashlib,importlib.util,sys
import numpy as np

P=Path(__file__).resolve().parent.parent
READER=P/'phone_episode_acquisition_v1/native_phone_reader.py'
READER_SHA='877aa4c5d51bff09e0a6824804eef3f2b335a8dbda26dcc3d0235bb3af57b841'
if hashlib.sha256(READER.read_bytes()).hexdigest()!=READER_SHA:raise ValueError('Frozen native reader changed')
spec=importlib.util.spec_from_file_location('gradient_probe_frozen_reader',READER)
reader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reader;spec.loader.exec_module(reader)

def bits(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def observable59(value):
    if not value['training_eligible']:return np.full(59,np.nan)
    s=value['summary']
    f=np.r_[s['pad_min_normal_N2'],s['pad_endpoint_normal_N2'],s['phone_nonpad_max_normal_N']]
    if np.any(f<0) or not np.isfinite(f).all():raise ValueError('Real positive-normal force labels')
    return np.r_[value['physical_world_delta52'],s['pad_coverage2'],np.log1p(f)]

class NativeSavedActionProbe:
    def __init__(self,model_path,*,dt_s,expected_effective_model_digest=None):
        self.reader=reader.NativePhoneReader(model_path,dt_s=dt_s)
        if expected_effective_model_digest is not None and self.reader.initial_digest!=expected_effective_model_digest:
            raise ValueError('Same effective physical model/context')
        self.counts=dict(branches_attempted=0,branches_returned=0,state_sets=0,restore_forward=0)

    def run(self,state,operands,u20):
        """state is one frozen selection JSON row; operands are pinned F64 arrays."""
        r=self.reader;sid=int(state['state_id']);u=r._control(u20)
        sig=int(state['source_before_action_receipt']['signature'])
        vector=np.asarray(operands['source_integration_F64'][sid])
        if (sig!=r.signature or vector.dtype!=np.float64 or vector.shape!=(r.state_size,)
                or not np.isfinite(vector).all() or hashlib.sha256(vector.tobytes()).hexdigest()!=state['source_before_action_receipt']['sha256']):
            raise ValueError('Exact source before-action integration vector/spec')
        self.counts['branches_attempted']+=1
        d=r.mj.MjData(r.m);r.counts['MjData']+=1
        r.mj.mj_setState(r.m,d,vector,sig);self.counts['state_sets']+=1
        r.d=d;r.tick=int(state['source_before_action_receipt']['global_tick']);r.terminated=False
        r.initial_qpos=np.asarray(operands['initial_qpos27_F64'][sid]).copy()
        r.initial_state58=np.asarray(operands['initial_state58_F64'][sid]).copy()
        before=r.receipt()
        if not bits(before['integration_state'],vector):raise ValueError('Restore does not roundtrip complete state bytes')
        if not bits(d.qpos,operands['source_qpos27_F64'][sid]) or not bits(d.qvel,operands['source_qvel26_F64'][sid]):
            raise ValueError('Saved full qpos/qvel after restore')
        if not bits(r.state58(),operands['source_state58_F64'][sid]):raise ValueError('Same POST state58 operand')
        if float(d.time)!=float(state['source_before_action_receipt']['native_time_s']):raise ValueError('Native source clock bits')
        if np.any(d.qfrc_applied) or np.any(d.xfrc_applied) or np.any(d.warning.number):
            raise ValueError('No hypothetical aid or inherited warning')
        initial_pure=reader.codec.actual_state(r.initial_qpos,np.zeros(26,dtype=np.float64),r.ipos)
        if not bits(initial_pure,r.initial_state58):raise ValueError('Original episode safety baseline, no recentering')
        value=r.step_control(u)
        self.counts['branches_returned']+=1
        value['observable59']=observable59(value)
        value['state_id']=sid
        value['forward_counters']=dict(initial=r.counts['initial_forward'],observer=r.counts['observer_forward'],restore=self.counts['restore_forward'])
        value['hypothetical_branch_not_executed_actor_rollout']=True
        return value

    def audit(self):
        return dict(model=self.reader.audit(),probe_counts=dict(self.counts),
            restored_integration_is_full_MjData_qualified=False,
            scope='Direct state restoration is supported only by matching-action empirical comparisons for the selected contexts. No parameter updates, native aid or main robot trajectory.')

def baseline_checks(value,state,operands,*,trace=None):
    sid=int(state['state_id']);checks={
        'complete_H':value['complete_H20ms'] is True,
        'finite_eligible':value['training_eligible'] is True,
        'source_before_action_receipt':value['source_receipt']['sha256']==state['source_before_action_receipt']['sha256'],
        'source_after_action_receipt':value['source_after_action_receipt']['sha256']==state['original_source_after_action_receipt']['sha256'],
        'target_full_integration':bits(value['target_receipt']['integration_state'],operands['original_target_integration_F64'][sid]),
        'target_qpos27':bits(value['trace']['post_qpos27'][-1],operands['original_target_qpos27_F64'][sid]),
        'target_qvel26':bits(value['trace']['post_qvel26'][-1],operands['original_target_qvel26_F64'][sid]),
        'all59_labels':bits(value['observable59'],operands['baseline_observable59_F64'][sid]),
        'no_observer_or_restore_forward':all(v==0 for v in value['forward_counters'].values()),
    }
    if trace is not None:
        for name in ['post_qpos27','post_qvel26','post_state58','pre_pad_normal_N2','pre_phone_nonpad_normal_N',
                     'pre_table_normal_N','pre_energy2','contact_rows24','pre_native_time_s','post_native_time_s']:
            checks['all_ticks_'+name]=bits(value['trace'][name],trace[name])
    return checks
