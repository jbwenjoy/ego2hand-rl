"""Canonical WM input/context from an owned W1 snapshot,without source relabeling.

Pure numeric extraction only. This neither restores a runtime nor qualifies a
policy,model gradient or new source envelope for native execution.
"""
import numpy as np
import torch
from ..mechanics.pen_reward_chain import source_context,controller
from ..mechanics.controlled_pen_records import native_state55

SCHEMA='owned_W1_live_decision_input102_RAW_controller_context:1'


def from_owned_snapshot(snapshot,phase,case,raw):
    if (snapshot.get('schema')!='controlled_pen_owned_state:4' or type(snapshot.get('worlds')) is not int or snapshot['worlds']!=1
        or snapshot['metadata'].get('fixed_frame_features')!='LIVE' or type(phase) is not int or phase<1):
        raise ValueError('Own LIVE W1 snapshot and positive reference phase required')
    layout=case['model_layout'];arrays=snapshot['native']['data']['arrays'];warm=np.asarray(arrays['qacc_warmstart'])
    raw=np.asarray(raw,np.float32)
    if raw.shape!=(20,) or warm.shape!=(1,26) or not np.isfinite(raw).all() or not np.isfinite(warm).all():raise ValueError('Finite RAW20 and actual warm26 required')
    dt=float(case['dt']);native_dt=np.asarray(snapshot['native']['model']['arrays']['opt.timestep']).reshape(-1)
    if dt!=.0025 or native_dt.shape!=(1,) or native_dt[0]!=np.float32(dt):raise ValueError('Original eight native ticks/20ms contract required')
    # Only the existing pure context reader receives this minimal numeric view;
    # no old probe/checkpoint schema is fabricated or passed to any loader.
    context=source_context(dict(native=snapshot,phase=phase),case)
    if not phase<context['frames']:raise ValueError('Reference phase outside the interval chart')
    PD,_,applied=controller(torch.from_numpy(raw[None]),context)
    s=native_state55(arrays['qpos'][0],arrays['qvel'][0],case['qids'],case['dofs'],layout['object_qposadr'],layout['object_dofadr'])
    x=np.r_[s,warm[0],PD[0].numpy(),dt].astype(np.float32)
    if x.shape!=(102,) or not np.isfinite(x).all():raise ValueError('Finite original102-input chart required')
    return dict(schema=SCHEMA,input102=x,sampled_RAW20=raw.copy(),applied_RAW20=applied[0].numpy().copy(),actual_PD20=PD[0].numpy().copy(),
        controller_context=context,phase=phase,clipped_components=int(np.count_nonzero(abs(raw)>1)))
