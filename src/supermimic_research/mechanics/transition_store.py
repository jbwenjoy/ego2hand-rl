"""Persist new native transitions with explicit per-record trace/contact ranges."""
import numpy as np
from .transition_records import transition_xy
from ..artifacts import file_pin


def save_transitions(folder,records,dt,metadata):
    folder.mkdir();xs,ys=zip(*(transition_xy(t,dt) for t in records));arrays=dict(x=np.array(xs),y=np.array(ys))
    for key in ['source55','next55','warmstart26_before','warmstart26_after','source_native_state','next_native_state','mean_native_wrench_world_COM','mean_native_wrench_source_BODY_COM','mean_native_group_Fn_N','k','actual_duration_s','complete_H','terminated']:
        arrays[key]=np.array([t[key] for t in records])
    for key in ['action20','EMA_before20','EMA_after20','actual_control20']:arrays[key]=np.array([t['decision'][key] for t in records])
    arrays.update({key:np.array([m[key] for m in metadata]) for key in metadata[0]})
    ticks=np.cumsum([0]+[t['executed_native_steps'] for t in records]);contacts=np.cumsum([0]+[len(t['raw_contact_tick_plus23']) for t in records])
    np.savez_compressed(folder/'transitions.npz',**arrays)
    np.savez_compressed(folder/'trace.npz',**{key:np.concatenate([t['trace'][key] for t in records]) for key in records[0]['trace']},
        raw_contact_tick_plus23=np.concatenate([t['raw_contact_tick_plus23'] for t in records]),tick_ranges=np.c_[ticks[:-1],ticks[1:]],contact_ranges=np.c_[contacts[:-1],contacts[1:]])
    return dict(transitions=file_pin(folder/'transitions.npz'),trace=file_pin(folder/'trace.npz'),rows=len(records),native_steps=int(ticks[-1]),complete_rows=sum(t['complete_H'] for t in records))


def exploration_action(seed,k,source,grasp):
    rng=np.random.default_rng(seed);phase=rng.uniform(-np.pi,np.pi,20);frequency=rng.uniform(.3,2.,20)
    return .45*np.sin(2*np.pi*frequency*(k+.5)*.02+phase)+.25*np.tanh((grasp-source[:20])/.05-.02*source[20:40])
