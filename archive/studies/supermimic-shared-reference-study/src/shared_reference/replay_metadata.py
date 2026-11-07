"""Typed trajectory audit metadata, kept separate from actor/WM features."""
import numpy as np


def _integers(value,name):
    a=np.asarray(value)
    if a.dtype.kind not in "iuf" or not np.isfinite(a).all() or not np.array_equal(a,np.rint(a)):
        raise ValueError(name+" must be finite exactinteger metadata")
    return a.astype(np.int64)


def make_transition_metadata(registry,track_name,source_ids,*,use,memory_start,memory_end,memory_time,episode_ids,actual_u):
    t=registry.track(track_name)
    if use not in t["allowed_uses"]: raise ValueError("External holdout cannot enter train replay")
    sources=list(source_ids);order=list(t["source_order"])
    if any(s not in order for s in sources): raise ValueError("Trajectory not enrolled in selectedtrack")
    start,end,local,episode=(_integers(v,n).reshape(-1) for v,n in ((memory_start,"start"),(memory_end,"end"),(memory_time,"time"),(episode_ids,"episode")))
    n=len(sources)
    if any(len(v)!=n for v in (start,end,local,episode)): raise ValueError("Metadata batchsize")
    u=np.asarray(actual_u)
    if u.shape!=(n,20) or u.dtype.kind!="f" or not np.isfinite(u).all(): raise ValueError("Actualexecuted targetu20 must be supplied, no qgoal fallback")
    ids=np.array([order.index(s) for s in sources],dtype=np.int64)
    expected_start=np.array(t["trajectory_starts"])[ids];expected_end=np.array(t["trajectory_ends"])[ids]
    if not np.array_equal(start,expected_start) or not np.array_equal(end,expected_end): raise ValueError("Replay memory prefix belongs to anothertrajectory/catalog")
    selected=np.minimum(np.maximum(local,0),end-start-1)
    return {"track_trajectory_id":ids,"source_registry_id":np.array([registry.source_ids.index(s) for s in sources],dtype=np.int64),
            "native_start":start,"native_end":end,"raw_local_time":local,"local_reference_frame":selected,
            "native_global_reference_frame":start+selected,"source_frame":np.array([registry.entry(s)["source_frame_range_half_open"][0] for s in sources])+selected,
            "episode_id":episode,"actual_u":u.copy(),"track_name":track_name,"use":use,
            "registry_sha256":registry.sha256,"metadata_is_network_input":False,"training_integration_pending":True}


def replay_window_mask(metadata,*,done,valid):
    """Terminaltransition retained; afterreset/invalidstates never crosseswindow."""
    done,valid=np.asarray(done),np.asarray(valid)
    if done.dtype!=np.bool_ or valid.dtype!=np.bool_ or done.shape!=valid.shape or done.ndim!=2: raise ValueError("Expected booleanB,L masks")
    ids=np.asarray(metadata["track_trajectory_id"]);episodes=np.asarray(metadata["episode_id"])
    times=np.asarray(metadata["raw_local_time"]);starts=np.asarray(metadata["native_start"]);ends=np.asarray(metadata["native_end"])
    if any(x.shape!=done.shape for x in (ids,episodes,times,starts,ends)): raise ValueError("Window metadata shape")
    change=(ids[:,1:]!=ids[:,:-1])|(episodes[:,1:]!=episodes[:,:-1])
    if np.any(change&~done[:,:-1]): raise ValueError("Unmarked trajectory/episode switch; not a valid supervisedwindow")
    continuous=(~done[:,:-1])&valid[:,:-1]
    if np.any(((times[:,1:]!=times[:,:-1]+1)|(starts[:,1:]!=starts[:,:-1])|(ends[:,1:]!=ends[:,:-1]))&continuous):
        raise ValueError("Referencephase/prefix drift inside a continuing replaywindow")
    alive=np.cumprod(np.c_[np.ones((len(done),1),dtype=np.int64),~done[:,:-1]],axis=1)
    return alive*np.cumprod(valid,axis=1)
