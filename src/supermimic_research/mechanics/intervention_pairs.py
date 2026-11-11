"""Pair actual actuator interventions without conflating prefix seeds or states."""
import numpy as np


def intervention_pairs(raw):
    fields=['seed','dt','k','joint','amplitude'];keys=list(zip(*(raw[f].tolist() for f in fields)));pairs=[]
    for key in sorted(set(keys)):
        ids=np.array([i for i,v in enumerate(keys) if v==key]);sign=raw['sign'][ids]
        if len(ids)!=2 or sorted(sign.tolist())!=[-1,1]:raise ValueError('Exactly one minus/plus per seed/dt/k/joint/amplitude')
        lo=int(ids[sign==-1][0]);hi=int(ids[sign==1][0]);joint=int(key[3])
        if not np.array_equal(raw['x'][lo,:81],raw['x'][hi,:81]) or raw['x'][lo,-1]!=raw['x'][hi,-1] or not np.array_equal(raw['source_native_state'][lo],raw['source_native_state'][hi]):raise ValueError('Intervention pairs must share complete source state and numerical warmstart')
        changed=raw['x'][hi,81:101]-raw['x'][lo,81:101]
        if not 0<=joint<20 or np.flatnonzero(changed).tolist()!=[joint] or changed[joint]<=0:raise ValueError('Declared single actual control difference must be positive and nonzero')
        pairs.append((lo,hi,joint,changed[joint]))
    return np.array(pairs,dtype=np.float64)
