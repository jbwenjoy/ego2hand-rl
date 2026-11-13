"""Do not conflate intentionally different controllers with a teacher's aliases."""
import numpy as np


def public_action_aliases(observations,actions,contexts):
    all_groups={};context_groups={}
    for i,(obs,context) in enumerate(zip(observations,contexts)):
        key=obs.tobytes();all_groups.setdefault(key,[]).append(i);context_groups.setdefault((context,key),[]).append(i)
    mixed=[];same=[];conflicts=[]
    for ids in all_groups.values():
        labels=sorted(set(contexts[i] for i in ids))
        if len(ids)>1 and len(labels)>1:mixed.append(dict(rows=ids,contexts=labels))
    for (context,_),ids in context_groups.items():
        if len(ids)>1:
            same.append(ids);delta=float(np.ptp(actions[ids],axis=0).max())
            if delta>0:conflicts.append(dict(context=context,rows=ids,max_pairwise_action_difference=delta))
    return dict(mixed_controller_context_groups=mixed,same_context_alias_groups=len(same),same_context_conflicting_actions=conflicts,
        limitation='Different controllers intentionally produce different labels. Same-context cross-dt teacher conflicts may reflect numerical state, optimizer nonuniqueness or sensitivity; they do not prove physical public observations are insufficient for control.')
