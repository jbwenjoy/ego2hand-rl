"""Finger-indexed candidate patch correspondence, not contact truth or reward.

Unknown source geometry stays unknown. A match means only proximity and normal
agreement with an explicitly estimated patch set, never force feasibility.
"""
import numpy as np


def match_patches(robot_points,robot_normals,robot_groups,target_points,target_inward,target_groups,target_known,
                  *,groups=(1,2,3),distance_m=.003,normal_cos=.5):
    rp,rn=np.asarray(robot_points,float),np.asarray(robot_normals,float)
    tp,tn=np.asarray(target_points,float),np.asarray(target_inward,float)
    rg,tg=np.asarray(robot_groups),np.asarray(target_groups);known=np.asarray(target_known,dtype=bool)
    if rp.shape!=rn.shape or tp.shape!=tn.shape or rp.shape!=(len(rg),3) or tp.shape!=(len(tg),3) or known.shape!=(len(tp),):
        raise ValueError('Explicit point/normal/group/known shapes')
    if not np.isfinite(np.r_[rp.ravel(),rn.ravel(),tp[known].ravel(),tn[known].ravel()]).all():
        raise ValueError('Finite robot and known source geometry required')
    if distance_m<=0 or not -1<=normal_cos<=1:raise ValueError('Explicit positive radius and cosine')
    if (len(rn) and not np.allclose(np.linalg.norm(rn,axis=1),1,atol=1e-6)) or (known.any() and not np.allclose(np.linalg.norm(tn[known],axis=1),1,atol=1e-6)):
        raise ValueError('Normals must already be unit; do not silently repair')
    result=[]
    for group in groups:
        rids=np.flatnonzero(rg==group);tids=np.flatnonzero((tg==group)&known)
        if not len(tids):
            result.append(dict(group=int(group),source_geometry_known=False,matched=None,pairs=[],native_contacts=int(len(rids))))
            continue
        if not len(rids):
            result.append(dict(group=int(group),source_geometry_known=True,matched=False,pairs=[],native_contacts=0))
            continue
        distances=np.linalg.norm(rp[rids,None]-tp[tids],axis=2);dots=rn[rids]@tn[tids].T
        ii,jj=np.where((distances<=distance_m)&(dots>=normal_cos))
        result.append(dict(group=int(group),source_geometry_known=True,matched=bool(len(ii)),native_contacts=int(len(rids)),
            minimum_candidate_distance_m=float(distances.min()),best_normal_dot=float(dots.max()),
            pairs=[[int(rids[i]),int(tids[j])] for i,j in zip(ii,jj)]))
    all_known=all(x['source_geometry_known'] for x in result)
    return dict(groups=result,all_source_geometry_known=all_known,all_groups_matched=all(x['matched'] for x in result) if all_known else None,
        source_contact_truth_known=False,physical_success_certified=False)
