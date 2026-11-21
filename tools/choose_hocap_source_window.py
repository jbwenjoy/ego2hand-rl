"""Source-only deterministic pick/place window selection for supplementary evaluation."""
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.assets.proximity import SurfaceQuery


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1,strict=True)


def candidate_bounds(all_poses, object_index, vertices):
    target=all_poses[object_index]
    if len(target)<15 or not np.isfinite(all_poses).all():
        return None,"invalid source trajectory"
    matrices=Rotation.from_quat(target[:,:4]).as_matrix()
    bottoms=(vertices @ matrices.transpose(0,2,1) + target[:,None,4:])[:,:,2].min(1)
    height=bottoms-np.median(bottoms[:15])
    lift=next(((a,b) for a,b in runs(height>.020) if b-a+1>3),None)
    if lift is None:
        return None,"no four consecutive source frames more than 20 mm above initial support"
    first,end=lift
    lo=max(0, first-100)
    hi=min(len(target)-1, first+220)
    placement=next((a for a,b in runs(height<=.015) if a>end and b-a+1>=6 and a+5<=hi),None)
    if placement is None:
        return None,"no sustained source return to initial support range inside the bounded crop"
    return dict(probe_first=int(lo),probe_last=int(hi),lift_frame=int(first),placement_frame=int(placement),
                peak_relative_bottom_clearance_mm=float(1000*height[lo:hi+1].max())),None


def select_window(bounds, all_poses, object_index, mesh, fixed_camera_label_paths):
    first,last=bounds["probe_first"],bounds["probe_last"]
    query=SurfaceQuery(mesh.vertices,mesh.faces)
    distances=[]
    for path in fixed_camera_label_paths:
        with np.load(path,allow_pickle=False) as label:
            hand=label["hand_joints_3d"][0,[4,8,12,16,20]]
            if not np.isfinite(hand).all() or np.any(hand[:,2]<=0):
                return None,"missing right-hand joint data in the inspection crop"
            obj=label["obj_poses"][object_index]
            local=(hand-obj[:3,3])@obj[:3,:3]
            distances.append(float(query.nearest(local)[1].min()))
    distances=np.asarray(distances)
    if distances.shape!=(last-first+1,):
        raise ValueError("Expected contiguous source labels throughout the probe interval")
    sustained=np.flatnonzero(np.convolve((distances>=.040).astype(int),np.ones(6,int),mode="valid")==6)
    before=sustained[sustained+first+5 < bounds["lift_frame"]]
    after=sustained[sustained+first >= bounds["placement_frame"]]
    if not len(before) or not len(after):
        return None,"missing six detached fingertip frames on one or both sides of source lift/place"
    begin,end=int(first+before[-1]),int(first+after[0]+5)
    if end-begin+1>300:
        return None,"qualifying window exceeds 300 source frames"
    near=int((distances[begin-first:end-first+1]<=.010).sum())
    if near<8:
        return None,"fewer than eight frames with tip joints within 10 mm"
    delta=all_poses[:,begin:end+1,4:] - all_poses[:,begin:begin+1,4:]
    excursion=1000*np.linalg.norm(delta,axis=-1).max(1)
    others=np.delete(excursion,object_index)
    if excursion[object_index]<100:
        return None,"target source excursion below 100 mm"
    if np.max(others)>25:
        return None,"other source object excursion exceeds 25 mm"
    return dict(first=begin,last=end,count=end-begin+1,near_10mm_frames=near,
                excursion_mm=float(excursion[object_index]),other_excursion_max_mm=float(np.max(others)),
                start_end_tip_distance_mm=(1000*distances[[begin-first,end-first]]).tolist()),None
