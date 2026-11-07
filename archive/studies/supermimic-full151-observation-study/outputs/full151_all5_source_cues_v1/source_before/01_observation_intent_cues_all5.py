"""Saved observation/surface proximity cues, never contact target labels.

New all-five cue schema, not A2 objective targets. No robot q or generated-contact field is consumed. The exterior query has already
been performed by the pinned pen_outer_surface.py producer. This module exports
its saved source-rigid results without invoking that geometry implementation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

SCHEMA='full151_source_potential_five_cues:1'
FINGERS=np.array([1,2,3,4,5],np.int32)
LANDMARKS=np.array([3,7,11,15,19],np.int32)
OBS_FIELDS=('time_s','hand_landmarks_source_w','object_pose_source_w','landmark_confidence_known','object_confidence_known')
MAP_FIELDS=('time_s','source_to_fixed_wrist_R','source_to_fixed_wrist_translation_m','mapped_object_pose_w')
PFX='source_rigid_mapped__'
SURFACE_SUFFIXES=('points_w','points_o','closest_o_m','closest_w','outward_normal_o','normal_w','signed_distance_m','unsigned_distance_m','feature_index','tie_mask','tie_count','axis_radial_normal_ambiguity')
SAVED_FIELDS=('time_s','source_to_fixed_wrist_R','source_to_fixed_wrist_translation_m')+tuple(PFX+k for k in SURFACE_SUFFIXES)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}

def load_selected(path,fields):
    # np.load opens the zip; only these explicitly selected members are read.
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in fields}

def rotation64(q):
    q=np.asarray(q,np.float64)
    if q.shape[-1]!=4 or not np.isfinite(q).all() or np.any(np.abs(np.linalg.norm(q,axis=-1)-1)>1e-6):raise ValueError('Proper finite unit wxyz quaternion required')
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))

def _same(a,b,name,tol=0.):
    if np.shape(a)!=np.shape(b) or not np.allclose(a,b,rtol=0,atol=tol,equal_nan=False):raise ValueError('Saved source relation differs: '+name)

def extract_all5_cues(observations,mapping,saved):
    """Select151x5 source-rigid cues. Mapping arguments can enforce read allowlists."""
    time=np.asarray(observations['time_s']);points=np.asarray(observations['hand_landmarks_source_w'],np.float64)
    obj_source=np.asarray(observations['object_pose_source_w'],np.float64)
    if time.shape!=(151,) or points.shape!=(151,21,3) or obj_source.shape!=(151,7):raise ValueError('Only full151 source schema is accepted')
    _same(time,np.arange(151)*.02,'canonical151 time')
    if not np.isfinite(points).all() or not np.isfinite(obj_source).all():raise ValueError('Nonfinite source observation')
    R=np.asarray(mapping['source_to_fixed_wrist_R']);t=np.asarray(mapping['source_to_fixed_wrist_translation_m']);pose=np.asarray(mapping['mapped_object_pose_w'])
    _same(mapping['time_s'],time,'mapped time');_same(saved['time_s'],time,'surface time')
    if R.shape!=(151,3,3) or t.shape!=(151,3) or pose.shape!=(151,7):raise ValueError('Mapping shape differs')
    if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite(pose).all():raise ValueError('Nonfinite mapping')
    _same(R@np.swapaxes(R,-1,-2),np.broadcast_to(np.eye(3),R.shape),'proper R',1e-12)
    if np.any(np.linalg.det(R)<=0):raise ValueError('Mapping reflection is not accepted')
    _same(R,saved['source_to_fixed_wrist_R'],'saved surface R')
    _same(t,saved['source_to_fixed_wrist_translation_m'],'saved surface t')
    objR=rotation64(pose[:,3:]);sourceR=rotation64(obj_source[:,3:])
    _same(pose[:,:3],np.einsum('tij,tj->ti',R,obj_source[:,:3])+t,'object position mapping',1e-12)
    _same(objR,R@sourceR,'object rotation mapping',1e-12)
    mapped=np.einsum('tij,tkj->tki',R,points)+t[:,None,:]
    local=np.einsum('tji,tkj->tki',objR,mapped-pose[:,None,:3])
    _same(mapped,saved[PFX+'points_w'],'source rigid world points',1e-12)
    _same(local,saved[PFX+'points_o'],'source rigid body points',1e-12)
    selected={k:np.asarray(saved[PFX+k])[:,LANDMARKS].copy() for k in SURFACE_SUFFIXES}
    shape=(151,5)
    for key in ['signed_distance_m','unsigned_distance_m','feature_index','tie_count','axis_radial_normal_ambiguity']:
        if selected[key].shape!=shape:raise ValueError('Surface scalar shape differs: '+key)
    for key in ['points_w','points_o','closest_o_m','closest_w','outward_normal_o','normal_w']:
        if selected[key].shape!=shape+(3,):raise ValueError('Surface vector shape differs: '+key)
    closest_rebuilt=np.einsum('tij,tkj->tki',objR,selected['closest_o_m'])+pose[:,None,:3]
    normal_rebuilt=np.einsum('tij,tkj->tki',objR,selected['outward_normal_o'])
    for rebuilt,stored,name in [(closest_rebuilt,selected['closest_w'],'closest point BODY/world'),(normal_rebuilt,selected['normal_w'],'outward normal BODY/world')]:
        finite=np.isfinite(rebuilt).all(-1)&np.isfinite(stored).all(-1)
        _same(rebuilt[finite],stored[finite],name,1e-12)
    ties=selected['tie_mask'];tc=selected['tie_count'];feature=selected['feature_index']
    if ties.shape!=(151,5,13) or ties.dtype!=np.bool_ or tc.dtype.kind not in 'iu' or feature.dtype.kind not in 'iu':raise ValueError('Frozen13-patch branch/tie schema differs')
    if np.any(ties.sum(-1)!=tc):raise ValueError('Tie count does not match tie mask')
    signed=selected['signed_distance_m'];unsigned=selected['unsigned_distance_m']
    finite_dist=np.isfinite(signed)&np.isfinite(unsigned)&(unsigned>=0)
    if np.any(finite_dist & (np.abs(np.abs(signed)-unsigned)>1e-12)):raise ValueError('Signed/unsigned saved distance differs')
    valid_patch=(feature>=0)&(feature<13)&(tc>=1)
    point_valid=np.isfinite(selected['closest_o_m']).all(-1)&np.isfinite(selected['closest_w']).all(-1)&finite_dist&valid_patch
    normals=selected['outward_normal_o'];nn=np.linalg.norm(normals,axis=-1)
    normal_numeric_valid=np.isfinite(normals).all(-1)&np.isfinite(selected['normal_w']).all(-1)&(np.abs(nn-1)<=1e-6)&valid_patch
    axis_ambiguous=selected['axis_radial_normal_ambiguity'].astype(bool)
    branch_unknown=(~valid_patch)|(tc!=1)|axis_ambiguous
    normal_valid=normal_numeric_valid&(~branch_unknown)&point_valid
    lc=np.asarray(observations['landmark_confidence_known']);oc=np.asarray(observations['object_confidence_known'])
    if lc.shape!=(151,21) or oc.shape!=(151,) or lc.dtype!=np.bool_ or oc.dtype!=np.bool_:raise ValueError('Confidence-known masks must be genuine booleans')
    # Current source has no measured confidence. No proxy is promoted into one.
    if lc.any() or oc.any():raise ValueError('Frozen source confidence contract is all unknown')
    z=np.zeros(shape,bool)
    arrays={'time_s':time.copy(),'finger_ids':FINGERS.copy(),'landmark_indices':LANDMARKS.copy(),
            'source_landmark_source_w':points[:,LANDMARKS].copy(),'source_landmark_mapped_w':selected['points_w'],'source_landmark_object_body_m':selected['points_o'],
            'mapped_object_pose_w':pose.copy(),'source_to_fixed_wrist_R':R.copy(),'source_to_fixed_wrist_translation_m':t.copy(),
            'object_surface_closest_body_m':selected['closest_o_m'],'object_surface_closest_mapped_w':selected['closest_w'],
            'object_surface_outward_normal_body':normals,'object_surface_outward_normal_mapped_w':selected['normal_w'],
            'signed_distance_m':signed,'unsigned_distance_m':unsigned,'surface_feature_index':feature,'surface_tie_mask':ties,'surface_tie_count':tc,
            'surface_axis_radial_normal_ambiguity':axis_ambiguous,'surface_point_numeric_valid':point_valid,'surface_normal_numeric_valid':normal_numeric_valid,
            'surface_normal_valid_unambiguous':normal_valid,'geometric_branch_unknown':branch_unknown,
            'landmark_confidence_known':lc[:,LANDMARKS].copy(),'object_confidence_known':oc.copy(),
            'human_contact_known':z.copy(),'human_contact_present':z.copy(),'human_contact_unknown_mask':~z,
            'declared_robot_intent_known':z.copy(),
            'geometry_valid_mask':point_valid&normal_valid&(~branch_unknown),
            'declared_potential_mask':np.ones(shape,dtype=bool)}
    validate_cues(arrays)
    validate_potential_masks(arrays)
    return arrays

def validate_cues(arrays):
    for key in ['human_contact_known','human_contact_present','declared_robot_intent_known','landmark_confidence_known']:
        if arrays[key].shape!=(151,5) or arrays[key].dtype!=np.bool_ or arrays[key].any():raise ValueError('No contact labels, intent or measured confidence may be introduced: '+key)
    if arrays['human_contact_unknown_mask'].shape!=(151,5) or not arrays['human_contact_unknown_mask'].all():raise ValueError('All human contact states remain UNKNOWN')

def validate_potential_masks(arrays):
    expected=arrays['surface_point_numeric_valid']&arrays['surface_normal_valid_unambiguous']&~arrays['geometric_branch_unknown']
    if arrays['geometry_valid_mask'].shape!=(151,5) or arrays['geometry_valid_mask'].dtype!=np.bool_ or not np.array_equal(arrays['geometry_valid_mask'],expected):raise ValueError('Geometry-valid mask must be independent numeric/unambiguous geometry')
    if arrays['declared_potential_mask'].shape!=(151,5) or arrays['declared_potential_mask'].dtype!=np.bool_ or not arrays['declared_potential_mask'].all():raise ValueError('Fixed all5 potential membership is separate, not a contact observation')


def require_objective_loading(metadata):
    """Fail closed: this schema is an observation cue, never an objective spec."""
    raise ValueError('All-five source cues are not contact targets: target_ready=False and objective_loading_permitted=False')

def descriptive_summary(arrays):
    result=[]
    for j,(f,l) in enumerate(zip(FINGERS,LANDMARKS)):
        d=arrays['signed_distance_m'][:,j];p=arrays['object_surface_closest_body_m'][:,j];n=arrays['object_surface_outward_normal_body'][:,j]
        result.append({'finger_id':int(f),'landmark_index':int(l),'signed_distance_min_m':float(np.nanmin(d)),'signed_distance_max_m':float(np.nanmax(d)),
                       'objectlocal_closest_surface_shift_from_frame0_max_m':float(np.nanmax(np.linalg.norm(p-p[0],axis=-1))),
                       'objectlocal_normal_angle_from_frame0_max_deg':float(np.nanmax(np.rad2deg(np.arccos(np.clip(n@n[0],-1,1))))),
                       'geometric_branch_unknown_count':int(arrays['geometric_branch_unknown'][:,j].sum()),
                       'normal_unambiguous_count':int(arrays['surface_normal_valid_unambiguous'][:,j].sum())})
    return result
