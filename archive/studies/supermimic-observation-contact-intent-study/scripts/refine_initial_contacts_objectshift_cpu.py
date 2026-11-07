"""Independent H23 q20+object-translation geometry branch; no physics/GPU.

Default invocation validates the fixed recipe and initial residual only. Actual
optimization requires BOTH --run and an explicitly authorized recipe. Source
local exterior anchors and the prior selected hand material points stay frozen.
Object translation is a morphology/task-mapping revision, not an observation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'D:/workspace/supermimic-observation-retarget-study/scripts')
from native_filtered_contact_geometry import NativeFilteredContactGeometry
from retarget_palmroot_cpu import BONES
from material_anchors import MaterialAnchor, MaterialAnchorGeometry
from refine_initial_contacts_distal_cpu import geometry_only, contact_records


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def ball_translation(z):
    """Open 20mm Euclidean ball, with analytic Jacobian w.r.t. z (not t)."""
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (3,) or not np.isfinite(z).all():
        raise ValueError('Expected finite z shape(3,)')
    s = np.sqrt(1.+z@z)
    return .020*z/s, .020*(np.eye(3)/s-np.outer(z,z)/s**3)


def object_translation_distance_jacobian(g, evaluation):
    """Generated pair distance derivative: force-on-object normal; self=0.

    Stored selected_normal_a_to_b_w uses sorted catalogue order, not necessarily
    native contact.geom order. Translating object A contributes -n; object B +n.
    No generated contact means UNKNOWN gap and remains NaN here, never gap0.
    """
    derivative = np.full((len(g.pairs),3),np.nan)
    derivative[evaluation.generated_mask & (g.pair_kind=='hand_self')] = 0.
    for i in np.flatnonzero(evaluation.generated_mask & (g.pair_kind=='hand_object')):
        a,b = g.pairs[i]
        if (a in g.object_ids) == (b in g.object_ids):
            raise ValueError('Expected exactly one object geom in hand-object pair')
        derivative[i] = evaluation.selected_normal_a_to_b_w[i]*(1. if b in g.object_ids else -1.)
    return derivative


class ObjectShiftObjective:
    """Pure kinematic residual/Jacobian in x=[q20,z3], with fixed rotations."""
    def __init__(self, recipe):
        if recipe['schema'] != 'frame0_material_contact_objectshift:1':
            raise ValueError('Unexpected recipe schema')
        for name,pin in recipe['pins'].items():
            if sha(pin['path']) != pin['sha256']:
                raise ValueError('Source/input drift: '+name)
        self.recipe = recipe
        parent = read(recipe['pins']['parent_recipe']['path'])
        for name,pin in parent['pins'].items():
            if recipe['pins'].get(name)!=pin:
                raise ValueError('Parent pins must be flattened unchanged: '+name)
        if parent['schema']!='frame0_material_contact_hypothesis:2' or parent['fingers']!=[2,3]:
            raise ValueError('Only the frozen H23 distal-v2 parent is supported')
        if recipe['weights'] != {**parent['weights'],'object_translation':1.}:
            raise ValueError('Original weights plus translation prior1 must be preserved')
        if (recipe['translation_radius_m'],recipe['translation_prior_scale_m'],recipe['z_bound_abs']) != (.020,.010,10.):
            raise ValueError('Fixed translation-domain/prior contract changed')
        if recipe['max_nfev']!=120 or recipe['CPU_soft_wall_seconds']!=60:
            raise ValueError('Fixed CPU optimization budget changed')
        self.g = g = NativeFilteredContactGeometry(parent['pins']['model']['path'],parent['pins']['context']['path'])
        self.robot = g.robot
        self.material = MaterialAnchorGeometry(g)
        with np.load(parent['pins']['baseline']['path'],allow_pickle=False) as z:
            self.qbase=z['joint_pos_rad'][0].copy()
            self.pose0=z['mapped_object_pose_w'][0].copy()
            self.targets=z['morphology_normalized_target_points_w'][0].copy()
        vectors=self.targets[BONES[:,1]]-self.targets[BONES[:,0]]
        self.desired=vectors/np.linalg.norm(vectors,axis=1)[:,None]
        g.set_object_pose(self.pose0)
        # Use the exact parent selected material points, never reselect at a
        # moved object pose or read an upstream source robot-q trajectory.
        self.bindings=read(recipe['pins']['parent_bindings']['path'])
        if self.bindings['hypothesis_id']!='H23': raise ValueError('Expected H23 bindings')
        self.anchors=[]
        for a in self.bindings['frozen_material_anchors']:
            a=dict(a)
            for key in ('point_geom_local','support_normal_geom_local'):
                a[key]=np.array(a[key],dtype=np.float64);a[key].setflags(write=False)
            anchor=MaterialAnchor(**a)
            if anchor.geom_id not in g.hand_ids or anchor.geom_name!=g.model.geom(anchor.geom_id).name:
                raise ValueError('Frozen material geometry mismatch')
            self.anchors.append(anchor)
        if [a.finger for a in self.anchors]!=[2,3]: raise ValueError('Wrong anchor identity/order')
        self.goals0=np.array(self.bindings['target_material_points_w'],dtype=np.float64)
        self.normals=np.array(self.bindings['target_hand_normals_w'],dtype=np.float64)
        # Native geometric contacts only; same actual friction metadata as parent.
        with np.load(parent['pins']['context']['path'],allow_pickle=False) as z:
            g.model.geom_friction[:]=z['geom_friction'][0]
        self.lower=np.r_[self.robot.lower,[-10.]*3]
        self.upper=np.r_[self.robot.upper,[10.]*3]
        self.x0=np.r_[self.qbase,np.zeros(3)]
        self.last_x=None
        self.cached=None
        self.last_components=None

    def evaluate(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.shape!=(23,) or not np.isfinite(x).all():
            raise ValueError('Expected finite x shape(23,)')
        if self.last_x is not None and np.array_equal(x,self.last_x): return self.cached
        q=x[:20]
        dt,D=ball_translation(x[20:])
        pose=self.pose0.copy();pose[:3]+=dt
        self.g.set_object_pose(pose)
        native=self.g.evaluate(q)
        dt_jac=object_translation_distance_jacobian(self.g,native)
        contact_J=np.zeros((len(self.g.pairs),23))
        contact_J[:,:20]=native.residual_jacobian
        contact_J[native.active_mask,20:]=-dt_jac[native.active_mask]@D/.001
        features=[self.material.oriented_features(q,a) for a in self.anchors]
        points=np.array([a[0] for a in features])
        normals=np.array([a[2] for a in features])
        bone,boneJ=self.robot.bone_features(q)
        residual=np.r_[(bone-self.desired).ravel(),native.residual,
                       ((points-self.goals0-dt)/.001).ravel(),
                       (.25*(normals-self.normals)).ravel(),
                       .03*(q-self.qbase),.001*(q-self.robot.nominal),dt/.010]
        jacobian=np.vstack((np.c_[boneJ,np.zeros((45,3))],contact_J,
                            np.c_[np.vstack([a[1]/.001 for a in features]),np.tile(-D/.001,(len(features),1))],
                            np.c_[np.vstack([.25*a[3] for a in features]),np.zeros((3*len(features),3))],
                            np.c_[.03*np.eye(20),np.zeros((20,3))],
                            np.c_[.001*np.eye(20),np.zeros((20,3))],
                            np.c_[np.zeros((3,20)),D/.010]))
        if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
            raise ValueError('Nonfinite coupled geometry residual/Jacobian')
        self.last_x=x.copy();self.cached=(residual,jacobian)
        self.last_components={'native':native,'dt_m':dt.copy(),'dt_dz':D.copy(),
                              'distance_dt':dt_jac.copy(),'pose':pose.copy(),
                              'anchor_points_w':points.copy(),'anchor_normals_w':normals.copy()}
        return self.cached


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    recipe=read(args.recipe)
    if args.run and recipe.get('optimization_authorized') is not True:
        raise ValueError('No optimization authorization in recipe')
    if args.run and args.output is None: raise ValueError('--output required for --run')
    started=time.monotonic()
    with geometry_only():
        objective=ObjectShiftObjective(recipe)
        r,J=objective.evaluate(objective.x0)
        preflight={'status':'CPU_PREFLIGHT_ONLY_NOT_OPTIMIZED','script_sha256':sha(__file__),
                   'recipe_sha256':sha(args.recipe),'residual_shape':list(r.shape),'jacobian_shape':list(J.shape),
                   'all_finite':True,'initial_shift_m':[0.,0.,0.],'physical_reference_qualified':False,
                   'object_pose_fixed':False,'object_rotation_and_wrist_fixed':True,'GPU_used':False,'physics_steps':0}
        if not args.run:
            print(json.dumps(preflight,indent=2));return
        out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
        write(out/'recipe_snapshot.json',recipe)
        (out/'optimizer_source.py').write_bytes(Path(__file__).read_bytes())
        write(out/'parent_anchor_bindings.json',objective.bindings)
        def budget():
            if time.monotonic()-started>60: raise TimeoutError('Fixed60s CPU recipe; no automatic retry')
        def residual(x): budget();return objective.evaluate(x)[0]
        def jacobian(x): budget();return objective.evaluate(x)[1]
        try:
            result=least_squares(residual,objective.x0.copy(),jac=jacobian,bounds=(objective.lower,objective.upper),
                                 max_nfev=120,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        except Exception as error:
            # Preserve the last already evaluated iterate without any extra
            # collision query, optimizer restart, target change or retry.
            c=objective.last_components
            np.savez_compressed(out/'partial_last_evaluated.npz',x=objective.last_x,
                                residual=objective.cached[0],jacobian=objective.cached[1],
                                mapped_object_pose_w=c['pose'],object_translation_delta_w_m=c['dt_m'])
            write(out/'failure.json',{**preflight,'status':'ENGINEERING_FAILURE_PARTIAL_ONLY',
                  'error_type':type(error).__name__,'error':str(error),
                  'wall_seconds_excluding_imports':time.monotonic()-started,
                  'candidate_accepted_for_dynamics':False,'no_retry':True})
            raise
        final_r,final_J=objective.evaluate(result.x)
        c=objective.last_components;g=objective.g;native=c['native']
        records=contact_records(g)
        points=g.robot.forward(result.x[:20])[0].copy()
        bone=g.robot.bone_features(result.x[:20])[0]
        angles=np.rad2deg(np.arccos(np.clip(np.sum(bone*objective.desired,axis=1),-1,1)))
        summary={}
        for kind in ('hand_object','hand_self'):
            mask=native.generated_mask&(g.pair_kind==kind)
            summary[kind]={'generated_pairs':int(mask.sum()),'max_generated_depth_m':float(np.maximum(-native.signed_distance_m[mask],0).max()) if mask.any() else None}
        report={**preflight,'status':'COMPLETED_GEOMETRIC_OBJECT_TRANSLATION_HYPOTHESIS_ONLY',
                'optimizer_success':bool(result.success),'optimizer_status':int(result.status),
                'nfev':int(result.nfev),'cost':float(result.cost),
                'object_translation_delta_w_m':c['dt_m'].tolist(),'object_translation_norm_m':float(np.linalg.norm(c['dt_m'])),
                'source_rigid_mapping_object_pose_w':objective.pose0.tolist(),'revised_object_pose_w':c['pose'].tolist(),
                'object_rotation_exact':bool(np.array_equal(c['pose'][3:],objective.pose0[3:])),
                'material_point_goal_error_m':np.linalg.norm(c['anchor_points_w']-objective.goals0-c['dt_m'],axis=1).tolist(),
                'material_normal_goal_error_deg':np.rad2deg(np.arccos(np.clip(np.sum(c['anchor_normals_w']*objective.normals,axis=1),-1,1))).tolist(),
                'bone_angle_mean_deg':float(angles.mean()),'bone_angle_max_deg':float(angles.max()),
                'target_tip_error_mean_m':float(np.linalg.norm(points[[4,8,12,16,20]]-objective.targets[[4,8,12,16,20]],axis=1).mean()),
                'q_baseline_change_abs_max_rad':float(np.abs(result.x[:20]-objective.qbase).max()),
                'native_summary':summary,'native_contact_tie_slots':int(np.count_nonzero(native.generated_mask&(native.tie_count>1))),
                'hard_joint_excess_max_rad':float(np.maximum(np.maximum(g.robot.lower-result.x[:20],result.x[:20]-g.robot.upper),0).max()),
                'candidate_accepted_for_dynamics':False,'PD_target':None,'confidence':None,'contacts_known':False,
                'wall_seconds_excluding_imports':time.monotonic()-started,
                'scope':'Source-mapping translation design revision with unknown confidence. Geometry only; local native contact Jacobian and fitted surface anchors are not force/equilibrium/physical-grasp evidence.'}
        np.savez_compressed(out/'initialization_candidate.npz',joint_pos_rad=result.x[None,:20],
                            object_shift_latent_z=result.x[None,20:],object_translation_delta_w_m=c['dt_m'][None],
                            mapped_object_pose_w=c['pose'][None],original_mapped_object_pose_w=objective.pose0[None],
                            morphology_normalized_target_points_w=objective.targets[None],
                            fk_hand_points_w=points[None],bone_angle_error_deg=angles[None],
                            baseline_joint_pos_rad=objective.qbase[None],
                            selected_native_contact_distance_m=native.signed_distance_m[None],
                            generated_contact_mask=native.generated_mask[None],geometry_pair_indices=g.pairs,geometry_pair_kind=g.pair_kind,
                            material_goal_points_w=objective.goals0+c['dt_m'],material_goal_normals_w=objective.normals,
                            joint_lower_rad=g.robot.lower,joint_upper_rad=g.robot.upper,residual=final_r,jacobian=final_J)
        write(out/'native_contacts.json',records)
        report['output_npz_sha256']=sha(out/'initialization_candidate.npz')
        write(out/'report.json',report)
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
