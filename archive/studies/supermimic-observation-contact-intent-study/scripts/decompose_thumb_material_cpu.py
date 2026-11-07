"""Two fixed thumb point+normal FK fits; no collision, forces, physics or GPU."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import re
import sys
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('D:/workspace/supermimic-observation-retarget-study')
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(OLD/'scripts'))
from material_anchors import MaterialAnchor,MaterialAnchorGeometry
from retarget_palmroot_cpu import RobotKinematics

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mat(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q);w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')

def reach_bound(robot,anchor,finger_joint_ids,target):
    """Triangle-inequality bound about fixed joint1 anchor, with hinge offsets.

    An origin shift from rotation about local a is <=2||a||. At the root hinge
    itself, subtracting its fixed world anchor contributes only||a_root||.
    All rest/body quaternions are rotations and do not enlarge vector norms.
    """
    m,d=robot.model,robot.data;first=int(finger_joint_ids[0]);rootbody=int(m.jnt_bodyid[first])
    if int(m.jnt_type[first])!=int(mujoco.mjtJoint.mjJNT_HINGE) or int(m.body_jntnum[rootbody])!=1:
        return {'certificate_available':False,'reason':'Unsupported root joint layout'}
    path=[];body=anchor.body_id
    while body!=rootbody and body!=0:
        path.append(body);body=int(m.body_parentid[body])
    if body!=rootbody:return {'certificate_available':False,'reason':'Material geom is outside requested finger subtree'}
    rows=[];radius=float(np.linalg.norm(m.jnt_pos[first]))
    for b in reversed(path):
        joints=[];offset=float(np.linalg.norm(m.body_pos[b]))
        for j in range(int(m.body_jntadr[b]),int(m.body_jntadr[b]+m.body_jntnum[b])):
            if int(m.jnt_type[j])!=int(mujoco.mjtJoint.mjJNT_HINGE):return {'certificate_available':False,'reason':'Unsupported nonhinge descendant'}
            correction=2*float(np.linalg.norm(m.jnt_pos[j]));offset+=correction
            joints.append({'id':j,'name':m.joint(j).name,'jnt_pos':m.jnt_pos[j].tolist(),'rotation_offset_upper_m':correction})
        radius+=offset;rows.append({'body':m.body(b).name,'body_pos':m.body_pos[b].tolist(),'body_translation_norm_m':float(np.linalg.norm(m.body_pos[b])),'joints':joints})
    local_body=m.geom_pos[anchor.geom_id]+mat(m.geom_quat[anchor.geom_id])@anchor.point_geom_local
    radius+=float(np.linalg.norm(local_body));center=d.xanchor[first].copy();distance=float(np.linalg.norm(target-center))
    margin=distance-radius
    return {'certificate_available':True,'center_joint':m.joint(first).name,'center_w_m':center.tolist(),
      'root_jnt_pos':m.jnt_pos[first].tolist(),'root_anchor_offset_norm_m':float(np.linalg.norm(m.jnt_pos[first])),
      'descendant_terms':rows,'material_point_body_local':local_body.tolist(),'material_point_body_radius_m':float(np.linalg.norm(local_body)),
      'strict_radius_upper_m':radius,'target_distance_from_center_m':distance,'target_minus_bound_m':margin,
      'outside_strict_bound':margin>1e-9,
      'conclusion':'Fixed material target excluded even without joint-angle bounds' if margin>1e-9 else 'No unreachable certificate: target is inside this conservative sphere. Local optimization outcomes are not global proof.',
      'scope':'Fixed material point, fixed parent/wrist and original hinge/body tree; includes nonzero hinge-anchor rotation offsets. Does not certify contact with some other material point.'}

def main():
    started=time.monotonic();output=ROOT/'outputs/thumb_material_reachability_v1'
    if output.exists():raise FileExistsError('No overwrite of all-case mechanism evidence')
    rootrun=ROOT/'outputs/frame0_h123_distal_v2';recipepath=rootrun/'recipe_snapshot.json';bindingpath=rootrun/'frozen_anchor_bindings.json'
    recipe,bindings=read(recipepath),read(bindingpath);prior=read(rootrun/'report.json')
    original_recipe=ROOT/'configs/frame0-h123-distal-v2.json'
    assert sha(bindingpath)==prior['bindings_sha256'] and sha(original_recipe)==prior['recipe_sha256']
    assert read(original_recipe)==recipe,'Recipe snapshot content differs from hash-bound original config'
    for pin in recipe['pins'].values():assert sha(pin['path'])==pin['sha256'],'Frozen input/source mismatch'
    sources={str(p):sha(p) for p in [ROOT/'scripts/material_anchors.py',OLD/'scripts/retarget_palmroot_cpu.py',Path(__file__)]}
    forbidden={k:0 for k in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_geomDistance','mj_contactForce','mj_collision')}
    saved={k:getattr(mujoco,k) for k in forbidden}
    def deny(name):
        def fail(*args,**kwargs):forbidden[name]+=1;raise AssertionError('Forbidden API '+name)
        return fail
    case_plan=[{'case_id':f'finger{f}_{objective}_{start}','finger':f,'objective':objective,'start':start,'status':'NOT_RUN'} for f in (1,) for objective in ('B_point_plus_normal',) for start in ('baseline','bounds_midpoint')]
    output.mkdir(parents=True,exist_ok=False)
    write(output/'case_plan_before_optimization.json',{'cases':case_plan,'max_nfev':120,'CPU_soft_wall_seconds':60,'point_scale_m':.001,'normal_weight':.25,'other16_fixed':True,'automatic_retry':False})
    write(output/'recipe_snapshot.json',recipe)
    write(output/'frozen_anchor_bindings.json',bindings)
    (output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write(output/'source_pins_before_optimization.json',{'source':sources,'recipe_path':str(original_recipe),'recipe_sha256':sha(original_recipe),'bindings_path':str(bindingpath),'bindings_sha256':sha(bindingpath),'pins':recipe['pins']})
    arrays=[];bounds={};failure=None
    try:
        for k in forbidden:setattr(mujoco,k,deny(k))
        robot=RobotKinematics(recipe['pins']['model']['path']);m,d=robot.model,robot.data
        with np.load(recipe['pins']['context']['path'],allow_pickle=False) as z:
            for field in ('geom_size','geom_pos'):getattr(m,field)[:]=z[field][0]
        with np.load(recipe['pins']['baseline']['path'],allow_pickle=False) as z:qbase=z['joint_pos_rad'][0].copy();pose=z['mapped_object_pose_w'][0].copy()
        free=m.joint('object/cube_freejoint');d.qpos[int(free.qposadr[0]):int(free.qposadr[0])+7]=pose;robot.last_q=None;robot.forward(qbase)
        helper=MaterialAnchorGeometry(SimpleNamespace(robot=robot,model=m,data=d))
        anchors={}
        for k,b in enumerate(bindings['frozen_material_anchors']):
            a=MaterialAnchor(**{**b,'point_geom_local':np.asarray(b['point_geom_local'],float),'support_normal_geom_local':np.asarray(b['support_normal_geom_local'],float)})
            a.point_geom_local.setflags(write=False);a.support_normal_geom_local.setflags(write=False)
            assert a.geom_name==m.geom(a.geom_id).name and a.body_id==int(m.geom_bodyid[a.geom_id])
            if a.vertex_index is not None:
                mesh=int(m.geom_dataid[a.geom_id]);assert np.array_equal(a.point_geom_local,m.mesh_vert[int(m.mesh_vertadr[mesh])+a.vertex_index])
            anchors[a.finger]=(a,np.asarray(bindings['target_material_points_w'][k],float),np.asarray(bindings['target_hand_normals_w'][k],float))
        def independent(q,a):
            robot.forward(q);rb=mat(d.xquat[a.body_id]);rg=mat(m.geom_quat[a.geom_id]);pb=d.xpos[a.body_id]
            return pb+rb@(m.geom_pos[a.geom_id]+rg@a.point_geom_local),rb@rg@a.support_normal_geom_local
        for finger in (1,):
            a,target,normal=anchors[finger];jids=np.array([m.joint(f'robot/right_finger{finger}_joint{i}').id for i in range(1,5)])
            robot.forward(qbase);bounds[str(finger)]=reach_bound(robot,a,jids,target)
        for row in case_plan:
            if time.monotonic()-started>60:raise TimeoutError('Whole finite two-case CPU budget60s')
            finger=row['finger'];a,target,normal=anchors[finger];indices=np.arange((finger-1)*4,finger*4);lower,upper=robot.lower[indices],robot.upper[indices]
            x0=qbase[indices].copy() if row['start']=='baseline' else (lower+upper)/2
            if np.any(x0<lower) or np.any(x0>upper):raise ValueError('Frozen start outside joint limits; no silent clipping')
            begin=time.monotonic();use_normal=row['objective']=='B_point_plus_normal'
            def features(x):
                if time.monotonic()-started>60:raise TimeoutError('Whole CPU60s; no continuation/retry')
                q=qbase.copy();q[indices]=x;p,jp,n,jn=helper.oriented_features(q,a)
                return q,p,jp[:,indices],n,jn[:,indices]
            def residual(x):
                q,p,jp,n,jn=features(x)
                return np.r_[(p-target)/.001,.25*(n-normal)] if use_normal else (p-target)/.001
            def jacobian(x):
                q,p,jp,n,jn=features(x)
                return np.vstack((jp/.001,.25*jn)) if use_normal else jp/.001
            result=least_squares(residual,x0,jac=jacobian,bounds=(lower,upper),method='trf',max_nfev=120,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            q,p,jp,n,jn=features(result.x);pi,ni=independent(q,a);point_error=float(np.linalg.norm(pi-target));angle=float(np.rad2deg(np.arccos(np.clip(np.dot(ni,normal),-1,1))))
            other=np.setdiff1d(np.arange(20),indices)
            row.update(status='completed_local_optimization',optimizer_success=bool(result.success),optimizer_status=int(result.status),message=str(result.message),nfev=int(result.nfev),njev=int(result.njev),cost=float(result.cost),optimality=float(result.optimality),
              q4_start_rad=x0.tolist(),q4_result_rad=result.x.tolist(),point_w_m=pi.tolist(),target_w_m=target.tolist(),point_error_vector_m=(pi-target).tolist(),point_error_m=point_error,
              normal_w=ni.tolist(),normal_target_w=normal.tolist(),normal_error_deg=angle,
              hard_joint_excess_max_rad=float(np.maximum(np.maximum(lower-result.x,result.x-upper),0).max()),at_bound_components=(np.minimum(result.x-lower,upper-result.x)<1e-6).tolist(),
              other16_joint_exact_unchanged=bool(np.array_equal(q[other],qbase[other])),
              independent_point_vs_helper_max_abs_m=float(np.max(np.abs(pi-p))),independent_normal_vs_helper_max_abs=float(np.max(np.abs(ni-n))),
              point_within1mm=point_error<=.001,normal_within30deg=angle<=30.,wall_seconds=time.monotonic()-begin,
              collision_or_fit_evaluated=False,accepted_for_dynamics=False)
            arrays.append({'q20':q.copy(),'point':pi.copy(),'normal':ni.copy(),'point_Jacobian_3x4':jp.copy(),'normal_Jacobian_3x4':jn.copy()})
    except BaseException as exc:failure=repr(exc)
    finally:
        for k,v in saved.items():setattr(mujoco,k,v)
    checks={'all2_cases_retained':len(case_plan)==2,'all2_completed':all(r['status']=='completed_local_optimization' for r in case_plan),
      'all_other16_fixed':all(r.get('other16_joint_exact_unchanged',False) for r in case_plan),
      'all_solutions_hard_legal':all(r.get('hard_joint_excess_max_rad',1)==0 for r in case_plan),
      'independent_FK_recomputed':all(r.get('independent_point_vs_helper_max_abs_m',1)<1e-12 and r.get('independent_normal_vs_helper_max_abs',1)<1e-12 for r in case_plan),
      'forbidden_calls_zero':not any(forbidden.values()),'no_Torch_Warp_CUDA':not any(x in sys.modules for x in ('torch','warp','mujoco_warp')),
      'sources_unchanged':all(sha(p)==v for p,v in sources.items())}
    report={'status':'completed_finite_local_decomposition' if failure is None and all(checks.values()) else 'engineering_failed_partial','checks':checks,'failure':failure,
      'recipe':{'fingers':[1],'objectives':['point/.001 + .25*(normal-normal_target)'],'starts':['baseline','bounds_midpoint'],'optimized_variables':'Only that finger4hinges; other16baseline','max_nfev_per_case':120,'whole_CPU_soft_wall_seconds':60,'tolerances':1e-8,'collision_penalty':False,'bone_penalty':False,'automatic_retry':False},
      'cases':case_plan,'strict_reach_sphere_bounds':bounds,'sources':sources,'source_recipe':{'path':str(original_recipe),'sha256':sha(original_recipe)},'recipe_snapshot':{'path':str(recipepath),'sha256':sha(recipepath)},'frozen_bindings':{'path':str(bindingpath),'sha256':sha(bindingpath)},
      'preparation_note':'H123 distal-v2 material point and normal loaded frozen; no re-selection. All inputs/source/case-plan copied before optimization.',
      'wall_seconds':time.monotonic()-started,'forbidden_calls':forbidden,'physics_steps':0,'GPU_used':False,
      'scope':'Two local, bounded, deterministic thumb starts, not a global infeasibility/reachability proof. A point-only success does not imply orientation, collision/fit compatibility, contact, preload, grasp or dynamics. The conservative sphere can only exclude a fixed material target when violated.'}
    write(output/'report.json',report)
    if arrays:np.savez_compressed(output/'case_solutions.npz',**{k:np.asarray([a[k] for a in arrays]) for k in arrays[0]})
    print(json.dumps({'status':report['status'],'checks':checks,'results':[{k:r.get(k) for k in ('case_id','optimizer_success','nfev','point_error_m','normal_error_deg')} for r in case_plan],'sphere_bounds':{f:{k:v.get(k) for k in ('strict_radius_upper_m','target_distance_from_center_m','outside_strict_bound')} for f,v in bounds.items()}},indent=2));raise SystemExit(0 if report['status']=='completed_finite_local_decomposition' else 1)
if __name__=='__main__':main()
