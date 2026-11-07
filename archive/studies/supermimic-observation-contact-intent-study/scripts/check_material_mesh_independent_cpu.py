"""Actual enabled MESH material-point branch only; CPU FK/Jacobians, no physics."""
from pathlib import Path
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import re
import sys
import time
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('D:/workspace/supermimic-observation-retarget-study')
sys.path.insert(0,str(OLD/'scripts'));sys.path.insert(0,str(ROOT/'scripts'))
from native_filtered_contact_geometry import NativeFilteredContactGeometry
from material_anchors import MaterialAnchorGeometry,MaterialAnchor

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q);w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def main():
    started=time.perf_counter();output=ROOT/'outputs/material_mesh_independent_v1'
    if output.exists():raise FileExistsError('No overwrite of independent evidence')
    sources=[ROOT/'scripts/material_anchors.py',OLD/'scripts/native_filtered_contact_geometry.py',OLD/'scripts/retarget_palmroot_cpu.py']
    before={str(p):sha(p) for p in sources}
    model=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb')
    context=Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz')
    candidate=OLD/'outputs/native_contact_refined_geometry_v1/geometric_retarget.npz'
    forbidden={k:0 for k in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_geomDistance','mj_contactForce')}
    saved={k:getattr(mujoco,k) for k in forbidden}
    def deny(name):
        def fail(*a,**kw):forbidden[name]+=1;raise AssertionError('Forbidden physical/distance API: '+name)
        return fail
    try:
        for k in forbidden:setattr(mujoco,k,deny(k))
        native=NativeFilteredContactGeometry(model,context);g=MaterialAnchorGeometry(native);m,d=g.model,g.data
        with np.load(candidate,allow_pickle=False) as z:q=z['joint_pos_rad'][0].copy();pose=z['mapped_object_pose_w'][0].copy()
        native.set_object_pose(pose);g.robot.forward(q)
        anchors=[];records=[]
        for geom in native.hand_ids:
            geom=int(geom);name=m.geom(geom).name
            finger=re.fullmatch(r'robot/right_finger([1-5])_(link[1-4]|tip_sensor)_col',name)
            if not finger or int(m.geom_type[geom])!=int(mujoco.mjtGeom.mjGEOM_MESH):continue
            center=d.geom_xpos[geom].copy();rotation=d.geom_xmat[geom].reshape(3,3).copy()
            direction=pose[:3]-center;direction/=np.linalg.norm(direction);local_direction=rotation.T@direction
            point,construction,index=g._surface_point(geom,np.zeros(3),local_direction)
            mesh=int(m.geom_dataid[geom]);start,count=int(m.mesh_vertadr[mesh]),int(m.mesh_vertnum[mesh]);vertices=np.asarray(m.mesh_vert[start:start+count],float)
            # Independent world/body construction checks the compiled geometry
            # frame, never applies source-mesh scale/refpose a second time.
            body=int(m.geom_bodyid[geom]);br=matrix(d.xquat[body]);gr=matrix(m.geom_quat[geom]);bp=d.xpos[body].copy()
            world_vertices=(bp[:,None]+br@(m.geom_pos[geom,None].T+gr@vertices.T)).T
            support=world_vertices@direction;selected=bp+br@(m.geom_pos[geom]+gr@point)
            anchor=MaterialAnchor(int(finger[1]),geom,name,body,point.copy(),local_direction.copy(),construction,index,0.)
            anchors.append(anchor)
            records.append({'geom':name,'geom_id':geom,'mesh_id':mesh,'compiled_vertex_count':count,'vertex_index':int(index),
              'construction':construction,'support_plane_max_error_m':float(abs(selected@direction-support.max())),
              'selected_vertex_local_exact':bool(np.array_equal(point,vertices[index])),
              'compiled_frame_max_abs_error_m':float(np.max(np.abs(world_vertices-(center[:,None]+rotation@vertices.T).T))),
              'normal_in_support_cone':bool(np.max((vertices-point)@local_direction)<=1e-12),
              'vertex_radius_max_m':float(np.linalg.norm(vertices,axis=-1).max()),'geom_rbound_m':float(m.geom_rbound[geom]),
              'support_tie_vertices_1e_12':int(np.count_nonzero(np.abs(vertices@local_direction-np.max(vertices@local_direction))<=1e-12))})
        if not anchors:raise AssertionError('No actual enabled finger meshes covered')
        def independently_reconstruct(qvalue):
            g.robot.forward(qvalue);points=[];normals=[]
            for a in anchors:
                br=matrix(d.xquat[a.body_id]);gr=matrix(m.geom_quat[a.geom_id]);bp=d.xpos[a.body_id]
                points.append(bp+br@(m.geom_pos[a.geom_id]+gr@a.point_geom_local))
                normals.append(br@gr@a.support_normal_geom_local)
            return np.asarray(points),np.asarray(normals)
        points,normals=independently_reconstruct(q)
        jp=[];jn=[];feature_errors=[]
        for i,a in enumerate(anchors):
            hp,hj,hn,hjn=g.oriented_features(q,a);jp.append(hj);jn.append(hjn)
            feature_errors.append({'geom':a.geom_name,'point_m':float(np.max(np.abs(hp-points[i]))),'normal':float(np.max(np.abs(hn-normals[i])))})
        fd_rows=[]
        for eps in (1e-6,1e-5):
            fdp=np.empty((len(anchors),3,20));fdn=np.empty_like(fdp)
            for j in range(20):
                plus=q.copy();minus=q.copy();plus[j]+=eps;minus[j]-=eps
                pp,pn=independently_reconstruct(plus);mp,mn=independently_reconstruct(minus)
                fdp[:,:,j]=(pp-mp)/(2*eps);fdn[:,:,j]=(pn-mn)/(2*eps)
            fd_rows.append({'epsilon_rad':eps,'point_max_abs_error_m_per_rad':float(np.max(np.abs(np.asarray(jp)-fdp))),
                           'normal_max_abs_error_per_rad':float(np.max(np.abs(np.asarray(jn)-fdn)))})
        checks={'actual_all_enabled_finger_meshes_covered':len(records)==sum(bool(re.fullmatch(r'robot/right_finger[1-5]_(link[1-4]|tip_sensor)_col',m.geom(int(i)).name)) and int(m.geom_type[int(i)])==int(mujoco.mjtGeom.mjGEOM_MESH) for i in native.hand_ids),
          'actual_mesh_branch_not_capsule':all(r['construction']=='compiled_mesh_extreme_support_vertex' for r in records),
          'all_exact_compiled_vertices':all(r['selected_vertex_local_exact'] for r in records),
          'all_world_extreme_support_planes':all(r['support_plane_max_error_m']<=1e-10 and r['normal_in_support_cone'] for r in records),
          'all_compiled_body_geom_frame_transforms':all(r['compiled_frame_max_abs_error_m']<=1e-12 for r in records),
          'all_vertices_within_native_rbound':all(r['vertex_radius_max_m']<=r['geom_rbound_m']+1e-6 for r in records),
          'features_match_independent_body_transform':all(r['point_m']<=1e-12 and r['normal']<=1e-12 for r in feature_errors),
          'fixed_point_Jacobian_FD':all(r['point_max_abs_error_m_per_rad']<1e-7 for r in fd_rows),
          'fixed_normal_rotation_Jacobian_FD':all(r['normal_max_abs_error_per_rad']<1e-7 for r in fd_rows),
          'no_forbidden_calls':not any(forbidden.values()),'no_CUDA_Warp_Torch':not any(k in sys.modules for k in ('torch','warp','mujoco_warp')),
          'inputs_sources_unchanged':all(sha(p)==v for p,v in before.items())}
    finally:
        for k,v in saved.items():setattr(mujoco,k,v)
    report={'status':'passed_actual_mesh_local_material_checks' if all(checks.values()) else 'failed','checks':checks,
      'actual_mesh_anchor_count':len(records),'mesh_rows':records,'feature_errors':feature_errors,'finite_differences':fd_rows,
      'sources':before,'inputs':{str(p):sha(p) for p in (model,context,candidate)},'forbidden_calls':forbidden,
      'wall_seconds':time.perf_counter()-started,'GPU_used':False,'dynamics_steps':0,
      'scope':'Actual enabled mesh branch at the fixed baseline q. Anchors are deliberately fixed to each real mesh to cover the branch; this does not claim the observation-based select() chose those meshes, that ties are globally differentiable, or that a native contact/grasp is present.'}
    output.mkdir(parents=True,exist_ok=False);(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':checks,'actual_mesh_anchor_count':len(records),'finite_differences':fd_rows},indent=2));raise SystemExit(0 if all(checks.values()) else 1)
if __name__=='__main__':main()
