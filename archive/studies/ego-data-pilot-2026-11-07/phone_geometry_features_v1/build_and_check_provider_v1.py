"""One bounded compiled-metadata export + three TRAIN kinematics checks.

Default is JSON/hash metadata only. --run uses one MJB load, three fresh data
objects and exactly three mj_kinematics calls. No forward, comPos, step or NN.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
P = HERE.parent

def pin(path):
    p=Path(path).resolve(strict=True); b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def checked(v):
    p=Path(v['path']).resolve(strict=True)
    if pin(p)!=v: raise ValueError('File pin mismatch: '+str(p))
    return p

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def quat_R(q):
    w,x,y,z=map(float,q)
    if abs(np.dot(q,q)-1)>1e-12:raise ValueError('Compiled unit quaternion')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def preflight(protocol):
    p=read(protocol)
    if p['schema']!='phone_geometry_features_bounded_CPU_protocol:1':raise ValueError('Protocol schema')
    for v in p['inputs'].values():checked(v)
    for v in p['sources'].values():checked(v)
    if p['sources']['runner']!=pin(__file__):raise ValueError('Current runner bytes')
    report=read(checked(p['inputs']['acquisition_report']))
    audit=read(checked(p['inputs']['dataset_audit']))
    if report['schema']!='phone_episode_acquisition_result:1' or report['status']!='COMPLETE_GROUPED_NATIVE_ACQUISITION':raise ValueError('Acquisition header')
    if audit['engineering_qualified'] is not True or audit['dataset_report']!=p['inputs']['acquisition_report'] or not all(v is True for v in audit['checks'].values()):raise ValueError('Saved parent engineering review')
    if report['sample_identities']!=p['inputs']['identities'] or report['episodes']!=p['inputs']['episodes']:raise ValueError('Original metadata join')
    if p['fixtures']!=[dict(family_id=0,dt_s=.00015625,interval_index=0),dict(family_id=8,dt_s=.0003125,interval_index=2),dict(family_id=24,dt_s=.00015625,interval_index=5)]:raise ValueError('Fixed named TRAIN checks')
    out=Path(p['output'])
    if not out.is_absolute() or out.exists():raise ValueError('Fresh absolute output')
    return p,out

def export_constants(m,mj,p,out):
    if (m.nq,m.nv,m.nu,m.na,m.neq,m.nmocap,m.nflex,m.nplugin,m.ntendon)!=(27,26,20,0,0,0,0,0,0):raise ValueError('Original simple model layout')
    phone=m.body('hot3d_phone24').id; pg=m.geom('phone24_collision_convex').id
    if pg!=71 or [m.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)]!=[15,28]:raise ValueError('Actual compiled phone/two-pad identity')
    if not np.array_equal(m.actuator_trnid[:,0],np.arange(20)) or not np.array_equal(m.actuator_gear[:,0],np.ones(20)):raise ValueError('Direct same-order20 actuators')
    if np.any(m.body_mocapid!=-1):raise ValueError('No hidden mocap body poses')
    binding=read(checked(p['inputs']['binding']))
    bodies=[]; qindices=[]; axes=[]; anchors=[]
    for b in range(m.nbody):
        bodies.append(m.body(b).name)
        nj=int(m.body_jntnum[b]);j=int(m.body_jntadr[b]); qi=-1; ax=np.zeros(3);an=np.zeros(3)
        if b==phone:
            if nj!=1 or int(m.jnt_type[j])!=int(mj.mjtJoint.mjJNT_FREE) or m.jnt_qposadr[j]!=20:raise ValueError('Phone sole free joint')
        elif nj:
            if nj!=1 or int(m.jnt_type[j])!=int(mj.mjtJoint.mjJNT_HINGE) or not 0<=m.jnt_qposadr[j]<20:raise ValueError('One actual hinge per hand body')
            qi=int(m.jnt_qposadr[j]);ax=m.jnt_axis[j].copy();an=m.jnt_pos[j].copy()
            if abs(np.linalg.norm(ax)-1)>1e-12:raise ValueError('Compiled unit hinge axis')
        qindices.append(qi);axes.append(ax.tolist());anchors.append(an.tolist())
    if sorted(q for q in qindices if q>=0)!=list(range(20)):raise ValueError('Every actual hand hinge once')
    pairs=[]
    if m.npair!=22:raise ValueError('Exactly22 explicit boundary pairs')
    byname={r['name']:r for r in binding['explicit_pairs']}
    for i in range(m.npair):
        name=m.pair(i).name;g1=int(m.pair_geom1[i]);g2=int(m.pair_geom2[i])
        if pg not in (g1,g2) or g1==g2:raise ValueError('Every pair crosses phone boundary')
        expected=byname[name]
        if [m.geom(g1).name,m.geom(g2).name]!=[expected['geom1'],expected['geom2']]:raise ValueError('Binding/native name-order join')
        other=g2 if g1==pg else g1
        b=int(m.geom_bodyid[other]);anc=np.zeros(20,dtype=bool)
        while b:
            if qindices[b]>=0:anc[qindices[b]]=True
            b=int(m.body_parentid[b])
        pairs.append(dict(pair_id=i,name=name,geom1_id=g1,geom2_id=g2,phone_geom_id=pg,other_geom_id=other,
            other_geom_name=m.geom(other).name,designated_pad=other in (15,28),
            hand_hinge_ancestors20=anc.tolist(),not_a_full_H_causal_zero_certificate=True))
    arrays={};geoms=[]
    for g in sorted({pg}|{r['other_geom_id'] for r in pairs}):
        gt=int(m.geom_type[g]);kind=mj.mjtGeom(gt).name.replace('mjGEOM_','').lower()
        row=dict(geom_id=g,name=m.geom(g).name,body_id=int(m.geom_bodyid[g]),kind=kind,
            size_m=m.geom_size[g].tolist(),geom_local_position_body_m=m.geom_pos[g].tolist(),
            geom_local_quaternion_body_wxyz=m.geom_quat[g].tolist(),
            contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g]),
            actual_native_type=gt,collision_semantics='compiled geometric binding only; functional pad/compliance not certified')
        if gt==int(mj.mjtGeom.mjGEOM_MESH):
            mid=int(m.geom_dataid[g]);v=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);f=int(m.mesh_faceadr[mid]);nf=int(m.mesh_facenum[mid])
            vk=f'mesh_{mid}_vertices_geom_m';fk=f'mesh_{mid}_source_triangles'
            arrays[vk]=m.mesh_vert[v:v+nv].copy();arrays[fk]=m.mesh_face[f:f+nf].copy()
            if arrays[fk].size and (arrays[fk].min()<0 or arrays[fk].max()>=nv):raise ValueError('Compiled mesh face local-index convention')
            row.update(mesh_id=mid,vertices_key=vk,faces_key=fk,vertex_count=nv,triangle_count=nf,
                compiled_mesh_pos_diagnostic_only=m.mesh_pos[mid].tolist(),compiled_mesh_quat_diagnostic_only=m.mesh_quat[mid].tolist(),
                vertices_frame='compiled geom frame; apply geom world pose once, no second mesh_pos/quat',
                surface_scope='raw compiled vertices/source triangles; convex support hull collision, no exact SDF claim')
        elif gt==int(mj.mjtGeom.mjGEOM_CAPSULE):
            row.update(centerline_local_m=[[0,0,-float(m.geom_size[g,1])],[0,0,float(m.geom_size[g,1])]],radius_m=float(m.geom_size[g,0]))
        elif gt not in (int(mj.mjtGeom.mjGEOM_BOX),int(mj.mjtGeom.mjGEOM_PLANE)):
            raise ValueError('Unimplemented binding type: '+kind)
        geoms.append(row)
    path=out/'compiled_geometry_arrays.npz';np.savez_compressed(path,**arrays)
    meta=dict(schema='phone_compiled_geometry_constants:1',model=p['inputs']['model_mjb'],binding=p['inputs']['binding'],
        extractor=p['sources']['runner'],provider=p['sources']['provider'],native_version=mj.__version__,
        phone_body_id=phone,phone_geom_id=pg,designated_pad_geom_ids=[15,28],
        body_names=bodies,body_parentid=m.body_parentid.tolist(),body_joint_qindex=qindices,
        body_pos=m.body_pos.tolist(),body_R=np.stack([quat_R(q) for q in m.body_quat]).tolist(),
        body_joint_axis=axes,body_joint_anchor=anchors,joint_qpos0=m.qpos0[:20].tolist(),
        joint_names=[m.joint(i).name for i in range(20)],joint_range_rad=m.jnt_range[:20].tolist(),
        actuator_names=[m.actuator(i).name for i in range(20)],actuator_ctrl_range_rad=m.actuator_ctrlrange.tolist(),
        geom_names=[m.geom(i).name for i in range(m.ngeom)],geom_bodyid=m.geom_bodyid.tolist(),
        geom_pos=m.geom_pos.tolist(),geom_R=np.stack([quat_R(q) for q in m.geom_quat]).tolist(),
        phone_body_ipos=m.body_ipos[phone].tolist(),phone_body_iquat=m.body_iquat[phone].tolist(),
        phone_body_R_is_not_inertial_R=True,pairs=pairs,geometry_bindings=geoms,geometry_arrays=pin(path),
        source_frames='Native body/geom transforms, fixed wrist unchanged; pair origin phone COM, axes phone BODY',
        learned_parameters=0,normal_or_distance_queries=0)
    write(out/'compiled_constants.json',meta)
    return meta

def run(protocol):
    p,out=preflight(protocol)
    if p['CPU_run_authorized'] is not True:raise ValueError('Bounded CPU check not authorized')
    out.mkdir();write(out/'protocol.json',p)
    counts=dict(MJB_loads=0,MjData=0,mj_kinematics=0,mj_forward=0,mj_comPos=0,mj_step=0,
        raw_state_NPZ_reads=0,Torch_FK_calls=0,pure_tensor_q_AD_calls=0,NN_constructors=0,NN_forward=0,NN_AD=0,optimizer_updates=0,GPU_calls=0)
    began=time.perf_counter()
    try:
        import mujoco as mj
        import torch
        torch.set_num_threads(4)
        if mj.__version__!='3.11.0' or torch.cuda.is_initialized():raise ValueError('Pinned CPU-only native/torch scope')
        rng0=torch.get_rng_state().clone()
        m=mj.MjModel.from_binary_path(str(checked(p['inputs']['model_mjb'])));counts['MJB_loads']+=1
        meta=export_constants(m,mj,p,out)
        mod=loadmod('phone_actual_geometry_provider',checked(p['sources']['provider']))
        provider=mod.PhoneGeometryProvider(out/'compiled_constants.json')
        ids=read(checked(p['inputs']['identities']));eps=read(checked(p['inputs']['episodes']))
        codec=loadmod('phone_geom_source_codec',checked(p['sources']['codec']))
        selected=[];xs=[];us=[];hs=[];ns=[];natives=[];evidence={}
        for fixture in p['fixtures']:
            row=[r for r in ids if r['family_id']==fixture['family_id'] and r['dt_s']==fixture['dt_s'] and r['interval_index']==fixture['interval_index']]
            if len(row)!=1 or row[0]['split']!='train' or row[0]['training_eligible'] is not True:raise ValueError('Named actual TRAIN source')
            row=row[0]; ep=eps[row['episode_id']]
            with np.load(checked(row['trace']),allow_pickle=False) as z:
                x=z['source_state58'].copy();u=z['actual_control20'].copy()
            counts['raw_state_NPZ_reads']+=1
            if fixture['interval_index']==0:
                ref=ep['initial']
                with np.load(checked(ref),allow_pickle=False) as z:q=z['qpos27'].copy();qd=z['qvel26'].copy()
            else:
                prev=[r for r in ids if r['episode_id']==row['episode_id'] and r['interval_index']==row['interval_index']-1]
                if len(prev)!=1 or prev[0]['complete_H20ms'] is not True:raise ValueError('Actual previous POST exists')
                ref=prev[0]['trace']
                with np.load(checked(ref),allow_pickle=False) as z:q=z['post_qpos27'][-1].copy();qd=z['post_qvel26'][-1].copy()
            counts['raw_state_NPZ_reads']+=1
            reconstructed=codec.actual_state(q,qd,np.array(meta['phone_body_ipos']))
            if not np.array_equal(x,reconstructed):raise ValueError('Raw current qpos/qvel -> original state58 exact')
            d=mj.MjData(m);counts['MjData']+=1;d.qpos[:]=q;d.qvel[:]=qd
            mj.mj_kinematics(m,d);counts['mj_kinematics']+=1
            natives.append(dict(bp=d.xpos.copy(),bR=d.xmat.reshape(-1,3,3).copy(),gp=d.geom_xpos.copy(),gR=d.geom_xmat.reshape(-1,3,3).copy(),xaxis=d.xaxis.copy(),xanchor=d.xanchor.copy()))
            selected.append(dict(**fixture,sample_id=row['sample_id'],episode_id=row['episode_id'],trace=row['trace'],qpos_source=ref,
                current_state_phase='source before current action; initial or previous raw POST, no target state',split='train'))
            xs.append(x);us.append(u);hs.append(row['dt_s']);ns.append(row['requested_substeps'])
        x=torch.tensor(np.stack(xs),dtype=torch.float64);u=torch.tensor(np.stack(us),dtype=torch.float64)
        h=torch.tensor(hs,dtype=torch.float64);n=torch.tensor(ns,dtype=torch.int64)
        features=provider.features(x,u,h,n);counts['Torch_FK_calls']+=1
        mod.validate_features(features)
        poses=provider.poses(x);counts['Torch_FK_calls']+=1
        errors={}
        for output,key in [('body_position_world_m','bp'),('body_rotation_world','bR'),('geom_position_world_m','gp'),('geom_rotation_world','gR')]:
            expected=np.stack([v[key] for v in natives]);actual=poses[output].detach().numpy()
            errors[output]=float(np.max(abs(actual-expected)))
            evidence['native_'+key]=expected;evidence['torch_'+key]=actual
        f32poses=provider.poses(x.float());counts['Torch_FK_calls']+=1
        f32errors={k:float(np.max(abs(f32poses[k].numpy().astype('float64')-np.stack([v[key] for v in natives])))) for k,key in [('body_position_world_m','bp'),('geom_position_world_m','gp'),('geom_rotation_world','gR')]}
        # The derivative is geometric q -> relative sensor-center position,
        # not absolute-u -> dynamics and not a physical action Jacobian.
        q=x[0,:20].clone().requires_grad_(True)
        def f(qq):
            xx=torch.cat((qq,x[0,20:]))
            return provider.poses(xx)['pair_relative_translation_BODY_COM_m'][provider.pair_geom_ids.index(15)]
        jac=torch.autograd.functional.jacobian(f,q);counts['Torch_FK_calls']+=1;counts['pure_tensor_q_AD_calls']+=1
        target=natives[0]['gp'][15];analytical=np.zeros((3,20));body=int(m.geom_bodyid[15])
        while body:
            ji=meta['body_joint_qindex'][body]
            if ji>=0:analytical[:,ji]=np.cross(natives[0]['xaxis'][ji],target-natives[0]['xanchor'][ji])
            body=int(m.body_parentid[body])
        Rp=xs[0][43:52].reshape(3,3);analytical=Rp.T@analytical
        direction=torch.arange(1,21,dtype=torch.float64);direction/=torch.linalg.vector_norm(direction)
        fd=(f(q.detach()+1e-6*direction)-f(q.detach()-1e-6*direction))/(2e-6);counts['Torch_FK_calls']+=2
        jacerr=float(np.max(abs(jac.detach().numpy()-analytical)))
        fderr=float(torch.max(abs(jac@direction-fd)))
        checks=dict(raw_current_state_reconstruction=True,
            all_body_and_geom_F64_poses_match_native=max(errors.values())<=p['tolerances']['F64_pose_abs'],
            pose_and_geometry_known_but_gap_and_normal_unknown=True,
            input_q_Jacobian_matches_native_axis_anchor=jacerr<=p['tolerances']['q_Jacobian_abs'],
            nonzero_directional_FD_matches_Torch_AD=fderr<=p['tolerances']['directional_FD_abs'],
            Torch_rng_raw_bytes_unchanged=torch.equal(rng0,torch.get_rng_state()),
            CPU_no_NN_no_forward_no_steps=not torch.cuda.is_initialized() and counts['MJB_loads']==1 and counts['mj_kinematics']==3 and counts['mj_forward']==counts['mj_step']==0)
        evidence.update(source_state58_F64=np.stack(xs),actual_u20_F64=np.stack(us),pair_translation_BODY_COM_m=features['pair_relative_translation_BODY_COM_m'].numpy(),
            pair_rotation_BODY9=features['pair_relative_rotation_BODY9'].numpy(),q_Jacobian_Torch=jac.detach().numpy(),q_Jacobian_native_axis_anchor=analytical,
            directional_FD=fd.detach().numpy(),directional_AD=(jac@direction).detach().numpy(),Torch_rng_before=rng0.numpy(),Torch_rng_after=torch.get_rng_state().numpy())
        np.savez_compressed(out/'parity_and_geometric_derivative_arrays.npz',**evidence)
        write(out/'selected_TRAIN_current_states.json',selected)
        report=dict(schema='phone_geometry_features_bounded_CPU_result:1',status='PASS_CURRENT_GEOMETRY_FK_NOT_DISTANCE_OR_DYNAMICS_QUALIFICATION' if all(checks.values()) else 'FAIL_PRESERVED_NO_RETRY',
            protocol=pin(protocol),sources=p['sources'],inputs=p['inputs'],checks=checks,passed=sum(checks.values()),total=len(checks),counts=counts,
            F64_max_absolute_pose_errors=errors,F32_descriptor_max_absolute_pose_errors=f32errors,
            q_Jacobian_max_absolute_error=jacerr,directional_FD_max_absolute_error=fderr,
            geometry_counts={k:sum(v['kind']==k for v in meta['geometry_bindings']) for k in ['mesh','capsule','box','plane']},
            geometry_frame_primary_source='https://raw.githubusercontent.com/google-deepmind/mujoco/3.11.0/src/engine/engine_collision_convex.c#L659-L702',
            elapsed_s=time.perf_counter()-began,outputs={q.name:pin(q) for q in out.iterdir() if q.is_file()},
            scope='Actual compiled geometry/FK at3 named TRAIN current states; geometric q derivatives only. No gaps/SDF/contact normal/cause certificate/NN/physical step/Actor migration.')
        write(out/'report.json',report)
        print(json.dumps(dict(status=report['status'],passed=report['passed'],total=report['total'],counts=counts,report=str(out/'report.json'))))
        if not all(checks.values()):return 1
        return 0
    except Exception as e:
        write(out/'failure.json',dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(e),traceback=traceback.format_exc(),counts=counts,protocol=pin(protocol)))
        print(json.dumps(dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error=repr(e),counts=counts)));return 1

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:
        p,o=preflight(a.protocol);print(json.dumps(dict(status='METADATA_ONLY_NO_TORCH_NO_NATIVE',fixtures=p['fixtures'],output=str(o))));return 0
    return run(a.protocol)

if __name__=='__main__':raise SystemExit(main())
