"""Mapped-pose designed inertial wrench and native object-only cone screens.

CPU only. No force/velocity observations, physics rollout, optimizer, GPU,
warmseed trajectory splice, parameter tuning or inferred contact labels.
"""
from pathlib import Path
import argparse,contextlib,csv,hashlib,importlib.util,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import mujoco
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def rotations(q):return Rotation.from_quat(np.asarray(q)[..., [1,2,3,0]]).as_matrix()

def canonical_quaternions(q):
    q=np.array(q,dtype=float,copy=True)
    error=np.max(np.abs(np.linalg.norm(q,axis=-1)-1))
    if not np.isfinite(q).all() or error>1e-6:raise ValueError('Invalid wxyz quaternion')
    q/=np.linalg.norm(q,axis=-1,keepdims=True);flips=[]
    for i in range(1,len(q)):
        if q[i]@q[i-1]<0:q[i]*=-1;flips.append(i)
    return q,float(error),flips

def derivative1(x,dt):return np.gradient(x,dt,axis=0,edge_order=2)

def derivative2(x,dt):
    x=np.asarray(x);a=np.empty_like(x)
    a[1:-1]=(x[2:]-2*x[1:-1]+x[:-2])/dt**2
    a[0]=(2*x[0]-5*x[1]+4*x[2]-x[3])/dt**2
    a[-1]=(2*x[-1]-5*x[-2]+4*x[-3]-x[-4])/dt**2
    return a

def world_angular_velocity(q,dt):
    q,_,_=canonical_quaternions(q);R=rotations(q)
    intervals=[]
    for i in range(len(q)-1):
        local=np.zeros(3);mujoco.mju_subQuat(local,q[i+1],q[i])
        intervals.append(R[i]@local/dt)
    intervals=np.array(intervals);w=np.empty((len(q),3))
    for i in range(1,len(q)-1):
        local=np.zeros(3);mujoco.mju_subQuat(local,q[i+1],q[i-1])
        w[i]=R[i-1]@local/(2*dt)
    w[0]=1.5*intervals[0]-.5*intervals[1]
    w[-1]=1.5*intervals[-1]-.5*intervals[-2]
    return w,intervals

def demand(pose,t,model,ctx,obj):
    q,norm_error,flips=canonical_quaternions(pose[:,3:]);R=rotations(q);dt=.020
    com=pose[:,:3]+np.einsum('tij,j->ti',R,model.body_ipos[obj])
    v=derivative1(com,dt);a=derivative2(com,dt);w,interval_w=world_angular_velocity(q,dt);alpha=derivative1(w,dt)
    vi=savgol_filter(com,9,3,deriv=1,delta=dt,axis=0,mode='interp')
    ai=savgol_filter(com,9,3,deriv=2,delta=dt,axis=0,mode='interp')
    wi=savgol_filter(w,9,3,deriv=0,delta=dt,axis=0,mode='interp')
    alphai=savgol_filter(w,9,3,deriv=1,delta=dt,axis=0,mode='interp')
    Ri=rotations(np.asarray(ctx['body_iquat'][0,obj])[None])[0]
    inertial_R=R@Ri;inertia=np.asarray(ctx['body_inertia'][0,obj],dtype=float)
    Iw=np.einsum('tij,j,tkj->tik',inertial_R,inertia,inertial_R)
    mass=float(ctx['body_mass'][0,obj]);gravity=np.asarray(model.opt.gravity,dtype=float)
    def wrench(accel,omega,angular_accel):
        return np.c_[mass*(accel-gravity),np.einsum('tij,tj->ti',Iw,angular_accel)+np.cross(omega,np.einsum('tij,tj->ti',Iw,omega))]
    return {'time_s':t,'object_pose_mapped_w':pose,'quaternion_continuous_wxyz':q,'body_rotation_w':R,'COM_world_m':com,
            'raw_v_world_m_s':v,'raw_a_world_m_s2':a,'raw_omega_world_rad_s':w,'raw_alpha_world_rad_s2':alpha,'interval_omega_world_rad_s':interval_w,
            'sensitivity_v_world_m_s':vi,'sensitivity_a_world_m_s2':ai,'sensitivity_omega_world_rad_s':wi,'sensitivity_alpha_world_rad_s2':alphai,
            'I_world_kg_m2':Iw,'raw_required_wrench_world_N_Nm':wrench(a,w,alpha),
            'sensitivity_required_wrench_world_N_Nm':wrench(ai,wi,alphai),
            'static_required_wrench_world_N_Nm':np.broadcast_to(np.r_[-mass*gravity,[0.,0.,0.]],(len(t),6)).copy()}, {'quaternion_norm_error_before_numerical_normalization':norm_error,'sign_flipped_frames':flips,'mass_kg':mass,'principal_inertia_kg_m2':inertia.tolist(),'body_iquat_wxyz':np.asarray(ctx['body_iquat'][0,obj]).tolist(),'body_ipos_m':model.body_ipos[obj].tolist(),'gravity_model_world_m_s2':gravity.tolist()}

def api_contract(model):
    q=np.array([.7,.2,-.4,.3]);q/=np.linalg.norm(q);R=rotations(q[None])[0]
    vb=np.array([.3,-.2,.4]);vw=np.array([.1,.2,-.3]);dt=1e-4;nextq=q.copy()
    mujoco.mju_quatIntegrate(nextq,vb,dt);local=np.zeros(3);mujoco.mju_subQuat(local,nextq,q)
    independent_world=Rotation.from_matrix(rotations(nextq[None])[0]@R.T).as_rotvec()/dt
    pos1=model.qpos0.copy();j=model.joint('object/cube_freejoint');qa=int(j.qposadr[0]);va=int(j.dofadr[0]);pos1[qa+3:qa+7]=q
    vel=np.zeros(model.nv);vel[va:va+3]=vw;vel[va+3:va+6]=vb;pos2=pos1.copy();mujoco.mj_integratePos(model,pos2,vel,dt)
    recovered=np.zeros(model.nv);mujoco.mj_differentiatePos(model,recovered,dt,pos1,pos2)
    t=np.arange(151)*.02;polynomial=np.c_[t*t,3*t+1,np.zeros(len(t))]
    ws=np.array([.2,-.1,.5]);Rs=Rotation.from_rotvec(t[:,None]*ws).as_matrix()@R
    qs=Rotation.from_matrix(Rs).as_quat()[:,[3,0,1,2]];qs[::3]*=-1
    estimated,_=world_angular_velocity(qs,.02)
    checks={'installed_quatIntegrate_right_body_velocity':float(np.max(np.abs(local/dt-vb)))<1e-10,
            'world_omega_equals_Rbody_times_native_omega':float(np.max(np.abs(independent_world-R@vb)))<1e-10,
            'free_qvel_translation_is_world':float(np.max(np.abs((pos2[qa:qa+3]-pos1[qa:qa+3])/dt-vw)))<1e-10,
            'free_qvel_angular_is_body_local':float(np.max(np.abs(pos2[qa+3:qa+7]-nextq)))<1e-12,
            'differentiatePos_roundtrip_native_qvel':float(np.max(np.abs(recovered-vel)))<1e-10,
            'FD_polynomial_velocity_all_endpoints':float(np.max(np.abs(derivative1(polynomial,.02)-np.c_[2*t,3*np.ones(len(t)),np.zeros(len(t))])))<1e-10,
            'FD_polynomial_acceleration_all_endpoints':float(np.max(np.abs(derivative2(polynomial,.02)-[2,0,0])))<1e-9,
            'SO3_world_constant_omega_and_sign_invariance':float(np.max(np.abs(estimated-ws)))<1e-10,
            'SO3_constant_world_omega_alpha_zero':float(np.max(np.abs(derivative1(estimated,.02))))<1e-9}
    header=Path(mujoco.__file__).parent/'include/mujoco/mujoco/mujoco.h'
    return {'checks':checks,'Mujoco_version':mujoco.__version__,'installed_header':{'path':str(header),'sha256':sha(header)},
            'installed_API_docstrings':{n:getattr(mujoco,n).__doc__ for n in ('mju_subQuat','mju_quatIntegrate','mj_differentiatePos','mj_integratePos')},
            'errors':{'subQuat_body_error':float(np.max(np.abs(local/dt-vb))),'world_conversion_error':float(np.max(np.abs(independent_world-R@vb))),'integrate_differentiate_qvel_error':float(np.max(np.abs(recovered-vel)))},
            'source_URLs':['https://raw.githubusercontent.com/google-deepmind/mujoco/3.11.0/src/engine/engine_util_spatial.c','https://raw.githubusercontent.com/google-deepmind/mujoco/3.11.0/src/engine/engine_support.c'],
            'scope':'Math-only API identity tests: integratePos is kinematic coordinate integration, not mj_step or physical simulation. Free translational qvel is world; rotational qvel is body-local. Wrench computations use explicit world omega=R*body omega.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,default=ROOT/'protocols/dynamic-wrench-v1.json');ap.add_argument('--output',type=Path,default=ROOT/'outputs/dynamic_wrench_v1');args=ap.parse_args()
    start=time.perf_counter();p=read(args.protocol);pins=p['inputs'];out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name,pin in pins.items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Input/source drift: '+name)
    if pins['candidate']['sha256']!='539d6247576b9b2e3fb1b143cf17b6ad507e3baea73c5fafcda4e221d9230c8f':raise ValueError('Not original151frame539d')
    with np.load(pins['candidate']['path'],allow_pickle=False) as z:
        candidate={k:z[k].copy() for k in p['candidate_fields']}
    with np.load(pins['observations']['path'],allow_pickle=False) as z:
        observations={k:z[k].copy() for k in p['observation_read_allowlist']}
    t=candidate['time_s'];pose=candidate['mapped_object_pose_w'];q=candidate['joint_pos_rad']
    if t.shape!=(151,) or pose.shape!=(151,7) or q.shape!=(151,20) or np.max(np.abs(t-np.arange(151)*.02))>1e-12:raise ValueError('Canonical151x20ms contract differs')
    if not all(np.isfinite(v).all() for v in list(candidate.values())+list(observations.values())):raise ValueError('Nonfinite input')
    Rmap=candidate['source_to_fixed_wrist_R'];translation=candidate['source_to_fixed_wrist_translation_m']
    checks={'same_observation_time':np.array_equal(t,observations['time_s']),
            'mapped_position_reconstructs_source_Rt':np.max(np.abs(np.einsum('tij,tj->ti',Rmap,observations['object_pose_source_w'][:,:3])+translation-pose[:,:3]))<1e-12,
            'mapped_rotation_reconstructs_source_Rt':np.max(np.abs(Rmap@rotations(observations['object_pose_source_w'][:,3:])-rotations(pose[:,3:])))<1e-12}
    model=mujoco.MjModel.from_binary_path(pins['model']['path']);obj=model.body('object/cube').id
    with np.load(pins['context']['path'],allow_pickle=False) as z:
        ctx={k:z[k].copy() for k in ('body_mass','body_inertia','body_iquat','geom_friction')}
    if not all(np.isfinite(x).all() and np.array_equal(x,np.broadcast_to(x[:1],x.shape)) for x in ctx.values()):raise ValueError('Context not shared finite')
    arrays,physical=demand(pose,t,model,ctx,obj);contract=api_contract(model);checks.update(contract['checks']);write(out/'installed_API_contract.json',contract)
    def quantity_summary(a):return {'frame0':a[0].tolist(),'norm_frame0':float(np.linalg.norm(a[0])),'norm_max':float(np.linalg.norm(a,axis=-1).max()),'norm_median':float(np.median(np.linalg.norm(a,axis=-1)))}
    initial={k:quantity_summary(arrays[k]) for k in ('raw_v_world_m_s','raw_a_world_m_s2','raw_omega_world_rad_s','raw_alpha_world_rad_s2')}
    for name in ('raw','sensitivity'):
        initial[name+'_force_N']=quantity_summary(arrays[name+'_required_wrench_world_N_Nm'][:,:3]);initial[name+'_torque_Nm']=quantity_summary(arrays[name+'_required_wrench_world_N_Nm'][:,3:])
    write(out/'initial_kinematics_and_demand.json',{'status':'DESIGNED_MODEL_DEMAND_ONLY_NOT_OBSERVED','physical':physical,'quantities':initial,'confidence':None,'protocol_sha256':sha(args.protocol)})
    print(json.dumps({'stage':'kinematics_complete','quantities':initial,'physical':physical},indent=2),flush=True)
    # Source-pinned existing cone implementation, with all failures retained.
    spec=importlib.util.spec_from_file_location('fixed_native_LP',pins['native_LP']['path']);lp=importlib.util.module_from_spec(spec);spec.loader.exec_module(lp)
    sys.path.insert(0,str(Path(pins['native_geometry']['path']).parent));from native_filtered_contact_geometry import NativeFilteredContactGeometry
    forbidden={name:0 for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_contactForce','mj_geomDistance')}
    original={name:getattr(mujoco,name) for name in forbidden}
    def deny(name):
        def f(*a,**kw):forbidden[name]+=1;raise AssertionError('Forbidden physics/force/gapquery '+name)
        return f
    for name in forbidden:setattr(mujoco,name,deny(name))
    conditions=('static','raw','sensitivity');targets={'static':arrays['static_required_wrench_world_N_Nm'],'raw':arrays['raw_required_wrench_world_N_Nm'],'sensitivity':arrays['sensitivity_required_wrench_world_N_Nm']}
    rows=[];raw_records=[];screen=[];matrices=[];weights={name:[] for name in conditions}
    try:
        g=NativeFilteredContactGeometry(pins['model']['path'],pins['context']['path']);g.model.geom_friction[:]=ctx['geom_friction'][0]
        if g.model.npair or g.model.flg_adhesion or np.any(g.model.geom_adhesion) or g.model.opt.cone!=mujoco.mjtCone.mjCONE_PYRAMIDAL:raise ValueError('Unsupported material/cone overrides')
        for i in range(151):
            g.set_object_pose(pose[i]);e=g.evaluate(q[i]);contacts=[];environment=0;positive=0
            for n,c in enumerate(g.data.contact):
                a,b=map(int,c.geom);isobj_a=a in g.object_ids;isobj_b=b in g.object_ids
                if not(isobj_a or isobj_b):continue
                hand=b if isobj_a else a
                if hand not in g.hand_ids:environment+=1;continue
                pa,pb=int(g.model.geom_priority[a]),int(g.model.geom_priority[b])
                if pa==pb:dim=max(int(g.model.geom_condim[a]),int(g.model.geom_condim[b]));f3=np.maximum(ctx['geom_friction'][0,a],ctx['geom_friction'][0,b])
                else:selected=a if pa>pb else b;dim=int(g.model.geom_condim[selected]);f3=ctx['geom_friction'][0,selected]
                friction=np.array([f3[0],f3[0],f3[1],f3[2],f3[2]],dtype=float)
                if dim!=c.dim or not np.array_equal(friction,c.friction):raise ValueError('Native material mix mismatch')
                frame=(1. if isobj_b else -1.)*c.frame.reshape(3,3).copy()
                if np.max(np.abs(frame@frame.T-np.eye(3)))>1e-10:raise ValueError('Native nonunit frame')
                record={'frame':i,'time_s':float(t[i]),'native_contact_index':n,'geom':[a,b],'names':[g.model.geom(a).name,g.model.geom(b).name],'distance_m':float(c.dist),'position_world_m':c.pos.tolist(),'frame_rows_force_on_object_world':frame.tolist(),'friction5':friction.tolist(),'condim':dim,'phase':'CPU_native_geometric_contact_not_solved_force'}
                raw_records.append(record);positive+=int(c.dist>0)
                contacts.append({'id':f'f{i}_c{n}','frame':frame,'point':c.pos.copy(),'friction':friction,'condim':dim})
            com=arrays['COM_world_m'][i];results={};A=None
            for name in conditions:
                if contacts:result,A=lp.solve(contacts,com,targets[name][i])
                else:
                    A=np.empty((6,0));result={'solver_status':2,'solver_message':'Empty generated hand-object contact set; no nonzero wrench in this available cone. Absent distances unknown.','feasible_in_native_object_wrench_screen':False,'native_edges':0,'normal_force_upper_cap_N':None}
                results[name]=result;weights[name].append(np.array(result.get('edge_weights_N',[])))
            matrices.append(A.copy());screen.append({'frame':i,'time_s':float(t[i]),'contacts':len(contacts),'positive_distance_potential_contacts':positive,'object_environment_contacts':environment,'conditions':results})
            ho=e.generated_mask&(g.pair_kind=='hand_object');hs=e.generated_mask&(g.pair_kind=='hand_self')
            row={'frame':i,'time_s':float(t[i]),'endpoint':i in (0,150),'native_hand_object_contacts':len(contacts),'native_hand_object_pairs':int(ho.sum()),'native_hand_self_pairs':int(hs.sum()),'object_environment_contacts':environment,'positive_distance_potential_contacts':positive,
                 'hand_object_max_depth_m':float(np.maximum(-e.signed_distance_m[ho],0).max()) if ho.any() else None,'hand_self_max_depth_m':float(np.maximum(-e.signed_distance_m[hs],0).max()) if hs.any() else None,'tied_generated_pairs':int(np.count_nonzero(e.generated_mask&(e.tie_count>1)))}
            for name in conditions:
                row[name+'_feasible']=results[name]['feasible_in_native_object_wrench_screen'];row[name+'_solver_status']=results[name]['solver_status'];row[name+'_required_force_norm_N']=float(np.linalg.norm(targets[name][i,:3]));row[name+'_required_torque_norm_Nm']=float(np.linalg.norm(targets[name][i,3:]))
            rows.append(row)
        checks['native_queries151_no_clock_or_velocity']=g.collision_query_count==151 and g.data.time==0 and not np.any(g.data.qvel)
        checks['native_absent_gap_semantics_NaN']=bool(np.isnan(e.signed_distance_m[~e.generated_mask]).all())
    finally:
        for name,method in original.items():setattr(mujoco,name,method)
    checks['no_physics_force_gapquery_calls']=not any(forbidden.values());checks['no_Torch_Warp_import']=not any(k.split('.')[0] in ('torch','warp','mujoco_warp') for k in sys.modules)
    maxedges=max(A.shape[1] for A in matrices);padded=np.full((151,6,maxedges),np.nan)
    for i,A in enumerate(matrices):padded[i,:,:A.shape[1]]=A
    arrays['native_wrench_matrix_world_padded']=padded;arrays['native_edges_per_frame']=np.array([A.shape[1] for A in matrices])
    for name in conditions:
        arr=np.full((151,maxedges),np.nan)
        for i,w in enumerate(weights[name]):arr[i,:len(w)]=w
        arrays[name+'_LP_edge_weight_candidate_N_padded']=arr
        arrays[name+'_LP_feasible']=np.array([r[name+'_feasible'] for r in rows])
    np.savez_compressed(out/'designed_wrench_151.npz',**arrays)
    with (out/'all_frames.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'all_frame_LP_results.json',screen);write(out/'native_geometric_contacts.json',raw_records)
    counts={name:{'feasible_frames':sum(r[name+'_feasible'] for r in rows),'total_frames':151,'interior_feasible_frames':sum(r[name+'_feasible'] for r in rows[1:-1]),'interior_frames':149,'frame0_feasible':rows[0][name+'_feasible'],'frame150_feasible':rows[-1][name+'_feasible']} for name in conditions}
    report={'status':'CPU_DESIGNED_WRENCH_AND_GEOMETRIC_SCREENS_COMPLETE' if all(checks.values()) else 'CPU_CONTRACT_CHECK_FAILED','checks':{k:bool(v) for k,v in checks.items()},'physical':physical,'initial_quantities':initial,'screen_counts_descriptive_not_independent_N':counts,
            'positive_distance_contact_records':sum(r['positive_distance_potential_contacts'] for r in rows),'no_contact_frames':sum(r['native_hand_object_contacts']==0 for r in rows),'inputs':pins,'protocol_sha256':sha(args.protocol),'analysis_source_sha256':sha(__file__),'observation_read_fields':p['observation_read_allowlist'],'candidate_read_fields':p['candidate_fields'],'raw_files_sha256':{n:sha(out/n) for n in ('designed_wrench_151.npz','all_frames.csv','all_frame_LP_results.json','native_geometric_contacts.json','installed_API_contract.json')},
            'confidence':None,'is_real_ego':False,'measured_force_or_velocity':False,'physical_reference_qualified':False,'force_reference_labels_created':False,'GPU_used':False,'physics_steps':0,'forbidden_API_calls':forbidden,'wall_seconds':time.perf_counter()-start,
            'limits':['Mapped poses include time-varying palm rebase: this is a fixed-palm task redesign, not the original video inertial trajectory.','Finite differences and inertia are design-model quantities; endpoints and observation noise are uncertain; source metric/camera/confidence unqualified.','Sensitivity omega/alpha filtering does not define a completely reconstructed smoothed orientation trajectory.','Object-only unbounded-normal native-pyramid wrench existence excludes robot/PD/selfcontact equilibrium, compliance/contact constitutive force, stability and continuous path execution.','Generated geometric contacts are not measured load/contact labels. Absent pair gaps remain unknown. Positive contactdist, if present, is an explicitly retained potential-contact relaxation.','Static or dynamic screen infeasibility applies to this per-frame contact cone/design demand, not a claim that the source motion or every retargeting is impossible.','All151 correlated frames and all3 fixed conditions preserved; no frame-level independent sample, retuning or best-condition selection.']}
    write(out/'report.json',report);print(json.dumps({'stage':'complete','counts':counts,'checks_passed':sum(checks.values()),'checks_total':len(checks),'wall_seconds':report['wall_seconds']},indent=2));raise SystemExit(not all(checks.values()))


if __name__=='__main__':main()
