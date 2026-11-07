"""One151-frame declared surface cone membership screen; static arrays only."""
from pathlib import Path
import argparse,csv,hashlib,json,os,time
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rotation(q):
    q=np.asarray(q,np.float64)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-6:raise ValueError('Static actual inertial quaternion invalid')
    q=q/np.linalg.norm(q);w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def rays(point,n):
    n=n/np.linalg.norm(n,axis=-1,keepdims=True);sg=np.where(n[:,2]>=0,1.,-1.)
    a=-1/(sg+n[:,2]);b=n[:,0]*n[:,1]*a
    t1=np.stack((1+sg*n[:,0]**2*a,sg*b,-sg*n[:,0]),-1);t1/=np.linalg.norm(t1,axis=-1,keepdims=True)
    t2=np.stack((b,sg+n[:,1]**2*a,-n[:,1]),-1);t2/=np.linalg.norm(t2,axis=-1,keepdims=True)
    angle=np.arange(8)*(2*np.pi/8)
    f=n[:,None,:]+.1*(np.cos(angle)[None,:,None]*t1[:,None,:]+np.sin(angle)[None,:,None]*t2[:,None,:])
    f/=np.linalg.norm(f,axis=-1,keepdims=True);f=np.concatenate((f,n[:,None,:]),axis=1)
    return f.reshape(27,3),np.cross(point[:,None,:],f).reshape(27,3)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args();p=read(a.protocol)
    for value in p['source_pins'].values():
        if pin(value['path'])['sha256']!=value['sha256']:raise ValueError('Source/input changed')
    if p['schema']!='full151_declared_H123_dynamic_cone_LP:1' or p['equality_residual_gate_N']!=1e-7 or p['solver_options']!={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}:raise ValueError('Frozen numerical contract changed')
    if not a.run:print(json.dumps({'status':'METADATA_ONLY_NO_LP','GPU_started':False}));return
    os.environ['CUDA_VISIBLE_DEVICES']='-1';out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'execution_source.py').write_bytes(Path(__file__).read_bytes());save(out/'protocol_snapshot.json',p)
    import mujoco
    from scipy.optimize import linprog
    forbidden={}
    for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce','mj_kinematics','mj_comPos'):
        def deny(*args,_name=name,**kwargs):forbidden[_name]=forbidden.get(_name,0)+1;raise AssertionError('Forbidden native call '+_name)
        setattr(mujoco,name,deny)
    start=time.perf_counter();target=load(p['targets']['path']);ref=load(p['reference']['path']);ctx=load(p['context']['path']);model=mujoco.MjModel.from_binary_path(p['model']['path']);checks={}
    def check(k,v):
        checks[k]=bool(v)
        if not checks[k]:raise ValueError('Invalid fixed input '+k)
    free=np.flatnonzero(model.jnt_type==int(mujoco.mjtJoint.mjJNT_FREE));check('unique_object_freejoint',len(free)==1);obj=int(model.jnt_bodyid[free[0]])
    check('COM0_target_and_staticmodel',np.array_equal(model.body_ipos[obj],target['object_com_offset_body_m']) and np.array_equal(target['object_com_offset_body_m'],np.zeros(3)))
    check('actual_context_shared128',all(np.array_equal(ctx[k],np.broadcast_to(ctx[k][:1],ctx[k].shape)) for k in ('body_mass','body_inertia','body_iquat')))
    mass=float(ctx['body_mass'][0,obj]);diag=ctx['body_inertia'][0,obj].astype(float);iquat=ctx['body_iquat'][0,obj].astype(float);Qi=rotation(iquat);I=Qi@np.diag(diag)@Qi.T;gravity=np.asarray(model.opt.gravity,float)
    check('actual_mass_inertia_positive',mass>0 and (diag>0).all() and np.isfinite(I).all())
    check('gravity_enabled_no_object_gravcomp',not(model.opt.disableflags&int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)) and model.body_gravcomp[obj]==0)
    va=int(model.jnt_dofadr[free[0]])
    check('object_no_damping_stiction_fluid_spring',not np.asarray(model.dof_damping[va:va+6]).any() and not np.asarray(model.dof_frictionloss[va:va+6]).any() and model.jnt_stiffness[free[0]]==0 and model.opt.density==0 and model.opt.viscosity==0)
    required=('object_rotation_w','object_a_world_m_s2','object_omega_world_rad_s','object_alpha_world_rad_s2')
    missing=[k for k in required if k not in ref]
    if missing:
        save(out/'report.json',{'status':'UNKNOWN_MISSING_ANALYTIC_INPUTS','missing':missing,'LP_calls':0,'no_zero_imputation':True});return
    R=ref['object_rotation_w'][::4];acc=ref['object_a_world_m_s2'][::4];omega=ref['object_omega_world_rad_s'][::4];alpha=ref['object_alpha_world_rad_s2'][::4]
    check('analytic151_finite_and_time_binding',R.shape==(151,3,3) and np.isfinite(R).all() and all(x.shape==(151,3) and np.isfinite(x).all() for x in (acc,omega,alpha)) and np.array_equal(ref['planned_time601_s'][::4],target['time_s']) and np.array_equal(ref['mapped_object_pose_w'],target['mapped_object_pose_w']))
    check('all151_H123_active_declared_not_observed',target['declared_intent_active'].shape==(151,3) and target['declared_intent_active'].all() and not target['human_contact_known'].any())
    radius=float(target['object_nominal_enclosing_radius_m']);check('frozen_radius_once',radius==p['radius_m'] and radius>0)
    F=np.einsum('nji,nj->ni',R,mass*(acc-gravity));wb=np.einsum('nji,nj->ni',R,omega);ab=np.einsum('nji,nj->ni',R,alpha);Iw=np.einsum('ij,nj->ni',I,wb);tau=np.einsum('ij,nj->ni',I,ab)+np.cross(wb,Iw)
    request=np.concatenate((F,tau/radius),axis=1);A=np.empty((151,6,27));lambdas=np.full((151,27),np.nan);rows=[];support_error=[];lp_calls=0
    for k in range(151):
        force,moment=rays(target['point_com_body_m'][k],target['inward_normal_body'][k]);A[k]=np.concatenate((force,moment/radius),axis=1).T
        rebuilt=np.maximum(0,(A[k].T@target['support_basis'].T).max(axis=0));support_error.append(float(np.max(np.abs(rebuilt-target['support512'][k,0,0]))))
    check('Frisvad27rays_match_saved512_fingerprint',max(support_error)<=1e-12)
    for k in range(151):
        row={'frame':k,'time_s':float(target['time_s'][k]),'status':'SOLVER_UNKNOWN','solver_status':None,'lambda_sum_N':None,'lambda_min_N':None,'residual_inf_N_equivalent':None,'force_residual_inf_N':None,'torque_residual_inf_Nm':None}
        try:
            lp_calls+=1;sol=linprog(np.ones(27),A_eq=A[k],b_eq=request[k],bounds=(0,None),method='highs',options=p['solver_options']);row.update(solver_status=int(sol.status),solver_message=str(sol.message))
            if sol.status==2:row['status']='INFEASIBLE'
            elif sol.status==0 and sol.success and sol.x is not None and np.isfinite(sol.x).all():
                lam=np.asarray(sol.x,float);res=A[k]@lam-request[k];lambdas[k]=lam
                row.update(lambda_sum_N=float(lam.sum()),lambda_min_N=float(lam.min()),residual_inf_N_equivalent=float(np.max(np.abs(res))),force_residual_inf_N=float(np.max(np.abs(res[:3]))),torque_residual_inf_Nm=float(np.max(np.abs(res[3:]*radius))))
                if lam.min()>=0 and row['residual_inf_N_equivalent']<=1e-7:row['status']='FEASIBLE'
                else:row['solver_message']+='; returned point fails frozen original-units primal gate; not clipped'
        except Exception as exc:row['error']=repr(exc)
        rows.append(row)
    check('exactly151_LP_attempts_no_retry',lp_calls==151)
    check('no_forbidden_native_calls',not forbidden)
    check('all_input_source_pins_unchanged',all(pin(v['path'])['sha256']==v['sha256'] for v in p['source_pins'].values()))
    np.savez_compressed(out/'matrices_and_witnesses.npz',A_N_equivalent=A,required_wrench_N_equivalent=request,lambda_N=lambdas,required_force_body_N=F,required_moment_body_Nm=tau,omega_body_rad_s=wb,alpha_body_rad_s2=ab,I_body_kg_m2=I,actual_mass_kg=np.array(mass),gravity_world_m_s2=gravity)
    save(out/'frame_results.json',rows)
    with (out/'frame_results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    count={s:sum(r['status']==s for r in rows) for s in ('FEASIBLE','INFEASIBLE','SOLVER_UNKNOWN')}
    report={'schema':'full151_declared_H123_dynamic_cone_LP_report:1','status':'COMPLETED_DECLARED_CONE_DIAGNOSTIC','protocol':pin(a.protocol),'source':pin(__file__),'source_pins':p['source_pins'],'checks':checks,'passed':sum(checks.values()),'total':len(checks),'frame_status_counts':count,'LP_calls':lp_calls,'object_body_id':obj,'actual_mass_kg':mass,'actual_inertia_diagonal_kg_m2':diag.tolist(),'actual_inertia_quaternion_wxyz':iquat.tolist(),'I_body_kg_m2':I.tolist(),'gravity_world_m_s2':gravity.tolist(),'radius_m':radius,'support512_reconstruction_max_abs_error':max(support_error),'wall_seconds':time.perf_counter()-start,'results':pin(out/'frame_results.json'),'matrices':pin(out/'matrices_and_witnesses.npz'),'physical_reference_qualified':False,'forceclosure_qualified':False,'GPU_used':False,'native_FK_collision_forward_step_contactforce_Env_calls':0,'scope':'DeclaredH123 projected surface points/inward normals,mu.1 unitforce9rays/contact, no directtorsion or arbitrary force cap. No robot configuration, actual native contact, PD/actuator bounds or contact constitutive law. FEASIBLE is a nonnegative mathematical inertial-wrench witness only; INFEASIBLE excludes this declared polyhedral cone, not all grasps/motion/physical dynamics. Nonzero512 directional support is not cone membership. One source151 dependent frames, not151 independent samples.'}
    save(out/'report.json',report);print(json.dumps({'status':report['status'],'counts':count,'checks':f"{len(checks)}/{len(checks)}",'LP_calls':lp_calls,'report':pin(out/'report.json')},indent=2))

if __name__=='__main__':main()
