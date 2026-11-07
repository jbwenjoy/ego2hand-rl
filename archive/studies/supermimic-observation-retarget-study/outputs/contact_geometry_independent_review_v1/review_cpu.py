"""Independent local derivative/sign/body-point review; no optimizer or physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco

ROOT=Path('D:/workspace/supermimic-observation-retarget-study')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from contact_penetration_geometry import ContactGeometry,ALLOWANCE_M,NORMALIZATION_M

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')

def main():
    started=time.perf_counter()
    model=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb')
    context=Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz')
    candidate=ROOT/'outputs/cached_landmarks_palmroot_v2/geometric_retarget.npz'
    parent=ROOT/'outputs/contact_jacobian_cpu_v1/report.json';old=json.loads(parent.read_text())
    sourcefiles=[ROOT/'scripts/contact_penetration_geometry.py',ROOT/'scripts/check_contact_geometry_jac_cpu.py',ROOT/'scripts/retarget_palmroot_cpu.py',model,context,candidate,parent]
    before={str(p):sha(p) for p in sourcefiles}
    checks={'inputs_match_original_check':sha(model)==old['model_sha256'] and sha(context)==old['context_sha256'] and sha(candidate)==old['candidate_sha256'],
            'source_matches_original_check':sha(sourcefiles[0])==old['geometry_module_sha256'] and sha(sourcefiles[1])==old['script_sha256']}
    # Forbid these Python entry points throughout the independent query. No
    # collision forward pass, simulation, inverse dynamics or force extraction.
    forbidden={name:0 for name in ['mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_contactForce','mj_collision']}
    def deny(name):
        def f(*args,**kwargs):forbidden[name]+=1;raise AssertionError('Forbidden '+name)
        return f
    for name in forbidden:setattr(mujoco,name,deny(name))
    g=ContactGeometry(model,context);m,d=g.model,g.data
    checks['26hand_x3object_no_mount']=g.pairs.shape==(78,2) and len(set(g.pairs[:,0]))==26 and all('mount' not in g.names[h] for h,o in g.pairs)
    with np.load(context,allow_pickle=False) as z:
        checks['exact_context_geom_size_pos']=np.array_equal(m.geom_size,z['geom_size'][0]) and np.array_equal(m.geom_pos,z['geom_pos'][0])
    with np.load(candidate,allow_pickle=False) as z:qs=z['joint_pos_rad'].copy();poses=z['mapped_object_pose_w'].copy()
    rootcases=np.load(ROOT/'outputs/contact_jacobian_cpu_v1/cases.npz',allow_pickle=False)
    rows=[];arrays={}
    for frame in [0,75,150]:
        g.set_object_pose(poses[frame]);q=qs[frame].copy();dist,jac=g.distance_and_jacobian(q);dist=dist.copy();jac=jac.copy()
        active=dist < -ALLOWANCE_M-.0001
        residual,rjac=g.penetration_residual(q);residual=residual.copy();rjac=rjac.copy()
        checks[f'frame{frame}_nonattraction_inactive_zero']=np.array_equal(residual[dist>=-ALLOWANCE_M],np.zeros((dist>=-ALLOWANCE_M).sum())) and not rjac[dist>=-ALLOWANCE_M].any()
        checks[f'frame{frame}_root_arrays_reproduced']=np.array_equal(dist,rootcases[f'frame{frame}_distances']) and np.array_equal(jac,rootcases[f'frame{frame}_analytic'])
        errors={};reserrors={};fullfd=None
        for epsilon in [1e-5,2e-5,5e-5]:
            fd=np.empty_like(jac);rfd=np.empty_like(jac)
            for j in range(20):
                values=[]
                for sign in [1,-1]:
                    qj=q.copy();qj[j]+=sign*epsilon;g.robot.forward(qj)
                    # Recompute direct scalar queries without module caching or
                    # its derivative path, then differentiate scalar residual.
                    values.append(np.array([mujoco.mj_geomDistance(m,d,int(h),int(o),1.,None) for h,o in g.pairs]))
                fd[:,j]=(values[0]-values[1])/(2*epsilon)
                rfd[:,j]=(np.maximum(-values[0]-ALLOWANCE_M,0)-np.maximum(-values[1]-ALLOWANCE_M,0))/(2*epsilon*NORMALIZATION_M)
            errors[str(epsilon)]=float(np.max(np.abs(fd[active]-jac[active])))
            resmask=np.abs(dist+ALLOWANCE_M)>.0001
            reserrors[str(epsilon)]=float(np.max(np.abs(rfd[resmask]-rjac[resmask])))
            if epsilon==2e-5:fullfd=fd.copy()
        g.robot.forward(q)
        direct=[];normerrors=[];witnesserrors=[];point_errors=[];reverse_errors=[]
        for k in np.flatnonzero(active):
            h,o=map(int,g.pairs[k]);segment=np.zeros(6);value=mujoco.mj_geomDistance(m,d,h,o,1.,segment)
            normal=(segment[3:]-segment[:3])/value
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));body=int(m.geom_bodyid[h]);mujoco.mj_jac(m,d,jp,jr,segment[:3],body)
            direct.append(float(np.max(np.abs(-normal@jp[:,g.robot.va]-jac[k]))));normerrors.append(abs(np.linalg.norm(normal)-1));witnesserrors.append(abs(np.linalg.norm(segment[3:]-segment[:3])-abs(value)))
            reversed_segment=np.zeros(6);rv=mujoco.mj_geomDistance(m,d,o,h,1.,reversed_segment)
            reverse_errors.append({'pair':int(k),'distance_error_m':abs(rv-value),'normal_reversal_error':float(np.max(np.abs((reversed_segment[3:]-reversed_segment[:3])/rv+normal)))})
            # Freeze the material witness point in the hand body's local frame.
            # Its world-point derivative must equal mj_jac at the same point.
            R=d.xmat[body].reshape(3,3).copy();local=R.T@(segment[:3]-d.xpos[body]);pfd=np.empty((3,20))
            for j in range(20):
                pts=[]
                for sign in [1,-1]:
                    qj=q.copy();qj[j]+=sign*1e-6;g.robot.forward(qj);pts.append(d.xpos[body]+d.xmat[body].reshape(3,3)@local)
                pfd[:,j]=(pts[0]-pts[1])/(2e-6)
            point_errors.append(float(np.max(np.abs(pfd-jp[:,g.robot.va]))));g.robot.forward(q)
        row={'frame':frame,'active_pairs_deeper_than_0_6mm':int(active.sum()),'max_distance_Jac_error_m_per_rad_by_epsilon':errors,'max_residual_Jac_error_per_rad_away_from_hinge':reserrors,'module_vs_independent_signed_normal_Jac_error_m_per_rad':max(direct),'max_rigid_material_witness_point_Jac_error_m_per_rad':max(point_errors),'max_normal_unit_error':max(normerrors),'max_negative_witness_length_error_m':max(witnesserrors),'reversed_geom_queries':reverse_errors,'active_pairs':[{'index':int(k),'hand':g.names[g.pairs[k,0]],'object':g.names[g.pairs[k,1]],'distance_m':float(dist[k]),'twenty_joint_Jac_norm_m_per_rad':float(np.linalg.norm(jac[k]))} for k in np.flatnonzero(active)],'minimum_margin_to_residual_hinge_m':float(np.min(np.abs(dist+ALLOWANCE_M)))}
        checks[f'frame{frame}_local_FD_unchanged_1e3_gate']=all(e<=1e-3 for e in errors.values())
        checks[f'frame{frame}_world_bodypoint_Jac']=max(point_errors)<1e-7 and max(direct)<1e-12
        checks[f'frame{frame}_signed_normal_consistent']=max(normerrors)<1e-6 and max(witnesserrors)<1e-8
        rows.append(row);arrays.update({f'frame{frame}_distance_m':dist,f'frame{frame}_Jac_module':jac,f'frame{frame}_centralFD_2e5':fullfd,f'frame{frame}_residual':residual})
    # Controlled geometries resolve fromto sign independently of the robot IK.
    sphere=[]
    for center in [1.5,3.]:
        sm=mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><geom name="a" type="sphere" size="1" pos="0 0 0"/><geom name="b" type="sphere" size="1" pos="{center} 0 0"/></worldbody></mujoco>');sd=mujoco.MjData(sm);mujoco.mj_kinematics(sm,sd)
        for first,second in [(0,1),(1,0)]:
            seg=np.zeros(6);v=mujoco.mj_geomDistance(sm,sd,first,second,10.,seg);n=(seg[3:]-seg[:3])/v
            expected=(center-2);direction=1 if first==0 else -1
            sphere.append({'center_distance':center,'first':first,'second':second,'signed_distance':v,'signed_fromto_normal':n.tolist(),'expected_distance':expected,'passed':abs(v-expected)<1e-10 and np.allclose(n,[direction,0,0],rtol=0,atol=1e-10)})
    checks['signed_fromto_sphere_separated_penetrating_both_orders']=all(row['passed'] for row in sphere)
    checks['no_dynamics_forward_collision_force_API']=not any(forbidden.values())
    checks['query_clock_zero_velocity_zero']=d.time==0 and not d.qvel.any()
    checks['input_and_source_files_unchanged']={str(p):sha(p) for p in sourcefiles}==before
    np.savez_compressed(OUT/'independent_cases.npz',**arrays)
    risks=[
        {'severity':'BLOCKING_FOR_ANY_GLOBAL_QUERY_QUALIFICATION','finding':'Root reports the first all151frame fixedweight refinement failed Inconsistent penetrating-pair witness at a newly visited optimizer state. This independent review does not hide or bypass that failure; only three prespecified original poses are checked. Failed snapshot review remains pending.','source':'contact_penetration_geometry.py:63-75; check_contact_geometry_jac_cpu.py:24-40'},
        {'severity':'SCOPE','finding':'Residual permits up to0.5mm pair overlap; zero residual is not strict nonintersection. Only fixed model geometry and object pose are measured. No contact-force/normal-force/stability/friction/gravity/dynamics claim.','source':'contact_penetration_geometry.py:15-16,82-88'},
        {'severity':'NUMERICAL','finding':'The squared hinge is piecewise smooth; its residual Jacobian is nondifferentiable at -0.5mm. Closest-feature switches/EPA degeneracy can also invalidate a local distance derivative. Original FD gate excludes depths -0.5..-0.6mm, so it cannot qualify threshold neighborhoods or optimizer paths.','source':'check_contact_geometry_jac_cpu.py:37; contact_penetration_geometry.py:65,85-87'},
        {'severity':'GEOMETRIC','finding':'78 separate convex-primitive pair constraints express overlap avoidance, not an exact signed distance to the entire pen union. Duplicated/adjacent primitive residuals can be counted more than once. Keeping object/palm fixed can make overlap irreducible by the20 finger DOFs; refinement must retain infeasibility and correspondence tradeoffs.','source':'contact_penetration_geometry.py:37-40; retarget_palmroot_cpu.py:64-69'},
        {'severity':'INPUT_CONTRACT','finding':'distance_and_jacobian(q) lacks an explicit finite shape(20,) check and RobotKinematics assigns by NumPy broadcasting. A scalar or length1 vector can be silently broadcast. The planned bounded least_squares path supplies20values, but module API is narrower in practice than an explicit general contract.','source':'contact_penetration_geometry.py:54-57; retarget_palmroot_cpu.py:83-87'},
        {'severity':'OBSERVATION_SCOPE','finding':'Mapped object metric scale/camera/right-hand and fixed-wrist task redesign remain unqualified. Unknown observed contact stays unknown. Candidate joint_pos_rad is a prior geometric IK solution, not a source robot joint label or executed robot state.','source':'retarget_palmroot_cpu.py:119-136,184-193,203-228'}]
    report={'status':'PASSED_THREE_STATE_LOCAL_SIGN_AND_JACOBIAN_REVIEW_GLOBAL_QUERY_UNQUALIFIED','checks':checks,'all_local_checks_passed':all(checks.values()),'rows':rows,'signed_fromto_sphere_cases':sphere,'risks':risks,'pending_failed_optimizer_snapshot':True,'model':pin(model),'context':pin(context),'candidate':pin(candidate),'source_files':{p.name:pin(p) for p in sourcefiles[:3]},'source_report':pin(parent),'independent_code':pin(__file__),'cases':pin(OUT/'independent_cases.npz'),'MuJoCo_version':mujoco.__version__,'query_options':g.query_options,'API_docstring':mujoco.mj_geomDistance.__doc__,'forbidden_call_counts':forbidden,'dynamics_steps':0,'known_observed_contacts_read':False,'source_robot_q_labels_read':False,'physical_reference_qualified':False,'wall_seconds_including_imported_module_setup':time.perf_counter()-started}
    write(OUT/'report.json',report);print(json.dumps({'status':report['status'],'all_local_checks_passed':report['all_local_checks_passed'],'check_count':len(checks),'rows':[{k:v for k,v in row.items() if k in ['frame','active_pairs_deeper_than_0_6mm','max_distance_Jac_error_m_per_rad_by_epsilon','max_rigid_material_witness_point_Jac_error_m_per_rad']} for row in rows],'failed':[k for k,v in checks.items() if not v]},indent=2));raise SystemExit(not all(checks.values()))

if __name__=='__main__':main()
