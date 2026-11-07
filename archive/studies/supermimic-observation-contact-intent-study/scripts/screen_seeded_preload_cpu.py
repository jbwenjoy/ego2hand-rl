"""Canonical FP32 static prerequisite screen and explicit blocked PD design.

The two saved-q64 object screens are independently owned elsewhere. This code
uses the canonical choice fixed before any physics result, reconstructs q32
native geometry and the hand matrices, and never runs forward/step/forces/GPU.
An object-cone failure blocks both zero-stiction and relaxed hand design; it is
not a statement that every control or changing contact configuration must fail.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys
import time
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('D:/workspace/supermimic-observation-retarget-study')
WARM=Path('D:/workspace/supermimic-observation-warmseed-grounding-study')
CONE=Path('D:/workspace/supermimic-observation-grounding-study/scripts/diagnose_rest_wrench_feasibility_cpu_v2.py')
CONE_SHA='ba3f0f934ff7cc2ce6d674c92f3a755487bcc0994f04736664c5ceb87f31fdca'
sys.path.insert(0,str(OLD/'scripts'))
from native_filtered_contact_geometry import NativeFilteredContactGeometry

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item())+'\n',encoding='utf-8')
def main():
    started=time.perf_counter();out=ROOT/'outputs/seeded_preload_cpu_v1'
    if out.exists():raise FileExistsError('No overwrite or repeat of the conditional static design')
    plan=read(WARM/'configs/warmseed-rest-qref-review-v1.json');canonical=ROOT/'outputs/frame0_h23_seeded_baseline_v3'
    assert plan['candidate']['path']==str(canonical/'initialization_candidate.npz')
    sourcepins={'canonical_candidate':plan['candidate'],'model':plan['model'],'context':plan['context'],
      'pure_projection':plan['source_pins']['pure_projection'],'original_pure_projection':plan['source_pins']['original_pure_projection'],
      'canonical_native_contacts':pin(canonical/'native_contacts.json'),'canonical_candidate_report':pin(canonical/'report.json'),
      'canonical_bindings':pin(canonical/'frozen_anchor_bindings.json'),'native_geometry':pin(OLD/'scripts/native_filtered_contact_geometry.py'),
      'kinematics':pin(OLD/'scripts/retarget_palmroot_cpu.py'),'native_pyramid_helper':pin(CONE),'screen_source':pin(__file__)}
    for k,v in sourcepins.items():assert sha(v['path'])==v['sha256'],'Source/input drift '+k
    assert sha(CONE)==CONE_SHA
    spec=importlib.util.spec_from_file_location('pinned_native_cone',CONE);cone=importlib.util.module_from_spec(spec);spec.loader.exec_module(cone)
    with np.load(plan['candidate']['path'],allow_pickle=False) as z:q64=z['joint_pos_rad'].reshape(-1,20)[0].copy();pose64=z['mapped_object_pose_w'].reshape(-1,7)[0].copy()
    q32,pose32=q64.astype(np.float32),pose64.astype(np.float32)
    saved={k:getattr(mujoco,k) for k in ('mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_contactForce')};forbidden={k:0 for k in saved}
    def deny(k):
        def fail(*args,**kwargs):forbidden[k]+=1;raise AssertionError('Forbidden dynamics/force API '+k)
        return fail
    try:
        for k in saved:setattr(mujoco,k,deny(k))
        geom=NativeFilteredContactGeometry(plan['model']['path'],plan['context']['path']);m,d=geom.model,geom.data
        physical={}
        with np.load(plan['context']['path'],allow_pickle=False) as z:
            for key in ('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm'):
                physical[key]=z[key][0].copy();assert np.array_equal(z[key],np.broadcast_to(z[key][:1],z[key].shape));getattr(m,key)[:]=physical[key]
        geom.set_object_pose(pose32.astype(float));geom.robot.last_q=None;native=geom.evaluate(q32.astype(float))
        robot=geom.robot;obj=m.body('object/cube').id;object_geoms=set(geom.object_ids.tolist());hand=set(geom.hand_ids.tolist())
        allcontacts=[];contacts=[];unsupported=[]
        for i,c in enumerate(d.contact):
            g1,g2=map(int,c.geom);hasobj=g1 in object_geoms or g2 in object_geoms;hashand=g1 in hand or g2 in hand
            category='hand_object' if hasobj and hashand else 'hand_self' if g1 in hand and g2 in hand else 'other'
            row={'contact_id':i,'geom':[g1,g2],'names':[m.geom(g1).name,m.geom(g2).name],'category':category,
              'distance_m':float(c.dist),'position_w':c.pos.tolist(),'frame_rows_geom0_to_geom1_w':c.frame.reshape(3,3).tolist(),
              'effective_condim':int(c.dim),'friction5':c.friction.tolist()};allcontacts.append(row)
            if category!='hand_object':unsupported.append('Non-HO contact would require additional generalized-force terms: '+str(row['geom']));continue
            sign=1. if g2 in object_geoms else -1.;frame=sign*c.frame.reshape(3,3).copy();hg=g1 if g1 in hand else g2
            assert int(c.dim) in (1,3,4,6) and np.isfinite(frame).all()
            priorities=(int(m.geom_priority[g1]),int(m.geom_priority[g2]))
            if priorities[0]!=priorities[1]:selected=g1 if priorities[0]>priorities[1] else g2;mu=np.asarray(m.geom_friction[selected]);condim=int(m.geom_condim[selected])
            else:mu=np.maximum(m.geom_friction[g1],m.geom_friction[g2]);condim=int(max(m.geom_condim[g1],m.geom_condim[g2]))
            expected=np.array([mu[0],mu[0],mu[1],mu[2],mu[2]])
            assert condim==int(c.dim) and np.array_equal(expected,c.friction),'Actual material mix mismatch'
            contacts.append({'id':str(i),'frame':frame,'point':c.pos.copy(),'friction':c.friction.copy(),'condim':condim,'hand_body':int(m.geom_bodyid[hg])})
        assert contacts,'No canonical native HO contacts'
        assert int(m.opt.cone)==int(mujoco.mjtCone.mjCONE_PYRAMIDAL) and m.npair==0 and not m.flg_adhesion
        gravity=np.asarray(m.opt.gravity,np.float32).astype(float);mass=float(physical['body_mass'][obj]);com=d.xipos[obj].copy();target=np.r_[-mass*gravity,np.zeros(3)]
        object_result,A=cone.solve(contacts,com,target)
        # Same q32 state, all bodies included. Object/free-body contributions
        # have zero columns on the robot20 DOFs and cannot be double counted.
        body_jac=[];body_com=[];body_g=[]
        for b in range(m.nbody):
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacBodyCom(m,d,jp,jr,b)
            J=jp[:,robot.va];body_jac.append(J);body_com.append(d.xipos[b].copy());body_g.append(J.T@(float(physical['body_mass'][b])*gravity))
        tau_g=np.sum(body_g,axis=0);Gblocks=[];direct_blocks=[]
        for c in contacts:
            world,local=cone.native_pyramid_rays(c['frame'],c['point'],com,c['friction'],c['condim'])
            force=c['frame'].T@local[:3];direct=c['frame'].T@local[3:]
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,c['point'],c['hand_body'])
            Gblocks.append(-jp[:,robot.va].T@force-jr[:,robot.va].T@direct);direct_blocks.append(direct)
        G=np.concatenate(Gblocks,axis=1)
        jids=np.array([m.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
        aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j).item() for j in jids]);kp=physical['actuator_gainprm'][aids,0].astype(float);bias=physical['actuator_biasprm'][aids].astype(float)
        lower=np.maximum(m.jnt_range[jids,0],m.actuator_ctrlrange[aids,0]);upper=np.minimum(m.jnt_range[jids,1],m.actuator_ctrlrange[aids,1])
        lower32,upper32=lower.astype(np.float32),upper.astype(np.float32);innerlo=lower32+np.float32(.020);innerhi=upper32-np.float32(.020)
        actual_force_bounds=m.actuator_forcerange[aids].astype(np.float32).astype(float)
        joint_force_bounds=m.jnt_actfrcrange[jids].astype(np.float32).astype(float)
        tau_lo=np.maximum(kp*(innerlo.astype(float)-q32.astype(float)),np.maximum(actual_force_bounds[:,0],joint_force_bounds[:,0]))
        tau_hi=np.minimum(kp*(innerhi.astype(float)-q32.astype(float)),np.minimum(actual_force_bounds[:,1],joint_force_bounds[:,1]))
        mujoco.mj_transmission(m,d)
        passive={'body_gravcomp':m.body_gravcomp.tolist(),'joint_stiffness':m.jnt_stiffness.tolist(),
          'dof_damping':m.dof_damping[robot.va].astype(np.float32).tolist(),'damping_effect_at_designed_qd0':'exactly0',
          'robot_dof_frictionloss':m.dof_frictionloss[robot.va].astype(np.float32).tolist(),'frictionloss_nonzero_count':int(np.count_nonzero(m.dof_frictionloss[robot.va])),
          'static_friction_policy':'Explicit zero-stiction restricted witness versus independent bounded zeta in[-loss,+loss] relaxed existence; neither predicts actual solved friction.',
          'tendons':int(m.ntendon),'equality_constraints':int(m.neq),'flex':int(m.nflex),'fluid_density':float(m.opt.density),'fluid_viscosity':float(m.opt.viscosity),
          'joint_min_hard_margin_rad':float(np.minimum(q32-m.jnt_range[jids,0],m.jnt_range[jids,1]-q32).min()),'joint_margin':m.jnt_margin[jids].tolist(),
          'joint_limit_reaction':'Excluded only because candidate is strictly outside all limit/margin activation bands; not modeled for a limit-active candidate.'}
        model_checks={'no_gravcomp':not np.any(m.body_gravcomp),'no_springs':not np.any(m.jnt_stiffness),'no_tendon_equality_flex':m.ntendon==m.neq==m.nflex==0,
          'fluid0':m.opt.density==m.opt.viscosity==0,'gravity_enabled':not (int(m.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)),
          'gear_unit_joint':np.array_equal(m.actuator_gear[aids],np.tile([1.,0,0,0,0,0],(20,1))) and np.all(m.actuator_trntype[aids]==int(mujoco.mjtTrn.mjTRN_JOINT)),
          'affine_position_PD':bool(np.all(m.actuator_gaintype[aids]==int(mujoco.mjtGain.mjGAIN_FIXED)) and np.all(m.actuator_biastype[aids]==int(mujoco.mjtBias.mjBIAS_AFFINE)) and np.all(m.actuator_dyntype[aids]==int(mujoco.mjtDyn.mjDYN_NONE)) and np.all(kp>0) and not bias[:,0].any() and np.array_equal(bias[:,1],-kp) and not bias[:,3:].any()),
          'transmission_length_q':np.array_equal(d.actuator_length[aids],q32.astype(float)),'no_actuator_plugins':bool(np.all(m.actuator_plugin[aids]<0)),
          'q_strictly_outside_limit_bands':bool(np.all(q32>m.jnt_range[jids,0]+m.jnt_margin[jids]) and np.all(q32<m.jnt_range[jids,1]-m.jnt_margin[jids])),
          'force_limits_enabled':bool(np.all(m.actuator_forcelimited[aids]) and np.all(m.jnt_actfrclimited[jids])),
          'no_other_generated_contacts':not unsupported,'q32_hard_legal':bool(np.all(q32>=m.jnt_range[jids,0]) and np.all(q32<=m.jnt_range[jids,1]))}
    finally:
        for k,v in saved.items():setattr(mujoco,k,v)
    assert not any(forbidden.values())
    # Use the exact pinned CPU projection kernel to certify physical bounds;
    # this source does not turn a failed object prerequisite into a preload.
    sys.path.insert(0,str(Path(plan['source_pins']['pure_projection']['path']).parent))
    import torch
    from target_inset import target_inset
    qref=target_inset(torch.tensor(q32,device='cpu'),torch.tensor(lower32,device='cpu'),torch.tensor(upper32,device='cpu'),.020).numpy()
    expected_qref=np.minimum(np.maximum(q32,innerlo),innerhi)
    assert np.array_equal(qref,expected_qref) and not torch.cuda.is_initialized()
    status='BLOCKED_OBJECT_STATIC_WRENCH_INFEASIBLE' if not object_result['feasible_in_native_object_wrench_screen'] else 'Q32_OBJECT_FEASIBLE_REVIEW_REQUIRED_AFTER_Q64_NEGATIVE'
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'static_matrices.npz',A_object_wrench=A,G_hand_generalized_contact=G,gravity_robot_tau=tau_g,
      all_body_COM_Jacobian=np.asarray(body_jac),all_body_COM_world=np.asarray(body_com),all_body_gravity_tau=np.asarray(body_g),body_mass=physical['body_mass'],
      target_object_wrench=target,object_COM_world=com,q32=q32,pose32=pose32,kp=kp,bias=bias,outer_lower64=lower,outer_upper64=upper,
      inset_lower32=innerlo,inset_upper32=innerhi,tau_lower=tau_lo,tau_upper=tau_hi,actuator_force_bounds=actual_force_bounds,joint_force_bounds=joint_force_bounds,
      static_frictionloss_bounds=m.dof_frictionloss[robot.va].astype(np.float32))
    report={'status':status,'canonical_selection':'baseline-derived fixed before quality; midpoint remains separate sensitivity, never substitutes canonical',
      'canonical_q32_object_native_LP':object_result,'q32_native_contacts':allcontacts,'q64_contacts_saved':read(canonical/'native_contacts.json'),
      'q32_contact_count':len(allcontacts),'canonical_self_generated_count':sum(c['category']=='hand_self' for c in allcontacts),
      'passive_terms':passive,'model_contract_checks':model_checks,'forbidden_calls':forbidden,'model_inputs_unsupported':unsupported,
      'equilibrium_convention':'A lambda + [m*g,0]=0 for object. G maps compressive forces/directtorques onto HAND generalized forces, so tau_act=-tau_g-G*lambda-zeta.',
      'G_scope':'Jp at each actualcontactpoint includes lever effect; Jr maps DIRECT contact torque only, not r_cross_F again.',
      'wholehand_zero_stiction_LP':'NOT_SOLVED: canonical q64 object prerequisite independently failed; this q32 reconstruction is not allowed to replace the canonical negative.',
      'wholehand_bounded_stiction_LP':'NOT_SOLVED: same object prerequisite. Nonzero20 frictionloss terms are retained explicitly in matrices/contract, never silently set0.',
      'softcontact_depth_force_constitutive_law_enforced':False,'normal_force_upper_cap_N':None,
      'PD_after_cast_residual':None,'u0_design_created':False,'physical_reference_qualified':False,
      'scope':'Static object-wrench necessary condition at this frozen contact configuration only. Not a force prediction, motion impossibility, actual solved friction or full physical feasibility proof.',
      'source_pins':sourcepins,'matrix_npz_sha256':sha(out/'static_matrices.npz'),'CPU_wall_seconds':time.perf_counter()-started,'GPU_used':False,'physics_steps':0}
    write(out/'report.json',report)
    design={'schema':'seeded_static_PD_preload:1','status':status,'feasible':False,'canonical_candidate':plan['candidate'],
      'q0_rad':q64.tolist(),'q0_FP32_rad':q32.tolist(),'object_pose_w':pose64.tolist(),'object_pose_w_FP32':pose32.tolist(),
      'u0_rad':None,'C2_margin_rad':.020,'target_lower_rad':lower.tolist(),'target_upper_rad':upper.tolist(),
      'inset_lower_rad':innerlo.tolist(),'inset_upper_rad':innerhi.tolist(),'source_pins':sourcepins,
      'canonical_q32_object_LP_feasible':object_result['feasible_in_native_object_wrench_screen'],'normal_force_upper_cap_N':None,
      'u0_reference_C2q0_not_preload':qref.tolist(),'no_dynamic_qualification':True,'report_sha256':sha(out/'report.json'),
      'explanation':'No fixed PD preload is exported because the canonical saved contact-set object-wrench prerequisite failed; q32 result is separately retained. Do not clip, change contacts or use relaxed hand friction to bypass object balance.'}
    write(out/'designed_target_u0.json',design)
    print(json.dumps({'status':status,'q32_object_LP_feasible':object_result['feasible_in_native_object_wrench_screen'],'native_contacts':len(allcontacts),'model_contracts':model_checks,'preload_created':False,'design_sha256':sha(out/'designed_target_u0.json')},indent=2))
if __name__=='__main__':main()
