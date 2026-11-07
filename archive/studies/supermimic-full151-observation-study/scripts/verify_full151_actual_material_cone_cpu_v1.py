"""One passive MJB/context read and NumPy cone algebra, no native state/query."""
from pathlib import Path
import argparse,hashlib,json,os,re,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ENV=Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages')
MODEL=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb')
CONTEXT=Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz')
def pin(p):p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def main(output):
    start=time.perf_counter();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);checks={};calls={}
    import mujoco
    for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_kinematics','mj_comPos','mj_collision','mj_geomDistance','mj_contactForce','mju_encodePyramid','mju_decodePyramid'):
        calls[name]=0
        def deny(*a,_name=name,**kw):calls[_name]+=1;raise AssertionError('Passive material verification forbids '+_name)
        setattr(mujoco,name,deny)
    m=mujoco.MjModel.from_binary_path(str(MODEL))
    with np.load(CONTEXT,allow_pickle=False) as z:actual=z['geom_friction'].copy()
    def check(name,value):checks[name]=bool(value)
    check('exact_frozen_model_and_context',pin(MODEL)['sha256']=='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd' and pin(CONTEXT)['sha256']=='5edd125973015c034d45e47092786b7fac105a06af52864582afa08d8660b89a')
    check('actual_geomfriction_shared_FP32',actual.dtype==np.float32 and actual.shape==(128,m.ngeom,3) and np.isfinite(actual).all() and actual.tobytes()==np.ascontiguousarray(np.broadcast_to(actual[:1],actual.shape)).tobytes())
    hand=[g for g in range(m.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',m.geom(g).name)]
    enabled=[g for g in hand if m.geom_contype[g] or m.geom_conaffinity[g]]
    obj=m.body('object/cube').id;objects=np.flatnonzero(m.geom_bodyid==obj).tolist()
    def row(g):return {'geom_id':int(g),'name':m.geom(g).name,'body_id':int(m.geom_bodyid[g]),'enabled':bool(m.geom_contype[g] or m.geom_conaffinity[g]),'contype':int(m.geom_contype[g]),'conaffinity':int(m.geom_conaffinity[g]),'priority':int(m.geom_priority[g]),'condim':int(m.geom_condim[g]),'solmix_nominal':float(m.geom_solmix[g]),'solref_nominal':m.geom_solref[g].tolist(),'solimp_nominal':m.geom_solimp[g].tolist(),'friction3_nominal':m.geom_friction[g].tolist(),'friction3_actual_FP32':actual[0,g].astype(float).tolist()}
    table=[row(g) for g in hand+objects]
    explicit=[]
    for k in range(m.npair):
        g1,g2=int(m.pair_geom1[k]),int(m.pair_geom2[k])
        if (g1 in hand and g2 in objects) or (g2 in hand and g1 in objects):explicit.append({'pair_id':k,'geom1':g1,'geom2':g2,'dim':int(m.pair_dim[k]),'friction5':m.pair_friction[k].tolist()})
    check('bound21_enabled_hand_and3_object_geoms',len(enabled)==21 and len(objects)==3 and all(m.geom_contype[g] or m.geom_conaffinity[g] for g in objects))
    check('native_pyramidal_cone',int(m.opt.cone)==int(mujoco.mjtCone.mjCONE_PYRAMIDAL)==0)
    override=bool(int(m.opt.enableflags)&int(mujoco.mjtEnableBit.mjENBL_OVERRIDE))
    check('no_contact_parameter_override',not override and not explicit)
    check('hand_priority1_condim4_object_priority0_condim3',all(m.geom_priority[g]==1 and m.geom_condim[g]==4 for g in enabled) and all(m.geom_priority[g]==0 and m.geom_condim[g]==3 for g in objects))
    uniform=all(np.array_equal(actual[0,g],actual[0,enabled[0]]) and m.geom_priority[g]==m.geom_priority[enabled[0]] and m.geom_condim[g]==m.geom_condim[enabled[0]] and m.geom_solmix[g]==m.geom_solmix[enabled[0]] and np.array_equal(m.geom_solref[g],m.geom_solref[enabled[0]]) and np.array_equal(m.geom_solimp[g],m.geom_solimp[enabled[0]]) for g in enabled)
    check('all_enabled_hand_material_solver_fields_uniform',uniform)
    pair_rows=[]
    for h in enabled:
        for o in objects:
            # Installed collision_core.py:359-380, strictly higher handpriority.
            friction3=actual[0,h].astype(float);friction5=np.maximum(float(mujoco.mjMINMU),friction3[[0,0,1,2,2]])
            solref=m.geom_solref[h] if (m.geom_solref[h,0]>0 and m.geom_solref[o,0]>0) else np.minimum(m.geom_solref[h],m.geom_solref[o])
            pair_rows.append({'hand_geom':h,'object_geom':o,'mix':1.,'condim':4,'friction5':friction5.tolist(),'solref':solref.tolist(),'solreffriction':[0.,0.],'solimp':m.geom_solimp[h].tolist(),'geometry_pair_mask_allows':bool((m.geom_contype[h]&m.geom_conaffinity[o]) or (m.geom_contype[o]&m.geom_conaffinity[h]))})
    mu=np.array(pair_rows[0]['friction5']);check('all63_hypothetical_HO_materials_identical',len(pair_rows)==63 and all(v['friction5']==pair_rows[0]['friction5'] and v['condim']==4 for v in pair_rows))
    check('minimum_friction_clamp_inactive',np.all(actual[0,enabled[0]]>=float(mujoco.mjMINMU)))
    # Local rows [Fn,Ft1,Ft2,Tn,Tt1,Tt2]. Each native pyramid coordinate has
    # normal force1; torsion columns consume normal force, not a free torque.
    A=np.zeros((6,6));A[0]=1.
    for i in range(3):A[i+1,2*i]=mu[i];A[i+1,2*i+1]=-mu[i]
    check('six_columns_normalforce_one',A.shape==(6,6) and np.array_equal(A[0],np.ones(6)))
    check('torsion_columns_have_normalforce_and_no_sliding',np.array_equal(A[:3,4:],np.array([[1.,1.],[0.,0.],[0.,0.]])) and np.array_equal(A[3,4:],mu[2]*np.array([1.,-1.])) and not A[4:].any())
    tests=np.array([[0,0,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,2,0],[.5,.5,0,0,0,0],[1,2,3,4,5,6]],float)
    decoded=np.c_[tests.sum(1),mu[0]*(tests[:,0]-tests[:,1]),mu[1]*(tests[:,2]-tests[:,3]),mu[2]*(tests[:,4]-tests[:,5]),np.zeros((len(tests),2))]
    check('NumPy_matrix_matches_installed_decode_formula',np.array_equal((A@tests.T).T,decoded))
    lhs=np.abs(decoded[:,1])/mu[0]+np.abs(decoded[:,2])/mu[1]+np.abs(decoded[:,3])/mu[2]
    check('shared_L1_friction_torsion_budget',np.all(lhs<=decoded[:,0]+1e-12))
    bad=np.array([1.,mu[0],0.,mu[2],0.,0.]);check('Cartesian_sliding_torsion_product_counterexample_rejected',abs(bad[1])/mu[0]+abs(bad[3])/mu[2]>bad[0])
    check('pure_normal_available_as_half_first_two_rays',np.array_equal(.5*(A[:,0]+A[:,1]),np.array([1.,0,0,0,0,0])))
    check('no_free_torsion_at_zero_normalforce',np.all(A[0]>0) and A[3,4]!=0 and np.all(A[4:]==0))
    backend=ENV/'mujoco_warp/_src';headers=ENV/'mujoco/include/mujoco/mujoco'
    sources={'producer':pin(__file__),'collision_driver':pin(backend/'collision_driver.py'),'collision_core':pin(backend/'collision_core.py'),'constraint':pin(backend/'constraint.py'),'support':pin(backend/'support.py'),'solver':pin(backend/'solver.py'),'types':pin(backend/'types.py'),'io':pin(backend/'io.py'),'header_model':pin(headers/'mjmodel.h'),'header_contact':pin(headers/'mjdata.h'),'header_api':pin(headers/'mujoco.h'),'header_enum':pin(headers/'mjtype.h')}
    np.savez_compressed(out/'material_arrays.npz',candidate_hand_ids=np.array(hand,np.int32),enabled_hand_ids=np.array(enabled,np.int32),object_geom_ids=np.array(objects,np.int32),geom_priority=m.geom_priority.copy(),geom_condim=m.geom_condim.copy(),geom_solmix=m.geom_solmix.copy(),geom_solref=m.geom_solref.copy(),geom_solimp=m.geom_solimp.copy(),geom_friction_nominal=m.geom_friction.copy(),actual_geom_friction_FP32=actual[0],A_local_normalforce_normalized=A)
    write(out/'all_candidate_materials.json',{'hand_candidate_count':len(hand),'enabled_hand_count':len(enabled),'disabled_hand_count':len(hand)-len(enabled),'object_count':len(objects),'model_npair':int(m.npair),'HO_explicit_pair_overrides':explicit,'rows':table,'resolved_enabled_HO_pairs':pair_rows})
    refs={'explicit_pair_precedence':{'source':'collision_core','lines':[335,342]},'priority_condim_friction_solmix':{'source':'collision_core','lines':[349,380]},'solref_solimp_mix_and_minmu':{'source':'collision_core','lines':[383,397]},'driver_delegates_material_contact_params':{'source':'collision_core','lines':[428,453]},'pyramid_row_count':{'source':'constraint','lines':[2684,2700]},'pyramidal_J_angular_normal_and_sign':{'source':'constraint','lines':[3144,3221]},'decode_normal_sum_and_other_directions':{'source':'support','lines':[326,347]},'contact_force_force_torque_frame_rotation':{'source':'support','lines':[368,392]},'pyramid_inequality_multiplier_nonnegative':{'source':'solver','lines':[403,463]},'friction_field_units_order':{'source':'header_contact','lines':[39,45]},'geom_friction_slide_spin_roll':{'source':'header_model','lines':[462,478]},'minimum_friction':{'source':'header_model','lines':[26,26]},'native_pyramid_C_API':{'source':'header_api','lines':[1390,1394]},'cone_enum':{'source':'header_enum','lines':[188,191]}}
    check('no_native_state_queries_Torch_or_Warp_imports',not any(calls.values()) and not any(n.split('.')[0] in ('torch','warp','mujoco_warp','mjlab') for n in sys.modules))
    contract={'schema':'full151_actual_material_native_pyramid_contract:1','status':'VERIFIED_PASSIVE_CONSTANTS_AND_NUMPY_ALGEBRA' if all(checks.values()) else 'UNKNOWN_CONTRACT_NOT_QUALIFIED',
      'inputs':{'model':pin(MODEL),'actual_context':pin(CONTEXT)},'sources':sources,'source_line_refs':refs,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'mujoco_version':mujoco.__version__,
      'model_cone':int(m.opt.cone),'model_cone_name':'PYRAMIDAL','contact_override_enabled':override,'mjMINMU':float(mujoco.mjMINMU),'material_table':pin(out/'all_candidate_materials.json'),'material_arrays':pin(out/'material_arrays.npz'),
      'enabled_hand_count':len(enabled),'object_count':len(objects),'mix_rule':'No explicitHO pair overrides; handpriority1>objectpriority0: mix1, condim fromhand4, friction3 fromactualhand (NOT componentwise max with object and NOT solmix-averaged friction). solmix only matters equalpriority solref/solimp; see conditional solref minformat branch.',
      'condim':4,'friction5':mu.tolist(),'friction5_order':['tangent1','tangent2','spin_about_normal','roll1_inactive_condim4','roll2_inactive_condim4'],
      'friction_units':['dimensionless','dimensionless','m=Nm/N','m_inactive','m_inactive'],
      'local_wrench_row_order':['Fn_N','Ft1_N','Ft2_N','Tn_Nm','Tt1_Nm','Tt2_Nm'],
      'ray_column_order':['tangent1+','tangent1-','tangent2+','tangent2-','normal_torsion+','normal_torsion-'],
      'A_local6x6':A.tolist(),'lambda_units':'N of compressive normalforce. Fn=sum(lambda); minsumlambda is minsumFn, not a unit-total-force norm.',
      'decode_formula':'Fn=sum(lambda0..5); Ft1=mu0*(l0-l1); Ft2=mu1*(l2-l3); Tn=mu2*(l4-l5); Tt1=Tt2=0.',
      'coupled_cone_formula':'Fn>=0; abs(Ft1)/mu0+abs(Ft2)/mu1+abs(Tn)/mu2<=Fn; no independent torsion atFn0.',
      'BODY_transform':'Assume proper hypotheticalframe rows[n_inward,t1,t2]. F_BODY=frame.T@F_local; directtau_BODY=frame.T@T_local; tau_COM=cross(point_COM_BODY,F_BODY)+directtau_BODY. LP may divide tau_COM and requiredtau by sameenclosingradius once.',
      'frame_scope':'n_inward is force onobject fromsource declaredprojection. Frisvad tangents are a fixed hypothetical nonmeasured frame. Do not label them actual native solved contactframe. Native measured contactnormal points geom0→geom1; force/torque-onobject uses+ forobjectgeom1(index1) and- forobjectgeom0(index0).',
      'forbidden_alternatives':['NoCartesian8edge-sliding×independenttorsion combinations: they violate sharednativepyramid budget.','No standalone zero-Fn torsion columns.','Do not unit-normalize forceparts ofthese columns: preserve Fn1 and normalforceobjective.','No rolling dimensions forcondim4.','Do not substitute supportrepresentationmu.1/unitforce9rays for this native-parameter6ray contract.'],
      'UNKNOWN_boundaries':['Hypothetical source contacts maynotexist or be reachable; no robotq/contact/frame measurements produced.','Uniformenabled materials are verified for this fixedasset/context only. Changedpair overrides/priority/condim/material/conetype require newcontract; mixedunassigned materials areUNKNOWN.','No normalforce upperbound, actuator/handtorque/contactconstitutivelaw/solverconvergence proof is provided.','Pyramidal eligibility depends on hypothetical tangentframe; evenuniformslidingmu doesnot remove finitepyramid orientation dependence.','Disabledcollisiongeoms are listed but not used as realcontact candidates. Absence/masks are not positivegap or humanrelease certificates.'],
      'native_calls':calls,'passive_MJB_reads':1,'passive_context_NPZ_reads':1,'LP_calls':0,'Env_created':False,'Torch_Warp_imported':False,'GPU_used':False,'wall_s':time.perf_counter()-start}
    write(out/'material_cone_contract.json',contract);print(json.dumps({'status':contract['status'],'passed':contract['passed'],'total':contract['total'],'enabled':len(enabled),'candidate':len(hand),'friction5':mu.tolist(),'contract':pin(out/'material_cone_contract.json')},indent=2));return 0 if all(checks.values()) else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();raise SystemExit(main(a.output))
