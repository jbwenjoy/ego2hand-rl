"""One original-pose FK contract. No forward, collision, RNE, LP or step."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import traceback

ROOT=Path(__file__).resolve().parent
P=ROOT.parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def plain(v):
    import numpy as np
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(plain(v),f,indent=2,allow_nan=False)


def main():
    anchors_path=P/'contact_patch_v7/paired_surface_anchors_v1.json'
    binding_path=P/'hand2_holding_v1/compiled_nominal_v2/binding.json'
    binding=json.loads(binding_path.read_text(encoding='utf8'))
    protocol={'schema':'single_original_pose_assembly_FK_contract:9',
        'source_pins':{'assembly':pin(ROOT/'assembly_kinematics_v9.py'),'checker':pin(__file__),
            'initial_loader':pin(P/'hand2_stage1/scene_loader_v1.py')},
        'inputs':{'anchors':pin(anchors_path),'binding':pin(binding_path),'compiled_model':binding['model']},
        'CPU_FK_authorized':True,'maximum_pose_queries':1,'maximum_mj_kinematics':1,'maximum_mj_comPos':1,
        'forward_RNE_passive_geometry_J_LP_steps_GPU_authorized':False,
        'comparison_absolute_bound_m_or_dimensionless':5e-12,
        'tangent_rule':'Original directed first phone triangle edge, numerical tangent orthogonalization only; normal antiparallel branch is separate.',
        'model_loading':'One standalone original compiled model, shared with the assembly object. No second comparison model is loaded; normal caller passes its already-owned e.',
        'automatic_retry':False}
    pp=ROOT/'original_pose_contract_protocol_v1.json';write(pp,protocol)
    out=ROOT/'original_pose_contract_v1';out.mkdir();started=time.perf_counter();phase='load';kin=None;model_reads=0
    try:
        import numpy as np
        import mujoco
        from scipy.spatial.transform import Rotation
        from types import SimpleNamespace
        from assembly_kinematics_v9 import AssemblyKinematics
        for rec in [*protocol['source_pins'].values(),*protocol['inputs'].values()]:assert pin(rec['path'])==rec
        model=mujoco.MjModel.from_binary_path(binding['model']['path']);model_reads+=1
        path=Path(protocol['source_pins']['initial_loader']['path'])
        spec=importlib.util.spec_from_file_location('original_initial_arrays_for_assembly_v9',path)
        loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
        initial=loader.initial_state()
        e=SimpleNamespace(m=model,initial=initial,Rp=initial['T_world_from_palm'][:3,:3].copy(),
            pp=initial['T_world_from_palm'][:3,3].copy(),pid=model.body('right_palm_link').id,
            oid=model.body('hot3d_phone24').id,phone=model.geom('phone24_collision_convex').id,
            pads=[model.geom(f'right_finger{i}_tip_sensor_col').id for i in (1,2)])
        kin=AssemblyKinematics(e,anchors_path=anchors_path)
        a=json.loads(anchors_path.read_text(encoding='utf8'));qpos=np.asarray(a['source_pose']['qpos27'],dtype=float)
        body_R=np.asarray(a['source_pose']['phone_body_R_world']);body_p=np.asarray(a['source_pose']['phone_root_world_m'])
        t=e.Rp.T@(body_p-e.pp);rv=Rotation.from_matrix(e.Rp.T@body_R).as_rotvec()
        x=np.r_[qpos[:8],t,rv]
        phase='one_original_pose_query';r=kin.query(x,qpos[8:20])
        expected={
            'A_hand_world_m':np.asarray([v['surfaces']['hand']['initial_projected_point_world_m'] for v in a['anchors']]),
            'B_phone_world_m':np.asarray([v['surfaces']['phone']['initial_projected_point_world_m'] for v in a['anchors']]),
            'n_hand_world':np.asarray([v['surfaces']['hand']['initial_outward_normal_world'] for v in a['anchors']]),
            'n_phone_world':np.asarray([v['surfaces']['phone']['initial_outward_normal_world'] for v in a['anchors']])}
        errors={k:float(np.max(np.abs(r[k]-v))) for k,v in expected.items()}
        checks={k+'_matches_saved_reference':v<=5e-12 for k,v in errors.items()}
        T=r['phone_tangents_world2x2x3'];normal=r['n_phone_world']
        checks.update(original_rest12_exact=np.array_equal(r['q20'][8:],qpos[8:20]),
            unit_tangent_rows=np.max(np.abs(np.linalg.norm(T,axis=2)-1))<=5e-12,
            tangent_rows_mutually_orthogonal=np.max(np.abs(np.einsum('ni,ni->n',T[:,0],T[:,1])))<=5e-12,
            tangent_rows_phone_normal_orthogonal=np.max(np.abs(np.einsum('nki,ni->nk',T,normal)))<=5e-12,
            right_handed_phone_surface_basis=np.max(np.abs(np.cross(T[:,0],T[:,1])-normal))<=5e-12,
            normal_eq_uses_exported_phone_tangents=np.array_equal(r['normal_eq2x2'],np.einsum('nki,ni->nk',T,r['n_hand_world'])),
            antidot_branch_explicit=np.array_equal(r['antiparallel_branch2'],r['antidot2']>0),
            one_FK_and_comPos_only=kin.counts['mj_kinematics']==kin.counts['mj_comPos']==kin.counts['query_returned']==1,
            no_dynamics_queries_or_J=all(kin.counts[k]==0 for k in ('mj_comVel','mj_rne','mj_passive','geometry_queries','jacobian_calls','mj_forward','LP_calls','integration_steps','learning_GPU')))
        audit=kin.audit_model();checks['model_audit']=audit['passed']
        assert all(checks.values()),{k:v for k,v in checks.items() if not v}
        write(out/'query.json',r)
        np.savez_compressed(out/'query_raw.npz',**{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
        report={'schema':'single_original_pose_assembly_FK_contract_result:9',
            'status':'PASS_FIXED_TRIANGLE_ASSEMBLY_FK_REFERENCE_NOT_GEOMETRY_OR_SUPPORT',
            'protocol':pin(pp),'checks':checks,'passed':int(sum(checks.values())),'total':len(checks),
            'maximum_reference_errors':errors,'counts':dict(kin.counts),'standalone_model_reads':model_reads,
            'model_audit':audit,'elapsed_seconds':time.perf_counter()-started,
            'original_pose_antidot2':r['antidot2'],'original_pose_eq10':r['eq10'],
            'outputs':{p.name:pin(p) for p in out.iterdir() if p.is_file()},
            'geometry_support_or_holding_qualified':False}
        write(out/'report.json',report)
        print(json.dumps({'report':pin(out/'report.json'),'passed':report['passed'],'total':report['total'],'counts':kin.counts,'reference_errors':errors}))
    except BaseException as error:
        write(out/'failure.json',{'phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
            'counts':None if kin is None else kin.counts,'standalone_model_reads':model_reads,'automatic_retry':False})
        raise


if __name__=='__main__':main()
