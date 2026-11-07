"""Independent saved NumPy support reconstruction; no production math import."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def selected(p,keys):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def main():
    source=ROOT/'outputs/full151_declared_H123_surface_targets_v1';out=ROOT/'outputs/full151_declared_targets_numpy_independent_v1'
    if out.exists():raise FileExistsError('No replacement audit')
    b=read(source/'binding_manifest.json');receipt=read(source/'receipt.json')
    with np.load(source/'intent_targets.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    cues=selected(Path(b['source_pins']['observation_cues']['path']),('time_s','mapped_object_pose_w','object_surface_closest_body_m','object_surface_outward_normal_body','surface_point_numeric_valid','surface_normal_valid_unambiguous','geometric_branch_unknown','signed_distance_m'))
    checks={}
    def check(k,v):checks[k]=bool(v)
    check('new_binding_exact151_not_old16',b['schema']=='full151_observation_declared_intent_binding:1' and b['reference_states']==151 and b['control_intervals']==150 and a['support512'].shape==(151,1,1,512))
    check('pinned_target_and_receipt_bind',b['target']==pin(source/'intent_targets.npz') and receipt['target']==b['target'] and receipt['binding_manifest']==pin(source/'binding_manifest.json'))
    check('all_declared_input_sources_current',all(pin(Path(p['path']))==p for p in b['source_pins'].values()))
    check('151_source_time_pose_exact',np.array_equal(a['time_s'],np.arange(151)*.02) and np.array_equal(a['time_s'],cues['time_s']) and np.array_equal(a['mapped_object_pose_w'],cues['mapped_object_pose_w']))
    geometry=cues['surface_point_numeric_valid']&cues['surface_normal_valid_unambiguous']&~cues['geometric_branch_unknown']
    check('geometry_and_H123_masks_separate',np.array_equal(a['geometry_valid_mask'],geometry) and a['declared_intent_mask'].dtype==np.bool_ and a['declared_intent_mask'].shape==(151,3) and a['declared_intent_mask'].all())
    active=geometry&a['declared_intent_mask'];known=active.any(1)
    check('active_known_phase_weight_exact',np.array_equal(a['declared_intent_active'],active) and np.array_equal(a['declared_intent_frame_known'],known) and np.array_equal(a['phase'],np.where(known,3,0)) and np.array_equal(a['phase_weight'],known.astype(float)))
    check('human_unknown_not_release_or_confidence',not any(a[k].any() for k in ('human_contact_known','human_contact_present','known_no_contact','landmark_confidence_known','object_confidence_known')) and a['human_contact_unknown_mask'].all())
    point=np.where(active[...,None],cues['object_surface_closest_body_m']-a['object_com_offset_body_m'],0.)
    normal=np.where(active[...,None],-cues['object_surface_outward_normal_body'],0.)
    check('COM_point_and_inward_normal_source_exact',np.array_equal(point,a['point_com_body_m']) and np.array_equal(normal,a['inward_normal_body']))
    radius=float(a['object_nominal_enclosing_radius_m']);basis=a['support_basis']
    check('finite_positive_radius_and_normals',np.isfinite(radius) and radius>.00745 and np.allclose(np.linalg.norm(normal,axis=-1)[active],1,rtol=0,atol=1e-6))
    basis_new=np.random.default_rng(0).standard_normal((512,6));basis_new/=np.linalg.norm(basis_new,axis=-1,keepdims=True)
    check('basis512_seed0_independent_exact',np.array_equal(basis,basis_new))
    n=normal/np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
    sign=np.where(n[...,2]>=0,1.,-1.);inv=-1/(sign+n[...,2]);xy=n[...,0]*n[...,1]*inv
    t1=np.stack((1+sign*n[...,0]**2*inv,sign*xy,-sign*n[...,0]),axis=-1);t1/=np.maximum(np.linalg.norm(t1,axis=-1,keepdims=True),1e-12)
    t2=np.stack((xy,sign+n[...,1]**2*inv,-n[...,1]),axis=-1);t2/=np.maximum(np.linalg.norm(t2,axis=-1,keepdims=True),1e-12)
    theta=np.arange(8)*(2*np.pi/8)
    rays=n[...,None,:]+.1*(np.cos(theta)[:,None]*t1[...,None,:]+np.sin(theta)[:,None]*t2[...,None,:]);rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-12)
    rays=np.concatenate((rays,n[...,None,:]),axis=-2)
    wrench=np.concatenate((rays,np.cross(point[...,None,:],rays)/radius),axis=-1)
    values=np.einsum('tked,nd->tken',wrench,basis);values=np.where(active[...,None,None],values,-np.inf)
    support=np.maximum(values.max(axis=(1,2)),0.);stored=a['support512'][:,0,0,:]
    err=float(np.abs(support-stored).max());check('independent_numpy_support512_matches',err<=2e-12)
    check('all151_nonzero512_and64_under_declared_mask',bool(known.any()) and np.all((support[known]>.001).any(1)) and np.all((support[known,::8]>.001).any(1)))
    check('raw_distances_preserved_without_contact_threshold',np.array_equal(a['raw_source_signed_distance_m'],cues['signed_distance_m']))
    check('targets_not_policy_robotq_or_force_claim',b['target_defined_from_policy_contacts'] is False and b['target_defined_from_generated_robot_contacts'] is False and b['robotq_used_to_define_targets'] is False and b['forceclosure_qualified'] is False and b['training_authorized'] is False and b['objective_loading_permitted'] is False)
    check('no_runtime_or_production_math_import',not any(k.split('.')[0] in ('torch','mujoco','warp','mujoco_warp','dex_supermimic','frozen_contact_math') for k in sys.modules))
    out.mkdir(parents=True);(out/'source_before.py').write_bytes(Path(__file__).read_bytes())
    report={'schema':'full151_declared_targets_independent_numpy:1','status':'PASS_DECLARED_NUMERIC_DATA_ONLY' if all(checks.values()) else 'FAIL_OR_MISSING','checks':checks,'passed':sum(checks.values()),'total':len(checks),'binding':pin(source/'binding_manifest.json'),'target':pin(source/'intent_targets.npz'),'auditor':pin(Path(__file__)),'independent_support_max_abs_error':err,'active_slots':int(active.sum()),'human_known_slots':int(a['human_contact_known'].sum()),'radius_m':radius,'scope':'Independent saved NumPy cone-ray fingerprint and source/mask/COM/normal reconstruction. Radius is a pinned input; static MJB derivation is in producer and separate demand-cone audit. No physical queries or real human labels. Nonzero support does not establish force closure, trajectory feasibility or a qualified proxy/gradient.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'support_max_error':err,'report':pin(out/'report.json')}))
    if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
