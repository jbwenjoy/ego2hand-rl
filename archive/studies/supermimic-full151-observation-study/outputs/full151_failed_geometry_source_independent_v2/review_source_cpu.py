"""Source-only review. Never imports or invokes the geometry producer."""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
OUT=Path(__file__).resolve().parent

def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pins(x):
    if isinstance(x,dict):
        if set(x)=={'path','sha256'}:yield x
        else:
            for v in x.values():yield from pins(v)
    elif isinstance(x,list):
        for v in x:yield from pins(v)

def same_pin(value):
    actual=pin(value['path'])
    return Path(actual['path']).samefile(value['path']) and actual['sha256']==value['sha256']

protocol_path=ROOT/'protocols/full151-failed-candidate-geometry-v1.json'
p=read(protocol_path);source_path=Path(p['sources']['producer']['path'])
s=source_path.read_text(encoding='utf-8-sig');g=Path(p['sources']['native_geometry']['path']).read_text(encoding='utf-8-sig')
k=Path(p['sources']['kinematics']['path']).read_text(encoding='utf-8-sig')
sourcepins=list(pins(p));checked=[{'pin':x,'match':same_pin(x)} for x in sourcepins]
parent=read(p['inputs']['failed_receipt']['path'])
checks={
    'all9_protocol_sources_inputs_match_current_bytes':len(checked)==9 and all(x['match'] for x in checked),
    'producer_actual_path_pin_match':pin(source_path)==p['sources']['producer'],
    'parent_stays_FAIL_with_no_READY':parent['status']=='CPU_CANDIDATE_GATES_FAIL_NO_QUALIFICATION' and not (Path(p['inputs']['failed_receipt']['path']).parent/'READY.json').exists(),
    'preconstruction_failed_parent_and_freshout_guards':'never READY' in s and 'out.mkdir(parents=True,exist_ok=False)' in s,
    '601_samples_explicit_clock_no_curve_extrapolation':"np.array_equal(times,np.arange(601)*.005)" in s and 'for i in range(601):' in s,
    'knot_source_quats_preserved_and_rotations_crosschecked':"poses[::4]=ref['mapped_object_pose_w']" in s and "np.max(np.abs(Rcheck-R))<=1e-12" in s,
    'forbidden_dynamics_force_distance_and_solver_wrapped_before_query_import':"['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce']" in s and "['linprog','minimize','least_squares']" in s and s.index('for namespace,names in')<s.index("'_full151_frozen_native_geometry'"),
    'allowed_kinematics_comPos_collision_counted':"['mj_kinematics','mj_comPos','mj_collision']" in s,
    'native_geo_actual_geom_size_pos_copied_original_collisionoptions': '("geom_size", "geom_pos")' in g and 'Original MJB collision numerical settings' in g and 'Frozen collision options were mutated' in g,
    'catalogue_no_mount_as_hand': 'right_palm_collision|right_finger[1-5]' in g and 'self.pairs' in g,
    'helper_state_remains_qvel0_time0_and_no_step': 'Geometry-only query requires clock0 and qvel0' in g and 'Collision query mutated physical state' in g,
    'absent_pair_distance_NaN_and_generated_mask_kept': 'distances = np.full(count, np.nan)' in g and "absent_native_pair_gaps_remain_NaN" in s,
    'sample_depth_does_not_grade_parent_or_issue_READY':"'physics_qualified':False,'contact_target_ready':False,'training_authorized':False" in s and 'READY.json' not in ''.join(line for line in s.splitlines() if 'write(' in line),
    'qualitythreshold_display_separate_from_engineeringstatus': "'sampled_depth3mm_pass'" in s and "'COMPLETED_SAMPLED_GEOMETRY_DIAGNOSTIC' if all(report['checks'].values())" in s,
    'known_constructor_and601_counter_contract_explicit':"allowed['mj_kinematics']==602 and allowed['mj_comPos']==602" in s and "query.collision_query_count==601 and allowed['mj_collision']==601" in s,
    'partial_query_mask_saved_and_monkeypatch_restored': 'GEOMETRY_DIAGNOSTIC_EXCEPTION_PRESERVED' in s and 'for (namespace,name),value in saved.items():setattr(namespace,name,value)' in s and 'queried=queried' in s,
    'macro21_semantics_not_BODY21_reference_export': 'for link in (1, 3, 4)' in k and 'right_finger{finger}_tip' in k and 'reference_qpos' not in s,
}
result={
    'schema':'full151_failed_candidate_geometry_source_independent_review:1',
    'status':'PASS_SOURCE_REVIEW_NO_EXECUTION_OR_GEOMETRY_QUALIFICATION',
    'protocol':pin(protocol_path),'producer':pin(source_path),'pin_checks':checked,
    'checks':checks,'all_checks_pass':all(checks.values()),
    'boundary':'All601 fixed samples are a generated-contact-depth diagnostic of prescribed FAILED candidate; not continuous clearance, positive gap, collision force, controller execution, READY or physics qualification.',
    'note':'fk_points_w uses the pinned RobotKinematics macro21(palm+link1/3/4/tip), not native BODY21. Must not relabel it for Reader ABI.',
    'scope':{'source_JSON_AST_hash_only':True,'NumPy_SciPy_MuJoCo_Torch_Warp_imported':any(x in sys.modules for x in ('numpy','scipy','mujoco','torch','warp')),'arrays_loaded':False,'query_FK_Env_WM_physics_GPU_solver':False,'old_source_modified':False},
    'review_source':pin(__file__),
}
result['pins_still_match_after_review']=all(same_pin(x) for x in sourcepins)
target=OUT/'report.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(target),'checks':len(checks),'all_pass':all(checks.values())}))
if not all(checks.values()) or not result['pins_still_match_after_review']:raise SystemExit(2)
