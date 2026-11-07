"""Progress handoff after bounded real-surface reference experiments."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

P=Path(__file__).resolve().parent;M=P.parent/'supermimic-chord-study'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    prev=P/'outputs/goal_extension_native_mpc_milestone_v5.json'
    for key,rec in read(prev)['evidence_pins'].items():
        if pin(rec['path'])!=rec:raise ValueError('Prior frozen evidence changed: '+key)
    files={
      'plan':M/'docs/research-plan-v0.25.md','previous_handoff':prev,
      'contact_surface_catalog':P/'contact_patch_v7/report.json',
      'compact_surface_catalog':P/'contact_patch_v7/catalog_compact_v1.json',
      'surface_localization':P/'contact_patch_v7/contact_localization_v1.json',
      'paired_actual_triangle_material_anchors':P/'contact_patch_v7/paired_surface_anchors_v1.json',
      'native_same_forward_contact_contract':P/'native_contact_design_v7/contract_v1/report.json',
      'native4ray_analytic_contract':P/'native_contact_design_v7/analytic_contract_v1/report.json',
      'body_axis_scan':P/'native_contact_design_v7/body_axis_scan_v1/report.json',
      'body_axis_scan_saved_review':P/'native_contact_design_v7/body_axis_scan_saved_recheck_v1/report.json',
      'saved_reference_seed_design':P/'reference_seed_design_v7/report.json',
      'axial_moment_necessary_bound':P/'reference_seed_design_v7/lever_bound_addendum_v1.json',
      'conditional_COM_axis_LP':P/'reference_seed_design_v7/axis_lever_LP_v1/actual_v1/report.json',
      'strict_support_source_only':P/'native_contact_design_v7/strict_support_lp_v7.py',
      'strict_support_guarded_source_only':P/'native_contact_design_v7/strict_support_lp_guarded_v7.py',
      'surface_pose_search':P/'surface_anchor_pose_v8/completed_v1/report.json',
      'surface_pose_search_source':P/'surface_anchor_pose_v8/solve_surface_anchor_pose_v8.py',
      'surface_pose_search_protocol':P/'surface_anchor_pose_v8/protocol_v1.json',
      'surface_pose_search_evaluation_log':P/'surface_anchor_pose_v8/completed_v1/evaluation_log.json',
      'surface_pose_algorithm_configuration':P/'surface_anchor_pose_v8/completed_v1/algorithm_surface_configuration.json',
      'surface_pose_counts_review':P/'surface_anchor_pose_v8/saved_count_review_v1/report.json',
      'surface_pose_saved_arithmetic':P/'surface_anchor_pose_v8/saved_recheck_v2/report.json',
      'surface_pose_six_candidate_figure':P/'surface_anchor_pose_v8/saved_recheck_v2/six_candidate_diagnostics_compact.png',
      'surface_pose_figure_receipt':P/'surface_anchor_pose_v8/saved_recheck_v2/compact_plot_receipt_v1.json'}
    terminal={
      'body_axis':P/'native_contact_design_v7/body_axis_scan_v1_terminal.json',
      'surface_pose':P/'surface_anchor_pose_v8/run1_terminal.json'}
    for key,path in terminal.items():
        t=read(path)
        if t['returncode']!=0 or t['stderr']['bytes']!=0:raise ValueError('Unexpected process terminal '+key)
        for k in ('stdout','stderr','source','protocol'):
            if pin(t[k]['path'])!=t[k]:raise ValueError('Changed process binding '+key)
        files[key+'_terminal']=path
    contract=read(files['native_same_forward_contact_contract']);analytic=read(files['native4ray_analytic_contract'])
    if (contract['passed'],contract['total'])!=(87,87) or (analytic['passed'],analytic['total'])!=(4,4):raise ValueError('Contracts')
    axis=read(files['body_axis_scan']);conditional=read(files['conditional_COM_axis_LP']);surface=read(files['surface_pose_search'])
    if axis['candidate_count']!=0 or axis['counts']['mj_forward_returned']!=3 or axis['LP_calls']!=3:raise ValueError('Three failed native reference hypotheses')
    if (conditional['LP_calls']!=1 or conditional['result']['solver_status']!=2
        or conditional['result']['delta_bodyZ_m'] is not None or conditional['result']['proposed_ctrl20'] is not None):
        raise ValueError('Preserve conditional LP infeasible and no pose fabricated')
    if (surface['holding_diagnostic_candidates']!=0 or len(surface['seed_results'])!=2
        or surface['counts']['mj_forward_returned']!=6 or surface['counts']['final_geometry_evaluations']!=6
        or surface['static_counts']['evaluations']!=878 or surface['LP_calls']!=877
        or surface['new_steps_training_GPU']!=0 or surface['model_audit']['passed'] is not True):raise ValueError('Bounded surface search scope')
    for seed in surface['seed_results']:
        if len(seed['shortlist'])!=3 or any(c['holding_diagnostic_eligible'] for c in seed['shortlist']):raise ValueError('All six candidates retained/rejected')
        for candidate in seed['shortlist']:
            if pin(candidate['result']['path'])!=candidate['result']:raise ValueError('Changed shortlisted record')
            files['surface_seed'+str(seed['seed'])+'_rank'+str(candidate['rank'])]=Path(candidate['result']['path'])
    for key in ('body_axis_scan_saved_review','surface_pose_counts_review','surface_pose_saved_arithmetic'):
        review=read(files[key])
        if review['passed']!=review['total'] or not all(v is True for v in review['checks'].values()):raise ValueError('Saved review incomplete '+key)
        if isinstance(review.get('plot'),dict):files[key+'_figure']=Path(review['plot']['path'])
    result={'schema':'ego_phone_goal_extension_progress:6','created_UTC':datetime.now(timezone.utc).isoformat(),
      'previous_goal_turn_classification':'progress','this_goal_turn_classification':'progress',
      'status':'ACTIVE_REAL_SURFACE_ANCHORS_MEASURED_GEOMETRY_SUPPORT_ASSEMBLY_NEXT',
      'native_goal_status_last_verified':'active','original_full_goal_complete':False,
      'real_ego_reference_and_official_object_asset_used':True,
      'new_learning_training_started':False,'new_native_holding_cases':0,'new_native_integration_steps':0,
      'new_native_forward_calls':11,'new_LP_calls':885,'strict_moment_box_LP_calls':0,
      'new_holding_diagnostic_candidates':0,'all_known_root_physics_processes_terminal':True,
      'new_training_first_healthy_checkpoint_pause_condition_triggered':False,
      'fixed_original_model_wrist_scale_gains_materials_solver_preserved':True,
      'functional_pad_or_finite_pressure_patch_calibrated':False,
      'mass_friction_COM_inertia_calibrated':False,'source_clip_GT_gravity_registration_verified':False,
      'new_ego_policy_or_WM_training_started':False,'holding_or_faithful_ego_qualified':False,
      'surface_search_count_scope':{'search_attempts':872,'successful_conditional_LP_logs':871,
        'one_deadline_attempt_without_full_log':True,'exact_failed_x14_or_internal_stopping_point_saved':False,
        'static_FK_counters':878,'final_native_forward':6,'final_native_LP':6,'strict_witness_LP':0},
      'next_action':[
        'First solve actual selected surface-pair assembly under complete geometry; do not trade normal/compression violations for a lower conditional wrench objective.',
        'Use two independent tangent-normal constraints per pair plus the opposite-normal branch. After geometry feasibility, optimize support under hard geometry and verify actual native contacts/PD/load/repeat.',
        'If the chosen facets cannot be assembled in finite search, explore recorded adjacent real triangle domains with explicit ego material deviation and UNKNOWN functional-pad semantics; do not invent a centroid/patch/torsion or unsupported third contact.',
        'Any nominal candidate still requires unchanged2s no-aid holding and two-dt/material-prior checks before61frame ego references and phone learning.',
        'Retain original controlled pen, SuperMimic/CHORD same-budget training, WM contact/gradient, independent/generalization, real-parameter and hardware-contract scope; next new training obeys checkpoint+ETA pause condition.'
      ],'evidence_pins':{k:pin(v) for k,v in files.items()},'finalizer':pin(__file__)}
    target=P/'outputs/goal_extension_surface_milestone_v6.json'
    with target.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'handoff':pin(target),'pins':len(files)}))


if __name__=='__main__':main()
