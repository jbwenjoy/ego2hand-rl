"""Pin staged-assembly evidence and phone transition contracts, goal stays active."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

P=Path(__file__).resolve().parent;M=P.parent/'supermimic-chord-study'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def checked(rec):
    if pin(rec['path'])!=rec:raise ValueError('Changed frozen evidence '+rec['path'])
    return Path(rec['path'])


def main():
    prev=P/'outputs/goal_extension_surface_milestone_v6.json'
    for rec in read(prev)['evidence_pins'].values():checked(rec)
    files={'plan':M/'docs/research-plan-v0.26.md','previous_handoff':prev,
      'assembly_FK_contract':P/'surface_assembly_v9/original_pose_contract_v1/report.json',
      'assembly_source_review':P/'surface_assembly_v9/design_review_v1.md',
      'assembly_source_review_addendum':P/'surface_assembly_v9/design_review_addendum_v2.md',
      'adjacent_face_candidates':P/'surface_assembly_v9/adjacent_candidates_v1/adjacent_candidates_v1.json',
      'v9_counts_review':P/'surface_assembly_v9/saved_count_review_v1/report.json',
      'v9_saved_arithmetic':P/'surface_assembly_v9/saved_recheck_v1/report.json',
      'v9_active_bounds':P/'surface_assembly_v9/active_bound_review_v1/report.json',
      'v9_antipodal_descriptor':P/'surface_assembly_v9/active_bound_review_v1/antipodal_geometry_v1.json',
      'ego_vs_CAD_phone_faces':P/'surface_assembly_v9/saved_recheck_v1/original_ego_vs_CAD_phone_faces_v1.json',
      'v10_source_derivation':P/'surface_assembly_v10/source_derivation_v1.json',
      'v10_exact_diff':P/'surface_assembly_v10/v9_to_v10_exact.diff',
      'v10_metadata_preflight':P/'surface_assembly_v10/metadata_preflight_v1/report.json',
      'v10_author_terminal_readback':P/'surface_assembly_v10/terminal_saved_readback_v1.json',
      'official_pad_semantics':P/'asset_semantics_followup_v10/report.json',
      'official_pad_source_manifest':P/'asset_semantics_followup_v10/source_manifest.json',
      'phone_dataset':P/'phone_transition_dataset_v1/dataset_v1/report.json',
      'phone_dataset_schema':P/'phone_transition_dataset_v1/dataset_v1/schema.json',
      'phone_dataset_saved_readback':P/'phone_transition_dataset_v1/saved_readback_v1.json',
      'phone_dataset_cross_cohort_aliases':P/'phone_transition_dataset_v1/overlap_audit_v1.json'}
    if (read(files['assembly_FK_contract'])['passed'],read(files['assembly_FK_contract'])['total'])!=(14,14):raise ValueError('Cheap-FK contract')
    totals={'native_forward':0,'strict_LP':0,'LP':0,'static_FK':0,'cheap_FK':1,'geometry_valid_final_records':0}
    for version,lp_count,static_count in ((9,717,713),(10,957,955)):
        root=P/f'surface_assembly_v{version}';result=root/'completed_v1/report.json';r=read(result)
        checked(r['protocol']);protocol=read(r['protocol']['path'])
        for rec in [*protocol['source_pins'].values(),*protocol['inputs'].values()]:checked(rec)
        terminal=root/'run1_terminal.json';t=read(terminal)
        if t['returncode']!=0 or t['stderr']['bytes']!=0:raise ValueError('Process terminal')
        for key in ('source','protocol','stdout','stderr'):checked(t[key])
        if (r['holding_diagnostic_candidates']!=0 or r['counts']['mj_forward_returned']!=4
            or r['counts']['final_geometry_evaluations']!=4 or r['LP_calls']!=lp_count
            or r['static_counts']['evaluations']!=static_count or r['new_steps_training_GPU']!=0
            or r['model_audit']['passed'] is not True):raise ValueError('Preserve experiment scope')
        files[f'v{version}_report']=result;files[f'v{version}_terminal']=terminal
        files[f'v{version}_attempt_log']=root/'completed_v1/attempts.jsonl'
        files[f'v{version}_surface_configuration']=root/'completed_v1/algorithm_surface_configuration.json'
        total_final=0
        for seed in r['seed_results']:
            for final in seed['finals']:
                f=read(checked(final['result']));c=f['candidate'];g=f['final_geometry'];s=f['strict_native_support_LP']
                if (not final['geometry_pass'] or not final['assembly_gate'] or final['holding_diagnostic_eligible']
                    or not g['geometry_gate_pass'] or not g['complete22_189'] or c['max_abs_eq10_scaled']>1e-5
                    or s is None or s['solver_status']!=2 or s['strict_force_moment_pass'] is not False
                    or f['candidate_state'] is not None or f['holding_diagnostic_eligible'] is not False):
                    raise ValueError('Preserve final geometry success but support/holding failure')
                files[f'v{version}_seed{seed["seed"]}_rank{final["rank"]}']=Path(final['result']['path']);total_final+=1
        if total_final!=4:raise ValueError('All final records retained')
        totals['native_forward']+=4;totals['strict_LP']+=4;totals['LP']+=lp_count
        totals['static_FK']+=static_count;totals['cheap_FK']+=r['cheap_FK_counts']['query_returned'];totals['geometry_valid_final_records']+=total_final
    for key in ('v9_counts_review','v9_saved_arithmetic','v10_author_terminal_readback'):
        r=read(files[key])
        if r['passed']!=r['total'] or any(v is not True for v in r['checks'].values()):raise ValueError('Incomplete saved review')
    data=read(files['phone_dataset']);alias=read(files['phone_dataset_cross_cohort_aliases']);back=read(files['phone_dataset_saved_readback'])
    if (data['passed'],data['total'])!=(8,8) or data['counts']['samples']!=322 or back['passed'] is not True:raise ValueError('Dataset contract')
    if (alias['cross_cohort_alias_group_count']!=6 or alias['cohort_union_group_count']!=1
        or alias['original_sample_count']!=322 or alias['all322_samples_retained'] is not True
        or alias['split_assigned'] is not False or alias['deduplication_applied'] is not False):raise ValueError('Preserve alias/leakage boundary')
    for name,rec in data['outputs'].items():checked(rec);files['phone_data_'+name]=Path(rec['path'])
    out={'schema':'ego_phone_goal_extension_progress:7','created_UTC':datetime.now(timezone.utc).isoformat(),
      'previous_goal_turn_classification':'progress','this_goal_turn_classification':'progress',
      'status':'ACTIVE_ASSEMBLY_VERIFIED_SOURCE_NORMAL_MAPPING_AND_NONOVERLAPPING_WM_DATA_NEXT',
      'native_goal_status_last_verified':'active','original_full_goal_complete':False,
      'real_ego_reference_and_official_object_asset_used':True,
      'new_learning_training_started':False,'new_native_holding_cases':0,'new_native_integration_steps':0,
      'new_native_forward_calls':totals['native_forward'],'new_LP_calls':totals['LP'],'new_strict_box_LP_calls':totals['strict_LP'],
      'new_static_FK_calls':totals['static_FK'],'new_cheap_FK_including_original_contract':totals['cheap_FK'],
      'geometry_and_hard_assembly_valid_final_records':totals['geometry_valid_final_records'],
      'records_are_independent_unique_grasps':False,'holding_diagnostic_candidates':0,
      'all_known_execution_processes_observed_terminal':True,
      'new_training_first_healthy_checkpoint_pause_condition_triggered':False,
      'original_physical_model_wrist_scale_gains_material_solver_preserved':True,
      'functional_facets_compliance_friction_mass_inertia_calibrated':False,
      'source_GT_gravity_registration_verified':False,
      'phone_WM_saved_transition_samples':322,'phone_WM_source_boundary_cohorts':7,
      'phone_WM_cross_cohort_exact_alias_groups':6,'phone_WM_alias_connected_prefix_groups':1,
      'phone_WM_independent_trajectory_or_Markov_qualified':False,'new_ego_policy_or_WM_training_started':False,
      'next_action':[
        'Rebuild source-normal-aware contact mapping using original ego object face/region evidence and uncertainty; millimetre-only CAD proximity changed phone surface normal by58–61degrees.',
        'Keep actual triangle/region and robot functional-pad semantics explicit. Use adjacent real faces or measured matching sensor metadata when available, not arbitrary centroids or added torsion.',
        'Any revised reference must recompute native contacts/J/PD/load and still pass original2s no-aid/two-dt/material-prior checks. Do not infer task benefit from lower static residual.',
        'Use full-orientation phone codec for later world-model input/output, but collect independent nonoverlapping episodes before trajectory validation; all current322 samples belong to one alias-connected failed-hold prefix.',
        'Retain original controlled-pen, same-budget SuperMimic/CHORD, WM contact/gradient,61frame/heldout/generalization, physical calibration and hardware-contract scope; next new learning obeys checkpoint+ETA pause condition.'
      ],'evidence_pins':{k:pin(v) for k,v in files.items()},'finalizer':pin(__file__)}
    target=P/'outputs/goal_extension_assembly_milestone_v7.json'
    with target.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print({'status':out['status'],'handoff':pin(target),'pins':len(files),'totals':totals})


if __name__=='__main__':main()
