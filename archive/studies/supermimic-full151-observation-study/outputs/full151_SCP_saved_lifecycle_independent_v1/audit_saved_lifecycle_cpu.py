"""Saved identity/lifecycle audit. No production or native modules imported."""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
RUN=ROOT/'outputs/full151_contact_coupled_scp_v1'
OUT=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pins(x):
    if isinstance(x,dict):
        if set(x)=={'path','sha256'}:yield x
        else:
            for v in x.values():yield from pins(v)
    elif isinstance(x,list):
        for v in x:yield from pins(v)
def samepin(x):return pin(x['path'])['sha256']==x['sha256']

r=read(RUN/'receipt.json');p=read(r['protocol']['path']);tool=read(RUN/'root_execution_observation.json');inventory=read(RUN/'terminal_artifact_inventory_v1.json')
started=read(RUN/'execution_started.json');live=read(RUN/'live_state.json')
source=Path(p['sources']['producer']['path']).read_text(encoding='utf-8-sig')
sourcechecks=[]
for k,v in p['sources'].items():
    archive=RUN/'source_before'/(k+Path(v['path']).suffix)
    sourcechecks.append({'key':k,'source':v,'archive':pin(archive),'match':samepin(v) and archive.read_bytes()==Path(v['path']).read_bytes()})
allpinchecks=[{'pin':x,'match':samepin(x)} for x in pins(p)]
artifacts=[{'pin':x,'actual':pin(x['path']),'match':samepin(x)} for x in r['artifacts']]
immutable=[x for x in artifacts if Path(x['pin']['path']).name!='live_state.json']
mutable=[x for x in artifacts if Path(x['pin']['path']).name=='live_state.json']
before={x['actual']['path']:x['actual'] for x in artifacts}
rows=[read(RUN/f'major{i:02d}'/'receipt.json') for i in range(1,9)]
parent=load(p['inputs']['parent_reference']['path'],['control_points_rad','mapped_object_pose_w','object_COM_world_m','object_rotation_w','object_v_world_m_s','object_a_world_m_s2','object_omega_world_rad_s','object_alpha_world_rad_s2','object_p_coefficients','object_phi_coefficients'])
final=load(RUN/'reference_candidate151.npz')
init_parent=load(p['inputs']['initialization']['path']);init=load(RUN/'init_requested.npz')
control=load(RUN/'control_candidate150.npz',['u_postC2_150_FP32_rad','planned_rate149_rad_s'])
P=parent['control_points_rad'];accepted=None;radius=.15;chain=[]
for row in rows:
    major=row['major'];center=load(RUN/f'major{major:02d}'/'subproblem.npz',['center_P'])['center_P']
    trial=load(RUN/f'major{major:02d}'/'trial_control_points.npz',['P'])['P']
    valid=(row['center_accepted_major']==accepted and row['radius_rad']==radius and exact(center,P))
    if row['accepted']:P=trial;accepted=major
    else:radius*=.5
    valid=valid and row['next_radius_rad']==radius
    chain.append({'major':major,'source_previous_accepted':row['center_accepted_major'],'accepted':row['accepted'],'radius':row['radius_rad'],'next_radius':row['next_radius_rad'],'identity_match':valid})
last=load(RUN/f'major{accepted:02d}'/'trial_control_points.npz',['P','eta_m'])
final_inputs=load(RUN/'final_geometry/native_grid.npz',['q601_rad','object_pose601_w'])
selected_inputs=load(RUN/f'major{accepted:02d}'/'geometry/native_grid.npz',['q601_rad','object_pose601_w'])
checks={
    'actual_protocol_pin_current':samepin(r['protocol']),
    'all24_source_input_pins_current':len(allpinchecks)==24 and all(x['match'] for x in allpinchecks),
    'all11_executed_sources_exact_archive_and_current':len(sourcechecks)==11 and all(x['match'] for x in sourcechecks),
    '103_original_immutable_artifact_pins_exact':len(immutable)==103 and all(x['match'] for x in immutable),
    'one_mutable_live_state_pin_mismatch_explicit':len(mutable)==1 and mutable[0]['match'] is False,
    'mutable_mismatch_matches_Root_sidecar':inventory['original_immutable_artifact_pins_matched']==103 and len(inventory['mutable_pin_mismatch'])==1 and inventory['mutable_pin_mismatch'][0]['original']==mutable[0]['pin'] and inventory['mutable_pin_mismatch'][0]['current']==mutable[0]['actual'],
    'source_order_explains_late_live_state_overwrite':source.index("report['artifacts']=[pin(f)")<source.index("write(out/'receipt.json',report);write(out/'live_state.json'"),
    'started_and_live_terminal_identity_consistent':started['protocol']==r['protocol'] and started['started_UTC']==r['started_UTC'] and live['stage']=='TERMINAL' and live['status']==r['status'] and live['cost']==r['cost'],
    '8_LP_attempted_returned_and_major_slots':r['cost']['LP_attempted']==r['cost']['LP_returned']==len(rows)==8 and [x['major'] for x in rows]==list(range(1,9)),
    'all_saved_permajor_receipts_match_parent':rows==r['majors'],
    'all8_normal_solver_terminal_and_raw_gate_recorded':all(x['solver_success'] is True and x['solver_status']==0 and x['linear_audit']['raw_primal_pass'] is True for x in rows),
    'all8_postC2_gate_recorded_pass':all(all(x['postC2_gates'].values()) for x in rows),
    'eta_positive_intermediate_did_not_block7accepts':all(x['eta_m']>1e-9 for x in rows) and sum(x['accepted'] for x in rows)==7,
    'chronological_center_trial_lastaccepted_chain_exact':all(x['identity_match'] for x in chain),
    'only_major6_rejected_and_radius_halved': [x['major'] for x in rows if not x['accepted']]==[6] and rows[5]['next_radius_rad']==.075 and rows[6]['radius_rad']==rows[7]['radius_rad']==.075,
    'final_P_is_lastaccepted8_not_historical_choice':accepted==r['last_accepted_major']==8 and exact(final['control_points_rad'],P) and exact(final['control_points_rad'],last['P']),
    'last_eta_matches_selected_trial':float(last['eta_m'])==r['last_accepted_eta_m']==rows[7]['eta_m'],
    'fresh_final_query_prescribes_same_q601_pose_as_lastaccepted':all(exact(final_inputs[k],selected_inputs[k]) for k in final_inputs),
    'fixed_P0P1_retained_exact':exact(final['control_points_rad'][:2],parent['control_points_rad'][:2]),
    'all7_requested_initial_arrays_byte_exact':len(init)==len(init_parent)==7 and set(init)==set(init_parent) and all(exact(init[k],init_parent[k]) for k in init),
    'all_prescribed_object_curves_and_design_derivatives_unchanged':all(exact(final[k],v) for k,v in parent.items() if k!='control_points_rad'),
    'full151_goals_and150u149updates_kept_no_prefixselection':final['q_goal_rad'].shape==(151,20) and control['u_postC2_150_FP32_rad'].shape==(150,20) and control['planned_rate149_rad_s'].shape==(149,20),
    '8major_limit_stop_is_not_fatal_or_softtimeout':r['algorithm_stop']=='MAJOR_LIMIT' and r['checks']['execution_without_fatal_numerical_stop'] is True,
    'only2_of28_final_gates_fail':len(r['checks'])==28 and [k for k,v in r['checks'].items() if v is not True]==['last_accepted_eta_final_gate','final_fresh_sampled_geometry'],
    'overall_FAILED_no_sample_READY_preserved':r['status']=='CPU_COUPLED_CANDIDATE_FAIL_RETAINED' and not (RUN/'READY_CPU_SAMPLED_ONLY.json').exists(),
    'no_physical_intent_or_training_qualification':all(r[k] is False for k in ('physical_qualified','contact_intent_target_ready','training_authorized')),
    '10grids6010collision6011FKCOM_reported':r['cost']['grid_audits']==10 and r['actual_native_calls']['allowed']['mj_collision']==6010 and r['actual_native_calls']['allowed']['mj_kinematics']==r['actual_native_calls']['allowed']['mj_comPos']==6011,
    'all_forbidden_API_counts_reported0':not any(r['actual_native_calls']['forbidden'].values()),
    'summed_solver_cost_matches_8receipts':abs(math.fsum(x['solver_wall_s'] for x in rows)-r['cost']['solver_wall_s'])<=1e-12 and 0<=r['cost']['solver_wall_s']<=r['wall_s'],
    'Root_tool_once_exit1_closed_no_retry_not_OSledger':tool['exit_code']==1 and tool['retained_session_state']=='CLOSED_TERMINAL_EXIT1' and tool['observed_Root_launches_this_turn']==1 and tool['retry'] is False and tool['PID_creation']=='NOT_SAMPLED' and tool['whole_external_wall_s'] is None,
    'Root_tool_pins_producer_protocol_receipt':samepin(tool['producer']) and tool['protocol']==r['protocol'] and tool['receipt']==pin(RUN/'receipt.json'),
    'actual_mask_minus1_logged_consistently':tool['CUDA_VISIBLE_DEVICES']==r['CUDA_VISIBLE_DEVICES']=='-1',
}
result={
    'schema':'full151_SCP_saved_lifecycle_independent_review:1',
    'status':'PASS_SELECTION_SOURCE_AND_FAILURE_PRESERVATION_WITH_ONE_EXPLICIT_MUTABLE_PROGRESS_PIN_MISMATCH',
    'candidate_status':'FAILED_GEOMETRY_AND_ETA_NOT_REQUALIFIED','receipt':pin(RUN/'receipt.json'),'protocol':r['protocol'],
    'checks':checks,'all_checks_pass':all(checks.values()),'source_checks':sourcechecks,'protocol_pin_checks':allpinchecks,
    'immutable_artifact_count':len(immutable),'mutable_artifact_checks':mutable,'root_execution_observation':pin(RUN/'root_execution_observation.json'),'terminal_inventory':pin(RUN/'terminal_artifact_inventory_v1.json'),
    'chronological_selection':chain,
    'independent_raw_math_audit_reference':pin(ROOT/'outputs/full151_scp_saved_independent_v1/report.json'),
    'cost':{'solver_wall_s':r['cost']['solver_wall_s'],'worker_wall_s':r['wall_s'],'external_whole_wall_s':None,'not_additive':True,'scheduled_grids':10,'prior164probe_is_separate':True},
    'scope':{'original_immutable_sources_and_selection_fields_recomputed':True,'arrays_loaded_for_identity_only':True,'LP_CSR_geometry_Phi_rawmath':'Separate Contact independent173audit; not recomputed here','runtime_API_counts':'Saved source-instrumented evidence, not independently observedlive native state','Root_launch_once_exitcode':'Pinned Root tool observation; no independentPID/Creation/OS startstop qualification','framework_imported':any(x in sys.modules for x in ('mujoco','torch','warp')),'solver_FK_query_Env_WM_physics_GPU':False},
    'review_source':pin(__file__),
}
result['all_original_run_artifacts_unchanged_pre_post']=all(pin(x)==v for x,v in before.items())
target=OUT/'report.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(target),'checks':len(checks),'all_pass':all(checks.values()),'mutable_pin_mismatches':len(mutable)}))
if not all(checks.values()) or not result['all_original_run_artifacts_unchanged_pre_post']:raise SystemExit(2)
