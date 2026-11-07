"""Freeze saved-input identities and CPU-only WM analysis rules; no model run."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path('D:/workspace/dex-supermimic-research/src/dex_supermimic')
PROTOCOL = ROOT/'protocols/rough-wm-onpolicy-cpu-diagnostic-v1.json'
OUT = ROOT/'outputs/rough_wm_cpu_protocol_review_v1'


def pin(path):
    p = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest()}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    if PROTOCOL.exists() or OUT.exists(): raise FileExistsError('Frozen artifact already exists')
    bundle_path = ROOT/'outputs/input_bundle_v1/bundle.json'
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    methods = ['A1L', 'P_ONLY', 'A2']
    models = []
    for method in methods:
        key = method.lower()
        run = ROOT/f'outputs/primary_{key}_seed0_v1'
        models.append({'method': method, 'seed': 0, 'iteration': 2000,
                       'checkpoint': pin(run/'train/model_final.pt'),
                       'training_config': pin(run/'config.json'),
                       'actual_physical_context': pin(run/'fixed_physics_context.npz'),
                       'final_health': pin(ROOT/f'outputs/health_primary_{key}_final_v1/report.json')})
    trajectories = []
    for name, directory in [('A0','native_a0_v2'), ('A1L','eval_primary_a1l_seed0_v1'),
                            ('P_ONLY','eval_primary_p_only_seed0_v1'), ('A2','eval_primary_a2_seed0_v1')]:
        folder = ROOT/'outputs'/directory
        files = ['raw_states.npz','control_events.json','solver_samples.json','config.json',
                 'evaluation.json','result.json','runtime.json','capacity_audit.json',
                 'requested_initialization_and_control.npz','actual_context_readbacks.npz',
                 'expected_context_fields.npz','nominal_model.mjb','controller_descriptor.json']
        trajectories.append({'name': name, 'directory': str(folder), 'state_rows': 61,
                             'physics_intervals': 60, 'control_transitions': 15,
                             'input_pins': {f:pin(folder/f) for f in files}})
    authorities = {
        'native_state': NATIVE/'rl/dex_supermimic/state.py',
        'native_sim_reader': NATIVE/'rl/dex_supermimic/sim.py',
        'native_pen_mdp': NATIVE/'rl/dex_supermimic/pen.py',
        'native_networks': NATIVE/'rl/dex_supermimic/networks.py',
        'native_runner': NATIVE/'rl/dex_supermimic/runner.py',
        'native_command': NATIVE/'tasks/pen_spin/mdp/commands.py',
        'rough_runtime': ROOT/'src/rough_prefix/runtime_v2.py',
        'rough_controller': ROOT/'src/rough_prefix/controller.py',
        'rough_bundle_loader': ROOT/'src/rough_prefix/bundle_v2.py',
        'training_protocol': ROOT/'protocols/rough-prefix-v4.json',
        'evaluation_protocol': ROOT/'protocols/rough-evaluation-v3.json',
    }
    p = {
        'schema':'rough_WM_onpolicy_saved_CPU_diagnostic_protocol:1',
        'status':'PREDECLARED_NOT_RUN',
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'purpose':'Bounded native104 one-step state/reward diagnostic on every saved control transition; no new physics, training, model selection or contact-dynamics certificate.',
        'scope':{'source_window_half_open':[200,216], 'duration_s':0.3,
                 'training_seed':0, 'methods':methods,
                 'trajectory_order':['A0','A1L','P_ONLY','A2'],
                 'lineage':'one previously seen controlled cached source; original ego video/camera/scale/confidence unknown',
                 'independent_training_replicates_per_method':1,
                 'independent_demonstration_N':None,
                 'same_source_posttraining_diagnostic':True,
                 'full151_or_shared10_qualified':False,
                 'physical_reference_qualified':False},
        'inputs':{'models':models,'trajectories':trajectories,
                  'input_bundle':pin(bundle_path),'reference_arrays':bundle['source_pins']['reference_arrays'],
                  'init_arrays':bundle['source_pins']['init_arrays'],
                  'control_arrays':bundle['source_pins']['control_arrays'],
                  'authoritative_context':bundle['context'],
                  'source_authorities':{k:pin(v) for k,v in authorities.items()}},
        'implementation_binding':{'loader':'PENDING_final_source_and_CPU_proof',
                                  'state_codec':'PENDING_final_source_and_CPU_proof',
                                  'run_script':'PENDING_final_source_and_CPU_proof',
                                  'require_separate_run_plan_pins_before_execution':True,
                                  'future_source_pin_updates_may_not_change_this_protocol':True},
        'execution':{'CPU_only':True,'CUDA_VISIBLE_DEVICES':'-1','torch_cuda_initialized_required':False,
                     'model_dtype':'float32','AMP':False,'device':'cpu','Env_created':False,
                     'physics_steps':0,'training_updates':0,'LP_or_optimization':False,
                     'allowed_model_geometry_operations':['read MJB topology/constants','optional mj_kinematics only for fixed-palm frame verification'],
                     'forbidden':['Warp construction','mj_step','mj_forward','mj_collision','geometry/contact query','new physical interventions','normalizer updates'],
                     'preserve_RNG_weights_normalizers_inputs':True,
                     'soft_worker_budget_seconds':120,'timeout_behavior':'retain partial artifacts; incomplete diagnostic is not complete; no automatic retry'},
        'transition_table':{
            'k':list(range(15)),'dt_s':0.02,'physics_dt_s':0.005,
            'x_index':'4*k','truth_x1_index':'4*k+4',
            'executed_u':'control_events[k].actualu[0], checked bitwise against raw ctrl_rad[4*k+1:4*k+5]',
            'do_not_use_as_u':'ctrl_rad[4*k] is prior-interval control (t0 initialization at k=0)',
            'phase_check':'event phase k; preaction_phase[0]=k; postaction_phase[0]=k+1; terminal only phase14',
            'sampling_phase':'raw states at20ms endpoints are pre-Manager-forward physical qpos/qvel; algebraic state reconstruction uses these saved states',
            'predictions_per_model':60,'total_WM_predictions':180,
            'all_paths_all_phases_required':True,'drop_or_select_by_error':False,
            'teacher_forcing':'every prediction begins at actual saved x, never previous predicted x',
            'state_teacher_label':'robot simulator states, not proprio-free human observations'},
        'state_codec':{
            'layout':[{'name':'q','slice':[0,20],'unit':'rad'}, {'name':'qd','slice':[20,40],'unit':'rad/s'},
                      {'name':'p_palm','slice':[40,43],'unit':'m'}, {'name':'R_palm','slice':[43,52],'unit':'row-major3x3'},
                      {'name':'v_palm','slice':[52,55],'unit':'m/s'}, {'name':'omega_palm','slice':[55,58],'unit':'rad/s'}],
            'p':'Rp.T@(p_object_bodyorigin_world-p_palm_world)',
            'R':'Rp.T@R_object_world',
            'v':'Rp.T@native_freejoint_linear_velocity_world',
            'omega':'Rp.T@R_object_world@native_freejoint_angular_velocity_body',
            'joint_mapping':'saved qpos/dof/position actuator IDs checked against MJB; do not reorder by a guessed name convention',
            'palm':'exact native robot root_link_pose_w topology; fixed mocap-to-palm transform must be proven from MJB, never assume every mocap is palm',
            'object_origin':'verify current body COM offset zero; if not zero, require explicit origin/velocity conversion or INVALID',
            'rotation_input':'finite wxyz unit quaternion, reject malformed; no silent repair of nonunit or improper matrices',
            'arithmetic':'decode/metrics may use float64; cast identical encoded x and actualu once to float32 for all WMs and preserve FP32 prediction inputs',
            'context':'validate actual saved8 fields and xi46 against frozen bundle; concat x58+xi46+zero174; not critic/task context368'},
        'baselines':[
            {'name':'identity','formula':'x1=x'},
            {'name':'constant_velocity','formula':'q1=q+dt*qd; qd1=qd; p1=p+dt*v; R1=Exp(dt*[omega_palm]x)@R; v1=v; omega1=omega',
             'rotation_side':'left multiplication because omega is expressed in fixed palm frame',
             'no_gravity_no_clamp_no_projection':True}],
        'metrics':{
            'retain_all_predictions_and_truth':True,
            'q_qd':['component RMSE','component max_abs','per-joint RMSE/max_abs'],
            'p_v_omega':['Euclidean norm per transition','mean','RMS','maximum'],
            'rotation':['SO3 geodesic per transition radians and degrees','mean','RMS','maximum'],
            'joint_extrema':['predicted and true max hard-limit excess','predicted and true >1mrad mask','per-joint extrema errors'],
            'normalization':'physical-unit metrics; no unit-mixed aggregate score',
            'aggregation':'each model x trajectory has15 dependent transitions; pooled60 is descriptive only; keep all4 path rows and allphase rows',
            'statistical_claims':'no independent N=15/60/180, no inferential CI/p-values or method-win selection from this diagnostic',
            'accuracy_acceptance_thresholds':None,
            'error_floor_interpretation':'native error_floor is a loss normalization scale lower bound, not a resolution/deadzone or causal explanation'},
        'object_reward':{
            'point_radius_m':0.1,'object_pos_weight':2,'position_scale_m':0.05,
            'object_rot_weight':8,'axis_endpoint_error_scale_m':0.04,
            'endpoint_error':'mean(norm(p_pred +/- .1*R_pred[:,0] - (p_ref +/- .1*R_ref[:,0]))) over corresponding two signed endpoints',
            'reward':'2*exp(-(norm(p-p_ref)/.05)^2) + 8*exp(-(endpoint_error/.04)^2)',
            'reference_phases':{'primary_native_training_semantics':'k (current preaction reference memory)','secondary_next_reference_tracking':'k+1'},
            'reference_coordinates':'same exact palm transform as state codec; mapped_object_pose_w[phase] from frozen reference arrays',
            'outputs':['predicted and actual next-state position term','predicted and actual next-state endpoint term','unscaled sum and prediction error','optional dt=.02 scaled separate column'],
            'not_full_native_reward':True,
            'excluded':['joint/link imitation','action_rate','torque','overcurrent/cage','selfcollision','geometry shaping','physical task-success criteria']},
        'selfcollision':{
            'native_output_dim':15,'native_output_kind':'selfcollision logits; not hand-object contact or force',
            'ground_truth_status':'UNKNOWN unless a separately pinned exact original sensor-label reconstruction proof is available',
            'current_raw_limit':'raw contacts lack ContactType/efc/sensordata; candidate existence or force>0 is not demonstrated equivalent to native CONSTRAINT-filtered sensor found',
            'phase':'native contact label snapshot is final solver preintegration (~t+15ms), distinct from x1 postintegration20ms',
            'when_unknown':'save predicted logits/probabilities only; BCE/F1/accuracy =null with UNKNOWN reason',
            'forbidden_substitutes':['hand-object contacts','force thresholds','independent geometry-intent target bundle']},
        'internal_gradient_check':{
            'scope':'mathematical autograd/finite-difference consistency only; no physics Jacobian qualification',
            'variables':'absolute executed u20 at fixed x/context/reference; no EMA/raw action differentiation or state feedback',
            'all180_predictions':'save finite flag and per-group Jacobian or object-reward gradient norm; retain nonfinite as diagnostic failure',
            'objective':'unscaled native object-only reward at primary phase k',
            'finite_difference_phases':[0,7,14],'all_models_and_all_paths':True,
            'random_direction_count_per_case':4,'PRNG':'NumPy default_rng','seed':20261104,
            'directions':'draw 20 normal components then L2 normalize, deterministic model/path/phase order; save exact vectors',
            'epsilon_rad_L2':[0.002,0.001], 'scheme':'central difference of the same frozen CPU WM objective',
            'lawful_domain':'C2_020 inner(hardjoint intersect actuator ctrlrange), actualu baseline must be in domain within declared input arithmetic tolerance',
            'out_of_domain':'record each proposed +/-epsilon target and crossing mask; skip that FD pair, never clamp/resample/replace direction',
            'outputs':['all finite autograd and FD values','absolute discrepancy','relative discrepancy with denominator max(abs(AD),abs(FD),1e-12)','epsilon disagreement','domain/skipped counts'],
            'pass_threshold_or_physical_direction_sign_gate':None,
            'no_physical_improvement_claim':True},
        'integrity_and_reporting':{
            'pre_post_input_and_source_SHA':True,'original_sources_and_saved_data_unmodified':True,
            'structural_failure':'INVALID input/model group with exact reason; retain complete matrix including not-run entries; no silent sample removal',
            'missing_contact_truth_does_not_invalidate_state_prediction':True,
            'required_outputs':['runplan pins','strict loader receipts','all encoded state/u/truth tables','all3x4x15 predictions and errors','baseline predictions','reward k/k+1 tables','gradient/FD domain diagnostics','model/RNG/normalizer unchanged receipts','CPU/cost/scope report'],
            'cost':'report protocol preparation, checkpoint qualification/loading, codec and model diagnostic wall separately; nested intervals not added twice',
            'selection':'fixed actual final2000 checkpoints only; no best checkpoint, path, phase, dose, seed or retry selection',
            'result_scope':'same-source N1 controlled .3s observation-derived rough reference, not full151/shared/ego/hardware/contact-force qualification'},
        'generator':pin(__file__),
    }
    write_new(PROTOCOL,p)
    OUT.mkdir(parents=True)
    write_new(OUT/'READY.json',{'schema':'rough_WM_CPU_protocol_preparation:1',
              'status':'FROZEN_PROTOCOL_NO_WM_NUMERIC_DIAGNOSTIC_RUN',
              'protocol':pin(PROTOCOL),'generator':pin(__file__),
              'input_model_count':len(models),'saved_trajectory_count':len(trajectories),
              'planned_WM_transition_predictions':180,
              'GPU_used':False,'Torch_imported':'torch' in sys.modules,
              'Env_created':False,'physics_steps':0,'WM_forward_calls':0,
              'implementation_source_pins_pending':True,
              'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')})
    print(json.dumps({'protocol':pin(PROTOCOL),'READY':pin(OUT/'READY.json')},indent=2))


if __name__=='__main__':main()
