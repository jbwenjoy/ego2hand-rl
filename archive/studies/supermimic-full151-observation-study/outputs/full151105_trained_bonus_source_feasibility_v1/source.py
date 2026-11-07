"""Record a source-only, not-authorized preregistration; import no science code.

This writer reads JSON/Python text and hashes immutable inputs. It never opens
NPZ tensors, deserializes checkpoints, imports objectives or evaluates scores.
"""
from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
ROUGH = Path('D:/workspace/supermimic-rough-prefix-learning-study')
NATIVE = Path('D:/workspace/dex-supermimic-research')
OUT = ROOT/'outputs/full151105_trained_bonus_source_feasibility_v1'
PROTOCOL = ROOT/'protocols/full151105_trained_geometric_bonus_secant_CPU_prereg_v1.json'
FROZEN = ROUGH/'vendor/frozen_contact_math'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': sha(p)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact pin shape required')
    p = Path(value['path']).resolve(strict=True)
    require(p.is_file() and sha(p) == value['sha256'], 'Pin changed: '+str(p))
    return p


def same_pin(a, b):
    return checked(a).samefile(checked(b)) and a['sha256'] == b['sha256']


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    require(not OUT.exists() and not PROTOCOL.exists(), 'Fresh protocol/output only; no overwrite')
    OUT.mkdir(parents=True)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    checks = {}
    def fact(name, condition):
        checks[name] = bool(condition)
        require(condition, name)

    try:
        spec_path = ROOT/'configs/full151-v2-declared-H123-GPU-objective-spec-v1.json'
        spec = read(spec_path)
        configs = {m: ROOT/f'configs/full151-v3-primary2000-{m}-seed0-v1.json' for m in ('P_ONLY','A2')}
        training = {m: read(p) for m,p in configs.items()}
        fact('actual_two_training_configs_share_exact_binding_spec', all(same_pin(c['contact_binding'], pin(spec_path)) for c in training.values()))
        fact('actual_trained_spec2_authorizes_CPU_without_modifying_GPU_spec', spec['schema']=='full151_declared_H123_objective_spec:2' and spec['CPU_authorized'] is True and spec['GPU_authorized'] is True and spec['execution_scope']=='EXPLORATORY_MATCHED_TRAINING')
        source_pins = {k:v for k,v in spec['expected_code_pins'].items()}
        source_paths = {
            'writer': Path(__file__),
            'prior_hook':ROOT/'src/full151_native/prior_hook_v1.py',
            'learning':ROOT/'src/full151_native/learning_v2.py',
            'runtime':ROOT/'src/full151_native/runtime.py',
            'WM_loader':ROOT/'src/full151_native/world_model_checkpoint.py',
            'actor_consumer_used_inside_WM_loader':ROOT/'src/full151_native/checkpoint.py',
            'health_auditor':ROOT/'scripts/audit_full151_checkpoint_cpu_v2.py',
            'native_runner':NATIVE/'src/dex_supermimic/rl/dex_supermimic/runner.py',
            'physical_worker':ROOT/'src/full151_gradient/serial_probe105_v2.py',
            'original_object_scalar_analyzer_source_only':ROOT/'src/full151_gradient/analysis105_v1.py',
        }
        source_pins.update({k:pin(p) for k,p in source_paths.items()})
        for value in source_pins.values(): checked(value)
        sources = {k:Path(v['path']).read_text(encoding='utf-8-sig') for k,v in source_pins.items() if Path(v['path']).suffix=='.py'}
        for text in sources.values(): ast.parse(text)
        fact('all_selected_source_pins_current_and_Python_AST_valid', True)
        fact('R_math_actual_paths_not_M_lookalike', all(checked(source_pins['frozen__'+name]).samefile(FROZEN/name) for name in ('contact_objective.py','contact_geometry.py','penalty_only_objective.py','contacts.py','__init__.py')))
        fact('actual_binding_inherits_forward_and_exports_both_factories', 'class Full151DeclaredH123ContactObjective(GeometricContactObjective)' in sources['binding'] and 'def build_contact_objective(' in sources['binding'] and 'def build_penalty_only_objective(' in sources['binding'])
        fact('actual_binding_guards_cached_math_module_location', "raise ValueError('Foreign cached math module')" in sources['binding'])
        fact('actual_proxy_exposes_SDF_weights_support_without_extra_forward', all(k in sources['frozen__contact_geometry.py'] for k in ('sampled_signed_distance_m','soft_contact_weight','projected_contact_com_o_m','minimum_signed_distance_m','"support"','"presence"')))
        fact('actual_trained_forward_uses64_not_optional512', 'target = self.reference_support64[frame]' in sources['frozen__contact_objective.py'])
        fact('actual_P_ONLY_direct_P_not_total_minus_support', 'bonus = -.1 * penalty' in sources['frozen__penalty_only_objective.py'])
        fact('native_runner_nextstate_with_preaction_memory', sources['native_runner'].index('r, cage, _ = self.reward_step(x1, m, a, target, contact.sigmoid(), tau, soft=True)') < sources['native_runner'].index('next_memory = self.advance_memory(', sources['native_runner'].index('r, cage, _ = self.reward_step(x1, m, a, target, contact.sigmoid(), tau, soft=True)')))
        fact('actual_prior_only_soft_and_dt_once', 'if soft is not True:' in sources['prior_hook'] and 'DT * bonus' in sources['prior_hook'])
        fact('actual_WM_CPU_input_gradient_not_detached', 'return x1,contact' in sources['WM_loader'] and "not torch.cuda.is_initialized()" in sources['WM_loader'] and 'input_control_autograd_supported' in sources['WM_loader'])

        physical = ROOT/'outputs/full151105_gradient_empirical_gradient05_v2'
        inputs = {
            'objective_spec': pin(spec_path),
            'P_ONLY_training_config': pin(configs['P_ONLY']), 'A2_training_config':pin(configs['A2']),
            'input_bundle':spec['input_bundle'],
            'target':pin(ROOT/'outputs/full151_declared_H123_surface_targets_v1/intent_targets.npz'),
            'target_manifest':spec['binding_manifest'],
            'physical_config':pin(ROOT/'configs/full151105-gradient-empirical_gradient05-v2.json'),
            'physical_report':pin(physical/'physical_report.json'),
            'physical_arrays':pin(physical/'physical_arrays.npz'),
            'branch_index':pin(physical/'branch_index.json'),
            'snapshot_records':pin(physical/'snapshot_records.json'),
            'donor_records':pin(physical/'donor_records.json'),
            'baselines':pin(physical/'baseline_gates.json'),
            'physical_saved_review':pin(ROOT/'outputs/full151105_full_serial_saved_review_v1/report.json'),
            'physical_lifecycle_saved_review':pin(ROOT/'outputs/full151105_empirical_full05_saved_lifecycle_v2/report.json'),
            'prior_object_scalar_analysis_saved_review':pin(ROOT/'outputs/full151105_WM_analysis_saved_independent_v2/report.json'),
            'current_findings':pin(ROOT/'docs/full151_control_gradient_findings_2026-11-05.md'),
            'fixed_A2_checkpoint':pin(ROOT/'outputs/full151_v3_primary2000_A2_seed0_v1/train/model_final.pt'),
            'fixed_A2_health':pin(ROOT/'outputs/full151_v3_primary_A2_final2000_health_v1/report.json'),
            'earlier_WM_loader_compatibility_only':pin(ROOT/'outputs/full151_schema2_pair_WM_loader_CPU_v1/report.json'),
        }
        for value in inputs.values():checked(value)
        fact('fixed_A2_checkpoint_is_predeclared_final2000', inputs['fixed_A2_checkpoint']['sha256']=='70d0a21d5b2f0238ab745264bf24db08515fef65f072cbe00fb6893a33512a0c')
        physical_report=read(physical/'physical_report.json')
        fact('physical_source_scope_preserves_nonqualified_hiddenstate_and_gradient', physical_report['whole_graph_hidden_state_qualified'] is False and physical_report['gradient_qualified'] is False)
        fact('physical_saved_prerequisite_all_checks_true', all(v is True for v in read(checked(inputs['physical_saved_review']))['checks'].values()))

        protocol = {
            'schema':'full151105_trained_geometric_bonus_secant_CPU_preregistration:1',
            'status':'FROZEN_SOURCE_FEASIBLE_NOT_EXECUTED_NOT_AUTHORIZED',
            'created_UTC':datetime.now(timezone.utc).isoformat(),
            'execution_authorized':False,'GPU_run_authorized':False,'training_authorized':False,
            'sequential_exploratory_preregistration':True,
            'already_known_before_this_preregistration':{'native_object_scalar_groups':32,'stable_nonflat':9,'below_noise':23,'sign_agreement_among_stable_each_epsilon':[5,9],'all_epsilon_rows_retained':64,'use_to_choose_bonus_states_floors_or_model':False},
            'inputs':inputs,'source_pins':source_pins,
            'question':'Do the exact trained geometric P_ONLY and A2 functions yield matching empirical absolute-control secants and fixed-A2-WM input AD on the same next20ms states? This is not measured force or contact differentiation.',
            'authority':{
                'objective_factories':'full151_intent.binding_v3.build_penalty_only_objective and build_contact_objective',
                'device':'cpu','spec_is_exact_training_GPU_spec_read_unchanged':True,
                'frozen_math_directory':str(FROZEN),'constructor_only_subclass':'Full151DeclaredH123ContactObjective; inherited R GeometricContactObjective.forward unchanged',
                'P_ONLY_retains_eager_A2_base_geometry':True,'trained_support_basis_count':64,'basis_subset':'512 directions stride8; no512 rescore in this budget',
                'WM':'one strict105 A2 model_final.pt, iteration2000, load_world_model_checkpoint; health producer5950; no new Actor forward',
                'state_input':'saved native58; q20,qd20,p3,R9,v3,angular3; geometry consumes q/p/R only',
                'WM_inputs':'state58 + physicalxi46 + zero174 disturbance; absolute actualu20; no re-C2/EMA/clip/project',
                'native_material_vs_proxy':'objective mu=.1 support representation is unchanged and differs from native mu~.7/condim4. No new material/dynamic-cone eligibility mask.'},
            'dataset':{
                'ordered_labels':['A0/0','A0/49','A0/99','A0/149','A2/0','A2/49','A2/99','A2/149'],
                'state_count':8,'phases':[0,49,99,149],'joint_directions':[0,4,8,12],'eps_rad':[.002,.001],
                'all_physical_nextstates':{'count':160,'order':'donor_next_x[0:8], then branch_x_next[0:152] in saved branch_index row order','baseline_branch_count':24,'perturbed_branch_count':128},
                'exclude_dropped_or_far_or_wrong_contact_states':False,'select_prior_native_object_stable9':False,
                'all_comparisons':'32 dependent state/direction groups per score,64 epsilon pairs per score. Repeated score functions/epsilon rows are not independent samples.',
                'independence':{'source_count':1,'training_seed_count':1,'ego_independent_sources':0,'fresh_action_source_paths':2,'all_states_dependent':True}},
            'phase_contract':{
                'state':'nextstate x1 after four5ms steps/native forward; same native58 target20ms as WM',
                'memory':'float32 [B,1]: start=0,time=k,end=151; independent immutable copies per forward; no advance_memory',
                'reference_frame':'frozen(start+clamp_min(time,0)).minimum(end-1), hence frame=k',
                'last_transition':'k149 active, physical/WM x_next is endpoint150, use target149; no u150; do not zero this score',
                'terminal_target150':'present in source151, outside this primary scoring task',
                'not_original_reward_manager':'original15ms-derived native RewardManager score is secondary data and is not the geometric prior or main target'},
            'score_channels':{
                'raw_penetration_P':'(max(relu(-sampled_sdf_m-.003))/.001)^2, dimensionless normalized squared penetration, unbounded',
                'raw_support_S':'actual official_tolerance_support_match(current64,target64); dimensionless, original live/commanded threshold1e-3,tolerance.1,var.1',
                'weighted_penalty_BP':'actual P_ONLY factory return=-.1*P; no subtraction from A2 total',
                'weighted_support_CS':'phase_weight[k]*S; phase3 means declared active, not measured human motion/contact',
                'total_A2':'actual A2 factory return=CS-.1*P; numerical components and sum residual saved, no manually substituted primary score',
                'dt':'all five primary saved channels are unscaled; additionally .02 times weighted_BP,CS,total_A2 is a derived training reward contribution. dt exactly once on both physical/AD sides; no extra discount/survival scalar.',
                'physical_interpretation':'analytic sampled-proxy score of measured nextstate; not a force/contact-sensor reward measured by physics'},
            'forward_and_AD_budget':{
                'WM_loader_API_calls':1,'WM_forward_calls':1,'WM_forward_batch_size':8,
                'objective_factory_calls':2,
                'actual_objective_forward_calls':4,'objective_batches':['P_ONLY on physical160','A2 on physical160','P_ONLY on WMpred8','A2 on WMpred8'],
                'total_geometry_proxy_forward_calls':4,'geometry_row_evaluations':336,
                'u_AD_calls':4,'u_AD_scalars':['sum(actual_P_ONLY_bonus)','sum(actual_A2_bonus)','sum(A2_phase_weight*support_match)','sum(A2_raw_penetration_prior)'],
                'AD_graph':'same fixedA2 WM prediction/control tensor; retain graph for first3 VJPs, release last; params frozen; per-row independence source-reviewed; no additional forward',
                'raw_support_AD':'phase weights fixed1 for these declared active source frames, rawS and weightedS coincide; verify source table, do not silently divide by zero for future data',
                'Actor_forward':0,'Env':0,'Warp':0,'native_FK_collision_solver_physics':0,'optimizer':0,'input_FD':0,
                'passive_constructor_work_allowed_only_after_separate_run_authorization':'binding factories passively read nominal MJB/actual geometry NPZ, perform deterministic pure Torch FK buffer preparation and NumPy/SciPy ConvexHull sampling. Strict WM loader passively validates MJB and torch-loads checkpoint twice internally, including Actor construction without Actor forward. Record true counts; do not claim zero native imports.'},
            'proxy_output_capture':{
                'method':'read-only register_forward_hook on each actual geometry_proxy, return None; detach/clone capture outputs only, never change original return/AD graph. Hook implementation requires CPU contract before actual study.',
                'all_rows_saved':['sampled_signed_distance_m','soft_contact_weight','support64','presence6','minimum_signed_distance_m','projected_contact_com_o_m','inward_normal_o'],
                'comparisons':'physical donor8 versus WMpred8 paired, plus all branch152; retain full SDF/weights/support arrays and target64; report support live/commanded counts and maxweight, not just aggregate position RMS.',
                'near_surface_diagnostics':'from captured distances only: all sampled signed distances, per-row counts abs(sdf)<=.001m and<=.003m, outside relu(sdf)/.001, max softweight and support64. These bins are descriptive, not selection or qualification.',
                'guard':'4 hook captures with exact batch/order/shapes and no extra geometry calls; if unavailable fail/reportPENDING, do not implement an M lookalike or silently re-evaluate geometry'},
            'secants_and_AD':{
                'raw_physical_numerator':'F64(score_plus_FP32)-F64(score_minus_FP32); never subtract endpoints inFP32 first',
                'slope':'numerator/(2*eps)',
                'primary_AD':'dot(g_bonus_FP32 promotedF64,saved direction__d_eff_F64); d_eff=(plus_FP32-minus_FP32)/(2eps)',
                'nominal_AD_secondary':'same g dot nominal one-hot for joints0,4,8,12; retain clipping and center_offset for every epsilon',
                'control_domain':'actual saved controls/endpoints, no resample/C2/clip repair. Clipping is an orthogonal flag; raw allgroups and unclipped-only secondary summaries both retained.',
                'output_domain':'do not normalize WM R, clip joints, project nextstate to physical manifold, or discard finite dropped/far states. Report nonorthogonality/domain diagnostics without repair.',
                'component_checks':'save P from both actualmodules, factory buffer/deterministic-sample identity, BP+CS versus actualA2; derivative g_BP versus-.1*g_P and g_A2 versus g_CS-.1*g_P as arithmetic diagnostics, not rewrites of primary values'},
            'noise_and_scales_PREDECLARED_NEW_PROPOSED':{
                'provenance':'fixed before any bonus/SDF/support/WM evaluation; constants are new analysis design, not old physical tolerances and not a rigorous cumulative FP32 error theorem',
                'for_each_state_component':'three baselinebranch values b0,b1,b2 and fresh donor d in this component exact units. N_C=max(max(b)-min(b),max_i abs(b_i-d)); scale_C=max(1,abs(d),abs(b0),abs(b1),abs(b2))',
                'eps32':'2^-23','roundoff_proxy':'8*2^-23*scale_C (deterministic scale-aware FP32 representational guard, NOT proof of all arithmetic accumulated error)',
                'P_raw_numerator_floor':'max(1e-4,10*N_P,8*2^-23*scale_P)',
                'BP_weighted_numerator_floor':'.1*floor_P_raw; preserve exact unit scaling; also report actual BP baseline mismatch separately',
                'S_raw_numerator_floor':'max(1e-5,10*N_S,8*2^-23*scale_S)',
                'CS_weighted_numerator_floor':'max(1e-5,10*N_CS,8*2^-23*scale_CS); sourceweights1 verified, retain weighted notation',
                'A2_total_numerator_floor':'max(floor_BP+floor_CS,10*N_A2,8*2^-23*scale_A2), protect component cancellation; never select floor after seeing signagreement',
                'roundoff_diagnostic':'actual_BP baseline noise larger than .1*P floor is a disclosed representation inconsistency; stop eligibility for that channel/state, preserve raw, do not tune floor',
                'dt_floor':'.02*weighted unscaled floor; derived dt rows keep original classification, not a fresh1e-5 reward floor',
                'finite_primary_classification':'for each32group/component: both eps numerators abs(delta)>component floor, same nonzero slope sign,abs(s0-s1)/max(abs(s0),abs(s1),1e-12)<=.5 ->STABLE_NONFLAT; either numerator below/equal floor ->FLAT_OR_BELOW_NOISE; otherwiseNONFLAT_UNSTABLE. Clipped flag orthogonal, all retained.',
                'comparison_denominator':'max(abs(physical_slope),abs(AD_eff),component_floor/(2eps)); no divide by nearzero without the declaredfloor. Save absolute error regardless; ratio/sign eligibility explicitly flagged.',
                'zero_and_invalid':'zeroAD sign is no-sign not counted as agreement with nonzero physicalslope; undefined ratios=null+reason; nonfinite abort affected group with partial retained, never JSONNaN or fabricated zero',
                'all_score_floors_and_scales_saved':True},
            'preflight_and_readonly':{
                'fresh_process':'CUDA_VISIBLE_DEVICES=-1, require not torch.cuda.is_initialized before/after; CPU FP32, AMPFalse; original training GPUAMP gradients are not claimed bit-reproduced',
                'required_new_execution_runplan':'separate unexecuted consumer source/checker/complete transitive sourceclosure must be frozen and source-archived before Root authorizesrun; this protocol is not execution permission or existing loaderREADY',
                'all_inputs_and_sources_current':'strict pin shape, absolute samefile+SHA aliases; spec/training/fixedWM/health/actualxi/source/physicalreview bind; old103/104 and Mmath substitutes reject',
                'read_only':'pre/post file hashes, Python/NumPy/TorchCPU RNG, actual frozenWM state_dict+normalizers, all objective buffers, inputs and spec/targets; eval()/requires_gradFalse parameters, inputgradTrue only',
                'save_before_analysis':'raw channel scores, full proxyoutputs, gradients, controlendpoints/d_eff/center_offset/clipped,loader/factoryreceipts/digests before aggregateclassification',
                'failure':'new output nooverwrite; retain partial/error/source snapshot, allgroups NOT_RUN explicit; no automatic newversion/newphysics/retrain/retry; existing inputs and oldfailedrecords unchanged'},
            'interpretation_limits':[
                'Actual functions used during training, not learned contact estimators. P_ONLY/A2 objectives have no learned parameters; only the fixedA2 WM is learned.',
                'Proxy derivatives include piecewise nearest exterior, max winners,threshold1e-3,3mmhinge and support-max nonsmoothness. Secant/AD disagreement can mix WMerror,geometrickinks,FP32 quantization,controlclipping and unknown graphscratch. Not uniquely attributable to WM.',
                'Aggregate positionRMS versus1mmbandwidth alone proves no proxy mismatch; exact captured support/SDF/weights on all160 measurednextstates versus8WMpredictions must be reported.',
                'Saved sensor/contact/HO force/COM torque remain separate measured annotations. Do not use them as proxy truth, bonus eligibility mask, self15 target or forcegradient.',
                'No wholehiddenstate restore certificate, true infinitesimal physicalJacobian, forceclosure,dynamicfeasibility or taskbenefit qualification follows. Full05 only local finite-interval empirical baseline support.',
                'No stable-group selection from preceding native-object9/32 result; retain allstates, all32groups, both eps and both functiondefinitions.',
                'Nsource1/Nseed1/Nego0, dependent states/eps/functions; no CI/pvalue or general methodclaim. Prepolicy100 trajectories/numerics differed beforeprior, so this local diagnostic cannot establish originaltraining causalbenefit.',
                'Complementary Actor-coefficient study uses other archived inputs and no physicalsecant rescore; keep its prereg and conclusions separate.'
            ]
        }
        fact('no_score_or_model_was_evaluated_by_writer', 'torch' not in __import__('sys').modules and 'mujoco' not in __import__('sys').modules and 'warp' not in __import__('sys').modules)
        write_new(PROTOCOL,protocol)
        report={'schema':'full151105_trained_bonus_source_feasibility:1','status':'PASS_SOURCE_FEASIBILITY_PREREGISTRATION_ONLY_NOT_EXECUTED','checks':checks,'passed':sum(checks.values()),'total':len(checks),'protocol':pin(PROTOCOL),'source_pins':source_pins,'inputs':inputs,'scope':{'source_text_JSON_hashes_only':True,'NPZ_arrays_loaded':0,'checkpoint_deserializations':0,'objective_constructions':0,'objective_or_geometry_forwards':0,'WM_or_Actor_forwards':0,'autograd_calls':0,'Env_Warp_physics_GPU_calls':0,'new_scientific_numbers_generated':False,'run_implementation_ready':False,'source_feasible':True,'score_preregistered_before_evaluation':True,'future_CPU_requires_separate_Root_run_authorization':True}}
        write_new(OUT/'report.json',report)
        write_new(OUT/'READY_source_only.json',{'schema':'full151105_bonus_prereg_source_only:1','report':pin(OUT/'report.json'),'protocol':pin(PROTOCOL),'source_pins':source_pins,'CPU_execution_authorized':False,'GPU_execution_authorized':False,'actual_score_or_gradient_ready':False})
        print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(OUT/'report.json'),'protocol':pin(PROTOCOL),'writer':pin(__file__)},ensure_ascii=False))
    except BaseException as error:
        write_new(OUT/'failure.json',{'status':'FAILED_SOURCE_ONLY_PREREG_WRITER','error':repr(error),'checks':checks,'source':pin(__file__),'scientific_execution':False})
        raise


if __name__=='__main__':main()
