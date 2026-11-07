"""Write a new source-only preregistration; no backend/model/data conversion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'protocols/full151105-matched-native-GPU-reader-replay-prereg-v1.json'


def pin(p):
    p = Path(p).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    old = ROOT/'outputs/full151105_converter_forensic_CPU_v1'
    pins = json.loads((old/'output_pins.json').read_text())
    forensic = json.loads((old/'report.json').read_text())
    parent = ROOT/'protocols/full151105_solver_phase_penalty_observability_CPU_prereg_v2.json'
    p = {
        'schema': 'full151105_matched_native_GPU_reader_replay_preregistration:1',
        'status': 'FROZEN_SINGLE_MODE_SOURCE_ONLY_NOT_EXECUTION_AUTHORIZATION',
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'GPU_run_authorized': False, 'CPU_conversion_authorized': False,
        'proxy_scoring_authorized': False, 'training_authorized': False,
        'writer': pin(__file__),
        'runplan_schema': 'full151105_matched_native_GPU_reader_replay_run:1',
        'role': 'full151105_native_GPU_state_replay',
        'fixed_mode': 'NATIVE_FP32_NO_AUTOCAST_CUDA_BATCH1',
        'question': 'Does replay of the exact saved inputs through the original unbound native Reader on CUDA with original single-world batch shape satisfy the unchanged full58 endpoint/anchor arithmetic rule?',
        'sequential_prior_knowledge': 'CPU B2260 replay failed original output-scaled8eps32 at five angular components/five endpoints. Conditional rounding envelopes are descriptive, not a certified cause or a replacement gate. This is one predeclared joint backend+B1 replay condition, not blind confirmation or a search over modes.',
        'inputs': {
            'forensic_config': forensic['config'], 'forensic_report': pins['report.json'],
            'forensic_saved_independent23': pin(ROOT/'outputs/full151105_converter_forensic_saved_independent_v1/report.json'),
            'raw_conversion_all2260': pins['raw_conversion_all2260.npz'],
            'endpoint_identities': pins['endpoint_identities.json'], 'anchor_identities': pins['anchor_identities.json'],
            'solver_position1808_old_rule': pins['solver_position1808_old_rule.npz'],
            'solver_rotation1808_old_rule': pins['solver_rotation1808_old_rule.npz'],
            'old_forensic_protocol': pin(ROOT/'protocols/full151105-converter-arithmetic-forensic-cpu-v1.json'),
            'old_dense_preregistration': pin(parent),
            'old_failed_config': pin(ROOT/'configs/full151105-solver-phase-penalty-CPU-v1.json'),
            'old_failed_record': pin(ROOT/'outputs/full151105_solver_phase_penalty_CPU_v1/failure.json'),
            'old_failure_saved_independent15': pin(ROOT/'outputs/full151105_solver_phase_converter_failure_saved_review_v2/report.json'),
            'conditioning_context_only': pin(ROOT/'outputs/full151105_converter_arithmetic_source_review_v1/interpretation_bounds_frozen_v1.json')},
        'source_pins': {
            'native_reader': pin('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/sim.py'),
            'native_state_pack': pin('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py'),
            'native_rotations': pin('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/rotations.py'),
            'physical_worker_source_only': pin(ROOT/'src/full151_gradient/serial_probe105_v2.py'),
            'forensic_consumer_source_only': pin(ROOT/'src/full151_gradient/converter_forensic105_v1.py')},
        'expected_counts': {
            'native_state_calls': 2260, 'native_state_returned': 2260,
            'native_state_batch_size': 1, 'converted_rows': 2260,
            'pre_rows': 1808, 'known_post20_rows': 452, 'endpoint_comparisons': 452,
            'anchor_comparisons': 160, 'solver_pose_comparisons': 1808,
            'native_CPU_state_calls': 0, 'new_MJB_loads': 0,
            'factory_calls': 0, 'objective_WM_Actor_forward': 0,
            'weights_loads': 0, 'AD_calls': 0, 'optimizer_steps': 0,
            'Env_created': 0, 'nativeFK_collision_queries': 0, 'physics_steps': 0},
        'input_and_GPU_ABI': {
            'authority': 'Use ONLY forensic23-validated raw_conversion_all2260 inputs/mapping. Its CPU converted_native58 is a diagnostic comparator, never the replay output.',
            'row_order': '1808 saved solver PRE qpos/qvel rows in original physical_samples order, followed by452 actual saved post20 endpoint rows; fixed authoritative Reader palm pos/R per row from the same artifact.',
            'byte_identity': 'Require finite FP32 qpos27/qvel26/palm3/palmR3x3, exact source bytes and current hashes before upload. Save input readback hashes or raw copies proving identical FP32 values; integer named-index/freejoint mapping is unchanged and separately typed/range/disjoint checked.',
            'explicit_new_ABI': 'Call original SimReader.state unbound with a scoped fakeReader whose env.sim.data.qpos/qvel are CUDA FP32 row views of shape1x27/1x26, robot/obj indexing are same CUDA integer indices, and palm_pos/palm_rot are exact CUDA FP32 row views1x3/1x3x3. Do not instantiate SimReader/Env, call the CPU validation wrapper with GPU tensors, or implement a hand-written replacement state converter.',
            'B1_required': 'Loop exactly2260 rows. Each native state call sees batch dimension1, matching original full05 one-world operator shape. Preloading input buffers on GPU is allowed; a B2260 state call, dummy state warmup, alternate B, precision mode or automatic second replay is forbidden.',
            'execution_context': 'One fresh Root-owned GPU process, CUDA device0 only, original torch.no_grad and explicit disabled CUDA autocast. Input/output/native operations remain FP32. No torch.compile, CUDA graph replay, WM/Actor or objective factory import.',
            'output': 'Each result must be shape1x58, CUDA FP32 and finite. Copy it to CPU without dtype conversion, preserve all2260 rows in a FP32 array and save raw output before eligibility diagnostics.',
            'same_material_and_pose_scope': 'No scene/model is created. MJB mapping is reused from forensic23, not reread or inferred from body IDs. Native body IDs, mocapid0 and qpos/dof addresses are separate namespaces.'},
        'flags_and_runtime': {
            'explicit_source_match': 'Set torch.set_num_threads(4) and torch.backends.cuda.matmul.allow_tf32=True, matching serial_probe105_v2.py:362 source-intended settings. This is a new observed replay setting, not a retrospective observation of historical runtime flags.',
            'record_before_after': ['actual CUDA device/name/capability and torch/CUDA versions', 'all tensor/operation observed dtypes and batch shapes',
                'autocast enabled/device dtype', 'allow_tf32 and float32_matmul_precision',
                'cudnn TF32/deterministic/benchmark', 'deterministic algorithms/warn-only',
                'BF16/FP16 reduced-precision and splitK flags or explicit UNAVAILABLE',
                'CUDA_VISIBLE_DEVICES, NVIDIA_TF32_OVERRIDE, TORCH_ALLOW_TF32_CUBLAS_OVERRIDE, CUBLAS_WORKSPACE_CONFIG'],
            'no_unrecorded_flag_changes': 'Only the two explicit source-match settings above and disabling autocast are prescribed. Record other inherited/default flags; do not tune them to pass.',
            'arithmetic_limit': 'Original native mm/mv use elementwise multiplication plus sum, not GEMM. A changed GPU/B1 result does not isolate TF32, hardware, batch-shape or a unique arithmetic cause.',
            'RNG': 'Capture Python/NumPy/TorchCPU and initialized CUDA RNG state arrays/bytes before/after the replay. Native deterministic algebra must not consume RNG; no manual restore to disguise mutation. Backend initialization effects are separately scoped.'},
        'unchanged_eligibility_rule': {
            'all452_endpoints': 'Replay output rows1808:2260 versus the SAME saved GPU native post20 references. All q20/qd20 exact FP32 bits; all other18 components abs(actual-ref)<=8*eps32*max(1,abs(actual),abs(ref)). Keep every element, row identity, units, signed delta, old bound and mask.',
            'all160_anchors': 'Replay outputs at exact saved anchor_pre_indices versus exact anchor_references. Same q/qd bits and other18 old8eps rule. Do not use endpointk+1 as a pre-anchor.',
            'all1808_solver_pose_source': 'Preserve original saved solver PRE pose versus PRE objectq positions/normalized rotations, all1808 original bounds/masks validated in forensic23. Recompute all saved-array differences/bounds/flags from those raw pose-pair artifacts and bind them to the unchanged raw PRE q/pose and source quaternion formula. This is an inherited saved-pose check, not a new GPU native state call, MJB query or pose-only substitute for full58.',
            'failure': 'ANY q/qd/8eps/solverpose/input/source/finite/count/capacity-qualified-source failure ->state_replay_qualified=False, dense_scoring_eligibility=False, factory/scoring calls0. Do not alter a threshold, select fewer components, fall back to pose-only or pick a second mode.',
            'finite_arithmetic_fail': 'For finite outputs complete the fixed2260 schedule, save every raw result and all masks, then return NOT_QUALIFIED. No scoring follows. Hard engineering/nonfinite/time failures preserve partial rows/completedmask and UNKNOWN/NOT_QUALIFIED; never retry.',
            'success': 'Only all unchanged gates can establish a new specifically matched-backend+B1 replay qualification. Even success grants no scoring authorization in this run; Root must separately review/freeze a downstream CPU proxy-input consumption contract. Old CPU converter FAIL remains immutable.'},
        'required_saved_outputs': [
            'Root config actualpath/SHA plus reserialized copy ownpin/strict parsed equality; full source inventory+ZIP/current/postchecks',
            'exact input/mapping bytes or verified upload/readback hashes and all2260 GPU-output CPU-byte-copy array',
            'endpoint452 and leading160 actual/reference/delta_F64/oldbound_F64/exactbit+8eps flags and allrow identities',
            'all1808 saved source solver pose actual/reference/delta/bound/flags with inherited-evidence scope explicit',
            'native state attempted/returned2260 and percallB1/dtype/device evidence; no fabricated module/physics counters',
            'runtime flags before/after, raw CPU/CUDA RNG bytes, partial rows/mask/checkpoints, stage elapsed costs',
            'state_replay_qualified and dense_scoring_eligibility separate; physical_gradient_qualified=False/wholehiddenstate_qualified=False'],
        'lifecycle': {'soft_seconds': 90, 'owned_watchdog_seconds': 120,
            'scope': 'Whole worker from startup including archive/import/upload/replay/checks; owned watchdog begins at Popen and includes launch. Root uses existing bounded f416/40b lifecycle with shared GPU lock, exact PID+OSCreation and12identity checks.',
            'partial': 'Save completed rows/mask every128 returned rows and finally via bounded same-payload atomic IO; soft/hard/error paths preserve partial, no canonical qualification.',
            'no_retry': True, 'one_process': True},
        'interpretation': ['This run tests native CUDA+B1 arithmetic fidelity on exactly the saved2260 inputs, not dynamics or reconstruction of hidden simulator state.',
            'All452+160 native full58 comparisons remain mandatory; solverpose agreement cannot replace them.',
            'Original CPU2260 replay FAIL, its saved failure audit, and all prior scientific/task failures remain unchanged.',
            'No new contact labels, pose optimization, physics, gradients, learned weights or training; no method-benefit or independent-N inference.',
            'Conditional rounding envelopes and F64 shadows remain descriptive only; no new acceptance bound.']}
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(p, f, indent=2, allow_nan=False)
    print(json.dumps({'protocol': pin(OUT), 'expected_counts': p['expected_counts'], 'GPU_run_authorized': False}, indent=2))


if __name__ == '__main__':
    main()
