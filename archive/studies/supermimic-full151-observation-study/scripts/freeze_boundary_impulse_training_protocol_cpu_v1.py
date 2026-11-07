"""Root metadata-only freeze of a new, bounded, independently reviewed job."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def pin(p):
    p = Path(p).absolute()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def passed(r):
    assert r['checks'] and all(v is True for v in r['checks'].values())
    assert r['passed'] == r['total'] == len(r['checks'])


def main():
    prior = read(ROOT/'protocols/full151105_boundary_component_initial_parity_CPU_run_v1.json')
    initial_review = ROOT/'outputs/full151105_boundary_component_initial_parity_saved_independent_v3/report.json'
    passed(read(initial_review))
    ready_path = ROOT/'outputs/full151105_boundary_impulse_training_preparation_CPU_v2/READY.json'
    ready = read(ready_path)
    passed(read(ready['cpu_report']['path']))
    assert ready['cpu_checks'] == {'passed': 3, 'total': 3}
    health_path = ROOT/'outputs/full151105_boundary_impulse_checkpoint_auditor_preparation_CPU_v1/READY.json'
    health = read(health_path)
    passed(read(health['cpu_report']['path']))
    assert health['cpu_checks'] == {'passed': 6, 'total': 6}
    review_path = ROOT/'outputs/full151105_boundary_impulse_training_source_independent_v1/report.json'
    source_review = read(review_path)
    passed(source_review)
    assert source_review['source_review_qualified'] is True
    output = ROOT/'outputs/full151105_boundary_impulse_matched_training_CPU_v1'
    assert not output.exists()
    inputs = {k: v for k, v in prior['inputs'].items() if k != 'recipe_draft'}
    inputs.update({
        'initial_parity_report': pin(ROOT/'outputs/full151105_boundary_component_initial_parity_CPU_v1/report.json'),
        'initial_parity_saved_review': pin(initial_review),
        'initial_parity_baseline': pin(ROOT/'outputs/full151105_boundary_component_initial_parity_CPU_v1/unwrapped_baseline_forward152.npz'),
    })
    assert len(inputs) == 12
    sources = dict(prior['source_pins'])
    for label, document in [('trainer', ready), ('checkpoint_health', health)]:
        for key, value in document['source_pins'].items():
            sources[label+'_'+key] = value
        sources[label+'_proof'] = document['cpu_report']
    sources.update({
        'trainer_READY': pin(ready_path), 'checkpoint_health_READY': pin(health_path),
        'training_source_review': pin(review_path),
        'initial_saved_auditor_v3': pin(ROOT/'scripts/audit_boundary_initial_parity_saved_cpu_v3.py'),
        'protocol_builder': pin(Path(__file__)),
    })
    for value in list(inputs.values()) + list(sources.values()):
        assert pin(value['path'])['sha256'] == value['sha256']
    recipe = pin(ROOT/'protocols/full151105_boundary_impulse_matched_training_recipe_v1.json')
    assert recipe['sha256'] == 'ed5e6cc378f80d94e1771607b8fd4ed1198ce31eec80e17c0465d9933ae4f855'
    scientific = {
        'schema': 'full151105_boundary_impulse_matched_training_preregistration:1',
        'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'FROZEN_BEFORE_ACTUAL_NEW_TRAINING', 'recipe': recipe,
        'input_pins': inputs, 'source_pins': sources,
        'comparison': 'Identical actual initial WM, zero14 head, frozen context/norm, full152 order, public losses, AdamW and 1000 update budget. Only consistency coefficient differs: control0/method1. Both compute and log all losses.',
        'paired_checkpoint_iterations': [100, 1000],
        'execution_budget': {'own_initial_torch_load': 1, 'native_WM_constructor': 1, 'raw_WM_deepcopies': 2, 'native_WM_forward': 2000, 'fullbatch_rows': 152, 'predicted_rows': 304000, 'backward': 2000, 'optimizer_steps': 2000, 'extra_validation_forward': 0, 'Actor_119f_MJB_Env_GPU_physics': 0},
        'first_training_forward': 'Each arm must match saved actual e427 initial native58/self15 bits and positive zero14; reuses training forward. Stop on any failure, never relax or automatically retry.',
        'health': 'At first paired100 load exactly own initial_pair_state plus two AFTER100 checkpoints once via separate saved-only reader; audit raw Adam and named PRE100 pre/post clip gradients, frozen norm/context, raw this-run RNG, original updates prefix and all16 native+aux parameters. No model reconstruction or FW.',
        'goal_pause': 'After first paired100 healthy serialized checkpoints and saved ETA, pause Goal per explicit user request. This job is explicitly authorized to continue to fixed1000 independently in background. No additional research after Goal pause.',
        'ETA': 'Measured first100 paired costs; trim first3, predict remaining900 pairs plus measured save and explicit20s finalization allowance. Planning range0.85..1.3, not confidence interval or speedup.',
        'no_resume_retry_or_model_selection': True,
        'no_scientific_success_gate_in_this_health_experiment': True,
        'limits': ['Old152 rows from one cached robot lineage; not independent trajectories or successful human demonstration.', 'r_saved includes rawR9 observer/solver defect; nonzero angular defect is retained.', 'self15 has no public label loss and may change after fitting.', 'No physical-gradient, force-capacity, task success, CHORD benefit, ego generalization or whole-hidden qualification.', 'Original reference and previous full151 task failures remain; subsequent saved metrics must report point/impulse/consistency tradeoffs and actual task evaluation remains pending.'],
    }
    prereg_path = ROOT/'protocols/full151105_boundary_impulse_matched_training_CPU_prereg_v1.json'
    write(prereg_path, scientific)
    protocol = {
        'schema': 'full151105_boundary_impulse_matched_training_CPU_protocol:1',
        'role': 'matched_public_aux_vs_cross_target_consistency',
        'CPU_run_authorized': True, 'GPU_run_authorized': False,
        'automatic_retry': False, 'background_continue_to_final': True,
        'preregistration': pin(prereg_path), 'recipe': recipe,
        'inputs': inputs, 'source_pins': sources,
        'output_prefix': 'full151105_boundary_impulse_matched_training_CPU_',
        'output': str(output),
        'user_pause_condition': scientific['goal_pause'],
    }
    protocol_path = ROOT/'protocols/full151105_boundary_impulse_matched_training_CPU_run_v1.json'
    write(protocol_path, protocol)
    print(json.dumps({'preregistration': pin(prereg_path), 'protocol': pin(protocol_path), 'output': str(output), 'actual_data_model_FW_AD_optimizer': 0}))


if __name__ == '__main__':
    main()
