"""Root metadata-only final-component evaluation preregistration and job."""
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


def allchecks(value):
    assert value['checks'] and all(v is True for v in value['checks'].values())
    assert value['passed'] == value['total'] == len(value['checks'])


def main():
    training_path = ROOT/'protocols/full151105_boundary_impulse_matched_training_CPU_run_v1.json'
    training = read(training_path);old = training['inputs']
    run = Path(training['output'])
    assert read(run/'result.json')['status'] == 'COMPLETED_FIXED1000_MATCHED_COMPONENT_TRAINING'
    health_path = ROOT/'outputs/full151105_boundary_impulse_final1000_saved_independent_v1/report.json'
    health = read(health_path);allchecks(health)
    assert health['iteration'] == 1000 and health['final_iteration_scope'] is True and health['own_CPU_checkpoint_loads'] == 3
    ready_path = ROOT/'outputs/full151105_boundary_component_final_evaluation_preparation_CPU_v2/READY.json'
    ready = read(ready_path);allchecks(read(ready['cpu_report']['path']))
    source_review_path = ROOT/'outputs/full151105_boundary_component_final_evaluation_source_independent_v2/report.json'
    source_review = read(source_review_path);allchecks(source_review)
    assert source_review['source_review_qualified'] is True
    contract = pin(ROOT/'docs/full151_boundary_component_final_evaluation_contract_v1.md')
    assert contract['sha256'] == '3ad8c12324e37795fc13217d699f323ac99a88719bdb8cfa6f9bfada01b899f6'
    stats = pin(ROOT/'scripts/describe_boundary_component_predictions_saved_cpu_v1.py')
    assert stats['sha256'] == '7487e16b819271fd3b6d3d6562d4802fd97216afa4e3f666f705948f88cce036'
    stats_ready_path = ROOT/'outputs/full151105_boundary_component_statistics_contract_preparation_v1/READY.json'
    stats_ready = read(stats_ready_path)
    allchecks(read(stats_ready['cpu_report']['path']))
    inputs = {k: old[k] for k in ('target_report','target_arrays','target_saved8','target_source_inventory','initial_validation','initial_parity_report')}
    inputs.update({
        'target_identities': pin(Path(old['target_report']['path']).parent/'group152_identities.json'),
        'original_initial_component': old['initial_component'],
        'initial_saved_parity10': old['initial_parity_saved_review'],
        'initial_saved_baseline': old['initial_parity_baseline'],
        'training_protocol': pin(training_path), 'training_result': pin(run/'result.json'),
        'final_pair_manifest': pin(run/'paired_checkpoint_1000.json'), 'final_saved_health5': pin(health_path),
    })
    assert len(inputs) == 14
    sources = dict(training['source_pins'])
    for k, v in ready['source_pins'].items():sources['final_eval_'+k] = v
    sources.update(scientific_contract=contract, saved_statistics=stats,
                   final_eval_READY=pin(ready_path), final_eval_proof=ready['cpu_report'],
                   final_eval_source_review=pin(source_review_path), protocol_builder=pin(Path(__file__)))
    sources['saved_statistics_READY'] = pin(stats_ready_path)
    sources['failed_eval_v1'] = pin(ROOT/'outputs/full151105_boundary_component_final_evaluation_CPU_v1/failure.json')
    sources['eval_v1_ABI_diagnosis'] = pin(ROOT/'outputs/full151105_boundary_final_eval_adapter_failure_diagnosis_v1.json')
    for k, v in stats_ready.get('source_pins', {}).items():sources['saved_statistics_'+k] = v
    for k in ('cpu_report', 'report'):
        if k in stats_ready and type(stats_ready[k]) is dict and set(stats_ready[k]) == {'path','sha256'}:
            sources['saved_statistics_proof'] = stats_ready[k]
    for value in list(inputs.values())+list(sources.values()):
        assert pin(value['path'])['sha256'] == value['sha256']
    output = ROOT/'outputs/full151105_boundary_component_final_evaluation_CPU_v2'
    assert not output.exists()
    prereg = {
        'schema': 'full151105_boundary_component_final_evaluation_preregistration:1',
        'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'FROZEN_BEFORE_ACTUAL_AFTER1000_EVALUATION',
        'previous_failed_v1_preserved': sources['failed_eval_v1'],
        'compatibility_correction': 'Checkpoint input_pins maps explicitly to actual protocol inputs; source_pins maps to source_pins. V1 consumed3loads/1ctor/2rawcopies but zeroFW. V2 scientific metrics/weights/cohort unchanged; no prior predictions exist.',
        'contract': contract, 'input_pins': inputs, 'source_pins': sources,
        'roles': ['unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency'],
        'budget': {'own_initial_plus_final_torch_loads':3,'native_WM_constructors':1,'raw_WM_deepcopies':2,'native_B152_FW':3,'aux_head_FW':2,'predicted_rows':456,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0},
        'baseline_auxiliary': 'Algebraic positive-zero initial template already actual-function-qualified; no third head FW. Initial native58/self15 from this job must bitmatch e427 saved baseline without extra FW.',
        'metrics': 'All152 and13 fixed overlapping cohorts. Common captured nativeFP32 point/mode, F64 z/impulse/cross-target/public losses; raw-unit six blocks and R9 separately; both-source J/K and totals; strict mode>0, zero-negative/ties; self15 initial drift and nativeFP32 soft penalty proxy. Saved-only F64 rescores explicitly distinct from nativeFP32 captures.',
        'no_new_self15_true_label_in_this_native_job': True,
        'known_data_scope': 'Fixed training152 old cached robot lineage, 24baseline+128direction, A0/A2 x k0/49/99/149. No donor/fresh enrollment or unseen trajectories; PRE1000 logs are weightsafter999, this eval is AFTER1000. No posthoc checkpoint selection/weight tuning or statistical significance.',
        'failure_policy': 'Preserve original outputs/evidence, whole-cohort failure on required-domain/finite/source gate. No row filtering, projection, teacher repair, automaticretry or extraFW.',
        'success_scope': 'Only complete fixed saved-component prediction and descriptive metrics. No task/CHORD/Actor, physicalgradient, forcecapacity, wholehidden, humanforce, ego or generalization qualification.',
        'next': 'Independent saved-only full-cohort arithmetic review before scientific interpretation; cachedsensor15 label evaluation is a separately preregistered saved-only stage with its own phase semantics and no additional NN FW.',
    }
    prereg_path = ROOT/'protocols/full151105_boundary_component_final_evaluation_CPU_prereg_v2.json'
    write(prereg_path, prereg)
    protocol = {
        'schema':'full151105_boundary_component_final_evaluation_CPU_protocol:1',
        'role':'fixed_initial_and_two_final1000_component_evaluation',
        'CPU_run_authorized':True,'GPU_run_authorized':False,'training_authorized':False,
        'automatic_retry':False,'preregistration':pin(prereg_path),'inputs':inputs,'source_pins':sources,
        'output_prefix':'full151105_boundary_component_final_evaluation_CPU_','output':str(output),
        'budget':prereg['budget'],'no_accuracy_gate_or_automatic_winner':True,
    }
    protocol_path = ROOT/'protocols/full151105_boundary_component_final_evaluation_CPU_run_v2.json'
    write(protocol_path, protocol)
    print(json.dumps({'preregistration':pin(prereg_path),'protocol':pin(protocol_path),'output':str(output),'actual_tensor_loads_FW':0}))


if __name__ == '__main__':
    main()
