"""Publish the bounded public15 final-evaluation SOURCE review; stdlib only."""
from __future__ import annotations
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
OUT = ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_source_independent_v1'

def pin(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def check_pin(value):
    assert pin(value['path'])['sha256'] == value['sha256'], value

def main():
    assert not OUT.exists(), 'Fresh review output required'
    own = {
        'consumer': ('src/full151_gradient/boundary_public_self15_final_evaluation105_v1.py', '81ad78161c75b1720a77aead414ccb6c5a447b0f614e5e597a940aaa1ea77419'),
        'CLI': ('scripts/evaluate_full151_boundary_public_self15_components_final_cpu_v1.py', '551f3d803a63d21cba59936a87795cc1028d0ff1e5f5365788cc74009fd5a1ad'),
        'checker': ('scripts/check_full151_boundary_public_self15_components_final_cpu_v1.py', 'd2e7fc550a5bf0515af241469ee81cf9bc0739126b5f90cecd5781cdd2c61cfc'),
        'statistics': ('scripts/describe_public_self15_predictions_saved_cpu_v1.py', 'fb4465bcf12ae98db966420ce24b25961673f523353119ffa4ec9a91843cc2c8'),
        'scientific_contract': ('docs/full151_boundary_public_self15_final_evaluation_contract_v1.md', '94268782d3f1cb282e123145cd85245d8f51ee8eabddc9e872c279e27e4d9046'),
        'public15_loss_helper': ('src/full151_gradient/boundary_public_self15_training105_v1.py', '796c360539961143fe1f588fd18eb08e4b168d141b34daee23da802a72194e7e'),
    }
    sources = {}
    for key, (relative, expected) in own.items():
        sources[key] = pin(ROOT/relative)
        assert sources[key]['sha256'] == expected, key
    evaluator_ready = pin(ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_preparation_CPU_v1/READY.json')
    assert evaluator_ready['sha256'] == 'd93b25e56a2194ae6e62d65abeb4a5c136e1ad17de3aef8e31aed536e407edd8'
    statistics_ready = pin(ROOT/'outputs/full151105_public_self15_statistics_independent_preparation_v1/READY.json')
    assert statistics_ready['sha256'] == '5d18826456e10544ec5fa4c31b75de2d500ef593494b2e53866527f38e4efcc0'
    er, sr = read(evaluator_ready['path']), read(statistics_ready['path'])
    for label, ready, n in [('evaluator', er, 3), ('statistics', sr, 1)]:
        assert ready['cpu_checks'] == {'passed': n, 'total': n}
        for k, value in ready['source_pins'].items():
            check_pin(value)
            sources[label+'_'+k] = value
        check_pin(ready['cpu_report'])
        report = read(ready['cpu_report']['path'])
        assert report['passed'] == report['total'] == n
        assert len(report['checks']) == n and all(type(v) is bool and v for v in report['checks'].values())
    health = pin(ROOT/'outputs/full151105_boundary_public_self15_final1000_saved_independent_v1/report.json')
    assert health['sha256'] == '8566053e9cfdad2024575b0fe9a67a12e3f9c5f2d0d9190fbf95d1c2c25b8619'
    h = read(health['path'])
    assert h['schema'] == 'full151105_boundary_public_self15_checkpoint_saved_review:1'
    assert h['status'] == 'PASS_SAVED_PAIRED_PUBLIC_SELF15_COMPONENT_HEALTH_NOT_ACCURACY'
    assert h['final_iteration_scope'] is True and h['iteration'] == 1000
    assert all(type(v) is bool and v for v in h['checks'].values())
    text = Path(sources['consumer']['path']).read_text(encoding='utf-8')
    tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    segment = lambda name: ast.get_source_segment(text, functions[name])
    metadata, load, forward, run = map(segment, ['metadata_preflight','load_final_into','forward_three','run_evaluation'])
    helper = Path(sources['public15_loss_helper']['path']).read_text(encoding='utf-8')
    stats = Path(sources['statistics']['path']).read_text(encoding='utf-8')
    ast.parse(stats)
    checks = {
        'frozen_current_sources_and_delegated_3_and_independent_1_proofs_bound_without_rerun': True,
        'strict_17_alias_new_family_fixed_AFTER1000_actual_health_and_protocol_inputs_map':
            'Exact17 semantic inputs' in text and 'Actual final1000 saved health, not accuracy' in metadata
            and "protocol_key='inputs' if key=='input_pins' else key" in load
            and "payload['checkpoint_tag']=='AFTER_OPTIMIZER_I'" in load,
        'original152_label_only_JSON_identity_and_old5_new15_combined_training_digests':
            'Only exact label-only JSON' in metadata and 'trainer.validate_target_identity_arrays(identities,gids,substeps,np)' in run
            and 'trainer.cached_label_rows' in run and 'Exact original152 inputs/old5 labels/cached15 labels match actual completed training' in run,
        'three_own_loads_one_strict_native_ctor_two_pre_hook_rawcopies_three_B152_two_heads_no_AD':
            run.count('torch.load(') == 2 and 'for arm in ARMS:payloads[arm]=torch.load' in run
            and run.count('h8.make_initial_eval_component(') == 1 and run.count('component.make_matched_components(') == 1
            and 'with torch.inference_mode():' in forward and 'raw(x,u,dynamics)' in forward
            and 'modules[arm](x,u,dynamics)' in forward and 'target' not in forward,
        'baseline_zero_aux_algebraic_with_actual_raw58_self15_cached_bit_parity':
            "torch.zeros((152,2,6),dtype=torch.float32)" in forward and "torch.zeros((152,2),dtype=torch.float32)" in forward
            and 'Raw initial same saved function/precision; no extra FW' in run,
        'frozen796c_eight_loss_fields_native_row_pair_FP32_public_order_and_capture_aliases':
            "point_row=squared.mean(-1);point=point_row.mean()" in helper
            and "native_pair=native_row.mean();self_scalar=self_row.mean()" in helper
            and "loss['L_mode_FP32']=mode_elements.mean()" in helper
            and "public=(native_pair+r['lambda_mode']*loss['L_mode_FP32'])+r['lambda_imp']*loss['L_imp_F64']" in helper
            and "'native_point_delta52_FP32':operands['native_D52_FP32']" in text,
        'captured_native_point_D_pred_target_not_common_pre_Log_replay':
            "delta=difference(values['prediction58_FP32'],target)" in run
            and 'chart=c.chart_guard' in run and 'common_pre_losses' not in run,
        'normalizer_context_input_raw_RNG_storage_and_weight_digest_evidence_scopes':
            'normalizer_buffers_before.npz' in run and 'normalizer_buffers_after.npz' in run
            and 'physical_context_buffers.npz' in run and 'parity.named_storage_inventory(models)' in run
            and "h8.cf.save_raw_rng(out,'before'" in run and "h8.cf.save_raw_rng(out,'after'" in run
            and 'state_digests_before' in run and 'state_digests_after' in run,
        'statistics_39_cohorts_456_rows_native52_R9_units_self15_recall_and_distinct_F64_rescore':
            'all39_public15_model_cohort_summary.csv' in stats and 'all456_public15_model_row_metrics.csv' in stats
            and 'per_feature_positive_recall' in stats and 'per_feature_positive_precision' in stats
            and 'captured_native_pair_row_FP32_mean_F64' in stats and 'full_public_F64_rescore' in stats
            and 'F64 rescoring from captured FP32 nativeD/logits' in stats,
        'cached15_phase_no_generalization_physics_gradient_or_winner_upgrade':
            'cached lastsolver~15ms sensors, not post20 geometry' in run
            and 'different consistency-weighted total is not a winner criterion' in run
            and "'physical_force_or_gradient_qualified':False" in run,
    }
    assert all(type(v) is bool and v for v in checks.values()), checks
    sources['source_reviewer'] = pin(__file__)
    assert not any(name in sys.modules for name in ('numpy','torch','mujoco','warp'))
    OUT.mkdir()
    snapshot = OUT/'reviewed_source_bytes'
    snapshot.mkdir()
    for k in own:
        shutil.copyfile(sources[k]['path'], snapshot/(k+'_'+Path(sources[k]['path']).name))
    report = {
        'schema':'full151105_boundary_public_self15_final_evaluation_source_independent:1',
        'status':'PASS_SOURCE_AND_BOUND_PURE_PROOFS_NOT_ACTUAL_EVALUATION',
        'checks':checks,'passed':len(checks),'total':len(checks),'source_review_qualified':True,
        'source_pins':sources,'scientific_contract':sources['scientific_contract'],
        'evaluator_READY':evaluator_ready,'evaluator_cpu_report':er['cpu_report'],
        'independent_statistics_READY':statistics_ready,'independent_statistics_cpu_report':sr['cpu_report'],
        'actual_final_health_metadata':health,
        'reviewed_function_lines':{name:{'line':node.lineno,'end_line':node.end_lineno} for name,node in functions.items()},
        'actual_predictions_or_new_metric_arrays_read':False,'actual_weights_loaded':0,'actual_model_FW_AD_physics':0,
        'scope':[
            'Source review plus hash binding of delegated evaluator 3/3 and independently executed synthetic statistics 1/1; neither proof rerun here.',
            'No evaluator metadata_preflight or run called. Root must bind its actual authorized protocol and perform actual run separately.',
            'Native D52 and its quaternion Log are source/capture evidence; no independent native Log calculation. Nonrotation coordinates use target minus prediction.',
            'Cached native self15 target is approximately lastsolver 15ms, not POST20 geometry/HO or human contact truth.',
            'Runtime model-weight/context integrity digests remain producer assertions until actual artifacts exist; raw saved norm/RNG/output bits permit later arithmetic review.',
            'All 152 known training rows and overlapping 13 cohorts; no unseen-trajectory, task, physical-gradient, hidden-state, or accuracy qualification.'
        ],
    }
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(OUT/'report.json'),'passed':len(checks),'total':len(checks)}))

if __name__ == '__main__':
    main()
