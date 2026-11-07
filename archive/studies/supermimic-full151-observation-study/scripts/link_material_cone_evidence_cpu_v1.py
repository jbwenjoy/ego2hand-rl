"""Link immutable material and native-C algebra evidence, stdlib only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151_actual_material_cone_contract_v1'


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    contract_path = OUT / 'material_cone_contract.json'
    algebra_path = ROOT / 'outputs/native_pyramidal4_algebra_CPU_v3/report.json'
    source_path = ROOT / 'outputs/native_pyramid_binding_source_review_v1/report.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    algebra = json.loads(algebra_path.read_text(encoding='utf-8'))
    checks = {
        'passive_actual_material_contract_17_of_17': contract['passed'] == contract['total'] == 17 and all(contract['checks'].values()),
        'explicit_installed_C_ABI_algebra_9_of_9': algebra['passed'] == algebra['total'] == 9 and all(algebra['checks'].values()),
        'actual_material_scope_distinct_from_arbitrary_decoder_fixture': algebra['material_value_claim'] is False,
        'model_contact_runtime_not_created_in_decoder_fixture': algebra['native_model_or_data_instances'] == 0,
    }
    result = {
        'schema': 'full151_actual_material_native_pyramid_READY:1',
        'status': 'READY_FOR_SEPARATELY_AUTHORIZED_CPU_CONE_SCREEN' if all(checks.values()) else 'NOT_READY',
        'checks': checks,
        'material_contract': pin(contract_path),
        'producer': contract['sources']['producer'],
        'native_C_ABI_algebra': pin(algebra_path),
        'retained_Python_binding_failure': algebra['prior_python_wrapper_failure'],
        'official_matching_binding_source_review': pin(source_path),
        'README': pin(OUT / 'README.md'),
        'linker': pin(__file__),
        'material_binding': {
            'condim_pointer': '/condim',
            'local_template_pointer': '/A_local6x6',
            'local_template_storage': 'contact_components_by_ray_columns',
            'local_component_order': ['force_normal', 'force_tangent1', 'force_tangent2', 'torque_normal', 'torque_tangent1', 'torque_tangent2'],
        },
        'authority_scope': 'Fixed passive material and native cone encoding only. Hypothetical contact frame. Not contact existence, force capacity, dynamics or physical qualification.',
        'new_LP_calls': 0,
        'new_native_queries_or_physics': 0,
    }
    target = OUT / 'READY.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'READY': pin(target)}, indent=2))


if __name__ == '__main__':
    main()
