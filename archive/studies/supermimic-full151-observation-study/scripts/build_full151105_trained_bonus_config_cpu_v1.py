"""Root metadata-only authorizer for one exact trained-bonus CPU comparison."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from full151_gradient import trained_bonus_analysis105_v1 as b


def main():
    protocol_path = ROOT / 'protocols/full151105_trained_geometric_bonus_secant_CPU_prereg_v1.json'
    ready_path = ROOT / 'outputs/full151105_trained_bonus_preparation_CPU_v1/READY.json'
    assert b.pin(protocol_path)['sha256'] == '9d48ae798471638bb6bbc3e47d80112b185de68fd861e9eac5449286428413b7'
    assert b.pin(ready_path)['sha256'] == '96861d975695b3a3714bf8d7f348f2f60e212f07104c5cf0b5cea37d9b49188a'
    protocol = b.read(protocol_path); ready = b.read(ready_path)
    proof = b.read(b.checked(ready['cpu_report'])); b.all_checks(proof)
    assert proof['passed'] == 66
    sources = b.required_sources(protocol)
    sources.update(config_builder=b.pin(__file__), CPU_READY=b.pin(ready_path), CPU_contract=ready['cpu_report'], preregistration=b.pin(protocol_path))
    for value in sources.values(): b.checked(value)
    output = ROOT / 'outputs/full151105_trained_bonus_CPU_v1'
    config = ROOT / 'configs/full151105-trained-bonus-CPU-v1.json'
    assert not output.exists() and not config.exists()
    case = {'schema': b.SCHEMA, 'role': b.ROLE, 'protocol': b.pin(protocol_path), 'output': str(output),
            'CPU_run_authorized': True, 'cpu_threads': 1, 'source_pins': sources, 'cpu_ready': b.pin(ready_path)}
    with config.open('x', encoding='utf-8') as file:
        json.dump(case, file, indent=2, ensure_ascii=False); file.write('\n')
    print(json.dumps({'config': b.pin(config), 'source_pins': len(sources), 'model_or_objective_executed': False}, indent=2))


if __name__ == '__main__':
    main()
