"""Root-only metadata authorizer for one two-pass GPU precision diagnostic."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def pin(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--ready', required=True, type=Path)
    args = parser.parse_args()
    protocol = read(args.protocol); ready = read(args.ready)
    assert protocol['schema'] == 'full151105_GPU_precision_preregistration:1'
    assert protocol['execution_authorized'] is False
    proof = read(ready['cpu_report']['path'])
    assert proof['passed'] == proof['total'] == len(proof['checks']) > 0
    assert all(v is True for v in proof['checks'].values())
    previous = ROOT / 'configs/full151105-A2-WM-physical-gradient-CPU-v1.json'
    assert pin(previous)['sha256'] == 'c7f84bfbff5e3bd62f8f75f4f297917948cc208bc7d8b7f2796c96c0f76f7cbf'
    old_config = read(previous)
    old_output = Path(old_config['output'])
    sources = {'CPU__' + k: v for k, v in old_config['source_pins'].items()}
    sources.update({'GPU_preparation__' + k: v for k, v in ready['source_pins'].items()})
    sources.update({'GPU_prereg__' + k: v for k, v in protocol['source_pins'].items()})
    sources.update(config_builder=pin(__file__), GPU_protocol=pin(args.protocol),
                   GPU_CPU_READY=pin(args.ready), GPU_CPU_contract=ready['cpu_report'])
    for value in sources.values():
        actual = pin(value['path'])
        assert actual['sha256'] == value['sha256'] and Path(value['path']).samefile(actual['path'])
    output = ROOT / 'outputs/full151105_A2_GPU_precision_v1'
    config = ROOT / 'configs/full151105-A2-GPU-precision-v1.json'
    assert not output.exists() and not config.exists()
    c = dict(schema='full151105_GPU_precision_diagnostic_run:1', role='full151105_GPU_precision_diagnostic',
             evaluation_seed=0, device='cuda:0', allow_tf32=True,
             modes=['FP32_STORAGE_TF32_ALLOWED', 'BF16_NET_FP32_INTEGRATE_TF32_ALLOWED'],
             GPU_run_authorized=True, training_authorized=False, Env_or_physics_authorized=False,
             automatic_retry=False, compile_fallback=False, soft_worker_seconds=60, hard_process_seconds=90,
             output=str(output), protocol=pin(args.protocol), source_pins=sources,
             CPU_ready=pin(args.ready), CPU_contract_report=ready['cpu_report'],
             CPU_prerequisite_config=pin(previous), CPU_report=pin(old_output / 'report.json'),
             CPU_saved_review=pin(ROOT / 'outputs/full151105_WM_analysis_saved_independent_v2/report.json'),
             CPU_output_pins=pin(old_output / 'output_pins.json'),
             CPU_predictions=pin(old_output / 'WM_predictions_gradients.npz'),
             scope='One Root-owned isolated GPU process; one original strict CPU WM validation and two explicit native GPU modes on the same eight states, two uAD calls. No Env, physics, Actor forward, optimizer, compile, retry or hidden/physical-gradient certificate. Single-pass backend/precision differences retain unquantified repeatability limits.')
    with config.open('x', encoding='utf-8') as file:
        json.dump(c, file, indent=2, ensure_ascii=False); file.write('\n')
    print(json.dumps({'config': pin(config), 'source_pins': len(sources), 'GPU_executed': False}, indent=2))


if __name__ == '__main__':
    main()
