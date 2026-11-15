"""Read-only diagnostics of the final fixed-budget pen checkpoint."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.learning.pen_response import PenResponseWM
from supermimic_research.learning.pen_stage import metrics, write
from supermimic_research.learning.pen_diagnostics import rollout_diagnostics, network_derivative_diagnostics


def main(a):
    h = json.loads(a.health.read_text()); cp = verify_file_pin(h['checkpoint'])
    if h['schema'] != 'pen_response_continuation_health:1' or not all(h['checks'].values()) or h['training_running']:
        raise ValueError('Healthy terminal continuation required')
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    for pin in ck['sources'] + ck['parent_sources'] + [ck['prepared'], ck['review'], ck['config']]: verify_file_pin(pin)
    if ck['update'] != 2000 or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Final2000 and fresh unified output required')
    data = {r: dict(np.load(verify_file_pin(p))) for r,p in ck['dataset_pins'].items()}
    device = torch.device('cuda:0'); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    a.output.mkdir(); models = {}; reports = {}; outputs = {}
    for mode, saved in ck['arms'].items():
        m = PenResponseWM(saved['scales'], saved['head_scales'], saved['physics'], width=saved['width'], coupled=saved['coupled']).to(device)
        m.load_state_dict(saved['model'], strict=True); m.eval(); models[mode] = m
        one_step = {}
        for role, d in data.items():
            mask = d['valid']
            one_step[role], _ = metrics(m, torch.as_tensor(d['x'][mask], dtype=torch.float32, device=device),
                                       torch.as_tensor(d['y'][mask], dtype=torch.float64, device=device))
        rollout, arrays = rollout_diagnostics(m, data['CHECK'], device)
        derivative, derivative_arrays = network_derivative_diagnostics(m, data['CHECK'], device)
        path = a.output/(mode+'_diagnostics.npz')
        np.savez_compressed(path, **arrays, **{'derivative_'+k:v for k,v in derivative_arrays.items()})
        outputs[mode] = file_pin(path)
        reports[mode] = dict(one_step=one_step, recorded_control_rollout=rollout, network_derivative=derivative)
    result = dict(schema='pen_response_final_diagnostics:1', health=file_pin(a.health), checkpoint=file_pin(cp),
        sources=[file_pin(__file__), file_pin(ROOT/'src/supermimic_research/learning/pen_diagnostics.py')],
        dataset_pins=ck['dataset_pins'], outputs=outputs, models=reports, optimizer_updates=0, native_integrations=0,
        contract='All contiguous valid CHECK windows at predeclared H1/2/5/10, no crossing episodes. First state/warm observed, later states/warm predicted; actual recorded PD targets and native dt only. No teacher forcing, clipping, reference reset or object aid. Overlapping windows are correlated, not independent samples.',
        limitation='Single fixed initial state/clip with held-out action streams; failed/nonexpert physical episodes. Network AD/FD consistency is not native physical gradient correctness. No closed-loop Actor, task success or ego-generalization result.')
    write(a.output/'report.json', result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'), models=reports)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--health', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
