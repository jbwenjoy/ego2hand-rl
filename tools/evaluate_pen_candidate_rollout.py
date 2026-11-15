"""Recorded-control recursion on actual development tails, no simulation/fit."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.observed_parent_records import pack_observed_controls
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.learning.pen_stage import write
from supermimic_research.learning.pen_response import PenResponseWM
from supermimic_research.learning.pen_diagnostics import rollout_diagnostics, window_indices


def load(pin):
    return dict(np.load(verify_file_pin(pin)))


def main(a):
    h = json.loads(a.health.read_text())
    cp = verify_file_pin(h['checkpoint'])
    if (h['schema'] != 'pen_response_continuation_health:1' or h['update'] != 2000
            or not all(h['checks'].values()) or h['training_running']):
        raise ValueError('Healthy terminal2000 required')
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    for pin in ck['sources'] + ck['parent_sources'] + [ck['prepared'], ck['review'], ck['config']]:
        verify_file_pin(pin)
    prep = json.loads(verify_file_pin(ck['prepared']).read_text())
    parent_path = verify_file_pin(prep['parents'])
    parent = json.loads(parent_path.read_text())
    if parent['response_data_role'] != 'FRESH_WM_PARENTS' or a.output.exists():
        raise ValueError('Fresh output and explicit parent cohort required')
    if not a.output.resolve().is_relative_to(ROOT / 'runs'):
        raise ValueError('Unified runs destination required')
    packed, cases, inputs = [], [], [file_pin(parent_path)]
    for case_id, c in enumerate(parent['cases']):
        z, physical = load(c['trace']), load(c['GPU_physical_context'])
        m = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        inputs += [c['trace'], c['GPU_physical_context'], c['host_topology']]
        sources = {p['k']: load(p['source']) for p in c['planning']}
        inputs += [p['source'] for p in c['planning']]
        jids = np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1, 6) for j in range(1, 5)])
        qids, dofs = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
        aids = np.array([np.flatnonzero(m.actuator_trnid[:, 0] == j)[0] for j in jids])
        decoded, labels = decode_saved_case(c, z, m, physical)
        N = len(z['post_qpos'])
        Fn, nonhand = np.zeros((N, 6)), np.zeros(N)
        body = c['model_layout']['object_body']
        for row in z['raw_contact_tick_plus23']:
            tick, g, other = map(int, row[:3])
            og, oh = m.geom_bodyid[g] == body, m.geom_bodyid[other] == body
            if og == oh:
                continue
            group = normal_group(m.geom(other if og else g).name or '')
            if group < 0:
                nonhand[tick] += max(0., float(row[18]))
            else:
                Fn[tick, group] += max(0., float(row[18]))
        tick_valid = ((np.linalg.norm(labels['force_residual_N'], axis=1) < 1e-4)
                      & (np.linalg.norm(labels['torque_residual_Nm'], axis=1) < 1e-5)
                      & (np.linalg.norm(labels['linear_impulse_residual_N_s'], axis=1) < 5e-7)
                      & ~z['overflow'].astype(bool) & (nonhand <= 1e-9)
                      & ~np.any(z['post_qfrc_applied'] != 0, axis=1)
                      & ~np.any(z['post_xfrc_applied'] != 0, axis=(1, 2)))
        if not decoded['checks']['valid_source_indices'] or not decoded['checks']['local_labels_exact']:
            raise ValueError('EFC indexing/local labels invalid')
        data = pack_observed_controls(c, z, sources, qids, dofs, aids, prep['physics'],
                                      labels['world_COM_wrench'], Fn, tick_valid, start=8)
        data['case_id'] = np.full(len(data['x']), case_id)
        packed.append(data)
        cases.append(dict(name=c['name'], dt=c['dt'], EFC=decoded, actual_intervals=c['actual_intervals'],
                          tail_intervals=len(data['x']), valid=int(data['valid'].sum()),
                          source_task_qualified=c['summary']['qualified'], last_control_excluded='Next actual warmstart unobserved'))
    data = {key: np.concatenate([d[key] for d in packed]) for key in packed[0]}
    a.output.mkdir()
    path = a.output / 'actual_tail_records.npz'
    np.savez_compressed(path, **data)
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reports, outputs = {}, {}
    for mode, saved in ck['arms'].items():
        model = PenResponseWM(saved['scales'], saved['head_scales'], saved['physics'],
                              width=saved['width'], coupled=saved['coupled']).to(device)
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        reports[mode], arrays = rollout_diagnostics(model, data, device)
        target = a.output / (mode + '_rollout.npz')
        np.savez_compressed(target, **arrays)
        outputs[mode] = file_pin(target)
    attempted = {}
    for horizon in (1, 2, 5, 10):
        attempted[str(horizon)] = dict(attempted=sum(max(0, len(d['x']) - horizon + 1) for d in packed),
                                       quality_valid=len(window_indices(data, horizon)))
    sources = [file_pin(__file__), file_pin(ROOT / 'src/supermimic_research/mechanics/observed_parent_records.py'),
               file_pin(ROOT / 'src/supermimic_research/learning/pen_diagnostics.py')]
    result = dict(schema='candidate_fit_actual_tail_recursion:1', health=file_pin(a.health), checkpoint=file_pin(cp),
                  prepared=ck['prepared'], inputs=inputs, sources=sources, records=file_pin(path),
                  cases=cases, window_coverage=attempted, models=reports, outputs=outputs,
                  optimizer_updates=0, native_integrations=0,
                  scope='All actual consecutive controls k8..K-2 in the four same development parent episodes. Actual source snapshot memory, never post_qacc. Only first state/warm observed in each window, actual held PD/dt supplied. No teacher forcing/reset/clipping/aid. Quality-masked, overlapping correlated windows; not untouched episodes, Actor task success or ego results.')
    write(a.output / 'report.json', result)
    print(json.dumps(dict(report=file_pin(a.output / 'report.json'), coverage=attempted,
                          errors={mode: {H: dict(windows=r['windows'], finite=r['finite_windows'],
                              COM_m=r.get('model', {}).get('COM_error_m'), angle_deg=r.get('model', {}).get('angle_error_deg'))
                              for H, r in model.items()} for mode, model in reports.items()})))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--health', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
