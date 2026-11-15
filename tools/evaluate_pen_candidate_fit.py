"""Read-only fixed-budget fit and native action-response evaluation."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.learning.pen_response import model_from_arm
from supermimic_research.learning.pen_stage import write
from supermimic_research.learning.candidate_response_evaluation import candidate_pairs, evaluate_pairs


def main(a):
    health = json.loads(a.health.read_text())
    cp = verify_file_pin(health['checkpoint'])
    if (health['schema'] not in ['pen_response_continuation_health:1','pen_response_continuation_health:2'] or health['update'] != 2000
            or not health['checks'] or not all(health['checks'].values()) or health['training_running']):
        raise ValueError('Healthy terminal2000 continuation required')
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    for pin in ck['sources'] + ck['parent_sources'] + [ck['prepared'], ck['review'], ck['config']]:
        verify_file_pin(pin)
    prep = json.loads(verify_file_pin(ck['prepared']).read_text())
    review = json.loads(verify_file_pin(ck['review']).read_text())
    if (prep['schema'] != 'pen_candidate_response_preparation:1' or not all(prep['checks'].values())
            or review['passed'] != review['total'] or verify_file_pin(review['prepared']) != verify_file_pin(ck['prepared'])
            or prep['outputs'] != ck['dataset_pins']):
        raise ValueError('Reviewed parent-split candidate dataset required')
    data = {role: dict(np.load(verify_file_pin(pin))) for role, pin in ck['dataset_pins'].items()}
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh unified output required')
    pairs = candidate_pairs(data['CHECK'])
    declared = {p['parent_id'] for p in prep['parents_detail'] if p['role'] == 'CHECK'}
    if set(data['CHECK']['parent_id']) != declared or set(data['TRAIN']['parent_id']) & declared:
        raise ValueError('Explicit parent role separation required')
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a.output.mkdir()
    reports, outputs = {}, {}
    for mode, saved in ck['arms'].items():
        model = model_from_arm(saved,device)
        model.eval()
        blocks = dict(delta52=slice(0, 52), object12=slice(40, 52), warm26=slice(52, 78),
                      source_BODY_wrench6=slice(78, 84), Fn6=slice(84, 90))
        fit = {}
        for role, d in data.items():
            with torch.no_grad():
                p = model.prediction(torch.as_tensor(d['x'], dtype=torch.float32, device=device)).cpu().numpy()
            scale = model.y_scale.cpu().numpy()
            mean = model.y_mean.cpu().numpy()
            fit[role] = {}
            subsets = dict(valid=d['valid'])
            if role == 'CHECK':
                subsets.update(all_including_uncertain=np.ones(len(p), bool), excluded_uncertain=~d['valid'])
            for name, mask in subsets.items():
                result = dict(rows=int(mask.sum()), finite_predictions=int(np.isfinite(p[mask]).all(axis=1).sum()),
                              labels_uncertain=name != 'valid')
                if mask.any() and np.isfinite(p[mask]).all():
                    result['normalized_MSE'] = {b: float(np.mean(((p[mask, s] - d['y'][mask, s]) / scale[s])**2)) for b, s in blocks.items()}
                    result['frozen_TRAIN_mean_MSE'] = {b: float(np.mean(((mean[s] - d['y'][mask, s]) / scale[s])**2)) for b, s in blocks.items()}
                fit[role][name] = result
        response, arrays = evaluate_pairs(model, data['CHECK'], pairs, device)
        path = a.output / (mode + '_responses.npz')
        np.savez_compressed(path, **arrays)
        outputs[mode] = file_pin(path)
        d=data['CHECK'];p=arrays['all_predictions'];per_parent={}
        for parent_id in sorted(set(d['parent_id'].tolist())):
            mask=(d['parent_id']==parent_id)&d['valid']
            row=dict(rows=int(mask.sum()),finite_predictions=int(np.isfinite(p[mask]).all(1).sum()))
            if mask.any() and np.isfinite(p[mask]).all():row['normalized_MSE']={b:float(np.mean(
                ((p[mask,s]-d['y'][mask,s])/scale[s])**2)) for b,s in blocks.items()}
            per_parent[str(parent_id)]=row
        reports[mode] = dict(one_step=fit, physical_finite_response=response,per_CHECK_parent=per_parent)
    result = dict(schema='pen_candidate_fit_evaluation:1', health=file_pin(a.health), checkpoint=file_pin(cp),
                  prepared=ck['prepared'], dataset_pins=ck['dataset_pins'], outputs=outputs, pairs=pairs, models=reports,
                  sources=[file_pin(__file__), file_pin(ROOT / 'src/supermimic_research/learning/candidate_response_evaluation.py')],
                  neural_updates=0, native_integrations=0,
                  scope='All declared CHECK parents and axis pairs. Same-clip development temporal holdout, failed/nonexpert episodes. Three repeats are not IID trials. Finite amplitudes and observed spread do not prove infinitesimal gradients. Excluded labels remain uncertain. No recursive trajectory, Actor, full-task or ego claim.')
    write(a.output / 'report.json', result)
    print(json.dumps(dict(report=file_pin(a.output / 'report.json'), attempted_pairs=len(pairs),
                          eligible_pairs=sum(p['eligible'] for p in pairs),
                          summaries={mode: r['physical_finite_response']['summary'].get('ALL') for mode, r in reports.items()})))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--health', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
