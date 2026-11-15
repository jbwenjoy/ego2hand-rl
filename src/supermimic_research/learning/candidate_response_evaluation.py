"""Read-only native finite-response checks at identical saved parent inputs."""
import numpy as np
import torch

from .pen_native_response import compare_native_response, vector_comparison
from .pen_response_evaluation import pooled_error


BLOCKS = dict(object12=slice(40, 52), source_BODY_wrench6=slice(78, 84), Fn6=slice(84, 90))


def candidate_pairs(data):
    """Keep all attempted axis pairs, including invalid or clipped zero spans."""
    x, y = data['x'], data['y']
    if x.shape != (len(x), 102) or y.shape != (len(x), 90):
        raise ValueError('Explicit candidate102/90 chart required')
    rows = []
    for parent in np.unique(data['parent_id']):
        ids = np.flatnonzero(data['parent_id'] == parent)
        lookup = {(int(data['repeat'][i]), int(data['candidate'][i])): int(i) for i in ids}
        if len(ids) != 171 or len(lookup) != 171 or set(lookup) != {
                (r, a) for r in range(3) for a in range(57)}:
            raise ValueError('Each parent needs exactly57 candidates and3 repeats')
        for axis in range(20):
            ip = [lookup[r, axis + 1] for r in range(3)]
            im = [lookup[r, axis + 21] for r in range(3)]
            xp, xm = x[ip], x[im]
            same = bool(np.array_equal(xp[:, :81], np.broadcast_to(xp[0, :81], (3, 81)))
                        and np.array_equal(xm[:, :81], xp[:, :81])
                        and np.all(xp[:, -1] == xp[0, -1]) and np.all(xm[:, -1] == xp[0, -1]))
            repeated_controls = bool(np.array_equal(xp[:, 81:101], np.broadcast_to(xp[0, 81:101], (3, 20)))
                                     and np.array_equal(xm[:, 81:101], np.broadcast_to(xm[0, 81:101], (3, 20))))
            half = (xp[0, 81:101] - xm[0, 81:101]) / 2
            epsilon = float(np.linalg.norm(half))
            checks = dict(exact_source_warm_dt=same, exact_repeated_PD=repeated_controls,
                          shared_valid=bool(np.all(data['valid'][ip + im])),
                          finite=bool(np.isfinite(x[ip + im]).all() and np.isfinite(y[ip + im]).all()),
                          nonzero_PD_span=bool(epsilon > 0))
            rows.append(dict(parent_id=int(parent), axis=axis, plus_indices=ip, minus_indices=im,
                             phase=int(data['phase'][ids[0]]), dt=float(xp[0, -1]),
                             PD_half_span_rad=epsilon, checks=checks, eligible=all(checks.values()),
                             midpoint=((xp[0] + xm[0]) / 2).tolist(),
                             PD_direction=(half / epsilon if epsilon > 0 else np.zeros(20)).tolist()))
    return rows


def evaluate_pairs(model, data, pairs, device):
    """Batch model JVPs; native targets are independently recorded, never fit."""
    eligible = [row for row in pairs if row['eligible']]
    if not eligible:
        return dict(attempted_pairs=len(pairs), eligible_pairs=0, rows=[], summary={}), {}
    x = torch.as_tensor(np.array([r['midpoint'] for r in eligible]), dtype=torch.float32, device=device)
    v = torch.zeros_like(x)
    v[:, 81:101] = torch.as_tensor(np.array([r['PD_direction'] for r in eligible]), dtype=x.dtype, device=device)
    _, jvp = torch.autograd.functional.jvp(model.prediction, x, v, create_graph=False)
    with torch.no_grad():
        prediction = model.prediction(torch.as_tensor(data['x'], dtype=torch.float32, device=device))
    ad, prediction = jvp.detach().cpu().numpy(), prediction.cpu().numpy()
    scales = model.y_scale.detach().cpu().numpy()
    rows = []
    for index, pair in enumerate(eligible):
        ip, im, epsilon = pair['plus_indices'], pair['minus_indices'], pair['PD_half_span_rad']
        native = (data['y'][ip].mean(0) - data['y'][im].mean(0)) / (2 * epsilon)
        secant = (prediction[ip[0]] - prediction[im[0]]) / (2 * epsilon)
        row = {k: pair[k] for k in ['parent_id', 'axis', 'phase', 'dt', 'PD_half_span_rad']}
        row['model_finite'] = bool(np.isfinite(ad[index]).all() and np.isfinite(secant).all())
        row['AD'], row['secant'] = {}, {}
        if row['model_finite']:
            for block, sl in BLOCKS.items():
                row['AD'][block], _ = compare_native_response(
                    data['y'][ip, sl], data['y'][im, sl], ad[index, sl], scales[sl], epsilon)
                row['secant'][block] = vector_comparison(secant[sl] / scales[sl], native[sl] / scales[sl])
        rows.append(row)
    summary = {}
    for parent in ['ALL'] + sorted({r['parent_id'] for r in rows}):
        selected = [r for r in rows if parent == 'ALL' or r['parent_id'] == parent]
        summary[str(parent)] = dict(attempted_eligible=len(selected), finite=sum(r['model_finite'] for r in selected))
        for method in ['AD', 'secant']:
            summary[str(parent)][method] = {}
            for block in BLOCKS:
                finite = [r for r in selected if r['model_finite']]
                signal = [r for r in finite if r['AD'][block]['contrast_exceeds_observed_spread']]
                summary[str(parent)][method][block] = dict(
                    all_finite=pooled_error([r[method][block] for r in finite]),
                    exceeds_observed_spread=pooled_error([r[method][block] for r in signal]))
    return dict(attempted_pairs=len(pairs), eligible_pairs=len(eligible), rows=rows, summary=summary), dict(
        JVP=ad, all_predictions=prediction)
