"""New-backend numeric packing without reinterpreting old dataset schemas.

This is not file/provenance admission: the caller must verify the raw cohorts,
their registrations, backend/source receipts and independent physical review.
Empty declared groups remain in the audit. Only valid TRAIN rows and complete
physical TRAIN pairs supply normalization or supervision.
"""
import numpy as np
from .pen_response import fit_pen_scales
from .response_coverage_review import review_coverage

ROLES = {
    'TRAIN': 'TRAIN_CONTROLLED_PHYSICAL_RESPONSE',
    'VALIDATE': 'VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT',
}


def backend_response_data(reports, data, directions, contract):
    if set(reports) != set(ROLES) or set(data) != set(ROLES) or set(directions) != set(ROLES):
        raise ValueError('Separate new-backend TRAIN and NEVER_FIT validation required')
    audits = {}; source_sets = {}
    for role, expected in ROLES.items():
        report, arrays = reports[role], data[role]
        if (report.get('schema') != 'backend_pen_response_preparation:1'
                or report['role'] != expected or report['collision_backend'] != contract
                or not report['checks'] or not all(report['checks'].values())
                or report['passed'] != report['total'] or report['total'] != len(report['checks'])
                or report['scales_fitted'] or report['CHECK_response_rows_used']
                or report['validation_rows_used_for_TRAIN']
                or report['actual_native_integrations'] or report['optimizer_steps']):
            raise ValueError('Completed explicit new-backend physical preparation required')
        n = report['total_rows']; mask = arrays['valid']
        if (arrays['x'].shape != (n, 102) or arrays['y'].shape != (n, 90)
                or arrays['heads'].shape != (n, 90) or mask.shape != (n,)
                or mask.dtype != np.bool_ or int(mask.sum()) != report['qualified_rows']
                or report['rejected_rows'] != np.flatnonzero(~mask).tolist()
                or not all(np.isfinite(arrays[k]).all() for k in ['x', 'y', 'heads'])):
            raise ValueError('Finite aligned rows and unchanged physical masks required')
        sources = list(dict.fromkeys(g['source_name'] for g in report['response_groups']))
        source_sets[role] = set(sources)
        # Fixed units here serve structural checking only. The diagnostics do
        # not select pairs and cannot enter the fitted TRAIN normalizers.
        audits[role] = review_coverage(arrays, report['metadata'], report['response_groups'],
                                      directions[role], sources, np.ones(90))
        if report['qualified_response_groups'] != sum(g['qualified'] for g in report['response_groups']):
            raise ValueError('Original qualified-group denominator/count required')
    if source_sets['TRAIN'] & source_sets['VALIDATE']:
        raise ValueError('Disjoint physical source identities required')

    train = data['TRAIN']; pairs = []; audit = []
    for gi, group in enumerate(reports['TRAIN']['response_groups']):
        audit.append(dict(TRAIN_group_index=gi, source_name=group['source_name'],
                          direction=group['direction'], epsilon=group['epsilon'],
                          complete=group['complete'], eligible=group['qualified']))
        if not group['qualified']:
            continue
        plus = np.asarray(group['rows']['1'], dtype=np.int64)
        minus = np.asarray(group['rows']['-1'], dtype=np.int64)
        yp, ym = train['y'][plus].mean(0), train['y'][minus].mean(0)
        epsilon = group['epsilon']; j = group['direction']
        pairs.append(dict(x_plus=train['x'][plus[0]], x_minus=train['x'][minus[0]],
                          y_plus=yp, y_minus=ym, finite_contrast=(yp - ym)/(2*epsilon),
                          epsilon=epsilon, direction20=directions['TRAIN'][j], direction_index=j,
                          TRAIN_group_index=gi, TRAIN_row_plus=plus, TRAIN_row_minus=minus))
    if not pairs or int(train['valid'].sum()) < 2:
        raise ValueError('No eligible new TRAIN point/pair supervision')
    packed = {k: np.asarray([p[k] for p in pairs]) for k in pairs[0]}
    subsets = {f'rank{rank}': np.flatnonzero(packed['direction_index'] < rank) for rank in [4, 20]}
    for key, indices in subsets.items():
        if not len(indices):
            raise ValueError('No eligible matched '+key+' TRAIN response subset')
        if packed['TRAIN_group_index'][indices].tolist() != audits['TRAIN']['eligible_pair_groups'][key]:
            raise ValueError('Pair membership differs from independent physical eligibility')
    scales, heads = fit_pen_scales(train['x'][train['valid']], train['y'][train['valid']],
                                   train['heads'][train['valid']])
    return dict(scales=scales, head_scales=heads, pairs=packed, pair_subsets=subsets, pair_audit=audit)
