"""Pure independent dense saved-row reconstruction; no geometry or model call.

The threshold, source FP32 P arithmetic and native exact-tie rules are frozen.
This module has no file loader or execution of future dense outputs.
"""
import numpy as np


def logical_world_rotation_from_saved_flat9(value, expected_rows=1808):
    a = np.asarray(value)
    if a.shape != (expected_rows, 9) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('Original saved row-major rotation must be finite FP32 rows x9')
    return a.reshape(expected_rows, 3, 3)


def contact_record(row, hand, objects, covered, minima=None):
    contacts = row['contacts']; n = row['nacon']; raw = row['raw_wp_contact']
    if type(n) is not int or not 0 <= n <= 192 or type(row['nefc']) is not int or not 0 <= row['nefc'] <= 4096:
        raise ValueError('Invalid saved perstep capacities')
    if len(contacts) != n or set(raw) != {'geom', 'type', 'worldid', 'efc_address'}:
        raise ValueError('Missing/malformed contact inventory is not empty')
    if n == 0:
        if contacts or any(type(v) is not list or v for v in raw.values()):
            raise ValueError('Malformed knownempty raw contact')
    else:
        g, w, typ = (np.asarray(raw[k]) for k in ('geom', 'worldid', 'type'))
        if g.shape != (n, 2) or w.shape != (n,) or typ.shape != (n,) or any(v.dtype.kind not in 'iu' for v in (g, w, typ)) or (w != 0).any() or (g < 0).any() or (typ < 0).any():
            raise ValueError('Invalid raw contact geom/world/type')
        if not np.array_equal(g, np.asarray([c['geom'] for c in contacts])):
            raise ValueError('Saved WP versus CPU contact order mismatch')
    pairs = []
    for c in contacts:
        a, b = c['geom']; d = c['dist']
        if type(a) is not int or type(b) is not int or type(d) not in (int, float) or not np.isfinite(d):
            raise ValueError('Unknown contact geometry/distance')
        if (a in hand and b in objects) or (b in hand and a in objects):
            pairs.append({'geom': [a, b], 'hand_geom': a if a in hand else b, 'depth_m': max(0., -float(d))})
    maximum = max((p['depth_m'] for p in pairs), default=0.)
    tied = [p for p in pairs if p['depth_m'] == maximum]
    violating = [p for p in pairs if p['depth_m'] > .003]
    def coverage(items):
        ids = sorted({p['hand_geom'] for p in items})
        return {'hand_geom_ids': ids, 'geom_set_gap_ids': [g for g in ids if g not in covered],
            # Producer field intentionally uses nominal F64 -.003 comparison;
            # source-FP32 boundary equality is independently retained elsewhere.
            'covered_proxy_undersensitivity_ids': [] if minima is None else [g for g in ids if g in covered and minima[g] >= -.003],
            'overlapping_not_exclusive': True}
    return {'native_generated_HO_depth_m': maximum, 'native_generated_HO_count': len(pairs),
        'native_HO_depth_gt3mm': maximum > .003, 'all_exact_tied_deepest_pairs': tied,
        'all_violating_pairs_SECONDARY': violating,
        'primary_deepest_tie_coverage': coverage(tied if maximum > .003 else []),
        'secondary_all_violating_coverage': coverage(violating)}

def summary(rows):
    return {'rows': len(rows), 'native_HO_depth_gt3mm': sum(r['native_HO_depth_gt3mm'] for r in rows),
        'proxy_P_positive': sum(r['proxy_P_positive'] for r in rows),
        'both_positive': sum(r['native_HO_depth_gt3mm'] and r['proxy_P_positive'] for r in rows),
        'native_positive_proxy_zero': sum(r['primary_miss'] for r in rows),
        'proxy_alert_without_nativeHO3mm': sum(r['proxy_alert_without_observed_native_HO3mm'] for r in rows),
        'both_nonpositive': sum(not r['native_HO_depth_gt3mm'] and not r['proxy_P_positive'] for r in rows),
        'primary_deepest_tie_geom_gap_rows': sum(r['primary_miss'] and bool(r['primary_deepest_tie_coverage']['geom_set_gap_ids']) for r in rows),
        'primary_deepest_tie_covered_undersensitivity_rows': sum(r['primary_miss'] and bool(r['primary_deepest_tie_coverage']['covered_proxy_undersensitivity_ids']) for r in rows),
        'overlapping_categories_not_independent_N': True}


def reconstruct_rows(samples, score, sample_geom_ids, hand, objects, ngeom, proxy_math):
    """Reconstruct every row using captured SDF and original raw contacts only.

    proxy_math is the independent frozen08c9 module, never a geometry object.
    Full1808/1344/21 cadence is enforced by the file adapter, not synthetic toys.
    """
    if type(ngeom) is not int or ngeom < 1 or not hand or not objects or set(hand) & set(objects):
        raise ValueError('Exact native catalog required')
    if not set(hand | objects) <= set(range(ngeom)):
        raise ValueError('Catalog native geom IDs out of range')
    recomputed = proxy_math.proxy_from_saved_sdf(score['sampled_sdf_m'], sample_geom_ids)
    if len(samples) != len(recomputed['P']):
        raise ValueError('Missing saved row cannot be dropped')
    ids = recomputed['geom_ids']; covered = set(map(int, ids)); rebuilt = []
    arrays_ok = all(np.asarray(score[b]).dtype == np.asarray(recomputed[a]).dtype
        and np.asarray(score[b]).shape == np.asarray(recomputed[a]).shape
        and np.asarray(score[b]).tobytes() == np.asarray(recomputed[a]).tobytes()
        for a, b in [('P', 'P'), ('BP', 'BP'), ('min_sdf_m', 'minimum_sdf_m'),
                     ('geom_ids', 'geom_ids'), ('pergeom_min_sdf_m', 'per_geom_min_sdf_m')])
    for i, s in enumerate(samples):
        for pair in s['raw_wp_contact']['geom']:
            if len(pair) != 2 or any(type(g) is not int or not 0 <= g < ngeom for g in pair):
                raise ValueError('Raw native geom ID malformed/unknown')
        minimum = recomputed['min_sdf_m'][i]
        P, BP = recomputed['P'][i], recomputed['BP'][i]
        native = contact_record(s, hand, objects, covered,
            {int(g): float(recomputed['pergeom_min_sdf_m'][i, j]) for j, g in enumerate(ids)})
        native.update(row_id=i, path=s['path'], mode=s['mode'], k=s['phase'], branch_id=s['branch_id'],
            substep=s['substep'], canonical_pre_tick=s['canonical_pre_tick'], source_frame=s['source_frame'],
            P=float(P), BP=float(BP), minimum_sampled_sdf_m=float(minimum),
            proxy_sample_depth_m=max(0., -float(minimum)), proxy_P_positive=bool(P > 0),
            proxy_min_sdf_below_minus3mm=bool(minimum < -.003),
            proxy_min_sdf_exactly_minus_FP32_allowance=bool(minimum == np.float32(-.003)),
            native_depth_minus_nominal_3mm_m=native['native_generated_HO_depth_m']-.003,
            primary_miss=bool(native['native_HO_depth_gt3mm'] and not P > 0),
            proxy_alert_without_observed_native_HO3mm=bool(P > 0 and not native['native_HO_depth_gt3mm']))
        rebuilt.append(native)
    statistics = {'all1808': summary(rebuilt)}
    for path in ('A0', 'A2'):
        for mode in ('donor', 'baseline', 'direction'):
            statistics[path+'/'+mode] = summary([r for r in rebuilt if r['path'] == path and r['mode'] == mode])
    return {'rows': rebuilt, 'statistics': statistics, 'captured_proxy_arrays_match': arrays_ok,
            'recomputed_proxy': recomputed}
