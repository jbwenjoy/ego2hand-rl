"""Root's separate saved arithmetic formulation; no MJB/backend/model reload.

This is not an independent investigator's audit. Model constants and omitted-term
eligibility retain the producer's source-backed one-read metadata scope.
"""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np


def pin(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(value):
    p = Path(value['path']).resolve(strict=True)
    assert pin(p)['sha256'] == value['sha256']
    return p


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    base = a.run_dir.resolve(strict=True)
    report = read(base/'report.json')
    for v in read(base/'output_pins.json').values():
        checked(v)
    for v in report['input_pins'].values():
        checked(v)
    meta = report['metadata']
    source = [json.loads(line) for line in checked(report['input_pins']['physical_samples']).read_text(encoding='utf8').splitlines()]
    catalog = read(checked(report['input_pins']['geometry_catalog']))
    obj = {g['id'] for g in catalog['native_geom_catalog'] if g['native_body_id'] == meta['object_body_id']}
    hand = {g['id'] for g in catalog['native_hand_geoms']}
    masses = []
    for path in ('A0', 'A2'):
        with np.load(checked(report['input_pins'][path+'_actual_readbacks']), allow_pickle=False) as z:
            for stage in ('after_copy', 'after_set_const_0'):
                masses.append(float(z[stage+'__body_mass'][0, meta['object_body_id']]))
    assert len(set(masses)) == 1 and masses[0] == meta['actual_WP_mass_kg']
    mass = masses[0]
    dt = float(np.float32(meta['nominal_options']['timestep']))
    gravity = np.zeros(3) if meta['gravity_disabled'] else np.asarray(meta['nominal_options']['gravity'], dtype='f4').astype('f8')
    assert dt == meta['dt_wp32_s'] and np.array_equal(gravity, meta['gravity_effective_world_m_s2'])
    with np.load(base/'all1808_momentum_arrays.npz', allow_pickle=False) as z:
        actual = {k: z[k].copy() for k in z.files}
    assert len(source) == 1808
    names = ('FHO_N_F64', 'Fother_N_F64', 'Ftotal_N_F64', 'F_known_valid_contacts_N_F64',
             'delta_p_Ns_F64', 'Fgravity_N_F64', 'Jcontact_Ns_F64', 'Jgravity_Ns_F64',
             'unassigned_known_terms_Ns_F64', 'closure_residual_Ns_F64', 'force_residual_N_F64')
    expected = {k: [] for k in names}
    eligible_rows = []
    contact_count = invalid_external = 0
    old_HO_discrepancies = []
    for sample in source:
        parts = {'HO': [], 'other': []}
        complete = {'HO': True, 'other': True}
        all_valid = []
        raw = sample['raw_wp_contact']
        for ci, c in enumerate(sample['contacts']):
            contact_count += 1
            left, right = c['geom']
            assert c['geom'] == raw['geom'][ci]
            if (left in obj) == (right in obj):
                continue  # no external object force, including canceled internal pairs
            group = 'HO' if left in hand or right in hand else 'other'
            slots = raw['efc_address'][ci]
            f = c['force6']
            frame = c['frame']
            valid = bool(raw['type'][ci] & 1) and raw['worldid'][ci] == 0 and c['efc_address'] >= 0
            valid = valid and any(s >= 0 for s in slots) and all(s == -1 or 0 <= s < sample['nefc'] for s in slots)
            valid = valid and len(f) == 6 and np.shape(frame) == (3, 3) and np.isfinite(f).all() and np.isfinite(frame).all()
            if not valid:
                complete[group] = False
                invalid_external += 1
                continue
            sign = 1 if right in obj else -1
            # Separate formulation: scalar compensated sums of basis rows.
            force = [sign * math.fsum(float(frame[i][j])*float(f[i]) for i in range(3)) for j in range(3)]
            parts[group].append(force)
            all_valid.append(force)
        def sum3(values):
            return np.asarray([math.fsum(row[j] for row in values) for j in range(3)], dtype='f8')
        ho = sum3(parts['HO']) if complete['HO'] else np.full(3, np.nan)
        other = sum3(parts['other']) if complete['other'] else np.full(3, np.nan)
        force = ho + other
        pre = np.asarray(sample['pre_qvel'], dtype='f4')[meta['dofadr']:meta['dofadr']+3].astype('f8')
        post = np.asarray(sample['post_qvel'], dtype='f4')[meta['dofadr']:meta['dofadr']+3].astype('f8')
        dp = mass*(post-pre) if meta['COM0_isolated_freebody'] else np.full(3, np.nan)
        fg = mass*gravity
        jc, jg = dt*force, dt*fg
        unallocated = dp-jc-jg
        eligible = bool(meta['COM0_isolated_freebody'] and all(complete.values()) and meta['noncontact_metadata_closed'] and sample['external_force_max'] == 0)
        residual = unallocated if eligible else np.full(3, np.nan)
        values = (ho, other, force, sum3(all_valid), dp, fg, jc, jg, unallocated, residual, residual/dt)
        for k, v in zip(names, values):
            expected[k].append(v)
        eligible_rows.append(eligible)
        old_HO_discrepancies.append(ho-np.asarray(sample['hand_object_force_on_object_w_N'], dtype='f8'))
    comparisons = {}
    for k, values in expected.items():
        e = np.asarray(values)
        b = actual[k]
        assert e.shape == b.shape == (1808, 3)
        assert np.array_equal(np.isnan(e), np.isnan(b))
        finite = np.isfinite(e) & np.isfinite(b)
        delta = np.abs(e[finite]-b[finite])
        # Only floating arithmetic agreement, not a scientific residual gate.
        bound = 64*np.finfo('f8').eps*np.maximum(1., np.maximum(np.abs(e[finite]), np.abs(b[finite])))
        assert bool((delta <= bound).all()), k
        comparisons[k] = {'maximum_absolute_formulation_difference': float(delta.max()) if delta.size else None,
                          'finite_scalar_values': int(finite.sum()), 'NaN_scalar_values': int(np.isnan(e).sum()),
                          'floating_arithmetic_agreement_only': True}
    assert np.array_equal(np.asarray(eligible_rows), actual['closure_interpretation_eligible'])
    assert contact_count == report['all_generated_contact_rows']
    out = a.output.resolve()
    assert not out.exists()
    out.mkdir()
    np.savez_compressed(out/'root_recomputed_arrays.npz', **{k: np.asarray(v) for k, v in expected.items()},
                        HO_minus_original_saved_aggregate_N=np.asarray(old_HO_discrepancies))
    d = {'schema': 'full151105_object_momentum_root_saved_crosscheck:1', 'status': 'PASS_ROOT_SEPARATE_SAVED_ARITHMETIC_NOT_INDEPENDENT_INVESTIGATOR',
         'producer': pin(base/'report.json'), 'checker': pin(__file__), 'rows': len(source), 'contacts': contact_count,
         'invalid_external_contacts': invalid_external, 'closure_eligible_rows': sum(eligible_rows), 'comparisons': comparisons,
         'HO_sum_vs_original_aggregate_maxabs_N': float(np.nanmax(np.abs(old_HO_discrepancies))),
         'metadata_omission_scope': 'One-read producer metadata/source-backed; not independently reloaded. No residual accuracy qualification.',
         'new_MJB_Torch_model_forward_AD_Env_physics_GPU': 0, 'outputs': {'root_recomputed_arrays.npz': pin(out/'root_recomputed_arrays.npz')}}
    (out/'report.json').write_text(json.dumps(d, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({k:d[k] for k in ('status','rows','contacts','closure_eligible_rows','HO_sum_vs_original_aggregate_maxabs_N')}))


if __name__ == '__main__':
    main()
