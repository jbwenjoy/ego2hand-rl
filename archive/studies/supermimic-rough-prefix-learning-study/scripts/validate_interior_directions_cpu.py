"""Boundary and provenance contracts; no WM or physics execution."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rough_prefix.interior_directions import prepare_interior_directions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checks = {}
    lo, hi = np.full(20, -1, np.float32), np.full(20, 1, np.float32)
    c = np.zeros((4, 20), np.float32)
    c[0] = lo
    c[1, :10] = lo[:10]
    c[2, :10] = hi[:10]
    c[3, 0] = np.float32(1 - .002)
    before = [x.copy() for x in (c, lo, hi)]
    rng_before = np.random.get_state()
    p = prepare_interior_directions(c, lo, hi, seed=20261104)
    q = prepare_interior_directions(c, lo, hi, seed=20261104)
    checks['exact_boundary_zero_dim_preserved_NOT_RUN'] = bool(not p['free_mask'][0].any() and not p['valid_pair'][0].any())
    checks['lower_face_normals_excluded'] = bool(not p['free_mask'][1, :10].any() and p['free_mask'][1, 10:].all())
    checks['upper_face_normals_excluded'] = bool(not p['free_mask'][2, :10].any() and p['free_mask'][2, 10:].all())
    checks['epsilon_boundary_without_guard_excluded'] = bool(not p['free_mask'][3, 0])
    checks['nonfree_vector_coordinates_exact_zero'] = bool((p['vectors'] * ~p['free_mask'][:, None, :]==0).all())
    checks['nonzero_vectors_unit_L2'] = bool(np.allclose(np.linalg.norm(p['vectors'][1:], axis=-1), 1, atol=4e-16, rtol=0))
    checks['all_nonzero_pairs_in_domain_both_eps'] = bool(p['valid_pair'][1:].all() and not p['crossing_mask'].any())
    checks['all_FP32_targets_exact_declared_arithmetic'] = bool(np.array_equal(p['plus'], (c.astype(float)[:,None,None,:]+p['epsilons'][None,None,:,None]*p['vectors'][:,:,None,:]).astype(np.float32)))
    checks['inactive_coordinates_unchanged_both_signs'] = bool(np.all(np.where(p['free_mask'][:,None,None,:], True, p['plus']==c[:,None,None,:])) and np.all(np.where(p['free_mask'][:,None,None,:], True, p['minus']==c[:,None,None,:])))
    checks['effective_quantized_direction_saved'] = bool(np.array_equal(p['effective_vectors'], (p['plus'].astype(float)-p['minus'].astype(float))/(2*p['epsilons'][None,None,:,None])))
    checks['deterministic_every_array'] = all(np.array_equal(p[k],q[k]) for k in p)
    checks['inputs_unchanged'] = all(np.array_equal(x,y) for x,y in zip((c,lo,hi),before))
    rng_after = np.random.get_state()
    checks['global_RNG_unchanged'] = all(np.array_equal(x,y) for x,y in zip(rng_before,rng_after))
    rejects = [dict(centers=c.astype(float)), dict(centers=np.full((1,20),2,np.float32)),
               dict(centers=np.full((1,20),np.nan,np.float32)), dict(lower=hi),
               dict(lower=lo.astype(float)), dict(seed=True), dict(count=0),
               dict(epsilons=(0,)), dict(epsilons=(.001,.001)), dict(guard_ulps=0)]
    for i, override in enumerate(rejects):
        kw=dict(centers=c,lower=lo,upper=hi,seed=20261104)
        kw.update(override)
        try: prepare_interior_directions(**kw)
        except ValueError: checks[f'invalid_input_{i}_rejects_without_repair'] = True
        else: checks[f'invalid_input_{i}_rejects_without_repair'] = False
    checks['Torch_Warp_MuJoCo_not_imported'] = not any(x in sys.modules for x in ('torch','warp','mujoco','mujoco_warp'))
    source = ROOT/'src/rough_prefix/interior_directions.py'
    report=dict(schema='rough_interior_direction_CPU_contract:1',checks=checks,
                passed=all(checks.values()),check_count=len(checks),physics_steps=0,
                WM_forwards=0,GPU_used=False,wall_seconds=time.perf_counter()-started,
                source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                scope='Synthetic boundary contracts, not physical or model gradient qualification.')
    np.savez_compressed(args.output/'boundary_fixture.npz',**p)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
