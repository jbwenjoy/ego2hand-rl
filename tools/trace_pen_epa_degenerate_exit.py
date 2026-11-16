"""Four diagnostic queries in pinned printf-only isolated EPA source copies."""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
import torch
import mujoco
import warp as wp
from mujoco_warp._src import collision_gjk, collision_convex, collision_core, types
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.isolated_epa_capacity import ALLOCATION
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from probe_retained_pen_collision import build_fixture


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(a):
    started = time.perf_counter(); pair_path = ROOT / 'runs/pen_epa_pair_capacity_v2/report.json'
    translation_path = ROOT / 'runs/pen_epa_translation_v1/report.json'
    pair = json.loads(pair_path.read_text()); translation = json.loads(translation_path.read_text())
    for r in [pair, translation]: verify_source_snapshot(json.loads(verify_file_pin(r['source_receipt']).read_text()), live=True)
    pr = json.loads(verify_file_pin(pair['capacity_results']).read_text())
    tr = json.loads(verify_file_pin(translation['rows']).read_text())
    def select(rows, **keys):
        found = [r for r in rows if all(r.get(k) == v for k, v in keys.items())]
        if len(found) != 1: raise ValueError('One pinned reference case required')
        return found[0]
    cases = [dict(label='BASE24_OVERFLOW', capacity=24, reference=select(pr, scene='INCIDENT_BASE', capacity=24)),
        dict(label='BASE48_SILENT_MISSING', capacity=48, reference=select(pr, scene='INCIDENT_BASE', capacity=48)),
        dict(label='LOCAL24_SILENT_MISSING', capacity=24, reference=select(tr, scene='QUAT3_ULP-1', placement='GEOM1_ORIGIN')),
        dict(label='PARENT24_CONTACT_CONTROL', capacity=24, reference=select(pr, scene='PARENT_RAW', capacity=24))]
    gjk_path, convex_path = Path(collision_gjk.__file__), Path(collision_convex.__file__)
    original_gjk, original_convex = gjk_path.read_text(), convex_path.read_text()
    source_pins = [file_pin(gjk_path), file_pin(convex_path)]
    edits = [
        ('      if dist2 == 0.0:\n        idx = -1',
         '      if dist2 == 0.0:\n        wp.printf("TRACE_EPA_DEGENERATE_FACE world=%d face=%d horizon=%d nvalid=%d\\n", worldid, pt.nface, pt.nhorizon, nvalid)\n        idx = -1'),
        ('  return 0.0, wp.vec3(), wp.vec3(), -1',
         '  wp.printf("TRACE_EPA_INVALID_RESULT world=%d idx=%d nface=%d nvalid=%d\\n", worldid, idx, pt.nface, nvalid)\n  return 0.0, wp.vec3(), wp.vec3(), -1')]
    traced = original_gjk
    for old, new in edits:
        if traced.count(old) != 1: raise ValueError('Exact unique source branch required')
        traced = traced.replace(old, new)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh trace output required')
    out.mkdir()
    receipt = freeze_sources([Path(__file__), ROOT / 'tools/probe_retained_pen_collision.py'], out / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    write(out / 'protocol.json', dict(cases=cases, max_pair_queries=4, native_integrations=0, optimizer_steps=0,
        source_reports=[file_pin(pair_path), file_pin(translation_path)], installed_sources=source_pins,
        source_edits=[dict(old=old, new=new) for old, new in edits],
        scope='printf-only _epa branch observations; convex imports redirected to the trace copy, scratch24/48 explicit. Require exact flag/count equivalence to uninstrumented reference. No production module/metadata/algorithm patch.'))
    attempted = completed = 0; rows = []
    try:
        folder = out / 'isolated_engine_sources'; folder.mkdir()
        gjk_copy = folder / 'collision_gjk.py'; gjk_copy.write_text(traced, encoding='utf-8', newline='\n')
        imported('isolated_pen_trace_gjk', gjk_copy)
        modules = {}; variants = [file_pin(gjk_copy)]
        for cap in [24, 48]:
            text = original_convex.replace('from mujoco_warp._src.collision_gjk import ', 'from isolated_pen_trace_gjk import ')
            if text == original_convex or text.count(ALLOCATION) != 1: raise ValueError('Exact trace import/allocation required')
            text = text.replace(ALLOCATION, f'epa_horizon = wp.empty(shape=(d.naccdmax, {cap}), dtype=int)')
            path = folder / f'convex_capacity{cap}.py'; path.write_text(text, encoding='utf-8', newline='\n')
            modules[cap] = imported(f'isolated_pen_trace_convex{cap}', path); variants.append(file_pin(path))
        wp.init()
        with wp.ScopedDevice('cuda:0'):
            parent = torch.load(verify_file_pin(pair['inputs']['parent']), map_location='cpu', weights_only=False)
            host = mujoco.MjModel.from_binary_path(str(verify_file_pin(pair['inputs']['host_model'])))
            model, data, construction = build_fixture(host, parent['native']['native']); write(out / 'construction.json', construction)
            ctx = collision_core.create_collision_context(data.naconmax)
            pairs = np.zeros((data.naconmax, 2), np.int32); pairs[0] = [1, 55]
            ctx.collision_pair.assign(pairs); ctx.collision_pairid.fill_(-1); ctx.collision_worldid.fill_(31)
            for case in cases:
                frame = np.load(verify_file_pin(case['reference']['artifact']))
                data.geom_xpos.assign(frame['geom_xpos']); data.geom_xmat.assign(frame['geom_xmat'])
                data.nacon.zero_(); data.ncollision.fill_(1); data.overflow.zero_()
                print('TRACE_CASE_BEGIN ' + case['label'], flush=True)
                attempted += 1
                modules[case['capacity']].convex_narrowphase(model, data, ctx, [(types.GeomType.CYLINDER, types.GeomType.MESH)])
                wp.synchronize(); completed += 1
                mask = data.overflow.numpy().tolist(); count = int(data.nacon.numpy()[0]); reference = case['reference']
                row = dict(label=case['label'], raw_mask=mask, nacon=count,
                    reference_flag_and_count_equal=mask == reference['raw_mask'] and count == reference['nacon'])
                rows.append(row); write(out / 'rows.json', rows)
                print('TRACE_CASE_END ' + json.dumps(row), flush=True)
                if not row['reference_flag_and_count_equal']: raise ValueError('Instrumentation changes reference behavior; trace not admitted')
            verify_source_snapshot(receipt, live=True)
            for pin in source_pins: verify_file_pin(pin)
            report = dict(schema='isolated_EPA_degenerate_exit_trace:1', source_receipt=file_pin(out / 'source_receipt.json'),
                protocol=file_pin(out / 'protocol.json'), source_variants=variants, construction=file_pin(out / 'construction.json'),
                rows=rows, attempted_pair_queries=attempted, actual_pair_queries=completed,
                actual_native_integrations=0, optimizer_steps=0, wall_s=time.perf_counter() - started,
                scope='printf-only isolated branch trace with matching raw masks/contact counts. Console branch lines need independent review. Not a production repair or exact original physical state recovery.')
            write(out / 'report.json', report); print(json.dumps(dict(report=file_pin(out / 'report.json'), queries=completed)), flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), attempted_pair_queries=attempted,
            actual_pair_queries=completed, actual_native_integrations=0, optimizer_steps=0, source_receipt=file_pin(out / 'source_receipt.json')))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); main(p.parse_args())
