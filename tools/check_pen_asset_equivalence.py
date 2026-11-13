"""Compare explicit-path pen loading against the immutable archived loader."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dex_supermimic.assets.pen import get_tracking_pen_spec
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.runtime.native_evaluation import model_digest_except_dt


def main(args):
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh unified audit output')
    historical = ROOT / 'archive/studies/dex-supermimic-research/src/dex_supermimic/assets/pen.py'
    spec = importlib.util.spec_from_file_location('archived_pen_asset_reference', historical)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    # This explicitly selected archived reference is used only for the audit.
    # Its original missing bundled path is redirected to the same pinned asset.
    legacy.PEN_XML = args.xml.resolve(strict=True)
    assets = [file_pin(p) for p in sorted(args.xml.resolve().parents[1].rglob('*')) if p.is_file()]
    cases = []
    for mass in [.03, .05]:
        old = legacy.get_tracking_pen_spec(mass=mass).compile()
        new = get_tracking_pen_spec(mass=mass, xml_path=args.xml.resolve()).compile()
        left, right = [model_digest_except_dt(m, np) for m in [old, new]]
        cases.append(dict(mass_kg=mass, old_model_digest=left, new_model_digest=right, exact=left == right))
    for pin in assets:
        verify_file_pin(pin)
    result = dict(schema='pen_explicit_asset_path_equivalence:1',
        sources=[file_pin(__file__), file_pin(historical), file_pin(ROOT / 'src/dex_supermimic/assets/pen.py'),
                 file_pin(ROOT / 'src/supermimic_research/runtime/native_evaluation.py')],
        assets=assets, cases=cases, passed=sum(c['exact'] for c in cases), total=len(cases),
        neural_updates=0, native_integration_steps=0,
        scope='All exposed compiled arrays and options except timestep hashed; both use identical timestep by default. Independent immutable archived loader, same original XML/textures. No physical calibration claim.')
    args.output.mkdir()
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(passed=result['passed'], total=result['total'])))
    if result['passed'] != result['total']:
        raise ValueError('Preserve failed asset equivalence')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--xml', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
