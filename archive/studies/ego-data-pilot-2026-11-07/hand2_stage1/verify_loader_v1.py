"""One CPU metadata reconstruction check; no MjData/forward/integration."""
from pathlib import Path
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import mujoco
import scene_loader_v1 as loader

ROOT = Path(__file__).resolve().parent


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    destination = ROOT / 'loader_reconstruction_v1.json'
    if destination.exists():
        raise ValueError('Fresh verification only')
    reference = loader.load_model()
    spec = loader.load_spec()
    reconstructed = spec.compile()
    arrays = {}
    for name in dir(reference):
        try:
            a, b = getattr(reference, name), getattr(reconstructed, name)
        except (AttributeError, TypeError):
            continue
        if isinstance(a, np.ndarray):
            passed = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
            arrays[name] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'bit_exact': passed}
    options = {}
    for name in dir(reference.opt):
        if name.startswith('_'):
            continue
        a, b = getattr(reference.opt, name), getattr(reconstructed.opt, name)
        if isinstance(a, np.ndarray):
            options[name] = {'value': a.tolist(), 'bit_exact': a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()}
        elif isinstance(a, (int, float, bool)):
            options[name] = {'value': a, 'equal': a == b}
    checks = {'model_arrays_bit_exact': all(x['bit_exact'] for x in arrays.values()),
              'options_equal': all(x.get('bit_exact', x.get('equal')) for x in options.values()),
              'names_bytes_equal': reference.names == reconstructed.names,
              'dimensions_equal': (reference.nq, reference.nv, reference.nu) == (reconstructed.nq, reconstructed.nv, reconstructed.nu),
              'fixed_palm_no_joint': int(reference.body_jntnum[reference.body('right_palm_link').id]) == 0,
              'one_freejoint_phone_only': [reference.joint(i).name for i in range(reference.njnt) if int(reference.jnt_type[i]) == 0] == ['phone24_freejoint'],
              'exactly_one_added_table': sum(reference.geom(i).name == 'stage1_single_table' for i in range(reference.ngeom)) == 1}
    report = {'schema': 'fixed_hand2_phone_exact_loader_reconstruction:1',
              'status': 'PASS_METADATA_RECONSTRUCTION' if all(checks.values()) else 'FAIL_PRESERVED',
              'checks': checks, 'model_arrays': arrays, 'actual_options': options,
              'source_pins': {'verifier': pin(__file__), 'loader': pin(loader.__file__),
                              'builder': pin(ROOT / 'build_hand2_phone_v2.py'),
                              'base_report': pin(ROOT / 'completed_v2/report.json')},
              'counts': {'binary_model_loads': 1, 'CPU_MjSpec_compiles': 1,
                         'MjData': 0, 'mj_forward': 0, 'mj_step': 0},
              'not_grasp_or_dynamics_qualification': True}
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checked_model_arrays': len(arrays),
                      'report': pin(destination)}))
    if not all(checks.values()):
        raise ValueError('Metadata reconstruction mismatch; preserved report')


if __name__ == '__main__':
    main()
