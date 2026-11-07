"""Compact post-audit saved numeric table; no new gates or backend calls."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'outputs/full151105_matched_native_GPU_reader_saved_independent_v3'
RUN = ROOT/'outputs/full151105_matched_native_GPU_reader_v2'


def pin(p):
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    (AUDIT/'numeric_table_source_before_execution.py').write_bytes(Path(__file__).read_bytes())
    review = json.loads((AUDIT/'report.json').read_text())
    if review['passed'] != review['total'] or not all(review['checks'].values()):
        raise ValueError('Actual independent saved audit must pass first')
    blocks = {'q_rad': (0, 20), 'qd_rad_s': (20, 40), 'p_m': (40, 43),
              'R_dimensionless': (43, 52), 'v_m_s': (52, 55), 'omega_rad_s': (55, 58)}
    table = {}
    for name in ('endpoint452_original_rule', 'anchor160_original_rule'):
        with np.load(RUN/(name+'.npz'), allow_pickle=False) as z:
            delta = z['difference_F64'].copy(); flags = z['old_flags'].copy()
        table[name] = {key: {'maximum_abs_difference': float(abs(delta[:, a:b]).max()),
            'failed_components': int((~flags[:, a:b]).sum()), 'failed_rows': int((~flags[:, a:b]).any(-1).sum())}
            for key, (a, b) in blocks.items()}
    value = {'schema': 'full151105_matched_reader_saved_difference_table:1', 'source': pin(Path(__file__)),
        'independent_review': pin(AUDIT/'report.json'), 'table': table,
        'GPU_vs_CPU_all2260_by_unit': review['details']['GPU_vs_CPU_descriptive_by_unit'],
        'new_gate': False, 'new_Reader_model_physics_GPU_calls': 0}
    with (AUDIT/'difference_summary.json').open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
    print(json.dumps({'summary': pin(AUDIT/'difference_summary.json'), 'table': table}, indent=2))


if __name__ == '__main__':
    main()
