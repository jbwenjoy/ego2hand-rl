"""CPU-only fixed milestone review;never resume physics or count new training."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.native_pen_ppo import tensor_tree_equal


def main(a):
    run = ROOT / 'runs/pen_long_control_training_v1'
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh CPU-only milestone output required')
    paths = dict(R_final_health=run / 'RESIDUAL/health_2000.json', S_first_health=run / 'SUPPORT/health_0001.json',
                 R_initial=run / 'RESIDUAL/initial_state.pt', S_initial=run / 'SUPPORT/initial_state.pt', source_receipt=run / 'source_receipt.json')
    R, S = (json.loads(paths[name].read_text()) for name in ['R_final_health', 'S_first_health'])
    rr = torch.load(verify_file_pin(R['checkpoint']), map_location='cpu', weights_only=False)
    ss = torch.load(verify_file_pin(S['checkpoint']), map_location='cpu', weights_only=False)
    ri = torch.load(paths['R_initial'], map_location='cpu', weights_only=False)
    si = torch.load(paths['S_initial'], map_location='cpu', weights_only=False)
    changed_metadata = copy.deepcopy(ri['metadata'])
    changed_metadata.update(arm='SUPPORT', support_weight=.2)
    checks = dict(
        R_final_counters=(rr['iteration'], rr['optimizer_steps'], rr['training_transitions']) == (2000, 64000, 4096000),
        S_first_counters=(ss['iteration'], ss['optimizer_steps'], ss['training_transitions']) == (1, 32, 2048),
        initial_modules_exact=tensor_tree_equal(ri['modules'], si['modules']),
        initial_empty_Adam_exact=tensor_tree_equal(ri['optimizer'], si['optimizer']) and not ri['optimizer']['state'],
        initial_PPO_config_exact=ri['config'] == si['config'],
        initial_normalizer_ready=ri['normalizers_ready'] and si['normalizers_ready'],
        initial_metadata_only_arm_weight_change=changed_metadata == si['metadata'],
        R_final_modules_finite=all(bool(torch.isfinite(v).all()) for m in rr['modules'].values() for v in m.values()),
        S_first_modules_finite=all(bool(torch.isfinite(v).all()) for m in ss['modules'].values() for v in m.values()),
        R_all_Adam_steps64000=bool(rr['optimizer']['state']) and all(int(v['step']) == 64000 for v in rr['optimizer']['state'].values()),
        S_all_Adam_steps32=bool(ss['optimizer']['state']) and all(int(v['step']) == 32 for v in ss['optimizer']['state'].values()),
        S_ordinary_reload_boolean_checks=all(v for v in S['first_checkpoint_reload'].values() if type(v) is bool),
        S_native_immediate_arrays=S['first_checkpoint_reload']['native']['all_array_bytes_equal'],
        S_native_immediate_scalars=S['first_checkpoint_reload']['native']['all_scalar_values_equal'])
    if not all(checks.values()):
        raise ValueError('Saved milestone mismatch: ' + repr(checks))
    verify_source_snapshot(json.loads(paths['source_receipt'].read_text()), live=True)
    out.mkdir()
    source = freeze_sources([Path(__file__)], out / 'source_snapshot', ROOT,
                            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    source_path = out / 'source_receipt.json'
    source_path.write_text(json.dumps(source, indent=2) + '\n', encoding='utf-8')
    report = dict(schema='pen_long_control_second_arm_milestone:2', checks=checks, passed=sum(checks.values()), total=len(checks),
        source_receipt=file_pin(source_path), inputs={k: file_pin(p) for k, p in paths.items()},
        R_final_checkpoint=R['checkpoint'], S_first_checkpoint=S['checkpoint'],
        extra_native_integrations=0, extra_optimizer_steps=0, long_training_final_actual_work=None,
        status='R_FINAL_S_FIRST_STATIC_MILESTONE_NOT_FINAL_TRIAL',
        scope='CPU saved-byte/counter/readback verification only;all embedded training work belongs to the original ongoing cohort,not additional work. Not final work accounting,task success,future W32 determinism or live-process proof. Parent polls session89759 separately.')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(report=file_pin(out / 'report.json'), checks=[report['passed'], report['total']])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
