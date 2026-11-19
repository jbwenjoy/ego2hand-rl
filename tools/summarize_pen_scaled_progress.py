"""Summarize complete rollout records without disturbing a running PPO process."""
import argparse
import json
from pathlib import Path


def summarize(run):
    windows = {}
    steps = 0
    seconds = 0.
    with (run / 'iterations.jsonl').open('rb') as stream:
        for expected, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            if row['iteration'] != expected:
                raise ValueError('Missing or duplicated rollout')
            histogram = row['phase_histogram_valid']
            count = sum(histogram)
            second = sum(histogram[185:])
            quarter = sum(histogram[277:])
            if len(histogram) != 369 or (count, second, quarter) != (row['valid_count'], row['valid_second_half'], row['valid_last_quarter']):
                raise ValueError('Inconsistent phase histogram')
            # Assign a rollout to the million-transition block containing its start.
            block = windows.setdefault(steps // 1_000_000, dict(steps=0, valid=0, second=0, quarter=0, max_phase=0))
            block['steps'] += row['transitions'] - steps
            block['valid'] += count
            block['second'] += second
            block['quarter'] += quarter
            block['max_phase'] = max(row['max_phase'], block['max_phase'])
            steps, seconds = row['transitions'], row['training_s']
    for block in windows.values():
        block['second_fraction_of_valid'] = block['second'] / block['valid'] if block['valid'] else None
    evals = []
    for path in sorted(run.glob('eval_*/report.json')):
        report = json.loads(path.read_bytes())
        evals.append({key:report[key] for key in ('iteration', 'successes', 'total')})
    return dict(transitions=steps, net_training_s=seconds, blocks_by_million=windows, completed_evaluations=evals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(json.dumps(summarize(parser.parse_args().run), indent=2))
