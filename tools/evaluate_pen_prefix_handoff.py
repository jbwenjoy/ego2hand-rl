"""Native clip_003 handoff ablation after the same frozen action-only prefix."""
import argparse
from pathlib import Path
import tomllib

import numpy as np

from evaluate_pen_prefix_w1 import DEFAULT_ACTOR, b, config, evaluate_cases, file_pin, load_selection, read, summarize_case, verify_file_pin, write_json


WEIGHTS = {'abrupt': [], 'blend': [i/8 for i in range(1,9)], 'hold': [0.]*8}


def summarize_handoff(row, plan, preload, ref):
    result = summarize_case(row, plan, preload, ref)
    prefix = plan[1]
    horizon = len(prefix)
    with np.load(verify_file_pin(row['handoff_controls']), allow_pickle=False) as controls:
        sent, applied = controls['sent_raw'], controls['applied_raw']
        actor, pd = controls['actor_mean'], controls['post_ctrl']
        count = len(sent)
        weights = WEIGHTS[row['arm']]
        expected = actor.copy()
        expected[:horizon] = prefix[:min(horizon, count)]
        for i, w in enumerate(weights[:max(0,count-horizon)]):
            expected[horizon+i] = (1-w)*prefix[-1] + w*np.clip(actor[horizon+i], -1, 1)
        np.testing.assert_allclose(sent, expected, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(applied, np.clip(sent, -1, 1))
        np.testing.assert_array_equal(controls['previous_raw'][1:], applied[:-1])
        weights = np.r_[np.zeros(horizon), weights, np.ones(count)][:count]
        np.testing.assert_array_equal(controls['actor_weight'], weights)
        def jump(z, at):
            return float(np.max(abs(z[at]-z[at-1]))) if count > at else None
        full_actor_phase = next((i+1 for i in range(horizon, count) if weights[i] == 1), None)
        diffs = np.abs(pd[horizon:,0,:]-pd[horizon-1:-1,-1,:]) if count>horizon else []
        result.update(controls_trace=row['handoff_controls'], full_actor_phase_reached=full_actor_phase,
            first_prefix_exit_raw_jump=jump(applied,horizon),
            first_prefix_exit_pd_jump_rad=float(np.max(diffs[0])) if len(diffs) else None,
            max_after_prefix_pd_jump_rad=float(np.max(diffs)) if len(diffs) else None,
            reached_phase17=bool(count>=17), last_actor_weight=float(weights[-1]))
    return result


def run(selection, actor_path, out, repetitions):
    if repetitions < 1:
        raise ValueError('Specify a positive number of repeats')
    root = b.ROOT/'runs'
    source = file_pin(selection)
    actor = file_pin(actor_path)
    preload = file_pin(root/'pen_refit_dev_preload_v1/report.json')
    proj = read(read(read(preload)['intent'])['source'])
    pose = next(r['variants']['projection']['trace'] for r in proj['rows'] if r['clip']=='clip_003')
    with np.load(verify_file_pin(pose), allow_pickle=False) as projected:
        plan = load_selection(source,projected['initial_hand_q'])['action_only']
    q, raw, _ = plan
    if len(raw) != 8:
        raise ValueError('Expected the registered eight-control action-only prefix')
    out.mkdir(parents=True,exist_ok=False)
    folder = out/'native'
    folder.mkdir()
    objective,item = config(folder,'clip_003',Path(preload['path']),initial_hand_q=q)
    adapter = Path(tomllib.loads(objective.read_text())['adapter_config'])
    cfg = tomllib.loads(adapter.read_text())
    scripts = Path(__file__).parent
    write_json(out/'intent.json',dict(clip='clip_003',selection=source, selected='action_only', actor=actor,
        repeats=repetitions, worlds=1, environment_support_weight=0., reference=item['series'],
        projected=cfg['projected_pin'], preload=cfg['control_pin'], objective=file_pin(objective), adapter=file_pin(adapter),
        prefix_controls=len(raw), actor_weights_after_prefix=WEIGHTS,
        scripts=[file_pin(scripts/f) for f in ('evaluate_pen_dev_clip.py','evaluate_pen_prefix_w1.py','evaluate_pen_prefix_handoff.py')],
        scope='All three controllers share eight raw prefix actions. Later weights combine the final prefix raw with the clipped Actor mean from the current native observation. After the final mixture control the Actor runs normally. The original strict W1 success and physical audit apply to the entire reference.'))
    write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json')))
    rows = evaluate_cases(folder,objective,item,dict.fromkeys(WEIGHTS,actor),dict.fromkeys(WEIGHTS,0.),
        repetitions,refit_export=True,raw_prefixes=dict.fromkeys(WEIGHTS,raw),handoff_weights=WEIGHTS)
    results = {}
    with np.load(verify_file_pin(cfg['control_pin'])) as ctrl, np.load(verify_file_pin(item['series'])) as ref:
        for arm in WEIGHTS:
            cases = [summarize_handoff(r,plan,ctrl['ctrl'],ref) for r in rows if r['arm']==arm]
            results[arm] = dict(cases=cases,full_count=sum(c['reference_completion'] for c in cases),
                strict_count=sum(c['strict_success'] for c in cases))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),native_report=file_pin(folder/'report.json'),
        clip='clip_003',arms=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--actor',type=Path,default=DEFAULT_ACTOR)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=3)
    args = parser.parse_args()
    run(args.selection.resolve(),args.actor.resolve(),args.out.resolve(),args.repetitions)
