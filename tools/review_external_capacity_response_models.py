"""Compare capacity response supervision on frozen, independently audited native branches."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from review_pen_capacity_action_response import ROOT, contains_pin, read, summarize
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.pen_capacity_forecast import capacity_mismatch, load_capacity_head
from supermimic_research.strict_json import write_json


def check_sources(native, absolute_fit, paired_fit):
    orig = read(absolute_fit['intent'])
    new = read(paired_fit['intent'])
    if new['data'] != orig['data'] or new['reference'] != orig['reference'] or new['reference'] != native['target_series']:
        raise ValueError('Sequence labels or capacity reference changed across comparisons')
    series = read(read(orig['data'])['collection'])
    sequence_actors = read(series['intent'])['actors']
    native_actors = read(read(native['source'])['intent'])['actors']
    if native_actors != dict(residual=sequence_actors['TEST2'], support=sequence_actors['TEST3']):
        raise ValueError('Native probe actors differ from the original two testing behaviors')
    held = {pin['sha256'] for pin in native_actors.values()}
    train = {sequence_actors[role]['sha256'] for role in ('A', 'B', 'C', 'D', 'VAL')}
    if train & held:
        raise ValueError('Training or validation actor also owns an external test trajectory')
    pairs_pin = file_pin(ROOT/'runs/pen_capacity_pair_labels_v1/report.json')
    if not contains_pin(new, pairs_pin):
        raise ValueError('Fit did not declare the expected paired-label bank')
    pairs = read(pairs_pin)
    read(pairs['native_report'])
    pair_intent_pin = file_pin(Path(pairs['native_report']['path']).parent/'intent.json')
    pair_actors = read(pair_intent_pin)['actors']
    if pair_actors != dict(TRAIN_A=sequence_actors['A'], TRAIN_B=sequence_actors['B'], TEST=native_actors['residual']):
        raise ValueError('Paired supervision actor assignment changed')
    if set(pairs['roles']) != set(pair_actors):
        raise ValueError('Unexpected paired-label roles')
    return dict(sequence_actors=sequence_actors, test_actors=native_actors, pair_actors=pair_actors,
                paired_labels=pairs_pin, paired_roles=pair_intent_pin)


def evaluate_group(head, row):
    cases = {}
    for kind in ('ASCEND', 'DESCEND', 'BASE'):
        pair = sorted((c for c in row['cases'] if c['kind'] == kind), key=lambda c: c['slot'])
        if len(pair) != 2 or not all(c['valid'] for c in pair):
            raise ValueError('Incomplete physical pair')
        np.testing.assert_array_equal(pair[0]['pd'], pair[1]['pd'])
        cases[kind] = pair

    to_t = lambda value: torch.as_tensor(value, dtype=torch.float64)
    state, target = to_t(row['state']), to_t(row['target'])
    pd = {name: to_t(items[0]['pd']) for name, items in cases.items()}
    torch.testing.assert_close((pd['DESCEND'] + pd['ASCEND']) / 2, pd['BASE'], atol=2e-7, rtol=0)
    step = pd['DESCEND'] - pd['ASCEND']
    if step.norm() <= 1e-8:
        raise ValueError('Transmitted PD perturbation was zero')

    def objective(control):
        return capacity_mismatch(head(state, control), target)

    u = pd['BASE'].clone().requires_grad_(True)
    diff = float(torch.autograd.grad(objective(u), u)[0] @ step)
    _, jvp = torch.autograd.functional.jvp(lambda v: head(state, v), u, step)
    with torch.no_grad():
        finite = float(objective(pd['DESCEND']) - objective(pd['ASCEND']))
        finite_vec = (head(state, pd['DESCEND']) - head(state, pd['ASCEND'])).numpy()

    vectors = {kind: np.array([c['capacity'] for c in seq]) for kind, seq in cases.items()}
    mismatches = {kind: np.mean(np.abs(v - target.numpy()), axis=1) for kind, v in vectors.items()}
    for kind, seq in cases.items():
        np.testing.assert_allclose(mismatches[kind], [c['native_l1'] for c in seq], atol=1e-12, rtol=0)
    native_delta = float(mismatches['DESCEND'].mean() - mismatches['ASCEND'].mean())
    native_vec = vectors['DESCEND'].mean(0) - vectors['ASCEND'].mean(0)
    return dict(
        phase=row['phase'], stage=row['stage'], source=row['source'],
        actual_change=native_delta, linear_prediction=diff, model_change=finite,
        sham_spread=float(abs(mismatches['BASE'][1]-mismatches['BASE'][0])),
        pd_norm=float(step.norm()), actual_vector=native_vec.tolist(),
        jvp_vector=jvp.numpy().tolist(), finite_vector=finite_vec.tolist(),
    )


def vector_stats(rows):
    native = np.array([r['actual_vector'] for r in rows])
    signal = np.linalg.norm(native)
    result = {}
    for name in ('jvp_vector', 'finite_vector'):
        pred = np.array([r[name] for r in rows])
        result[name] = dict(
            norm_ratio=float(np.linalg.norm(pred)/signal) if signal else None,
            error_over_zero=float(np.linalg.norm(pred-native)/signal) if signal else None,
            positive_dot=int(np.sum(np.sum(pred*native, axis=1)>0)), groups=len(rows),
        )
    return result


def main(paired, out):
    if out.exists():
        raise ValueError('The independent external review requires a new output path')
    torch.set_num_threads(1)
    old_review_pin = file_pin(ROOT/'runs/pen_capacity_action_response_v1/report.json')
    old_review = read(old_review_pin)
    native = read(old_review['inputs']['native'])
    old_fit = read(old_review['inputs']['fit'])
    paired_pin = file_pin(paired/'report.json')
    paired_fit = read(paired_pin)
    ownership = check_sources(native, old_fit, paired_fit)
    ck = {'old_sequence_only':old_review['inputs']['model']}
    ck.update({name:paired_fit['variants'][name]['model'] for name in ('absolute','response')})
    if len({p['sha256'] for p in ck.values()})!=3:
        raise ValueError('Expected three different learned states')
    frozen = {}
    for arm in ('residual','support'):
        group = [s for s in native['source_packets'] if s['arm']==arm]
        if len(group)!=10 or len({row['phase'] for row in group})!=10:
            raise ValueError('All original declared sources required')
        for row in group:
            for c in row['cases']:
                verify_file_pin(c['trace'])
        frozen[arm] = [r for r in group if r['status']=='included']
    results = {}
    for model_name, pin in ck.items():
        state=torch.load(verify_file_pin(pin), weights_only=True, map_location='cpu')
        head=load_capacity_head(state,dtype=torch.float64,device='cpu').requires_grad_(False)
        if head.use_history:
            raise ValueError('This comparison requires identical state-only inputs')
        arms = {}
        for arm, selected in frozen.items():
            rows = [evaluate_group(head,s) for s in selected]
            summary=summarize(rows)
            if model_name=='old_sequence_only':
                for k,v in summary.items():
                    np.testing.assert_allclose(v,old_review['summary'][arm]['groups']['DESCEND-ASCEND'][k],rtol=1e-9,atol=1e-10)
            arms[arm]=dict(n_declared=10, excluded=[dict(phase=s['phase'],reason=s['status']) for s in native['source_packets'] if s['arm']==arm and s['status']!='included'],
                           scalar=summary,capacity_vector=vector_stats(rows),rows=rows)
            print(model_name,arm,{k:summary[k] for k in ('n','sign_agrees','predicted_norm_over_observed','jvp_error_over_zero_response','finite_error_over_zero_response')},flush=True)
        results[model_name]=dict(checkpoint=pin,arms=arms)
    out.mkdir()
    write_json(out/'report.json',dict(schema='external_capacity_response_model_comparison:1',
        paired_fit=paired_pin,old_review=old_review_pin,native=old_review['inputs']['native'],
        reviewer=file_pin(__file__),ownership=ownership,results=results,
        scope='Two previously inspected test policies on one controlled reference. Each contrast uses the same native DESCEND minus ASCEND transmitted PD direction from a shaft-selected probe, k=phase-1. JVP is computed at BASE. Finite model endpoint contrast is retained. Two native numerical repeats per endpoint. No data fitting, GPU calls, native integration or Actor update.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paired_fit',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    main(args.paired_fit,args.out)
