"""Prospective eight- versus sixteen-control native screening before fresh W1 validation."""
import argparse
from pathlib import Path
import shutil
import tomllib

import numpy as np

from evaluate_pen_dev_clip import b, config, evaluate_cases, file_pin, read, verify_file_pin, write_json


CLIPS = ('clip_003', 'clip_002')
SCALES = {'actor_from_start': None, 'nominal': 1., 'zero': 0., 'half': .5, 'strong': 1.5}
HORIZONS = (8,16)


def register(out):
    out.mkdir(parents=True,exist_ok=False)
    root = b.ROOT/'runs'
    train = read(file_pin(root/'pen_multiclip_balanced_v1/report.json'))
    actor = train['actors']['balanced_kl']
    verify_file_pin(actor)
    preload = file_pin(root/'pen_refit_dev_preload_v1/report.json')
    cfgs, plans = {}, {}
    for clip in CLIPS:
        folder = out/'config'/clip
        folder.mkdir(parents=True)
        objective,item = config(folder,clip,Path(preload['path']))
        adapter = Path(tomllib.loads(objective.read_text())['adapter_config'])
        cfg = tomllib.loads(adapter.read_text())
        with np.load(verify_file_pin(cfg['control_pin'])) as servo, np.load(verify_file_pin(item['source'])) as source:
            initial_ema = (servo['ctrl']-source['reference_qpos'][0].astype(np.float32)).astype(np.float32)
        initial_raw = 2*initial_ema
        plans[clip] = {k:(np.empty((0,20),dtype=np.float32) if s is None else
            np.repeat(np.clip(s*initial_raw,-1,1)[None,:],8,axis=0)) for k,s in SCALES.items()}
        plan = folder/'raw_prefixes.npz'
        np.savez_compressed(plan,**plans[clip])
        cfgs[clip] = dict(objective=file_pin(objective), adapter=file_pin(adapter), evaluation=file_pin(folder/'controlled_evaluation_v4.toml'),
            projected=cfg['projected_pin'], control=cfg['control_pin'], item=item, prefixes=file_pin(plan))
    scripts = Path(__file__).parent
    write_json(out/'registration.json',dict(actor=actor,preload=preload,clips=cfgs,candidates=list(SCALES),
        scales=SCALES,horizons=HORIZONS,selection_repeats=1,validation_repeats=3,
        code=[file_pin(scripts/f) for f in ('evaluate_pen_prospective_prefix.py','evaluate_pen_dev_clip.py')],
        protocol='Fixed original free object, projected start, preload and reference per clip. Nominal RAW = 2*(preload ctrl - reference joint positions at frame 0). Hold scaled RAW for 8 controls; then use the same frozen balanced_kl Actor on the live observations. Actor-from-start is the zero-prefix control.',
        rank='Same pool and rollouts. At each budget maximize consecutive task- and physics-passing controls; next minimize mean of COM endpoint error / 5 mm plus SO3 geodesic / 5 degrees over observed controls up to budget (including the final observed failure). Quantize mean score to nearest 0.01 via floor(100*x+0.5). Break ties by listed candidate order.',
        prospective_claim='On EACH clip the 16-control winner must improve the median fresh continuous passing count by at least 2 controls over the 8-control winner without lowering strict success. Same winner receives zero difference. This is a development-screening test on two clips with numerical repeats.',
        budget='Screen once per candidate per clip using the full existing physical-trace recorder until the original termination. Rankings only receive separate horizon-censored ledgers. Record actual native integrations and separately count phase-limited controls. Freeze both winners before running fresh validation. Identical selected policies are evaluated once per clip, with three fresh numerical repeats.'))


def evaluate(folder, cfg, actor, prefixes, repetitions, source):
    folder.mkdir(parents=True)
    shutil.copyfile(verify_file_pin(cfg['evaluation']),folder/'controlled_evaluation_v4.toml')
    write_json(folder/'intent.json', dict(source=source,actor=actor,candidates=list(prefixes),repetitions=repetitions))
    return evaluate_cases(folder,verify_file_pin(cfg['objective']),cfg['item'],dict.fromkeys(prefixes,actor),
        dict.fromkeys(prefixes,0.),repetitions,refit_export=True,raw_prefixes=prefixes)


def censored_row(row, horizon):
    """Only pass the observed controls 1..horizon to the scoring function."""
    case = read(row['case'])
    checks = case['checks'][:horizon]
    observed = len(checks)
    ended = row['controls'] <= horizon
    stop_at_end = ended and row['stop'] != 'REFERENCE_COMPLETION'
    passed = sum(c['ok'] and not (stop_at_end and i==observed-1) for i,c in enumerate(checks))
    with np.load(verify_file_pin(case['trace'])) as tr:
        phase = np.arange(1,observed+1)
        position = tr['post_COM'][8*phase-1] - tr['COM_world_m'][phase]
        actual = tr['post_quaternion_wxyz'][8*phase-1]
        target = tr['quaternion_wxyz'][phase]
        dots = (actual*target).sum(1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(target,axis=1))
        pos = np.linalg.norm(position,axis=1)
        angle = np.rad2deg(2*np.arccos(np.clip(abs(dots),0,1)))
    cost = float((pos/.005+angle/5).mean())
    return dict(candidate=row['arm'],observed=observed,passing=passed,COM_endpoint_m=pos.tolist(),
        SO3_endpoint_deg=angle.tolist(),pose_cost=cost,cost_bin=int(np.floor(100*cost+.5)))


def rank(censor_pin, order):
    ledger = read(censor_pin)
    return min(ledger,key=lambda r:(-r['passing'],r['cost_bin'],order.index(r['candidate'])))['candidate']


def validation_row(row, frames):
    complete = row['stop']=='REFERENCE_COMPLETION'
    n = row['controls']-int(not complete)
    return dict(candidate=row['arm'],rep=row['rep'],passing=n,passing_s=.02*n,
        passed_halfway=2*n>=frames-1,complete=complete,strict=row['summary']['qualified'],
        observed=row['controls'],stop=row['stop'],reasons=row['reasons'],case=row['case'])


def run(out):
    register(out)
    registration = file_pin(out/'registration.json')
    protocol = read(registration)
    selections = {}
    budget = {'screen_controls':0,'screen_native_ticks':0,'screen_observed_by_horizon':{'8':0,'16':0},
              'validation_controls':0,'validation_native_ticks':0}
    for clip, cfg in protocol['clips'].items():
        with np.load(verify_file_pin(cfg['prefixes'])) as saved:
            prefixes = dict(saved)
        folder = out/'screen'/clip
        rows = evaluate(folder,cfg,protocol['actor'],prefixes,1,registration)
        native = read(file_pin(folder/'report.json'))
        budget['screen_controls'] += sum(r['controls'] for r in rows)
        budget['screen_native_ticks'] += native['native']
        choices = {}
        for h in HORIZONS:
            ledger = [censored_row(r,h) for r in rows]
            budget['screen_observed_by_horizon'][str(h)] += sum(r['observed'] for r in ledger)
            path = folder/f'view_{h}.json'
            write_json(path,ledger)
            pin = file_pin(path)
            choices[str(h)] = dict(winner=rank(pin,protocol['candidates']),censored_view=pin)
        selections[clip] = choices
    frozen = out/'winners.json'
    write_json(frozen,dict(registration=registration,selections=selections,screen_budget=budget.copy()))
    win_pin = file_pin(frozen)
    outcomes = {}
    for clip, choices in read(win_pin)['selections'].items():
        cfg = protocol['clips'][clip]
        ids = {'actor_from_start',*(c['winner'] for c in choices.values())}
        with np.load(verify_file_pin(cfg['prefixes'])) as saved:
            selected = {k:saved[k] for k in protocol['candidates'] if k in ids}
        folder = out/'validation'/clip
        rows = evaluate(folder,cfg,protocol['actor'],selected,3,win_pin)
        results = {k:[validation_row(r,cfg['item']['frames']) for r in rows if r['arm']==k] for k in selected}
        early,late = (results[choices[str(h)]['winner']] for h in HORIZONS)
        diff = float(np.median([r['passing'] for r in late])-np.median([r['passing'] for r in early]))
        strict_diff = sum(r['strict'] for r in late)-sum(r['strict'] for r in early)
        outcomes[clip] = dict(selected=choices,frames=cfg['item']['frames'],results=results,
            median_difference_controls=diff,criterion_met=diff>=2 and strict_diff>=0)
        budget['validation_controls'] += sum(r['controls'] for r in rows)
        budget['validation_native_ticks'] += read(file_pin(folder/'report.json'))['native']
    assert budget['screen_native_ticks']==8*budget['screen_controls']
    assert budget['validation_native_ticks']==8*budget['validation_controls']
    write_json(out/'report.json',dict(registration=registration,frozen_winners=win_pin,clips=outcomes,
        budget=budget,prospective_claim_met=all(r['criterion_met'] for r in outcomes.values())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    run(p.parse_args().out.resolve())
