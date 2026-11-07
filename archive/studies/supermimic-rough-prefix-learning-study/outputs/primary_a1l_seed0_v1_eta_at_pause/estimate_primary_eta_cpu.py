"""Read saved/live JSONL prefixes for a segmented, explicitly uncertain ETA."""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin
from rough_prefix.metadata import write


def complete_rows(path):
    data=path.read_bytes();lines=data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):lines.pop()
    return [json.loads(line) for line in lines],b''.join(lines)


def policy_cost(rows):
    values=[r for r in rows if r.get('profile/calls/policy',0)>0]
    if len(values)<2:return None
    # Drop the first three policy calls when enough samples exist; retain the
    # original graph/cold-start cost separately in measured elapsed time.
    first=values[2] if len(values)>4 else values[0];last=values[-1]
    calls=last['profile/calls/policy']-first['profile/calls/policy']
    return {'seconds_per_policy_update':(last['profile/seconds/policy']-first['profile/seconds/policy'])/calls,
            'measured_policy_calls':calls,'first_iteration':first['iteration'],'last_iteration':last['iteration']} if calls>0 else None


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--health',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();p=read(a.config);h=read(a.health)
    if p['method']!='A1L':raise ValueError('This ETA consumer uses only the declared A1L smoke policy proxy')
    if h['status']!='PASS_SAVED_ENGINEERING_HEALTH' or not all(h['checks'].values()) or h['iteration']!=100:
        raise ValueError('Require the actual healthy100 prefix')
    if h['config']!=pin(a.config):raise ValueError('ETA/health configuration differs')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    run=Path(p['output']);rows,prefix=complete_rows(run/'metrics.jsonl')
    if not rows:raise ValueError('No committed metrics')
    wm=[r for r in rows if r['iteration']<p['training']['wm_only_iters']]
    if len(wm)<20:raise ValueError('Need measured WM-only steady window')
    first=wm[max(5,len(wm)-51)];last=wm[-1]
    count=last['iteration']-first['iteration']
    base=(last['training/time']-first['training/time'])/count
    primary_policy=policy_cost(rows)
    if primary_policy is None:
        proxy_path=ROOT/'outputs/smoke_a1l_v2/metrics.jsonl';proxy_rows,proxy_raw=complete_rows(proxy_path)
        policy=policy_cost(proxy_rows);kind='Same-source A1L engineering smoke steady policy-phase proxy; primary policy not yet measured'
        policy_source={'file':pin(proxy_path),'prefix_sha256':hashlib.sha256(proxy_raw).hexdigest()}
    else:policy=primary_policy;kind='Measured primary policy-phase cost';policy_source={'run':str(run)}
    if policy is None or base<=0 or policy['seconds_per_policy_update']<=0:raise ValueError('Invalid measured rates')
    total=p['training']['max_iterations'];completed=rows[-1]['iteration']+1
    remaining=max(0,total-completed);policy_left=max(0,total-max(completed,p['training']['wm_only_iters']))
    estimate=remaining*base+policy_left*policy['seconds_per_policy_update']+20.
    now=datetime.now(timezone.utc);local=timezone(timedelta(hours=8))
    report={'schema':'rough_primary_segmented_ETA:1','as_of_utc':now.isoformat(),'config':pin(a.config),
            'health100':pin(a.health),'latest_committed_iteration':completed,'total_iterations':total,
            'base_WM_collect_log_holdout_seconds_per_iteration':base,
            'base_measurement_window':{'first_iteration':first['iteration'],'last_iteration':last['iteration'],'intervals':count},
            'policy_cost':{**policy,'kind':kind,'source':policy_source},
            'remaining_iterations':remaining,'remaining_policy_updates':policy_left,
            'unmeasured_future_checkpoint_and_phase_overhead_allowance_seconds':20.,
            'point_remaining_seconds':estimate,'rough_remaining_range_seconds':[estimate*.5,estimate*1.8],
            'point_finish_local_UTCplus8':(now+timedelta(seconds=estimate)).astimezone(local).isoformat(),
            'rough_finish_range_local_UTCplus8':[(now+timedelta(seconds=estimate*x)).astimezone(local).isoformat() for x in (.5,1.8)],
            'uncertainty':'Planning range only, not a confidence interval. Primary Actor phase is unmeasured until after200; smoke proxy may miss long-run costs. Hardware/load/runtime costs may change.',
            'first100_scope':'WM/failure-head prefix engineering health only; no Actor/Critic optimizer updates or policy-quality qualification.',
            'wall_scope':'Elapsed training wall, not GPU busy; historical reference construction/smoke costs separate.',
            'metrics_committed_prefix_sha256':hashlib.sha256(prefix).hexdigest(),'physics_steps_added':0,'source':pin(__file__)}
    (a.output/'metrics_prefix_snapshot.jsonl').write_bytes(prefix);write(a.output/'report.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
