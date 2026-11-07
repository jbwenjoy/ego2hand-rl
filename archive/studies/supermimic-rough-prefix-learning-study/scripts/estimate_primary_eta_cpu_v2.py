"""JSONL-only segmented ETA, with strictly same-arm smoke policy proxies.

The V1 cost/window/range/20s formula is retained. Optional snapshot mode replays
a pinned historical prefix; it does not publish a current completion forecast.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin
from rough_prefix.metadata import write
METHODS=('A1L','P_ONLY','A2')

def complete_rows(path):
    data=Path(path).read_bytes();lines=data.splitlines(keepends=True)
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

def require_method(method):
    if method not in METHODS:raise ValueError('Unknown primary method; only exact A1L/P_ONLY/A2 are supported')
    return method

def smoke_proxy_path(method):
    return ROOT/'outputs'/('smoke_'+require_method(method).lower()+'_v2')/'metrics.jsonl'

def validate_health(config_path,health_path):
    p=read(config_path);h=read(health_path);require_method(p.get('method'))
    if p.get('role')!='primary_matched' or h.get('status')!='PASS_SAVED_ENGINEERING_HEALTH' or not h.get('checks') or not all(type(v) is bool and v for v in h['checks'].values()) or type(h.get('iteration')) is not int or h['iteration']!=100:raise ValueError('Require actual primary healthy100 prefix')
    if h.get('config')!=pin(config_path) or h.get('method')!=p['method'] or type(h.get('seed')) is not int or h['seed']!=p['seed']:raise ValueError('ETA/health configuration, arm or seed differs')
    if Path(h['checkpoint']['path']).resolve()!=Path(p['output']).resolve()/'train/model_100.pt':raise ValueError('Healthy100 checkpoint belongs to a different primary run')
    return p,h

def validate_smoke_identity(primary,smoke):
    if smoke.get('method')!=require_method(primary.get('method')) or smoke.get('role')!='engineering_smoke' or smoke.get('seed')!=primary['seed'] or smoke.get('input_bundle')!=primary['input_bundle'] or smoke.get('controller')!=primary['controller']:raise ValueError('Cross-arm/source/seed smoke policy proxy is forbidden')

def smoke_policy_proxy(primary):
    path=smoke_proxy_path(primary['method']);config=path.parent/'config.json';smoke=read(config);validate_smoke_identity(primary,smoke)
    rows,raw=complete_rows(path);cost=policy_cost(rows)
    return cost,{'file':pin(path),'prefix_sha256':hashlib.sha256(raw).hexdigest(),'smoke_config':pin(config),'method':primary['method'],'cross_arm_proxy':False}

def validate_snapshot(snapshot,config_path,health_path,rows,prefix):
    # Existing V1 snapshots carry these exact pins in adjacent report.json.
    sidecar=Path(snapshot).parent/'report.json';r=read(sidecar)
    if r.get('config')!=pin(config_path) or r.get('health100')!=pin(health_path) or r.get('metrics_committed_prefix_sha256')!=hashlib.sha256(prefix).hexdigest() or not rows or r.get('latest_committed_iteration')!=rows[-1]['iteration']+1:raise ValueError('Snapshot/config/health/prefix identity differs; cross-arm replay is forbidden')
    return pin(sidecar)

def calculate(primary,rows,primary_policy,policy,kind,policy_source):
    if not rows:raise ValueError('No committed metrics')
    if [r.get('iteration') for r in rows]!=list(range(len(rows))):raise ValueError('Committed iteration prefix is not contiguous from0')
    wm=[r for r in rows if r['iteration']<primary['training']['wm_only_iters']]
    if len(wm)<20:raise ValueError('Need measured WM-only steady window')
    first=wm[max(5,len(wm)-51)];last=wm[-1];count=last['iteration']-first['iteration'];base=(last['training/time']-first['training/time'])/count
    if policy is None or base<=0 or policy['seconds_per_policy_update']<=0:raise ValueError('Invalid measured rates')
    total=primary['training']['max_iterations'];completed=rows[-1]['iteration']+1;remaining=max(0,total-completed);policy_left=max(0,total-max(completed,primary['training']['wm_only_iters']));estimate=remaining*base+policy_left*policy['seconds_per_policy_update']+20.
    return {'latest_committed_iteration':completed,'total_iterations':total,'base_WM_collect_log_holdout_seconds_per_iteration':base,'base_measurement_window':{'first_iteration':first['iteration'],'last_iteration':last['iteration'],'intervals':count},'policy_cost':{**policy,'kind':kind,'source':policy_source},'remaining_iterations':remaining,'remaining_policy_updates':policy_left,'unmeasured_future_checkpoint_and_phase_overhead_allowance_seconds':20.,'point_remaining_seconds':estimate,'rough_remaining_range_seconds':[estimate*.5,estimate*1.8]}

def estimate(config_path,health_path,metrics_snapshot=None):
    p,h=validate_health(config_path,health_path);run=Path(p['output']);metrics=Path(metrics_snapshot) if metrics_snapshot is not None else run/'metrics.jsonl';rows,prefix=complete_rows(metrics);historical=metrics_snapshot is not None
    snapshot_source=validate_snapshot(metrics,config_path,health_path,rows,prefix) if historical else None
    primary_policy=policy_cost(rows)
    if primary_policy is None:
        policy,source=smoke_policy_proxy(p);kind='Same-source '+p['method']+' engineering smoke steady policy-phase proxy; primary policy not yet measured'
    else:policy=primary_policy;kind='Measured primary policy-phase cost';source={'run':str(run),'method':p['method'],'cross_arm_proxy':False}
    numbers=calculate(p,rows,primary_policy,policy,kind,source);now=datetime.now(timezone.utc);local=timezone(timedelta(hours=8));seconds=numbers['point_remaining_seconds']
    report={'schema':'rough_primary_segmented_ETA:2','as_of_utc':now.isoformat(),'mode':'historical_snapshot_replay_not_current_ETA' if historical else 'live_committed_primary_prefix','method':p['method'],'config':pin(config_path),'health100':pin(health_path),**numbers,
            'point_finish_local_UTCplus8':None if historical else (now+timedelta(seconds=seconds)).astimezone(local).isoformat(),
            'rough_finish_range_local_UTCplus8':None if historical else [(now+timedelta(seconds=seconds*x)).astimezone(local).isoformat() for x in (.5,1.8)],
            'uncertainty':'Planning range only, not a confidence interval. Primary Actor phase is unmeasured until after200; same-arm smoke proxy may miss long-run costs. Hardware/load/runtime costs may change.',
            'first100_scope':'WM/failure-head prefix engineering health only; no Actor/Critic optimizer updates or policy-quality qualification.',
            'wall_scope':'Elapsed training wall, not GPU busy; historical reference construction/smoke costs separate.',
            'metrics_source':pin(metrics),'snapshot_identity_report':snapshot_source,'metrics_committed_prefix_sha256':hashlib.sha256(prefix).hexdigest(),'physics_steps_added':0,'GPU_used':False,'source':pin(__file__)}
    return report,prefix

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--health',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--metrics-snapshot',type=Path,help='Historical committed prefix; adjacent report.json must bind exact config/health/prefixSHA')
    a=ap.parse_args();report,prefix=estimate(a.config.resolve(strict=True),a.health.resolve(strict=True),a.metrics_snapshot.resolve(strict=True) if a.metrics_snapshot else None);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(a.output/'metrics_prefix_snapshot.jsonl').write_bytes(prefix);write(a.output/'report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
