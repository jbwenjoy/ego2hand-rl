"""Actual numerical-setting conformance, original control and failed task audit."""
from pathlib import Path
import argparse
import json
import sys
import tomllib
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.runtime.controlled_evaluation import summarize_trace


def main(args):
    if args.output.exists():
        raise ValueError('Fresh full numerical episode audit')
    r = json.loads(args.report.read_text())
    for pin in r['sources'] + r['task_sources']:
        verify_file_pin(pin)
    efc = json.loads(args.efc_report.read_text())
    if verify_file_pin(efc['run']).resolve() != args.report.resolve():
        raise ValueError('Exactly the original EFC review including failures')
    cfg = tomllib.loads(verify_file_pin(r['config']).read_text())
    base = tomllib.loads(verify_file_pin(r['base_config']).read_text())
    f = json.loads(verify_file_pin(r['factorial']).read_text())
    fc = tomllib.loads(verify_file_pin(f['config']).read_text())
    p = json.loads(verify_file_pin(r['proposal']).read_text())
    selected = np.load(verify_file_pin(p['artifacts']['selected']))
    qref = np.load(verify_file_pin(p['clip']))['reference_qpos']
    parent = json.loads(verify_file_pin(p['run']).read_text())
    source = np.load(verify_file_pin(next(c for c in parent['cases'] if c['name']=='PROJECTED_CONSTANT_RESIDUAL' and c['dt']==.005)['trace']))
    expected = {(v, dt, profile) for v in ['DEFAULT', r['actual_alternative']] for dt in cfg['dt'] for profile in cfg['profiles']}
    if {(c['numerical_variant'],c['dt'],c['profile']) for c in r['cases']}!=expected or len(r['cases'])!=len(expected):
        raise ValueError('Exactly the fixed16-case comparison')
    items, tables = [], []
    for c in r['cases']:
        z = dict(np.load(verify_file_pin(c['trace'])))
        physical = dict(np.load(verify_file_pin(c['GPU_physical_context'])))
        tables.append(physical)
        m = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        offset = np.zeros(20)
        if c['profile']!='BASELINE':
            k = int(c['profile'][1])
            offset = np.sin(np.pi*(np.arange(20)+.5)*k/20)
            offset *= .04/abs(offset).max() * (1 if c['profile'].endswith('PLUS') else -1)
        raw = np.tile((2*selected['delta20']).astype(np.float32),(len(z['actions20']),1))
        raw[:5] += offset.astype(np.float32)
        memory=selected['delta20'].astype(np.float32)
        targets=[]
        jids=m.actuator_trnid[selected['aids'],0]
        lo=np.maximum(physical['jnt_range'][0,jids,0],physical['actuator_ctrlrange'][0,selected['aids'],0])
        hi=np.minimum(physical['jnt_range'][0,jids,1],physical['actuator_ctrlrange'][0,selected['aids'],1])
        for phase, action in zip(z['pre_action_phase'],raw):
            memory=.5*np.clip(.5*action,-.6,.6)+.5*memory
            targets.append(np.clip(qref[int(phase)]+memory,lo,hi))
        request=fc['numerical_options'][c['numerical_variant']]
        receipt=c['numerical_options']
        wanted={k:request.get(k,default) for k,default in dict(iterations=10,ls_iterations=20,tolerance=1e-8,ls_tolerance=.01).items()}
        override=c['numerical_variant']!='DEFAULT' and cfg['device_override']
        actual_wanted={k:(wanted[k] if k in ['iterations','ls_iterations'] else float(np.float32(wanted[k] if k!='tolerance' or override else max(wanted[k],1e-6)))) for k in wanted}
        summary=summarize_trace(z,z,c['dt'],len(z['actions20'])==c['planned_intervals'],base)
        metric_error=max(abs(summary['metrics'][k]-c['summary']['metrics'][k]) for k in summary['metrics'])
        checks=dict(explicit_request=receipt['requested']==request,
            actual_device_policy=receipt['device_override']==override and receipt['GPU']==actual_wanted,
            actual_host_settings=receipt['host']==wanted,
            recorded_before_floor=receipt['before_override']['GPU']['tolerance']==float(np.float32(max(wanted['tolerance'],1e-6))),
            solver_iteration_counts=bool(np.all((z['post_solver_niter']>=0)&(z['post_solver_niter']<=receipt['GPU']['iterations']))),
            selected_pose=np.array_equal(z['initial_qpos'],selected['qpos'].astype(np.float32).astype(float)),
            original_velocity=np.array_equal(z['initial_qvel'],source['initial_qvel']),
            raw_actions=np.array_equal(raw,z['actions20']),
            original_EMA_PD=float(abs(np.repeat(targets,round(.02/c['dt']),axis=0)-z['post_ctrl'][:,selected['aids']]).max())<2e-7,
            original_phase=np.array_equal(z['pre_action_phase'],np.arange(1,len(z['actions20'])+1)),
            state_carries=np.array_equal(z['pre_qpos'][1:],z['post_qpos'][:-1]) and np.array_equal(z['pre_qvel'][1:],z['post_qvel'][:-1]),
            no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
            first_terminal=not np.any(z['terminated_truncated'][:-1]) and bool(z['terminated_truncated'][-1,0]),
            metrics=metric_error<1e-12, unchanged_task_gates=summary['gates']==c['summary']['gates'])
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt'])
        items.append(dict(name=c['name'],dt=c['dt'],checks={k:bool(v) for k,v in checks.items()},
            metric_error=metric_error,EFC_passed=sum(e['checks'].values()),EFC_total=len(e['checks']),
            duration_s=c['summary']['metrics']['actual_duration_s'],qualified=c['summary']['qualified']))
    same_tables=all(all(np.array_equal(t[k],tables[0][k]) for k in tables[0]) for t in tables)
    args.output.mkdir()
    result=dict(schema='pen_quality_full_saved_review:1',run=file_pin(args.report),source=file_pin(__file__),cases=items,
        passed=sum(sum(c['checks'].values()) for c in items),total=sum(len(c['checks']) for c in items),
        physical_tables_equal=same_tables,EFC_review=file_pin(args.efc_report),EFC_checks=[efc['passed'],efc['total']],
        combined_review_passed=all(all(c['checks'].values()) for c in items) and same_tables and efc['passed']==efc['total'],
        neural_updates=0,simulation_integration_steps=0,
        scope='Saved actual numerical arrays versus explicit floor/override policy, original selected-state/control/clock, physical tables, task metrics and EFC receipt status. All failed task and DEFAULT momentum gates remain failed; not calibrated global numerical repair.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','total','physical_tables_equal','EFC_checks','combined_review_passed']}))
    if not result['combined_review_passed']:raise ValueError('Preserve mixed full-case audit status')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--efc-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
