"""Audit the fresh eight-case matrix without treating task failure as success."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.runtime.controlled_evaluation import summarize_trace,write


def read(pin):return json.loads(verify_file_pin(pin).read_text())


def differences(a,b):
    if set(a)!=set(b):raise ValueError('Model keys differ')
    return sorted(k for k in a if a[k].shape!=b[k].shape or a[k].dtype!=b[k].dtype or a[k].tobytes()!=b[k].tobytes())


def main(a):
    if a.output.exists():raise ValueError('Fresh eight-case audit')
    runs=[json.loads(p.read_text()) for p in [a.baseline,a.tight]]
    reviews=[json.loads(p.read_text()) for p in [a.baseline_review,a.tight_review]]
    configs=[tomllib.loads(verify_file_pin(r['config']).read_text()) for r in runs]
    checks=dict(config_fixed_except_CCD={k:v for k,v in configs[0].items() if k!='collision_numerics'}=={k:v for k,v in configs[1].items() if k!='collision_numerics'},
        predeclared_CCD=[r['collision_numerics'] for r in runs]==[dict(ccd_iterations=50,ccd_tolerance=1e-6),dict(ccd_iterations=50,ccd_tolerance=1e-7)],
        candidate_table_exact=np.array_equal(np.load(verify_file_pin(runs[0]['candidates']))['raw_offsets'],np.load(verify_file_pin(runs[1]['candidates']))['raw_offsets']),
        original_proposal=runs[0]['proposal']==runs[1]['proposal'],base_config=runs[0]['base_config']==runs[1]['base_config'])
    receipts=[]
    for r in runs:
        maps=[]
        for key in ['source_receipt','installed_source_receipt','engine_source_receipt']:
            manifest=read(r[key]);verify_source_snapshot(manifest)
            maps.append({e['observed']['path']:(e['observed']['bytes'],e['observed']['sha256']) for e in manifest['records']})
        receipts.append(maps)
    checks['exact_active_installed_engine_sources']=receipts[0]==receipts[1]
    expected={(mode,dt) for mode in ['STATE_ONLY','STATE_PLUS_SUPPORT'] for dt in [.005,.0025]}
    for i,(r,v,p) in enumerate(zip(runs,reviews,[a.baseline,a.tight])):
        checks[f'profile{i}_exact_cells']={(c['name'],c['dt']) for c in r['cases']}==expected and len(r['cases'])==4
        checks[f'profile{i}_review_binding']=verify_file_pin(v['run'])==p.resolve()
        checks[f'profile{i}_fresh_task_imports']=r['source_import_enumeration_available'] and bool(r['task_sources'])
        for pin in r['task_sources']:verify_file_pin(pin)
    cells=[]
    for key in sorted(expected):
        pair=[next(c for c in r['cases'] if (c['name'],c['dt'])==key) for r in runs]
        models=[np.load(verify_file_pin(c['fixed_model_arrays'])) for c in pair]
        traces=[dict(np.load(verify_file_pin(c['trace']))) for c in pair]
        local=dict(only_CCD_array_diff=differences(*models)==['opt.ccd_tolerance'],all_model_scalars_exact=read(pair[0]['fixed_model_scalars'])==read(pair[1]['fixed_model_scalars']),
            initial_qpos_exact=np.array_equal(traces[0]['initial_qpos'],traces[1]['initial_qpos']),initial_qvel_exact=np.array_equal(traces[0]['initial_qvel'],traces[1]['initial_qvel']),
            reference_exact=all(np.array_equal(traces[0][k],traces[1][k]) for k in ['time_s','COM_world_m','quaternion_wxyz']),
            observed_layout_exact=read(pair[0]['observed_Data_layout'])==read(pair[1]['observed_Data_layout']))
        outcomes=[]
        for i,(c,z,model,run,review) in enumerate(zip(pair,traces,models,runs,reviews)):
            base=tomllib.loads(verify_file_pin(run['base_config']).read_text());layout=read(c['observed_Data_layout'])
            # Same trace dictionary used by the collector's task summary; no
            # dormant source scratch buffers or new metrics substituted here.
            dynamic={k:z[k] for k in z if k.startswith('pre_') or k.startswith('post_') or k=='overflow'}
            recomputed=summarize_trace(dynamic,{k:z[k] for k in ['time_s','COM_world_m','quaternion_wxyz']},c['dt'],c['summary']['gates']['duration_complete'],base)
            # Duration completion also checked independently against full clock.
            full_clock=len(z['post_qpos'])*c['dt']>=7.36-1e-9
            constructed=np.load(verify_file_pin(c['collision_numerics']['constructed_model_arrays']))
            changed=differences(constructed,model)
            policy=c['collision_numerics'];want=run['collision_numerics'];tol=want['ccd_tolerance']
            local.update({f'profile{i}_summary_metrics':recomputed['metrics']==c['summary']['metrics'],
                f'profile{i}_summary_gates':recomputed['gates']==c['summary']['gates'],
                f'profile{i}_full_clock':full_clock==c['summary']['gates']['duration_complete'] and c['planned_intervals']==368,
                f'profile{i}_device_CCD':model['opt.ccd_tolerance'].tobytes()==np.array([tol],np.float32).tobytes() and read(c['fixed_model_scalars'])['opt.ccd_iterations']==50,
                f'profile{i}_construction_CCD':policy['before']['host']==want and policy['before']['GPU']['ccd_tolerance']==float(np.float32(tol)),
                f'profile{i}_override_diffs':set(changed)<={'opt.ccd_tolerance'} and read(policy['constructed_model_scalars'])==read(c['fixed_model_scalars']),
                f'profile{i}_unchanged_diffs_metadata':changed==policy['changed_arrays'] and not policy['changed_scalars'],
                f'profile{i}_source_keys':all(set(np.load(verify_file_pin(d['source'])).files)==set(layout['arrays']) for d in c['planning']),
                f'profile{i}_observed_Data':layout['schema']=='observed_live_dynamic_Data_layout:1' and layout['layout']['arrays']==169 and layout['layout']['scalar_leaves']==8,
                f'profile{i}_57_budget':c['candidate_physics_ticks']==57*c['physics_steps'],
                f'profile{i}_no_aid_no_reset':c['initial_reset_count']==1 and c['resets_after_start']==0 and c['summary']['gates']['no_applied_force']})
            reviewed=next(e for e in review['cases'] if (e['mode'],e['dt'])==key)
            outcomes.append(dict(profile=['BASELINE','TIGHT'][i],duration_s=c['summary']['metrics']['actual_duration_s'],metrics=c['summary']['metrics'],task_qualified=c['summary']['qualified'],
                prediction_passed=sum(d['fidelity_passed'] for d in reviewed['decisions']),prediction_total=len(reviewed['decisions']),
                failed_predictions=[d for d in reviewed['decisions'] if not d['fidelity_passed']],candidate_ticks=c['candidate_physics_ticks'],actual_ticks=c['physics_steps'],wall_s=c['wall_s'],
                candidate_cuda_ms=sum(d['candidate_cuda_ms'] for d in c['planning'])))
        cells.append(dict(mode=key[0],dt=key[1],checks=local,outcomes=outcomes))
    passed=sum(checks.values())+sum(sum(c['checks'].values()) for c in cells);total=len(checks)+sum(len(c['checks']) for c in cells)
    a.output.mkdir();result=dict(schema='fresh_native_CCD_matrix_audit:1',source=file_pin(__file__),baseline=file_pin(a.baseline),tight=file_pin(a.tight),
        baseline_review=file_pin(a.baseline_review),tight_review=file_pin(a.tight_review),checks=checks,cells=cells,passed=passed,total=total,
        actual_integrations=sum(c['physics_steps'] for r in runs for c in r['cases']),candidate_integrations=sum(c['candidate_physics_ticks'] for r in runs for c in r['cases']),
        new_integrations=0,neural_updates=0,scope='Exact paired physics except declared CCD tolerance; fresh actual task metrics and source provenance. One episode/cell, unequal survival compute, no statistical policy-benefit/expert/ego/acceleration qualification; failed force predictions preserved.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[passed,total],actual=result['actual_integrations'],candidate=result['candidate_integrations'])))
    if passed!=total:raise ValueError('Matrix contract failed; retain report')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['baseline','tight','baseline-review','tight-review','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
