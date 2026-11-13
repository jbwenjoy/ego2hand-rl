"""Fixed120ms reference feedback extension, reusing original controller/math."""
from pathlib import Path
import argparse,json,sys,time,tomllib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.native_case import execute_native_case
from tools.run_mug_native_tracking import summarize


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main(args):
    config_path=args.config.resolve();extension=tomllib.loads(config_path.read_text());base_path=(config_path.parent/extension['base_config']).resolve();base=tomllib.loads(base_path.read_text());out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or extension['horizon']!=6 or extension['tasks']!=['REFERENCE'] or extension['controllers']!=['NATIVE_MPC']:raise ValueError('Fresh fixed H6 protocol')
    baseline_path=(config_path.parent/extension['baseline_report']).resolve();baseline=json.loads(baseline_path.read_text());review_path=(config_path.parent/extension['baseline_review']).resolve();review=json.loads(review_path.read_text())
    verify_file_pin(baseline['config'])
    if file_pin(base_path)!=baseline['config'] or (review['passed'],review['total'])!=(8,8) or verify_file_pin(review['run'])!=baseline_path:raise ValueError('Exact qualified-review baseline provenance')
    for p in baseline['sources']:verify_file_pin(p)
    merged={**base,**{k:extension[k] for k in ['horizon','tasks','controllers','max_total_oracle_steps']}};paths={k:(base_path.parent/base[k]).resolve() for k in ['oracle_parity','reference','binding']}
    parity=json.loads(paths['oracle_parity'].read_text());binding=json.loads(paths['binding'].read_text());rr=json.loads(paths['reference'].read_text());ref=dict(np.load(verify_file_pin(rr['output'])))
    if (parity['passed'],parity['total'])!=(18,18):raise ValueError('Frozen native parity')
    sources=[file_pin(ROOT/p) for p in ['tools/run_mug_native_horizon.py','src/supermimic_research/mechanics/native_case.py','tools/run_mug_native_tracking.py','src/supermimic_research/mechanics/native_tracking.py',
        'src/supermimic_research/mechanics/native_oracle.py','src/supermimic_research/mechanics/compound_transition.py','src/supermimic_research/mechanics/compound_contacts.py','src/supermimic_research/mechanics/residual_control.py','src/supermimic_research/mechanics/transition_store.py']]
    out.mkdir();write(out/'intent.json',dict(schema='native_reference_horizon_extension:1',sources=sources,config=file_pin(config_path),base_config=file_pin(base_path),effective_config=merged,baseline=file_pin(baseline_path),baseline_review=file_pin(review_path),inputs={k:file_pin(p) for k,p in paths.items()},
        changed_control_math=['prediction horizon3->6 only'],cost_scope='Same sum-of-state-residuals formula/coefficient values, with additional future states; once-per-action regularizers unchanged, no renormalization.',
        cost_comparison='30 oracle calls per decision, up to twice native integration versus H3; not matched computational budget or speedup evidence.',neural_updates=0,max_total_oracle_steps=extension['max_total_oracle_steps']))
    cases=[];steps=0;active_steps=0;start=time.perf_counter()
    for dt in base['dt']:
        case=execute_native_case(binding,ref,merged,'REFERENCE',dt,out/f'REFERENCE_NATIVE_MPC_{dt}',summarize,oracle_budget=extension['max_total_oracle_steps']-steps,on_progress=lambda x:print(json.dumps(x),flush=True));cases.append(case);steps+=case['oracle_steps'];active_steps+=case['execution']['native_steps']
        write(out/'progress.json',dict(completed_cases=len(cases),oracle_steps=steps,active_steps=active_steps,last=case,wall_s=time.perf_counter()-start));print(json.dumps(dict(dt=dt,duration=case['summary']['metrics']['actual_duration_s'],qualified=case['summary']['task_qualified'],termination=case['summary']['termination'])),flush=True)
    for pin in sources+[binding['model']]:verify_file_pin(pin)
    result=dict(schema='native_reference_horizon_extension:1',sources=sources,config=file_pin(config_path),base_config=file_pin(base_path),effective_config=merged,baseline=file_pin(baseline_path),baseline_review=file_pin(review_path),inputs={k:file_pin(p) for k,p in paths.items()},
        cases=cases,total_oracle_steps=steps,total_active_steps=active_steps,wall_s=time.perf_counter()-start,neural_updates=0,scope='Fixed horizon diagnostic, original reference/scales/box/init/physics/gates. Longer summed objective window and higher native cost, not a pure matched-time or learned-policy/CHORD success comparison.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),oracle_steps=steps,active_steps=active_steps,qualified=sum(c['summary']['task_qualified'] for c in cases))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
