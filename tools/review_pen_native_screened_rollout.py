"""Review sealed completed cases incrementally;final54 only after producer end."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import mujoco
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json
from supermimic_research.mechanics.pen_screened_review import ScreenedCaseReviewer


def read(p):return json.loads(Path(p).read_text())


def main(args):
    started=time.perf_counter();producer=ROOT/'runs/pen_native_screened_rollout_v1';out=ROOT/'runs/pen_native_screened_review_v1';registration=ROOT/'runs/pen_native_screened_rollout_registration_v1/registration.json';reg=read(registration);intent=read(producer/'intent.json')
    if intent['cases']!=reg['cases']:raise ValueError('Exact54 declared identity required')
    final_path=producer/'report.json';final=read(final_path) if final_path.exists() else None
    if not args.completed_only and (final is None or final['passed']!=final['total'] or not all(final['checks'].values())):raise ValueError('Final all54 producer report required;use completed-only for sealed partial cases')
    if (producer/'failure.json').exists() and not args.completed_only:raise ValueError('Producer failed;partial case evidence may be reviewed explicitly')
    available=54 if final is not None else read(producer/'work.json')['completed_cases'];declared=reg['cases'][:available]
    if not out.exists():
        out.mkdir();source=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',source);write_json(out/'intent.json',dict(registration=file_pin(registration),producer_source_receipt=file_pin(producer/'source_receipt.json'),declared_cases=reg['cases'],CPU_values_scaled_L2_bound=1e-5,CPU_RAW_scaled_L2_bound=1e-4))
    verify_source_snapshot(read(out/'source_receipt.json'),live=True);old=read(out/'intent.json')
    if old['registration']!=file_pin(registration) or old['producer_source_receipt']!=file_pin(producer/'source_receipt.json'):raise ValueError('Sealed review input changed')
    verify_source_snapshot(read(producer/'source_receipt.json'),live=True);fit=read(verify_file_pin(reg['inputs']['model_fit']));checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False);objective=tomllib.loads(verify_file_pin(reg['inputs']['objective_config']).read_text());adapter=(Path(reg['inputs']['objective_config']['path']).parent/objective['adapter_config']).resolve();reward=tomllib.loads(adapter.read_text())['reward'];torch.set_num_threads(4);reviewer=ScreenedCaseReviewer(checkpoint,reward,objective);case_reviews=[]
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('No native replay in saved review')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward replay in saved review')):
        for declaration in declared:
            name=declaration['name'];case_path=producer/name/'report.json';case_pin=file_pin(case_path);target=out/name
            if target.exists():
                result=read(target/'report.json')
                if result['producer_case']!=case_pin or result['passed']!=result['total'] or not all(result['checks'].values()):raise ValueError('Prior case review changed or failed;retain it')
            else:
                target.mkdir();t=time.perf_counter();case=read(case_path)
                if any(case[k]!=v for k,v in declaration.items()):raise ValueError('Declared case header changed')
                try:
                    result=reviewer.case(case,case_path.parent);verify_file_pin(case_pin);result.update(schema='pen_native_screened_case_saved_review:1',producer_case=case_pin,source_receipt=file_pin(out/'source_receipt.json'),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-t);write_json(target/'report.json',result)
                    if result['passed']!=result['total'] or not all(result['checks'].values()):raise ValueError('Saved case discrepancy retained: '+repr([k for k,v in result['checks'].items() if not v]))
                except Exception as e:
                    write_json(target/'failure.json',dict(type=type(e).__name__,message=str(e),producer_case=case_pin,actual_native_integrations=0,optimizer_steps=0));raise
            case_reviews.append(dict(name=name,report=file_pin(target/'report.json'),summary=result['summary']))
    verify_source_snapshot(read(out/'source_receipt.json'),live=True)
    partial=dict(schema='pen_native_screened_completed_case_review_progress:1',registration=file_pin(registration),source_receipt=file_pin(out/'source_receipt.json'),reviewed_cases=case_reviews,declared_total=54,complete=len(case_reviews)==54 and final is not None,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,invocation_wall_s=time.perf_counter()-started,
        scope='Independent saved completed cases only;partial review is not all54 completion or policy success. All native masks retained;CPU/device recomputation bounds are broad math checks,not changed physics gates.')
    write_json(out/'progress.json',partial)
    if not args.completed_only:
        reviewed=[read(verify_file_pin(x['report'])) for x in case_reviews]
        checks=dict(all54=len(case_reviews)==54,case_order=[x['name'] for x in case_reviews]==[x['name'] for x in reg['cases']],native_exact=final['actual_native_integrations']==final['attempted_native_integrations']==8*sum(x['summary']['native_queries'] for x in case_reviews),producer_complete=final['passed']==final['total'] and all(final['checks'].values()),all_case_reviews=all(x['passed']==x['total'] and all(x['checks'].values()) for x in reviewed))
        result=dict(partial,schema='pen_full54_native_screened_saved_review:1',run=file_pin(final_path),checks=checks,passed=sum(checks.values()),total=len(checks),original_physics_flags=sum(x['summary']['original_physics_flags'] for x in case_reviews),original_reward_flags=sum(x['summary']['original_reward_flags'] for x in case_reviews));write_json(out/'report.json',result)
        if not all(checks.values()):raise ValueError('Full54 review discrepancy retained')
    print(json.dumps(dict(progress=file_pin(out/'progress.json'),reviewed_cases=len(case_reviews),full54=partial['complete'],native=0,optimizer=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--completed-only',action='store_true');main(p.parse_args())
