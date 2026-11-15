"""Registered repeated fresh W1 mean execution; first repeat video, all failures."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from preview_pen_support_pilot import execute,render


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());tp=(cp.parent/cfg['training_report']).resolve();gp=(cp.parent/cfg['gates']).resolve();protocol=(cp.parent/cfg['protocol']).resolve()
    training=json.loads(tp.read_text());config=verify_file_pin(training['config']);out=a.output.resolve()
    registration_path=(cp.parent/cfg['registration']).resolve();registration=json.loads(registration_path.read_text())
    if registration['config']!=file_pin(cp) or registration['protocol']!=file_pin(protocol):raise ValueError('Pre-execution evaluation registration mismatch')
    verify_source_snapshot(json.loads(verify_file_pin(registration['source_receipt']).read_text()),live=True)
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['repetitions']!=3 or cfg['case_order']!=['INITIAL','RESIDUAL','SUPPORT'] or cfg['video_repetition']!=0:raise ValueError('Fresh declared nine-execution matrix')
    if training['passed']!=training['total'] or any(c['iterations']!=64 for c in training['arms']):raise ValueError('Completed registered trial required')
    verify_source_snapshot(json.loads(verify_file_pin(training['source_receipt']).read_text()),live=True)
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),ROOT/'tools/preview_pen_support_pilot.py',ROOT/'tools/render_mug_progress_video.py',cp,gp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    gates=tomllib.loads(gp.read_text());pins=[training['arms'][0]['initial']]+[arm['checkpoint'] for arm in training['arms']];cases=[]
    with torch.jit.optimized_execution(False):
        for repetition in range(3):
            folder=out/f'REP{repetition}';folder.mkdir()
            for label,pin in zip(cfg['case_order'],pins):
                case=execute(pin,config,gates,folder/label);case.update(name=f'REP{repetition}_{label}',policy=label,repetition=repetition);cases.append(case)
                print(json.dumps(dict(name=case['name'],duration_s=case['summary']['metrics']['actual_duration_s'],qualified=case['summary']['qualified'])),flush=True)
    video=render(cases[:3],out);aggregate={}
    for label in cfg['case_order']:
        values=[c['summary']['metrics']['actual_duration_s'] for c in cases if c['policy']==label]
        aggregate[label]=dict(duration_s=values,min_s=min(values),median_s=float(np.median(values)),max_s=max(values),qualified_count=sum(c['summary']['qualified'] for c in cases if c['policy']==label))
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_support_fresh_execution_repeats:1',config=file_pin(cp),protocol=file_pin(protocol),registration=file_pin(registration_path),training=file_pin(tp),source_receipt=file_pin(out/'source_receipt.json'),
        cases=cases,aggregate=aggregate,rendering=video,actual_native_integrations=sum(c['native_integrations'] for c in cases),optimizer_steps=0,scope=cfg['scope'])
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),aggregate=aggregate,native_integrations=result['actual_native_integrations'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
