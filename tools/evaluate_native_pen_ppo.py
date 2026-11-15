"""Evaluate initial/iteration2/iteration32 policies on the original strict task."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.native_pen_policy_evaluation import evaluate


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    first=json.loads(a.first.read_text());final=json.loads(a.final.read_text())
    if out.exists() or not out.is_relative_to(ROOT/'runs') or first['iteration']!=2 or final['iteration']!=32:raise ValueError('Fresh declared evaluation matrix')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),cp,(cp.parent/cfg['adapter_config']).resolve(),(cp.parent/cfg['gates']).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checkpoints=[first['initial_state'],first['checkpoint'],final['checkpoint']];cases=[]
    for label,pin in zip(cfg['cases'],checkpoints):
        verify_file_pin(pin)
        with torch.jit.optimized_execution(False):r=evaluate(pin,(cp.parent/cfg['adapter_config']).resolve(),tomllib.loads((cp.parent/cfg['gates']).read_text()),out/label)
        r['name']=label;cases.append(r);print(json.dumps(dict(name=label,duration=r['summary']['metrics']['actual_duration_s'],qualified=r['summary']['qualified'])),flush=True)
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='native_pen_PPO_strict_evaluation_matrix:1',config=file_pin(cp),first=file_pin(a.first),final=file_pin(a.final),
        source_receipt=file_pin(out/'source_receipt.json'),cases=cases,actual_native_integrations=sum(c['native_integrations'] for c in cases),actual_optimizer_steps=0,
        scope=cfg['scope'])
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),integrations=result['actual_native_integrations'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--first',type=Path,required=True);p.add_argument('--final',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
