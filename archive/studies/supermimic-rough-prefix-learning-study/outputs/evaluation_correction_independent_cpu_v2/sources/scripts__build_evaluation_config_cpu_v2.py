"""Freeze a selected format104 evaluation config; CPU metadata only."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin,checked
from rough_prefix.metadata import write

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--method',choices=['A1L','P_ONLY','A2'],required=True)
    ap.add_argument('--version',required=True);ap.add_argument('--health',type=Path,required=True)
    ap.add_argument('--actor-load',type=Path,required=True);ap.add_argument('--authorize',action='store_true')
    ap.add_argument('--evidence',type=Path,action='append',default=[]);a=ap.parse_args()
    train_path=ROOT/'configs'/(a.method.lower()+'-engineering_smoke-v2.json')
    train=read(train_path);protocol=ROOT/'protocols/rough-evaluation-v2.json'
    pins=dict(train['source_pins']);pins['training_protocol']=train['protocol'];pins['protocol']=pin(protocol)
    paths=[Path(__file__),ROOT/'scripts/evaluate_rough_prefix_pen_v2.py',ROOT/'scripts/run_rough_evaluation_v2.py']
    paths += [ROOT/'src/rough_prefix'/name for name in ('evaluation_v2.py','policy_probe.py','checkpoint_v2.py')]
    for path in paths:pins['evaluation__'+path.relative_to(ROOT).as_posix()]=pin(path)
    checkpoint=Path(train['output'])/'train/model_final.pt'
    for key,path in {'training_config':train_path,'checkpoint_health':a.health,
                     'actor_cpu_load':a.actor_load}.items():pins[key]=pin(path)
    for index,path in enumerate(a.evidence):pins['evaluation_evidence__'+str(index)]=pin(path)
    for value in pins.values():checked(value)
    p=read(protocol)
    config={'schema':'rough_prefix_policy_evaluation_run:1','role':'native_policy_eval',
            'method':a.method,'seed':train['seed'],'input_bundle':train['input_bundle'],
            'output':str(ROOT/'outputs'/('eval_smoke_'+a.method.lower()+'_'+a.version)),
            'checkpoint':pin(checkpoint),'checkpoint_iteration':10,'training_config':pin(train_path),
            'health_report':pin(a.health),'actor_load_report':pin(a.actor_load),
            'source_pins':pins,'protocol':pin(protocol),'GPU_run_authorized':a.authorize,'worlds':1,
            'hard_process_seconds':p['hard_process_seconds'],'soft_worker_seconds':p['soft_worker_seconds'],
            'automatic_retry':False,'compile_fallback':False,'evaluation_gates':p['evaluation_gates']}
    dest=ROOT/'configs'/(a.method.lower()+'-policy-evaluation-'+a.version+'.json')
    if dest.exists():raise FileExistsError('Never overwrite frozen evaluation configs')
    write(dest,config);print(pin(dest))

if __name__=='__main__':main()
