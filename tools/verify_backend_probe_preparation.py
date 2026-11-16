"""Numerical regression against genuine OLD-backend records, without relabeling."""
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.probe_row_review import review_recorded_row
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.learning.backend_probe_review import require_response_probe


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=ROOT/'runs/pen_backend_preparation_numeric_regression_v1'
    if out.exists():raise ValueError('Fresh numerical regression required')
    old_path=ROOT/'provenance/pen_fullrank_data_results/ledger.json';old=read(old_path)
    prior_path=ROOT/'provenance/pen_backend_response_design_v1/ledger.json';prior=read(prior_path)
    reg=read(verify_file_pin(prior['registration']));verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    protocol=ROOT/'docs/pen-backend-response-preparation-protocol-v1.md'
    paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/prepare_backend_pen_responses.py',protocol]
    out.mkdir();receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    records=[];checks={};inputs={};model_cache={};states={}
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('No native integration')) as steps,patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward reconstruction')) as forwards:
        for label,key in [('TRAIN','train_preparation'),('VALIDATE','validation_preparation')]:
            pp=verify_file_pin(old['roots'][key]);prep=read(pp);run=read(verify_file_pin(prep['probe']));cfg=tomllib.loads(verify_file_pin(prep['config']).read_text());data=dict(np.load(verify_file_pin(prep['rows'])));inputs[label]=dict(preparation=file_pin(pp),raw=prep['probe'],rows=prep['rows'])
            valid=np.flatnonzero(data['valid']);invalid=np.flatnonzero(~data['valid'])
            if len(valid)<2 or len(invalid)<2:raise ValueError('Genuine valid and invalid fixture coverage required')
            indices=[int(valid[0]),int(valid[-1]),int(invalid[0]),int(invalid[-1])]
            for i in indices:
                old_row=prep['metadata'][i];case=run['cases'][old_row['case_index']];entry=next(s for s in run['sources'] if s['name']==case['source_name'])
                if entry['source']['path'] not in states:states[entry['source']['path']]=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False)
                source=states[entry['source']['path']];modelkey=case['model']['path']
                if modelkey not in model_cache:model_cache[modelkey]=(mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model']))),dict(np.load(verify_file_pin(case['physical_context']))))
                model,physical=model_cache[modelkey];z=dict(np.load(verify_file_pin(case['trace'])));window=dict(np.load(verify_file_pin(case['window'])))
                packed,audit=review_recorded_row(case,z,window,source,model,physical,cfg)
                identity=f'{label}_{i}';local={k:packed[k].dtype==data[k][i].dtype and packed[k].tobytes()==data[k][i].tobytes() for k in ['x','y','heads']}
                local.update(mask=audit['valid']==bool(data['valid'][i]),all_original_gate_flags=audit['checks']==old_row['checks'],decoder_metrics=audit['decoder_metrics']==old_row['decoder_metrics'])
                checks.update({identity+'_'+k:bool(v) for k,v in local.items()})
                # Old source identity is deliberately NOT altered to enter the
                # new production path. Only the pure numerical audit is tested.
                try:require_backend_probe(source,dict(backend_id='not_this_old_source'))
                except ValueError:checks[identity+'_old_envelope_refused']=True
                else:checks[identity+'_old_envelope_refused']=False
                records.append(dict(cohort=label,row=i,original_valid=bool(data['valid'][i]),checks=local,case=case['name'],trace=case['trace'],window=case['window'],source=entry['source'],model=case['model'],physical_context=case['physical_context']))
            newcfg=tomllib.loads((ROOT/('configs/pen_backend_response_train_v1.toml' if label=='TRAIN' else 'configs/pen_backend_response_validation_v1.toml')).read_text())
            try:require_response_probe(run,newcfg,dict(backend_id='new'),'new-registration','new-config')
            except ValueError:checks[label+'_old_raw_production_refused']=True
            else:checks[label+'_old_raw_production_refused']=False
        checks['no_native_step_calls']=steps.call_count==0;checks['no_native_forward_calls']=forwards.call_count==0
    checks['eight_original_records']=len(records)==8
    checks['four_invalid_records_retained']=sum(not r['original_valid'] for r in records)==4
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='old_packet_numerical_regression:1',old_data_ledger=file_pin(old_path),prior_evidence=file_pin(prior_path),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
        protocol=file_pin(protocol),checks=checks,passed=sum(checks.values()),total=len(checks),records=records,actual_native_integrations=0,optimizer_steps=0,new_backend_rows=0,
        scope='Pure numerical audit regression on8 genuine OLD-backend records;4 failed masks retained,old identities never rebound. NOT new-backend native/data/gradient/Actor/task/ego qualification.')
    write(out/'report.json',result)
    if not all(checks.values()):raise ValueError('Numerical regression differs: '+repr(checks))
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],old_fixture_rows=8,new_backend_rows=0)),flush=True)


if __name__=='__main__':main()
