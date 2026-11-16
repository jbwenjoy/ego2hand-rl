"""Regression of pure numerical coverage review on old,unmodified fixtures."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.response_coverage_review import review_coverage


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=ROOT/'runs/pen_backend_coverage_regression_v1'
    if out.exists():raise ValueError('Fresh regression output required')
    oldledger=ROOT/'provenance/pen_fullrank_data_results/ledger.json';ledger=read(oldledger);golden_path=verify_file_pin(ledger['roots']['review']);golden=read(golden_path)
    scale=np.load(verify_file_pin(golden['frozen_descriptive_scales']))['y_scale'];registration_path=ROOT/'runs/pen_backend_data_review_registration_v1/registration.json';registration=read(registration_path)
    if registration['inputs']['descriptive_units']!=golden['frozen_descriptive_scales']:raise ValueError('Exact same frozen descriptive units required')
    verify_source_snapshot(read(verify_file_pin(registration['source_receipt'])),live=True)
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/review_backend_pen_responses.py',ROOT/'configs/pen_backend_data_review_v1.toml',ROOT/'docs/pen-backend-data-review-protocol-v1.md']
    out.mkdir();receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};cohorts={};inputs={}
    for name,key in [('TRAIN','train_preparation'),('VALIDATE','validation_preparation')]:
        prep_path=verify_file_pin(ledger['roots'][key]);r=read(prep_path);raw=read(verify_file_pin(r['probe']));data=dict(np.load(verify_file_pin(r['rows'])));directions=np.load(verify_file_pin(r['directions']))['directions']
        before={k:v.tobytes() for k,v in data.items()};metadata_before=json.dumps([r['metadata'],r['response_groups']],sort_keys=True)
        review=review_coverage(data,r['metadata'],r['response_groups'],directions,[s['name'] for s in raw['sources']],scale);old=golden['cohorts'][name]
        old_ranks=[(v['source'],v['epsilon'],v['eligible_directions'],v['declared']['rank'],v.get('actual_PD',{}).get('rank')) for v in old['source_ranks']]
        new_ranks=[(v['source'],v['epsilon'],v['eligible_directions'],v['declared_input']['numerical_rank'],v.get('actual_PD_input',{}).get('numerical_rank')) for v in review['source_spectra']]
        checks[name+'_all_input_ranks_and_eligible_directions']=new_ranks==old_ranks
        checks[name+'_all_finite_amplitude_comparisons_and_spreads']=review['finite_amplitude_screens']==old['finite_amplitude_screens']
        checks[name+'_physical_screen_counts']=review['physical_screen_direction_counts']==old['physical_screen_direction_counts']
        checks[name+'_original_arrays_unchanged']=all(data[k].tobytes()==v for k,v in before.items())
        checks[name+'_metadata_and_masks_not_relabelled']=json.dumps([r['metadata'],r['response_groups']],sort_keys=True)==metadata_before
        if name=='TRAIN':checks['TRAIN_eligibility_indices_exact']=review['eligible_pair_groups']==golden['TRAIN_eligible_pair_group_indices']
        cohorts[name]=dict(old_preparation=file_pin(prep_path),old_role=r['role'],rows=r['total_rows'],valid_rows=r['qualified_rows'],groups=len(r['response_groups']),
            eligible_group_counts={k:len(v) for k,v in review['eligible_pair_groups'].items()},coverage_and_finite_diagnostics=review)
        inputs[name]=dict(preparation=file_pin(prep_path),raw=r['probe'],arrays=r['rows'],directions=r['directions'])
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='old_response_coverage_numeric_regression:1',registration=file_pin(registration_path),old_ledger=file_pin(oldledger),old_review=file_pin(golden_path),
        fixed_old_descriptive_units=golden['frozen_descriptive_scales'],source_receipt=file_pin(out/'source_receipt.json'),inputs=inputs,checks=checks,passed=sum(checks.values()),total=len(checks),cohorts=cohorts,
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,new_backend_rows=0,
        scope='Pure numerical regression on genuine OLD-backend TRAIN/validation fixtures. Old identities/masks are unchanged;no new-backend production admission,gradient/Actor/task/ego or backend-improvement conclusion.')
    write(out/'report.json',result)
    if not all(checks.values()):raise ValueError('Preserved numerical review differs: '+repr(checks))
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],old_rows=sum(v['rows'] for v in cohorts.values()),new_backend_rows=0)),flush=True)


if __name__=='__main__':main()
