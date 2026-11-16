"""Role/backend/plant coverage review; validation never determines TRAIN pairs."""
import argparse
import json
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.explicit_collision_backend import load_contract
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.learning.backend_probe_review import require_response_probe
from supermimic_research.learning.backend_response_study import response_spec
from supermimic_research.learning.response_coverage_review import review_coverage


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    if (cfg['schema'],cfg['native_amplitude_relative_error'],cfg['native_amplitude_cosine'])!=(1,.25,.9):raise ValueError('Exact coverage/screen protocol required')
    p={k:(cp.parent/cfg[k]).resolve() for k in ['TRAIN','VALIDATE','backend_manifest','response_registration','frozen_scale_report','protocol','registration','output']}
    contract=load_contract(p['backend_manifest']);response_reg=read(p['response_registration']);old=read(p['frozen_scale_report']);scale=np.load(verify_file_pin(old['scales']))['y_scale']
    if old['scale_input_role']!='VALID_TRAIN_ONLY' or scale.shape!=(90,) or np.any(scale<=0) or not np.isfinite(scale).all() or response_reg['backend_id']!=contract['backend_id']:raise ValueError('Pinned old TRAIN-only descriptive units and same backend required')
    inputs={k:file_pin(p[k]) for k in ['backend_manifest','response_registration','frozen_scale_report','protocol']};inputs['descriptive_units']=old['scales']
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,p['protocol']]
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=p['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh review registration required')
        out.mkdir();freeze(out);write(p['registration'],dict(schema='backend_response_data_review_registration:1',config=file_pin(cp),inputs=inputs,future_preparations={k:str(p[k]) for k in ['TRAIN','VALIDATE']},
            source_receipt=file_pin(out/'source_receipt.json'),scope='Frozen review design only;no new data/physics/fitting. Old units are descriptive only.'))
        print(json.dumps(dict(registration=file_pin(p['registration']))),flush=True);return
    reg=read(p['registration'])
    if reg['config']!=file_pin(cp) or reg['inputs']!=inputs or reg['future_preparations']!={k:str(p[k]) for k in ['TRAIN','VALIDATE']}:raise ValueError('Registered review changed')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    roles={'TRAIN':'TRAIN_CONTROLLED_PHYSICAL_RESPONSE','VALIDATE':'VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT'}
    prepared={k:read(p[k]) for k in roles};arrays={};specs={};raws={}
    for key,r in prepared.items():
        if r.get('schema')!='backend_pen_response_preparation:1' or r['role']!=roles[key] or r['collision_backend']!=contract or r['registration']!=file_pin(p['response_registration']) or r['passed']!=r['total'] or not all(r['checks'].values()) or r['actual_native_integrations'] or r['optimizer_steps'] or r['scales_fitted'] or r['CHECK_response_rows_used'] or r['validation_rows_used_for_TRAIN']:raise ValueError('Completed explicit new-backend preparation/role required')
        spec=tomllib.loads(verify_file_pin(r['config']).read_text());directions,_,_=response_spec(spec);raw=read(verify_file_pin(r['probe']));require_response_probe(raw,spec,contract,file_pin(p['response_registration']),r['config'])
        if spec['role']!=roles[key] or response_reg['specs'][roles[key]]!=r['config']:raise ValueError('Raw/prepared config role mismatch')
        z=dict(np.load(verify_file_pin(r['rows'])))
        if z['x'].shape!=(r['total_rows'],102) or z['y'].shape!=(r['total_rows'],90) or z['heads'].shape!=(r['total_rows'],90) or z['valid'].dtype!=np.bool_ or z['valid'].shape!=(r['total_rows'],) or int(z['valid'].sum())!=r['qualified_rows'] or not all(np.isfinite(z[v]).all() for v in ['x','y','heads']):raise ValueError('Exact finite aligned arrays/masks required')
        for item in [raw,r]:verify_source_snapshot(read(verify_file_pin(item['source_receipt'])),live=True)
        arrays[key]=z;specs[key]=spec;raws[key]=raw
    if any(set(specs['TRAIN'][f])&set(specs['VALIDATE'][f]) for f in ['noise_seeds','source_phases']):raise ValueError('Validation identity overlaps TRAIN')
    out=p['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh review output required')
    out.mkdir();receipt=freeze(out);checks={};cohorts={};reference=None;names=set();training_groups=None;phase='source_model'
    try:
        for key,r in prepared.items():
            raw=raws[key];d,_,_=response_spec(specs[key]);checks[key+'_directions_exact']=np.array_equal(d,np.load(verify_file_pin(r['directions']))['directions'])
            for entry in raw['sources']:
                if entry['name'] in names:raise ValueError('Duplicate source identity across cohorts')
                names.add(entry['name'])
                if not entry['available']:continue
                state=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);require_backend_probe(state,contract)
                model=state['native']['native']['model']
                if reference is None:reference=model
                same=model['scalars']==reference['scalars'] and model['arrays'].keys()==reference['arrays'].keys() and all(v.shape==reference['arrays'][k].shape and v.dtype==reference['arrays'][k].dtype and v.tobytes()==reference['arrays'][k].tobytes() for k,v in model['arrays'].items())
                checks[entry['name']+'_full_plant']=same and len(model['arrays'])==404 and len(model['scalars'])==195
            phase=key+'_finite_coverage';review=review_coverage(arrays[key],r['metadata'],r['response_groups'],d,[s['name'] for s in raw['sources']],scale,
                relative_error=cfg['native_amplitude_relative_error'],cosine=cfg['native_amplitude_cosine'])
            if key=='TRAIN':training_groups=review['eligible_pair_groups']
            # Validation's diagnostics stay isolated;do not emit TRAIN indices
            # derived from validation labels or stability screens.
            review.pop('eligible_pair_groups');failed=Counter()
            for row in r['metadata']:
                if not row['valid']:
                    failed.update(k for k,v in row['checks'].items() if not v)
                    failed.update(k for k in ['geometry_legal','online_quality','packet_quality'] if not row[k])
            cohorts[key]=dict(role=roles[key],rows=r['total_rows'],qualified_rows=r['qualified_rows'],excluded_rows=r['rejected_rows'],unavailable_case_indices=r['unavailable_case_indices'],
                groups=len(r['response_groups']),qualified_groups=r['qualified_response_groups'],failed_gate_counts=dict(failed),**review)
        checks['no_scale_or_model_fit']=True;verify_source_snapshot(receipt,live=True)
        result=dict(schema='backend_response_data_review:1',config=file_pin(cp),registration=file_pin(p['registration']),inputs=inputs,preparations={k:file_pin(p[k]) for k in roles},collision_backend=contract,
            source_receipt=file_pin(out/'source_receipt.json'),cohorts=cohorts,TRAIN_eligible_pair_group_indices=training_groups,checks=checks,passed=sum(checks.values()),total=len(checks),
            actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,validation_rows_used_for_TRAIN=0,
            scope='New-backend source/plant/PD/physical-mask coverage and finite-output spectra/screens only. Old units descriptive,not new normalization. No model fitting/selection,gradient/Actor/task/ego admission.')
        write(out/'report.json',result)
        if not all(checks.values()):raise ValueError('Independent source/data review failed')
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],TRAIN_eligible={k:len(v) for k,v in training_groups.items()})),flush=True)
    except Exception as error:write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=0,optimizer_steps=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
