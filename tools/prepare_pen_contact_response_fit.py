"""Frozen common point/pair data for the registered contact-response fit."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_response_dataset import common_response_data,TRAIN_ROLE,CHECK_ROLE
from supermimic_research.learning.pen_response import INPUT_SCHEMA,OUTPUT_SCHEMA


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());tp=(cp.parent/cfg['train_preparation']).resolve();tr=json.loads(tp.read_text());kp=(cp.parent/cfg['check_review']).resolve();cr=json.loads(kp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['schema']!=2 or tr['role']!=TRAIN_ROLE or cr['role']!=CHECK_ROLE or tr['qualified_rows']!=808 or cr['qualified_rows']!=324 or tr['scales_fitted']:raise ValueError('Fresh registered common808/324 data;no role reinterpretation')
    train=dict(np.load(verify_file_pin(tr['rows'])));check=dict(np.load(verify_file_pin(cr['rows'])));directions=np.load(verify_file_pin(tr['directions']))['directions'];probes={role:json.loads(verify_file_pin(report['probe']).read_text()) for role,report in [('TRAIN',tr),('CHECK',cr)]}
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    model_ref=None;model_audits=[];checks={}
    for role,probe in probes.items():
        verify_source_snapshot(json.loads(verify_file_pin(probe['source_receipt']).read_text()))
        for entry in probe['sources']:
            if not entry['available']:continue
            state=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);verify_source_snapshot(state['native']['sources']);model=state['native']['native']['model']
            if model_ref is None:model_ref=model
            same=model['scalars']==model_ref['scalars'] and set(model['arrays'])==set(model_ref['arrays']) and all(np.array_equal(v,model_ref['arrays'][k]) for k,v in model['arrays'].items())
            checks[role+'_'+entry['name']+'_full_Model']=same and len(model['arrays'])==404
            model_audits.append(dict(role=role,name=entry['name'],source=entry['source'],arrays=len(model['arrays']),exact=same))
    built=common_response_data(tr,train,cr,check,directions)
    for name,arrays in [('TRAIN',train),('CHECK',check),('TRAIN_pairs',built['pairs']),('scales',built['scales']),('head_scales',built['head_scales'])]:np.savez_compressed(out/(name+'.npz'),**arrays)
    # Deliberately alter CHECK labels and rejected TRAIN labels in independent
    # copies:the common normalizers and pair supervision must remain bitwise.
    altered_train={k:v.copy() for k,v in train.items()};altered_check={k:v.copy() for k,v in check.items()}
    for key in ['y','heads']:
        altered_check[key][:,:84]+=1e8;altered_train[key][~train['valid'],:84]+=1e9
    negative=common_response_data(tr,altered_train,cr,altered_check,directions)
    for key in ['scales','head_scales','pairs']:checks['no_CHECK_or_invalid_TRAIN_in_'+key]=set(built[key])==set(negative[key]) and all(np.array_equal(v,negative[key][k]) for k,v in built[key].items())
    checks.update(TRAIN_mask_unchanged=np.array_equal(np.load(out/'TRAIN.npz')['valid'],train['valid']),CHECK_mask_unchanged=np.array_equal(np.load(out/'CHECK.npz')['valid'],check['valid']),
        declared_pairs=len(built['pairs']['epsilon'])==188,declared_groups=len(built['pair_audit'])==192,masked_groups=sum(not r['eligible'] for r in built['pair_audit'])==4,
        common_physics=all(s['physics']==probes['TRAIN']['sources'][0]['physics'] for p in probes.values() for s in p['sources'] if s['available']))
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='contact_pen_common_response_data:2',config=file_pin(cp),protocol=file_pin(protocol),TRAIN_preparation=file_pin(tp),CHECK_review=file_pin(kp),source_receipt=file_pin(out/'source_receipt.json'),
        outputs={role:file_pin(out/(role+'.npz')) for role in ['TRAIN','CHECK']},pairs=file_pin(out/'TRAIN_pairs.npz'),scales=file_pin(out/'scales.npz'),head_scales=file_pin(out/'head_scales.npz'),
        physics=probes['TRAIN']['sources'][0]['physics'],directions=tr['directions'],input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,full_Model_audits=model_audits,pair_audit=built['pair_audit'],
        TRAIN_valid=int(train['valid'].sum()),TRAIN_total=len(train['valid']),CHECK_valid=int(check['valid'].sum()),CHECK_total=len(check['valid']),TRAIN_pairs=188,masked_TRAIN_rows=tr['rejected_rows'],
        scale_input_role='VALID_TRAIN_ONLY',CHECK_role=cfg['CHECK_role'],checks={k:bool(v) for k,v in checks.items()},passed=sum(checks.values()),total=len(checks),
        actual_native_integrations=0,optimizer_steps=0,neural_updates=0,normalizer_fits=2,
        scope='One original reference:independent-noise TRAIN and inspected developmental CHECK,never final test. Two normalizer computations include an identical-result deliberate leakage perturbation check;no model/Actor optimization or targeted replay data used.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],TRAIN=[result['TRAIN_valid'],result['TRAIN_total']],CHECK=result['CHECK_valid'],pairs=188,neural_updates=0)),flush=True)
    if not all(checks.values()):raise ValueError('Retain common-data contract discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
