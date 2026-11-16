"""One full20 point bank,matched rank pair subsets,and validation isolation."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_fullrank_dataset import full_rank_response_data,TRAIN_ROLE,VALIDATION_ROLE
from supermimic_research.learning.pen_interventions import validate_full_rank_cohort
from supermimic_research.learning.pen_response import INPUT_SCHEMA,OUTPUT_SCHEMA


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());tp=(cp.parent/cfg['train_preparation']).resolve();vp=(cp.parent/cfg['validation_preparation']).resolve();rp=(cp.parent/cfg['data_review']).resolve();tr=json.loads(tp.read_text());vr=json.loads(vp.read_text());review=json.loads(rp.read_text());out=a.output.resolve()
    if (out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['schema']!=3 or tr['role']!=TRAIN_ROLE or vr['role']!=VALIDATION_ROLE
            or review['schema']!='full_rank_response_data_review:1' or review['passed']!=review['total'] or not all(review['checks'].values())
            or review['inputs']!={'TRAIN':file_pin(tp),'VALIDATE':file_pin(vp)} or review['validation_rows_used_for_TRAIN'] or review['optimizer_steps']):
        raise ValueError('Passing exact full20 TRAIN/reserved-validation/data-review inputs required')
    for report in [tr,vr]:
        spec=tomllib.loads(verify_file_pin(report['config']).read_text());validate_full_rank_cohort(spec)
        if spec['role']!=report['role']:raise ValueError('Prepared role differs from config')
        verify_source_snapshot(json.loads(verify_file_pin(report['source_receipt']).read_text()),live=True)
    train=dict(np.load(verify_file_pin(tr['rows'])));validation=dict(np.load(verify_file_pin(vr['rows'])));directions=np.load(verify_file_pin(tr['directions']))['directions']
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    built=full_rank_response_data(tr,train,vr,validation,directions)
    at={k:v.copy() for k,v in train.items()};av={k:v.copy() for k,v in validation.items()}
    for k in ['x','y','heads']:av[k]+=1e8;at[k][~train['valid']]-=1e9
    negative=full_rank_response_data(tr,at,vr,av,directions);checks={}
    for k in ['scales','head_scales','pairs','pair_subsets']:
        checks['no_validation_or_invalid_TRAIN_in_'+k]=built[k].keys()==negative[k].keys() and all(np.array_equal(v,negative[k][name]) for name,v in built[k].items())
    checks['pair_audit_unchanged']=built['pair_audit']==negative['pair_audit']
    for name,z in [('TRAIN',train),('VALIDATE',validation),('TRAIN_pairs',built['pairs']),('scales',built['scales']),('head_scales',built['head_scales']),('pair_subsets',built['pair_subsets'])]:np.savez_compressed(out/(name+'.npz'),**z)
    checks['pair_group_indices_match_review']=all(built['pairs']['TRAIN_group_index'][built['pair_subsets'][name]].tolist()==review['TRAIN_eligible_pair_group_indices'][name] for name in ['rank4','rank20'])
    checks.update(TRAIN_mask_unchanged=np.array_equal(np.load(out/'TRAIN.npz')['valid'],train['valid']),VALIDATE_mask_unchanged=np.array_equal(np.load(out/'VALIDATE.npz')['valid'],validation['valid']),rank20_contains_rank4=set(built['pair_subsets']['rank4'].tolist())<=set(built['pair_subsets']['rank20'].tolist()))
    raw=json.loads(verify_file_pin(tr['probe']).read_text());physics=next(s['physics'] for s in raw['sources'] if s['available'])
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='contact_pen_fullrank_response_data:3',config=file_pin(cp),protocol=file_pin(protocol),TRAIN_preparation=file_pin(tp),VALIDATE_preparation=file_pin(vp),data_review=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),outputs={r:file_pin(out/(r+'.npz')) for r in ['TRAIN','VALIDATE']},pairs=file_pin(out/'TRAIN_pairs.npz'),scales=file_pin(out/'scales.npz'),head_scales=file_pin(out/'head_scales.npz'),pair_subsets=file_pin(out/'pair_subsets.npz'),pair_audit=built['pair_audit'],physics=physics,directions=tr['directions'],input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,TRAIN_total=len(train['valid']),TRAIN_valid=int(train['valid'].sum()),VALIDATE_total=len(validation['valid']),VALIDATE_valid=int(validation['valid'].sum()),TRAIN_pairs=len(built['pairs']['epsilon']),pair_counts={k:len(v) for k,v in built['pair_subsets'].items()},scale_input_role='VALID_TRAIN_ONLY',validation_role=VALIDATION_ROLE,validation_rows_used_for_TRAIN=0,CHECK_response_rows_used=0,actual_native_integrations=0,optimizer_steps=0,neural_updates=0,normalizer_fits=2,checks=checks,passed=sum(checks.values()),total=len(checks),scope='One full20 valid-TRAIN point bank/scales and two pair-direction subsets;reserved development validation never fits or normalizes. Negative-copy check included. No NN/Actor/physics updates,task/ego/final-test or gradient qualification.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],TRAIN=[result['TRAIN_valid'],result['TRAIN_total']],VALIDATE=[result['VALIDATE_valid'],result['VALIDATE_total']],pairs=result['pair_counts'])),flush=True)
    if not all(checks.values()):raise ValueError('Retain fullrank data-isolation mismatch')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
