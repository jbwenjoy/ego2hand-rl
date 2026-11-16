"""Freeze eligible-TRAIN derivative RMS and a full-label isolation check."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import tomllib
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_fullrank_study import load_common
from supermimic_research.learning.pen_fullrank_dataset import full_rank_response_data
from supermimic_research.learning.pen_response_units import model_config,derivative_scale


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();spec=tomllib.loads(cp.read_text());model_config(spec);out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh immutable derivative-unit output required')
    pp,prep,data,pairs,subsets=load_common(cp,(cp.parent/spec['prepared_data']).resolve());point=np.load(verify_file_pin(prep['scales']))['y_scale']
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,(cp.parent/spec['protocol']).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    built=derivative_scale(pairs['finite_contrast'],point,role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')
    tr=json.loads(verify_file_pin(prep['TRAIN_preparation']).read_text());vr=json.loads(verify_file_pin(prep['VALIDATE_preparation']).read_text());direction=np.load(verify_file_pin(prep['directions']))['directions'];at={k:v.copy() for k,v in data['TRAIN'].items()};av={k:v.copy() for k,v in data['VALIDATE'].items()}
    for k in ['x','y','heads']:av[k]+=1e8;at[k][~at['valid']]-=1e9
    altered=full_rank_response_data(tr,at,vr,av,direction);negative=derivative_scale(altered['pairs']['finite_contrast'],altered['scales']['y_scale'],role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')
    checks={k+'_no_validation_or_invalid_TRAIN_labels':np.array_equal(v,negative[k]) for k,v in built.items()};checks['positive_finite_units']=bool(np.isfinite(built['response_scale']).all() and (built['response_scale']>0).all())
    np.savez_compressed(out/'response_units.npz',**built);verify_source_snapshot(receipt,live=True)
    r=dict(schema='valid_TRAIN_response_units:1',config=file_pin(cp),common_data=file_pin(pp),source_receipt=file_pin(out/'source_receipt.json'),pairs=prep['pairs'],point_scales=prep['scales'],arrays=file_pin(out/'response_units.npz'),input_role='VALID_TRAIN_PAIR_TARGETS_ONLY',pairs_used=949,validation_rows_used_for_TRAIN=0,CHECK_response_rows_used=0,response_floor_relative=.001,response_blocks=['object12','wrench6'],checks=checks,passed=sum(checks.values()),total=len(checks),normalizer_fits=2,response_unit_computations=2,actual_native_integrations=0,optimizer_steps=0,scope='Eligible TRAIN derivative units only,negative-label perturbation included. Point/head normalizers and forward objective unchanged;no new NN/Actor/native steps or generalization claim.')
    write(out/'report.json',r);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[r['passed'],r['total']])),flush=True)
    if not all(checks.values()):raise ValueError('Retain derivative-unit isolation failure')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
