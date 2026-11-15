"""Independent report arithmetic and predeclared stable-response subset summary."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    r=json.loads(a.review.read_text());cfg=tomllib.loads(verify_file_pin(r['config']).read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or r['passed']!=r['total']:raise ValueError('Fresh successful-readback screen summary')
    out.mkdir();receipt=freeze_sources([Path(__file__).resolve()],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};amplitudes={(p['source_name'],p['direction']):p for p in r['amplitude_comparisons']};pairs={(p['source_name'],p['direction'],p['epsilon']):p for p in r['response_pairs']}
    for i,row in enumerate(r['response_pairs']):
        other=pairs[(row['source_name'],row['direction'],.005 if row['epsilon']==.002 else .002)]
        for block in ['object12','wrench6','Fn6']:
            comp=amplitudes[(row['source_name'],row['direction'])]['blocks'][block]
            stable=bool(comp['relative_L2_error']<=cfg['native_amplitude_relative_error'] and comp['cosine'] is not None and comp['cosine']>=cfg['native_amplitude_cosine']
                and row['qualified'] and other['qualified'] and row['blocks'][block]['contrast_exceeds_observed_spread'] and other['blocks'][block]['contrast_exceeds_observed_spread'])
            checks[f'{i}_{block}_native_screen']=stable==row['blocks'][block]['two_amplitude_consistency_screen']
            for mode in r['aggregate']:
                model=row['models'][mode][block]
                expected=bool(stable and r['checks'][mode+'_physics'] and model['relative_L2_error']<cfg['model_AD_relative_error'] and model['cosine'] is not None and model['cosine']>=cfg['model_AD_cosine'])
                checks[f'{i}_{block}_{mode}_model_screen']=expected==model['candidate_diagnostic_screen']
    aggregate={}
    for mode in r['aggregate']:
        aggregate[mode]={}
        for block in ['object12','wrench6','Fn6']:
            aggregate[mode][block]={}
            for subset in ['ALL_QUALIFIED','TWO_AMPLITUDE_SCREEN']:
                rows=[p for p in r['response_pairs'] if p['qualified'] and (subset=='ALL_QUALIFIED' or p['blocks'][block]['two_amplitude_consistency_screen'])]
                numerator=sum(p['models'][mode][block]['absolute_L2_error']**2 for p in rows);denominator=sum(p['models'][mode][block]['target_L2']**2 for p in rows)
                value=float(np.sqrt(numerator/max(denominator,1e-24)))
                aggregate[mode][block][subset]=dict(pairs=len(rows),AD_relative_L2_error=value,zero_response_relative_L2_error=1.,candidate_screen_pairs=sum(p['models'][mode][block]['candidate_diagnostic_screen'] for p in rows))
                if subset=='ALL_QUALIFIED':checks[mode+'_'+block+'_aggregate']=bool(np.isclose(value,r['aggregate'][mode][block]['AD_relative_L2_error'],atol=1e-12,rtol=1e-12))
    checks['complete_pair_population']=len(pairs)==48 and len(amplitudes)==24
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='trained_actor_physical_screen_summary:1',review=file_pin(a.review),source_receipt=file_pin(out/'source_receipt.json'),aggregate=aggregate,
        checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Independent screen/count/normalized aggregate arithmetic. Stable finite-response subset was predeclared; neither it nor passing record checks establish a smooth physical Jacobian,trained-seed generalization or Actor-use qualification.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],aggregate=aggregate)),flush=True)
    if not all(checks.values()):raise ValueError('Retain screen arithmetic discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
