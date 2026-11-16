"""Recompute recovered target bytes,coverage,pairs and provenance independently."""
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    rp=ROOT/'runs/pen_interval_targets_v1/report.json';r=read(rp);out=ROOT/'runs/pen_interval_targets_review_v1';archive=ROOT/'provenance/pen_interval_targets_results_v1'
    if out.exists() or archive.exists() or r['schema']!='pen_saved_interval_targets:1' or r['passed']!=r['total'] or not all(r['checks'].values()):raise ValueError('Fresh completed recovery review required')
    common=read(verify_file_pin(r['inputs']['common_data']));reward=read(verify_file_pin(r['inputs']['reward_chain']));reg=read(verify_file_pin(r['registration']))
    if reg['inputs']!=r['inputs'] or reg['config']!=r['config']:raise ValueError('Exact registered interval inputs required')
    objective=tomllib.loads(verify_file_pin(r['inputs']['objective_config']).read_text());limits=np.array([.002,.0002,.001,.1],np.float64)
    if [objective[k] for k in ['object_penetration_m','self_penetration_m','joint_excursion_rad','nonhand_normal_N']]!=limits.tolist():raise ValueError('Original physical path units/limits required')
    checks=dict(recovery8=r['passed']==r['total']==8,no_fitting_or_simulation=r['actual_native_integrations']==r['optimizer_steps']==r['normalizer_fits']==0,
        exact_channels=r['channels']==['object_depth_max_m','self_depth_max_m','joint_excursion_max_rad','nonhand_normal_max_N'])
    reviewed={};operations=0
    for role in ['TRAIN','VALIDATE']:
        c=r['cohorts'][role];prep=read(verify_file_pin(common['preparations'][role]));raw=read(verify_file_pin(reward['inputs'][role]));data=dict(np.load(verify_file_pin(c['outputs'])))
        native=np.load(verify_file_pin(reward['cohorts'][role]['outputs']));mask=np.asarray([m['valid'] for m in prep['metadata']],bool);row_checks=[]
        if len(c['row_audits'])!=len(prep['metadata']):raise ValueError('All declared recovery rows required')
        for i,m in enumerate(prep['metadata']):
            case=raw['cases'][m['case_index']];window=np.load(verify_file_pin(case['window']));a=c['row_audits'][i]
            direct=np.r_[window['geometry_max'],window['nonhand_max_sum'][0]].astype(np.float64)
            loss=np.float32(np.log1p((direct/limits)**2).sum())
            # All sources here are within the registered reference interval;
            # recover the hard flag directly from recorded thresholds/error bits.
            failed=bool((direct>limits).any() or int(window['errors']) or int(np.asarray(window['global_error']).reshape(-1)[0]))
            penalty=np.float32(-objective['geometry_soft_weight']*loss-objective['geometry_failure_penalty']*np.float32(failed))
            local=dict(native_window_path_bytes=equal(data['path4'][i],direct),original_trace_geometry=np.array_equal(data['path4'][i,:3],np.asarray(m['geometry'])),
                transformed=np.array_equal(data['log1p_squared_ratio4'][i],np.log1p((direct/limits)**2)),loss=bool(data['geometry_loss'][i]==loss),
                geometry_penalty=bool(data['geometry_penalty'][i]==penalty),native_chain_penalty=abs(float(penalty)-float(native['values'][i,reward['channels'].index('geometry_penalty')]))<=2e-6,
                hard_failure=data['geometry_failure'][i]==failed==bool(window['objective_geometry_failure']),
                error_bits=int(data['errors'][i])==int(window['errors']) and int(data['global_error'][i])==int(np.asarray(window['global_error']).reshape(-1)[0]),
                row_identity=a['row_index']==m['row_index']==i and a['case_index']==data['case_indices'][i]==m['case_index'] and a['name']==case['name'] and a['source_name']==case['source_name'],
                all_extraction_checks=all(a['checks'].values()))
            operations+=len(local);row_checks.append(dict(row_index=i,checks={k:bool(v) for k,v in local.items()}))
        pairs=[];pair_checks=[]
        if len(c['groups'])!=len(prep['response_groups']):raise ValueError('Original empty groups required')
        for gi,g in enumerate(prep['response_groups']):
            p,m=g['rows']['1'],g['rows']['-1'];complete=len(p)==len(m)==2;qualified=bool(complete and mask[p+m].all());stored=c['groups'][gi]
            identity=(stored['group_index']==gi and stored['source_name']==g['source_name'] and stored['direction']==g['direction'] and stored['epsilon']==g['epsilon'] and stored['complete']==complete==g['complete'] and stored['qualified']==qualified==g['qualified'])
            fd=(data['path4'][p].mean(0)-data['path4'][m].mean(0))/(2*g['epsilon']) if complete else None
            correct=stored['finite_response_per_PD'] is None if fd is None else np.array_equal(np.asarray(stored['finite_response_per_PD']),fd)
            pair_checks.append(dict(group_index=gi,identity=bool(identity),finite_response=bool(correct)))
            operations+=2
            if qualified:pairs.append(fd)
        pairs=np.asarray(pairs).reshape(-1,4);coverage={}
        for j,channel in enumerate(r['channels']):
            values=data['path4'][mask,j];fd=pairs[:,j]
            coverage[channel]=dict(physical_valid_rows=len(values),positive_rows=int((values>0).sum()),zero_rows=int((values==0).sum()),maximum=float(values.max()),eligible_pairs=len(fd),
                nonzero_finite_responses=int((fd!=0).sum()),FD_RMS_per_PD=float(np.sqrt(np.mean(fd*fd))),FD_abs_max_per_PD=float(abs(fd).max()))
        checks[role+'_all_row_math']=all(all(x['checks'].values()) for x in row_checks)
        checks[role+'_all_group_math']=all(x['identity'] and x['finite_response'] for x in pair_checks)
        checks[role+'_coverage_exact']=coverage==c['coverage']
        checks[role+'_all_masks']=equal(mask,data['physical_valid']) and equal(mask,native['physical_valid']) and int(mask.sum())==reg['qualified_rows'][role]
        checks[role+'_all_counts']=len(row_checks)==reg['rows'][role] and len(pair_checks)==reg['expected_groups'][role]
        checks[role+'_original_failure_bits']=int(data['geometry_failure'].sum())==c['failure_rows'] and int(np.count_nonzero(data['errors']))==c['error_rows'] and int(np.count_nonzero(data['global_error']))==c['global_error_rows']
        reviewed[role]=dict(row_checks=row_checks,group_checks=pair_checks,coverage=coverage,rows=len(row_checks),valid_rows=int(mask.sum()),
            flags=[sum(sum(x['checks'].values()) for x in row_checks),sum(len(x['checks']) for x in row_checks)],
            positive_failure_rows=int(data['geometry_failure'].sum()),positive_error_rows=int(np.count_nonzero(data['errors'])),positive_global_error_rows=int(np.count_nonzero(data['global_error'])))
    tests=ROOT/'runs/pen_interval_targets_tests527_v1.log';checks['full527_tests']='Ran 527 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    receipts=[r['source_receipt'],reg['source_receipt'],common['source_receipt'],reward['source_receipt']]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Saved interval review discrepancy: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-interval-targets-results-2026-11-16.md','research-plan-v1.39.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],recovery_checks=[8,8],reviewed_rows=4860,arithmetic_checks=operations,
        cohorts={k:{n:v[n] for n in ['coverage','rows','valid_rows','flags','positive_failure_rows','positive_error_rows','positive_global_error_rows']} for k,v in reviewed.items()},
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='Saved interval target/window/reward correspondence only. Positive geometry coverage and response coverage explicitly counted,not learned gradient or hard-failure predictor qualification. Original impulse masks/all1440 groups retained;no Actor/ego claim.')
    write(out/'report.json',dict(schema='pen_interval_targets_independent_review:1',recovery=file_pin(rp),checks=checks,passed=sum(checks.values()),total=len(checks),cohorts=reviewed,summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    prior_path=ROOT/'provenance/native_full_rehash_results_v1/ledger.json';prior=read(prior_path);verified={}
    for pin in read(verify_file_pin(prior['verified_manifest']))['records']:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    roots=[rp,verify_file_pin(r['registration']),out/'report.json',out/'source_receipt.json',prior_path,tests]+docs
    todo=[file_pin(p) for p in roots]+receipts;allowed={rp.parent,Path(r['registration']['path']).parent,out}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write(archive/'summary.json',summary)
    write(archive/'ledger.json',dict(schema='pen_interval_targets_results_ledger:1',prior_evidence=file_pin(prior_path),recovery=file_pin(rp),registration=r['registration'],review=file_pin(out/'report.json'),
        summary=file_pin(archive/'summary.json'),documents=[file_pin(p) for p in docs],source_receipts=receipts+[file_pin(out/'source_receipt.json')],verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],arithmetic_checks=operations,pins=len(verified),cohorts=summary['cohorts'])),flush=True)


if __name__=='__main__':main()
