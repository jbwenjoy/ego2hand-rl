"""Independent saved partial-chain/projection/contrast arithmetic and archive."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def close(a,b):return a is b if a is None or b is None else bool(np.isclose(a,b,atol=1e-11,rtol=1e-9))


def native_targets(components,prep):
    targets={'OBSERVABLE_NATIVE':np.column_stack([components[:,4],components[:,4]+.2*components[:,6]]),'FULL_NATIVE':components[:,[7,8]]}
    fd={};screens={}
    for target,z in targets.items():
        table={};fd[target]={}
        for gi,g in enumerate(prep['response_groups']):
            p,m=g['rows']['1'],g['rows']['-1']
            if len(p)!=2 or len(m)!=2:continue
            # Independent four-row sum and direct repeat-deviation envelope.
            derivative=(z[p[0]]+z[p[1]]-z[m[0]]-z[m[1]])/(16*g['epsilon'])
            signal=abs((z[p[0]]+z[p[1]]-z[m[0]]-z[m[1]])/2)
            noise=abs(z[p[0]]-z[p[1]])/2+abs(z[m[0]]-z[m[1]])/2
            fd[target][gi]=derivative
            table[g['source_name'],g['direction'],g['epsilon']]=(g['qualified'],derivative,signal,noise)
        screens[target]={}
        for source,j in dict.fromkeys((g['source_name'],g['direction']) for g in prep['response_groups']):
            a,b=table.get((source,j,.002)),table.get((source,j,.005));flags=[]
            for k in range(2):
                flags.append(bool(a is not None and b is not None and a[0] and b[0] and b[1][k]!=0
                    and abs(a[1][k]-b[1][k])/abs(b[1][k])<=.25 and a[1][k]*b[1][k]>0
                    and a[2][k]>a[3][k] and b[2][k]>b[3][k]))
            screens[target][source,j]=flags
    return fd,screens


def main():
    out=ROOT/'runs/pen_RAW_reward_independent_review_v1';archive=ROOT/'provenance/pen_RAW_reward_gradient_results_v1'
    if out.exists() or archive.exists():raise ValueError('Fresh RAW reward review required')
    rp=ROOT/'runs/pen_backend_RAW_reward_gradient_v2/report.json';r=read(rp)
    original_path=ROOT/'runs/pen_backend_RAW_reward_gradient_v1/report.json';original=read(original_path)
    native=read(verify_file_pin(r['inputs']['native_reward_audit']));common=read(verify_file_pin(r['inputs']['common_data']))
    arrays=dict(np.load(verify_file_pin(r['arrays'])));checks=dict(root58=r['passed']==r['total']==58 and all(r['checks'].values()),
        no_updates=all(r[k]==0 for k in ['normalizer_fits','actual_native_integrations','optimizer_steps','Actor_updates']))
    old_arrays=dict(np.load(verify_file_pin(original['arrays'])))
    checks['original_group_statistics_exact']=r['cohorts']==original['cohorts']
    checks['all_original_base_arrays_exact']=all(a.dtype==arrays[k.replace('_values','_base_values') if k.endswith('_values') else k].dtype
        and a.shape==arrays[k.replace('_values','_base_values') if k.endswith('_values') else k].shape
        and a.tobytes()==arrays[k.replace('_values','_base_values') if k.endswith('_values') else k].tobytes() for k,a in old_arrays.items())
    arithmetic=[];aggregate_checks=[];curves={};screen_counts={}
    for role,c in r['cohorts'].items():
        prep=read(verify_file_pin(common['preparations'][role]));basis=np.load(verify_file_pin(prep['directions']))['directions']
        components=np.load(verify_file_pin(native['cohorts'][role]['outputs']))['values']
        targets,screens=native_targets(components,prep)
        screen_counts[role]={t:[sum(flags[k] for flags in d.values()) for k in range(2)] for t,d in screens.items()}
        checks[role+'_all_declared_groups']=all(len(v['groups'])==len(prep['response_groups']) for v in c['models'].values())
        curves[role]={}
        for name,m in c['models'].items():
            prefix=role+'_'+name+'_';v=arrays[prefix+'values'];gradient=arrays[prefix+'RAW']
            checks[name+'_'+role+'_full_and_base_score_shapes']=v.shape==(prep['total_rows'],2) and arrays[prefix+'base_values'].shape==(len(m['base_source_order']),2)
            chain=arrays[prefix+'PD_partial']*arrays[prefix+'controller_diagonal'][:,None,:]+arrays[prefix+'RAW_direct_partial']
            checks[name+'_'+role+'_saved_chain_exact']=gradient.dtype==chain.dtype and gradient.tobytes()==chain.tobytes()
            checks[name+'_'+role+'_quarter_gain']=bool(np.all(arrays[prefix+'controller_diagonal']==.25))
            order={source:i for i,source in enumerate(m['base_source_order'])};cache={}
            for gi,(g,row) in enumerate(zip(prep['response_groups'],m['groups'])):
                done=len(g['rows']['1'])==len(g['rows']['-1'])==2
                arithmetic.append(row['qualified']==g['qualified'] and row['complete']==g['complete']==done and row['group_index']==gi)
                if not done:arithmetic.append(row['metrics'] is None);continue
                p,q=g['rows']['1'],g['rows']['-1'];fd=(v[p[0]]+v[p[1]]-v[q[0]]-v[q[1]])/(16*g['epsilon'])
                source,j=g['source_name'],g['direction']
                ad=np.sum(gradient[order[source]].astype(np.float64)*basis[j][None],axis=-1) if source in order else None
                base=m['base_row_indices'][order[source]] if source in order else None
                base_valid=base is not None and prep['metadata'][base]['valid']
                arithmetic.append(row['base_row']==base and row['base_physical_valid']==base_valid)
                candidates={}
                for target in targets:
                    truth=targets[target][gi];candidates[target]=[]
                    for k in range(2):
                        score=row['metrics'][target][k];flag=screens[target][source,j][k]
                        relative=abs(ad[k]-truth[k])/abs(truth[k]) if ad is not None and truth[k]!=0 else None
                        accept=bool(g['qualified'] and base_valid and flag and relative is not None and relative<1 and ad[k]*truth[k]>0)
                        arithmetic.extend([close(score['native_FD_per_RAW'],float(truth[k])),close(score['model_FD_per_RAW'],float(fd[k])),
                            close(score['model_AD_per_RAW'],float(ad[k]) if ad is not None else None),score['physical_scalar_screen']==flag,
                            score['candidate_local_AD']==accept]);candidates[target].append(accept)
                cache[gi]=dict(fd=fd,ad=ad,base_valid=base_valid,candidates=candidates)
            curves[role][name]=m['aggregate']
            for target,scores in m['aggregate'].items():
                for k,(score,subsets) in enumerate(scores.items()):
                    for subset,observed in subsets.items():
                        ids=[i for i,g in enumerate(prep['response_groups']) if g['qualified'] and (subset=='ALL_PHYSICAL_VALID' or screens[target][g['source_name'],g['direction']][k])]
                        ad_ids=[i for i in ids if cache[i]['ad'] is not None]
                        t=np.asarray([targets[target][i][k] for i in ids]);a_t=np.asarray([targets[target][i][k] for i in ad_ids])
                        fd=np.asarray([cache[i]['fd'][k] for i in ids]);ad=np.asarray([cache[i]['ad'][k] for i in ad_ids]);ad_fd=np.asarray([cache[i]['fd'][k] for i in ad_ids])
                        den=float(t@t);ad_den=float(a_t@a_t);fd_den=float(ad_fd@ad_fd)
                        aggregate_checks.extend([observed['pairs']==len(ids),observed['AD_pairs']==len(ad_ids),observed['missing_AD_pairs']==len(ids)-len(ad_ids),
                            observed['invalid_native_base_pairs']==sum(not cache[i]['base_valid'] for i in ad_ids),
                            close(observed['AD_relative_L2'],float(np.linalg.norm(ad-a_t)/np.sqrt(ad_den)) if ad_den else None),
                            close(observed['model_FD_relative_L2'],float(np.linalg.norm(fd-t)/np.sqrt(den)) if den else None),
                            close(observed['AD_vs_model_FD_relative_L2'],float(np.linalg.norm(ad-ad_fd)/np.sqrt(fd_den)) if fd_den else None),
                            observed['same_sign_AD_pairs']==int(np.sum(ad*a_t>0)),
                            observed['candidate_local_AD_pairs']==sum(cache[i]['candidates'][target][k] for i in ids),
                            close(observed['normalized_native_squared_norm'],den),close(observed['AD_subset_native_squared_norm'],ad_den)])
    checks['all_partial_projection_FD_candidates']=all(arithmetic);checks['all_aggregate_denominators_and_ratios']=all(aggregate_checks)
    tests=ROOT/'runs/pen_RAW_reward_gradient_tests516_v1.log';checks['full516_tests']='Ran 516 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    receipts=[r['source_receipt'],read(verify_file_pin(r['registration']))['source_receipt'],original['source_receipt'],read(verify_file_pin(original['registration']))['source_receipt']]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Independent RAW review failed: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-RAW-reward-gradient-results-2026-11-16.md','research-plan-v1.36.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__)],out/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],arithmetic_checks=len(arithmetic)+len(aggregate_checks),
        source_counts={role:[c['observed_base_sources'],c['declared_sources']] for role,c in r['cohorts'].items()},
        scalar_screen_direction_counts=screen_counts,aggregate=curves,analysis_wall_s=r['wall_s'],original_analysis_wall_s=original['wall_s'],
        analysis_attempts=2,original_export_problem='Full-row score key overwritten by BASE-only values;original group statistics and all base arrays are exact against V2.',
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,Actor_updates=0,
        physical_status=r['physical_status'],scope='Numerically coherent one-step RAW partial chain;native reward directions remain inaccurate,especially across development source states. All six results retained,no Actor/full-reward/multistep/task/ego admission.')
    write(out/'independent_review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    prior_path=ROOT/'provenance/pen_reward_chain_results_v1/ledger.json';prior=read(prior_path);parent=read(verify_file_pin(prior['verified_manifest']))
    verified={}
    for pin in parent['records']:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    diagnosis=ROOT/'runs/pen_RAW_reward_export_diagnosis_v1'
    todo=[file_pin(p) for p in [rp,original_path,out/'independent_review.json',out/'source_receipt.json',prior_path,tests]+docs]+receipts
    todo += [file_pin(p) for p in diagnosis.iterdir() if p.is_file()]
    allowed={rp.parent,original_path.parent,out,verify_file_pin(r['registration']).parent,verify_file_pin(original['registration']).parent}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write(archive/'summary.json',summary)
    write(archive/'ledger.json',dict(schema='pen_RAW_reward_gradient_results_ledger:1',prior_evidence=file_pin(prior_path),
        summary=file_pin(archive/'summary.json'),analysis=file_pin(rp),review=file_pin(out/'independent_review.json'),
        source_receipts=receipts+[file_pin(out/'source_receipt.json')],documents=[file_pin(p) for p in docs],
        verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),added_native_integrations=0,added_optimizer_steps=0,normalizer_fits=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),arithmetic_checks=summary['arithmetic_checks'])),flush=True)


if __name__=='__main__':main()
