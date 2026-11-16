"""Recheck saved native utility arithmetic and retain failed launch costs."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def close(a,b):return a is None and b is None if a is None or b is None else bool(np.isclose(a,b,rtol=1e-12,atol=1e-12))


def main():
    out=ROOT/'runs/pen_native_suggestions_archive_v1';archive=ROOT/'provenance/pen_native_suggestions_results_v1'
    if out.exists() or archive.exists():raise ValueError('Fresh native suggestion archive required')
    roots=dict(native=ROOT/'runs/pen_native_suggestions_v2/report.json',review=ROOT/'runs/pen_native_suggestions_review_v1/report.json',
        bank=ROOT/'runs/pen_native_suggestions_bank_v1/report.json',registration=ROOT/'runs/pen_native_suggestions_registration_v1/registration.json',
        launch=ROOT/'runs/pen_native_suggestions_registration_v2/registration.json',failure=ROOT/'runs/pen_native_suggestions_failure_diagnosis_v1/report.json',comparison=ROOT/'runs/pen_native_suggestions_comparison_v1/report.json',figure=ROOT/'runs/pen_native_suggestions_figure_v1/report.json')
    d={k:read(p) for k,p in roots.items()};run,review,bank,reg,launch,failure=[d[k] for k in ['native','review','bank','registration','launch','failure']]
    comparison=d['comparison']
    checks=dict(native8=run['passed']==run['total']==8 and all(run['checks'].values()),review12=review['passed']==review['total']==12 and all(review['checks'].values()),
        comparison8=comparison['passed']==comparison['total']==8 and all(comparison['checks'].values()) and comparison['physical_review']==file_pin(roots['review']),
        figure_context=d['figure']['physical_review']==file_pin(roots['review']) and d['figure']['comparison']==file_pin(roots['comparison']),
        bank10=bank['passed']==bank['total']==10 and all(bank['checks'].values()),failure6=failure['passed']==failure['total']==6 and all(failure['checks'].values()),
        exact_report_links=run['bank']==file_pin(roots['bank'])==review['bank'] and run['registration']==file_pin(roots['registration']) and run['launch_registration']==file_pin(roots['launch']) and review['native']==file_pin(roots['native']) and review['prior_failure']==file_pin(roots['failure']),
        retained_failed_work=failure['reconstructed_actual_native_integrations']==failure['reconstructed_attempted_native_integrations']==8 and failure['reconstructed_successful_restores']==2 and failure['original_online_qualification']=='UNAVAILABLE_DO_NOT_INFER' and not failure['returned_source_reward_retained'],
        native12008=run['actual_native_integrations']==run['attempted_native_integrations']==12000 and run['total_cohort_native']==12008,
        restores1530=run['restore_attempts']==run['restore_completed']==1530,
        no_optimizer_or_epoch=not run['default_changed'] and not run['source_epoch_used'] and run['optimizer_steps']==run['attempted_optimizer_steps']==review['optimizer_steps']==0,
        all1800_1500=run['declared_controls']==bank['declared_controls']==1800 and run['executed_controls']==bank['available_controls']==review['total_rows']==1500,
        all432_360=len(review['groups'])==432 and sum(g['complete'] for g in review['groups'])==360,
        all12_invalid_center_groups=sum(g['complete'] and not g['original_base_physical_valid'] for g in review['groups'])==12 and all(not g['result']['eligible'] for g in review['groups'] if g['complete'] and not g['original_base_physical_valid']))
    rows={x['name']:x for x in review['rows']};groups=review['groups'];arithmetic=0
    for g in groups:
        key=g['source_name']+'_'+g['model']+'_'+g['goal']+'_'+str(g['epsilon']);computed={}
        if g['complete']:
            k=7 if g['goal']=='RESIDUAL' else 8;names=g['cases'];b,p,m=[np.asarray([rows[n]['native_components'][k] for n in chunk]) for chunk in [names[:2],names[2:4],names[4:6]]]
            bg,pg,mg=map(float,[b.mean(),p.mean(),m.mean()]);gain=pg-bg;sep=pg-mg;envelope=float(abs(p-pg).max()+abs(b-bg).max());spread=float(abs(p-pg).max()+abs(m-mg).max());eligible=bool(all(rows[n]['valid'] for n in names) and g['original_base_physical_valid']);fd=sep/(8*g['epsilon'])
            computed=dict(eligible=eligible,base_mean=bg,positive_mean=pg,negative_mean=mg,positive_gain=gain,negative_gain=mg-bg,positive_gain_repeat_envelope=envelope,separation_repeat_envelope=spread,native_FD_per_RAW=fd,
                improvement_exceeds_observed_repeat_envelope=bool(eligible and gain>envelope),descent_exceeds_observed_repeat_envelope=bool(eligible and gain< -envelope),ascent_beats_reverse_exceeds_observed_repeat_envelope=bool(eligible and sep>spread))
            checks[key+'_gain_math']=all(close(v,g['result'][n]) for n,v in computed.items());arithmetic+=len(computed)
            components=[np.asarray([rows[n]['native_components'] for n in chunk]).mean(0) for chunk in [names[:2],names[2:4],names[4:6]]]
            checks[key+'_component_math']=np.array_equal(np.asarray(g['positive_component_gain']),components[1]-components[0]) and np.array_equal(np.asarray(g['negative_component_gain']),components[2]-components[0]);arithmetic+=20
        else:checks[key+'_missing_retained']=g['result'] is None
    for s in review['summary']:
        key=s['role']+'_'+s['model']+'_'+s['goal']+'_'+str(s['epsilon']);selected=[g for g in groups if (g['role'],g['model'],g['goal'],g['epsilon'])==(s['role'],s['model'],s['goal'],s['epsilon'])]
        complete=[g for g in selected if g['complete']];eligible=[g for g in complete if g['result']['eligible']];results=[g['result'] for g in eligible];gain=np.array([x['positive_gain'] for x in results]);fd=np.array([x['native_FD_per_RAW'] for x in results]);ad=np.array([x['predicted_AD_per_RAW'] for x in results]);den=float(np.sum(fd**2))
        expected=dict(declared=len(selected),complete=len(complete),missing=len(selected)-len(complete),eligible=len(eligible),new_all_physics_invalid=sum(not g['new_all_physics_valid'] for g in complete),original_center_invalid=sum(not g['original_base_physical_valid'] for g in complete),
            positive_gain_above_repeat_envelope=sum(x['improvement_exceeds_observed_repeat_envelope'] for x in results),negative_gain_above_repeat_envelope=sum(x['descent_exceeds_observed_repeat_envelope'] for x in results),ascent_beats_reverse_above_repeat_envelope=sum(x['ascent_beats_reverse_exceeds_observed_repeat_envelope'] for x in results),
            positive_gain_count=int((gain>0).sum()),mean_positive_gain=float(gain.mean()) if len(gain) else None,median_positive_gain=float(np.median(gain)) if len(gain) else None,min_positive_gain=float(gain.min()) if len(gain) else None,max_positive_gain=float(gain.max()) if len(gain) else None,
            AD_relative_L2=float(np.linalg.norm(ad-fd)/np.sqrt(den)) if den else None,two_amplitude_screen_groups=sum(g['original_two_amplitude_repeat_screen'] for g in selected))
        checks[key+'_summary_math']=all(close(v,s[n]) for n,v in expected.items());arithmetic+=len(expected)
        checks[key+'_component_summary_math']=(s['mean_positive_component_gains'] is None if not eligible else np.array_equal(np.asarray(list(s['mean_positive_component_gains'].values())),np.mean([g['positive_component_gain'] for g in eligible],axis=0)));arithmetic+=10
    packed=dict(np.load(verify_file_pin(review['packets'])));cases=run['cases']
    checks['all1500_packet_masks']=len(packed['valid'])==1500 and np.array_equal(packed['valid'],[row['valid'] for row in review['rows']]) and all(cases[int(i)]['name']==row['name'] for i,row in zip(packed['case_index'],review['rows']))
    checks['all_saved_native_values']=np.array_equal(packed['native_components'],[row['native_components'] for row in review['rows']])
    checks['all_physical_valid_masks']=all(row['valid']==bool(row['physics']['valid'] and all(row['reward_checks'].values()) and all(row['score_checks'].values())) for row in review['rows'])
    for cell in comparison['cells']:
        selected=[g for g in groups if (g['role'],g['goal'],g['epsilon'])==(cell['role'],cell['goal'],cell['epsilon'])];names=sorted({g['source_name'] for g in selected});by_source={s:{g['model']:g for g in selected if g['source_name']==s} for s in names}
        common=[s for s in names if all(g['complete'] and g['result']['eligible'] for g in by_source[s].values())];key=cell['role']+'_'+cell['goal']+'_'+str(cell['epsilon']);checks[key+'_common_sources']=common==cell['common_source_names']
        for model,record in cell['models'].items():
            eligible=[by_source[s][model] for s in names if by_source[s][model]['complete'] and by_source[s][model]['result']['eligible']];verified=sum(g['result']['improvement_exceeds_observed_repeat_envelope'] for g in eligible)
            mean=float(np.mean([by_source[s][model]['result']['positive_gain'] for s in common])) if common else None
            checks[key+'_'+model+'_declared_and_common']=record['declared_sources']==len(names) and record['verified_improving_sources']==verified and record['verified_improving_fraction_of_declared']==verified/len(names) and record['common_eligible_sources']==len(common) and close(record['common_mean_gain'],mean);arithmetic+=5
        k=7 if cell['goal']=='RESIDUAL' else 8
        for model,record in cell['paired_against_RESPONSE_ONLY'].items():
            differences=[];better=worse=0
            for source in common:
                candidate=by_source[source][model];baseline=by_source[source]['RESPONSE_ONLY'];a=np.array([rows[n]['native_components'][k] for n in candidate['cases'][2:4]]);b=np.array([rows[n]['native_components'][k] for n in baseline['cases'][2:4]])
                difference=float(a.mean()-b.mean());envelope=float(abs(a-a.mean()).max()+abs(b-b.mean()).max());differences.append(difference);better+=difference>envelope;worse+=difference< -envelope;arithmetic+=2
            checks[key+'_'+model+'_paired_math']=record['sources']==len(common) and record['better_sources']==better and record['worse_sources']==worse and close(record['mean_gain_difference'],float(np.mean(differences)) if differences else None) and close(record['median_gain_difference'],float(np.median(differences)) if differences else None)
    tests=ROOT/'runs/pen_native_suggestions_tests544_v2.log';checks['full544_tests']='Ran 544 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    receipts=[x['source_receipt'] for x in d.values() if 'source_receipt' in x]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Native suggestion archive discrepancy: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-native-suggestions-results-2026-11-16.md','research-plan-v1.41.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__),ROOT/'tools/review_pen_native_suggestions.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],independent_gain_arithmetic_checks=arithmetic,declared_controls=1800,executed_controls=1500,physical_valid_rows=review['valid_rows'],physical_total_rows=review['total_rows'],physics_flags=review['physics_flags'],reward_flags=review['reward_flags'],
        native_integrations=12000,attempted_native_integrations=12000,prior_reconstructed_native_integrations=8,total_cohort_native_integrations=12008,prior_counter_reconstruction_scope=failure['reconstruction_basis'],
        native_restores=1530,prior_reconstructed_restores=2,total_cohort_restores=1532,optimizer_steps=0,normalizer_fits=0,added_review_native_integrations=0,
        bank_wall_s=bank['wall_s'],native_wall_s=run['wall_s'],physical_review_wall_s=review['wall_s'],groups=432,complete_groups=360,original_invalid_center_groups=12,outcomes=review['summary'],common_source_comparisons=comparison['cells'],
        default_changed=False,source_epoch_used=False,scope='All fixed native one-step suggestions/reverse/base controls,original gates and missing/invalid centers retained. V1 failed serialization preserved with explicitly reconstructed8 prior ticks;no Actor/multistep/ego/task/overall acceleration claim.')
    write_json(out/'independent_review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    prior_path=ROOT/'provenance/pen_path_study_results_v1/ledger.json';prior=read(prior_path);verified={}
    for pin in read(verify_file_pin(prior['verified_manifest']))['records']:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    roots_extra=[out/'independent_review.json',out/'source_receipt.json',prior_path,tests,ROOT/'runs/pen_native_suggestions_review_preflight_v1.log']+docs
    todo=[file_pin(p) for p in list(roots.values())+roots_extra]+receipts;allowed={p.parent for p in roots.values()}|{out}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write_json(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write_json(archive/'summary.json',summary)
    write_json(archive/'ledger.json',dict(schema='pen_native_suggestions_results_ledger:1',prior_evidence=file_pin(prior_path),roots={k:file_pin(p) for k,p in roots.items()},review=file_pin(out/'independent_review.json'),summary=file_pin(archive/'summary.json'),
        documents=[file_pin(p) for p in docs],source_receipts=receipts+[file_pin(out/'source_receipt.json')],verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),native_integrations=12000,prior_reconstructed_native=8,total_cohort_native=12008,optimizer_steps=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),physical=[review['valid_rows'],review['total_rows']],cohort_native=12008)),flush=True)


if __name__=='__main__':main()
