"""Independent terminal data/cost/checkpoint and saved-response arithmetic review."""
import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.backend_response_fit import MODELS,load_dataset,model_spec,new_arm,exact_context
from supermimic_research.learning.checkpoint import _adam_valid,_finite_tree
from supermimic_research.learning.world_model import raw_tree_equal
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def close(a,b):return a is b if a is None or b is None else bool(np.isclose(a,b,atol=1e-11,rtol=1e-10))
BLOCKS=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90))


def main():
    out=ROOT/'runs/pen_backend_response_results_review_v1';archive=ROOT/'provenance/pen_backend_response_results_v1'
    if out.exists() or archive.exists():raise ValueError('Fresh terminal results review required')
    names=dict(TRAIN='pen_backend_response_train_v1',VALIDATE='pen_backend_response_validation_v1',
        TRAIN_preparation='pen_backend_response_train_preparation_v1',VALIDATE_preparation='pen_backend_response_validation_preparation_v1',
        data_review='pen_backend_data_review_v1',common='pen_backend_common_data_v1',
        dataset_registration='pen_backend_common_data_registration_v1',model_registration='pen_backend_response_fit_registration_v1',
        fit='pen_backend_response_fit_v1',evaluation='pen_backend_response_evaluation_v1')
    roots={k:ROOT/'runs'/v/('registration.json' if k.endswith('registration') else 'report.json') for k,v in names.items()}
    reports={k:read(p) for k,p in roots.items()};fit=reports['fit'];evaluation=reports['evaluation'];checks={}
    cp=ROOT/'configs/pen_backend_response_fit_v1.toml';spec=model_spec(tomllib.loads(cp.read_text()))
    pp,prep,data,built,context,_=load_dataset(cp,spec)
    checks['common5']=prep['passed']==prep['total']==5 and all(prep['checks'].values())
    checks['review39']=reports['data_review']['passed']==reports['data_review']['total']==39 and all(reports['data_review']['checks'].values())
    checks['fit_exact_budget']=fit['update']==2000 and not fit['training_running'] and fit['live_optimizer_steps']==12000 and fit['clone_only_optimizer_steps']==24 and fit['optimizer_steps']==12024
    checks['all_attempted_updates_complete']=fit['work']==dict(live_attempted_calls=12000,live_completed_steps=12000,QA_attempted_calls=24,QA_completed_steps=24)
    checks['validation_final_only']=fit['validation_evaluation_updates']==[2000] and fit['health_probe_role']=='TRAIN' and all(
        all(set(v)==({'TRAIN','VALIDATE'} if u=='2000' else {'TRAIN'}) for v in curve.values()) for u,curve in fit['curve'].items())
    checks['owned_evaluation10']=evaluation['passed']==evaluation['total']==10 and all(evaluation['checks'].values()) and evaluation['fit']==file_pin(roots['fit'])
    receipts=[r['source_receipt'] for r in reports.values() if 'source_receipt' in r]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    for key in ['TRAIN','VALIDATE']:
        raw=reports[key];prepared=reports[key+'_preparation'];arrays=data[key]
        checks[key+'_raw8']=raw['passed']==raw['total']==8 and all(raw['checks'].values())
        checks[key+'_native_exact']=raw['actual_native_integrations']==raw['attempted_native_integrations']==raw['producer_native_integrations']+8*raw['available_cases'] and raw['optimizer_steps']==0
        checks[key+'_rows_and_masks']=len(arrays['x'])==prepared['total_rows']==raw['available_cases'] and int(arrays['valid'].sum())==prepared['qualified_rows'] and np.array_equal(arrays['valid'],[m['valid'] for m in prepared['metadata']])
        checks[key+'_all_declared_groups']=len(prepared['response_groups'])==40*len(raw['sources']) and sum(g['qualified'] for g in prepared['response_groups'])==prepared['qualified_response_groups']
    initial=torch.load(verify_file_pin(fit['initial_NN']),map_location='cpu',weights_only=False)
    checkpoint=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False)
    checks['all_six_owned_final_states']=checkpoint['schema']=='backend_response_fit_checkpoint:1' and checkpoint['update']==2000 and list(checkpoint['arms'])==MODELS
    for name,state in checkpoint['arms'].items():
        if state['schema']!='backend_pen_response_arm:1' or not exact_context(state['context'],context):
            raise ValueError('Own external dataset/backend required before read-only CPU view')
        # A CUDA generator's16-byte state is not a CPU generator's5056-byte
        # state. This view reads weights/Adam without migrating samplers or
        # claiming CPU continuation. Actual CUDA resume parity is in111/111.
        arm=new_arm(name,context,'cpu')
        frozen=all(raw_tree_equal(v,state['model'][k]) for k,v in arm['model'].named_buffers())
        owned=state['pair_contract']==arm['pair_contract'] and state['response_contract']==arm['response_contract']
        arm['model'].load_state_dict(state['model'],strict=True)
        arm['optimizer'].load_state_dict(copy.deepcopy(state['optimizer']))
        checks[name+'_CPU_state_readback']=state['update']==2000 and frozen and owned and _adam_valid(arm['optimizer'],arm['model'],2000,torch) and _finite_tree([state['model'],state['optimizer'],state['gradient_snapshot']],torch) and raw_tree_equal(state['model'],arm['model'].state_dict()) and raw_tree_equal(state['optimizer'],arm['optimizer'].state_dict())
        checks[name+'_all2000_logs']=len(checkpoint['logs'][name])==2000 and bool(np.isfinite(checkpoint['logs'][name]).all())
        checks[name+'_parameters_changed']=not raw_tree_equal(initial,arm['model'].net.state_dict())
        del arm
    for key,pin in fit['checkpoint_health'].items():
        health=read(verify_file_pin(pin));checks['health'+key+'_111']=health['passed']==health['total']==111 and all(health['checks'].values()) and health['clone_only_optimizer_steps_at_boundary']==12
    vr=reports['VALIDATE_preparation'];d=data['VALIDATE'];scale=context['scales']['y_scale']
    prediction=dict(np.load(verify_file_pin(evaluation['predictions'])));jvp=dict(np.load(verify_file_pin(evaluation['JVP'])))
    order=[(r['source_name'],r['direction']) for r in evaluation['JVP_source_order']];lookup={key:i for i,key in enumerate(order)}
    bases={}
    for i,m in enumerate(vr['metadata']):
        if m['profile']['direction']<0:bases.setdefault(m['source_name'],i)
    checks['all_actual_bases_and20_directions']=order==[(s,j) for s in bases for j in range(20)] and evaluation['AD_direction_evaluations']==6*len(order)
    checks['all480_groups_preserved']=len(evaluation['response_pairs'])==len(vr['response_groups'])==480 and evaluation['qualified_groups']==386
    screens={(r['source'],r['direction']):r['blocks'] for r in reports['data_review']['cohorts']['VALIDATE']['finite_amplitude_screens']}
    arithmetic=[];mask_checks=[];targets={};finite={};candidate={};zero=0
    for gi,(group,row) in enumerate(zip(vr['response_groups'],evaluation['response_pairs'])):
        source,j,epsilon=group['source_name'],group['direction'],group['epsilon'];p,m=group['rows']['1'],group['rows']['-1']
        complete=len(p)==len(m)==2;eligible=bool(complete and d['valid'][p+m].all())
        flags={b:bool((source,j) in screens and screens[source,j][b]['physical_screen']) for b in BLOCKS}
        mask_checks.append(row['group_index']==gi and row['source_name']==source and row['direction']==j and row['epsilon']==epsilon
            and row['rows']==group['rows'] and row['qualified']==eligible==group['qualified'] and row['complete']==complete
            and row['physical_screens']==flags and row['base_row']==bases.get(source)
            and row['base_row_valid']==bool(source in bases and d['valid'][bases[source]]))
        if not complete:
            mask_checks.append(row['models'] is None);continue
        # Separate four-row sum formula,not the evaluator's mean-difference helper.
        target=(d['y'][p[0]]+d['y'][p[1]]-d['y'][m[0]]-d['y'][m[1]])/(4*epsilon);targets[gi]=target
        for name in MODELS:
            fd=(prediction[name][p[0]]+prediction[name][p[1]]-prediction[name][m[0]]-prediction[name][m[1]])/(4*epsilon);finite[name,gi]=fd
            ad=jvp[name][lookup[source,j]] if (source,j) in lookup else None
            for b,sl in BLOCKS.items():
                t=target[sl]/scale[sl];tn=float(np.linalg.norm(t));f=fd[sl]/scale[sl];record=row['models'][name][b]
                arithmetic.append(close(record['model_finite_response']['relative_L2_error'],float(np.linalg.norm(f-t)/tn) if tn else None))
                if ad is None:
                    arithmetic.append(record['AD'] is None);candidate[name,gi,b]=False;continue
                a=ad[sl]/scale[sl];an=float(np.linalg.norm(a));relative=float(np.linalg.norm(a-t)/tn) if tn else None
                cosine=float(a@t/(an*tn)) if an*tn>1e-24 else None
                arithmetic.extend([close(record['AD']['relative_L2_error'],relative),close(record['AD']['cosine'],cosine)])
                accepted=bool(eligible and source in bases and d['valid'][bases[source]] and flags[b] and relative is not None and relative<1 and cosine is not None and cosine>=.9)
                candidate[name,gi,b]=accepted;arithmetic.append(record['candidate_AD_screen']==accepted)
                zero+=int(tn==0)
    checks['every_original_group_mask_and_screen']=all(mask_checks)
    checks['all_saved_response_and_candidate_arithmetic']=all(arithmetic)
    aggregate_checks=[]
    for name in MODELS:
        mask=d['valid']
        for b,sl in BLOCKS.items():
            mse=float(np.mean(((prediction[name][mask,sl]-d['y'][mask,sl])/scale[sl])**2))
            aggregate_checks.append(close(mse,evaluation['forward_normalized_MSE'][name]['VALIDATE'][b]))
            for screen in ['ALL_PHYSICAL_VALID','FROZEN_PHYSICAL_SCREEN']:
                ids=[i for i,r in enumerate(evaluation['response_pairs']) if r['qualified'] and (screen=='ALL_PHYSICAL_VALID' or r['physical_screens'][b])]
                ad_ids=[i for i in ids if (evaluation['response_pairs'][i]['source_name'],evaluation['response_pairs'][i]['direction']) in lookup]
                den=sum(float(np.dot(targets[i][sl]/scale[sl],targets[i][sl]/scale[sl])) for i in ids)
                ad_den=sum(float(np.dot(targets[i][sl]/scale[sl],targets[i][sl]/scale[sl])) for i in ad_ids)
                fd_err=sum(float(np.sum(((finite[name,i][sl]-targets[i][sl])/scale[sl])**2)) for i in ids)
                ad_err=sum(float(np.sum(((jvp[name][lookup[evaluation['response_pairs'][i]['source_name'],evaluation['response_pairs'][i]['direction']]][sl]-targets[i][sl])/scale[sl])**2)) for i in ad_ids)
                r=evaluation['aggregate'][name][b][screen]
                aggregate_checks.append((r['pairs'],r['AD_pairs'],r['finite_response_pairs'],r['missing_AD_pairs'])==(len(ids),len(ad_ids),len(ids),len(ids)-len(ad_ids)))
                aggregate_checks.extend([close(r['AD_relative_L2_error'],float(np.sqrt(ad_err/ad_den)) if ad_den else None),
                    close(r['model_finite_response_relative_L2_error'],float(np.sqrt(fd_err/den)) if den else None),
                    close(r['normalized_native_squared_norm'],den),close(r['AD_subset_normalized_native_squared_norm'],ad_den),
                    r['candidate_AD_pairs']==sum(candidate[name,i,b] for i in ids)])
    checks['all_forward_and_aggregate_formulas']=all(aggregate_checks)
    log=ROOT/'runs/pen_backend_fit_tests505_v1.log';checks['full505_tests']='Ran 505 tests' in log.read_text() and '\nOK\n' in log.read_text()
    if not all(checks.values()):raise ValueError('Independent result review failed: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-backend-response-results-2026-11-16.md','research-plan-v1.34.md','pen-backend-response-fit-protocol-v1.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)]+docs,out/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],TRAIN=[3202,3240],VALIDATE=[1591,1620],
        declared_groups=dict(TRAIN=960,VALIDATE=480),qualified_groups=dict(TRAIN=779,VALIDATE=386),
        actual_native_integrations=reports['TRAIN']['actual_native_integrations']+reports['VALIDATE']['actual_native_integrations'],
        WM_live_optimizer_steps=12000,WM_QA_optimizer_steps=24,WM_optimizer_steps=12024,Actor_updates=0,
        production_normalizer_computations_this_pipeline=7,TRAIN_response_unit_computations_this_pipeline=5,
        normalization_scope='Dataset original+isolation2;model registration/run/evaluation1 each;first failed independent review and corrected review1 each. Synthetic unit operations and historical studies excluded.',
        independent_arithmetic_checks=len(arithmetic)+len(aggregate_checks),zero_target_component_cases=zero,
        fit_and_QA_wall_s=fit['fit_and_QA_wall_s'],runner_wall_s=fit['total_runner_wall_s'],evaluation_wall_s=evaluation['wall_s'],
        forward=evaluation['forward_normalized_MSE'],aggregate=evaluation['aggregate'],
        physical_status='NOT_ADMITTED_FOR_ACTOR_OPTIMIZATION',
        scope='New-backend response supervision improves some local directional predictions,but forward/object/wrench/Fn6 tradeoffs persist. Same reference/fixed actor seed and viewed development validation;not full-gradient/task/ego/acceleration qualification.')
    write(out/'independent_review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,
        source_receipt=file_pin(out/'source_receipt.json'),actual_native_integrations=0,optimizer_steps=0,
        loader_normalizer_reconstructions=1,TRAIN_response_unit_computations=1))
    prior=ROOT/'provenance/pen_backend_train_response_v1/ledger.json';diagnosis=ROOT/'runs/pen_backend_fit_readiness_diagnosis_v1'
    write(diagnosis/'diagnosis.json',dict(problem='Optimizer.load_state_dict retained CPU moment storage shared between independent clones.',
        fix='Deep-copy the owned optimizer input before load;new storage-isolation and input-immutability regression.',
        production_model_updates_before_fix=0,old_failed_unit_log=file_pin(diagnosis/'original_unit_failure.log'),
        old_source=file_pin(diagnosis/'original_fit.py'),corrected_unit_log=file_pin(ROOT/'runs/pen_backend_fit_unit_v2.log')))
    todo=[file_pin(p) for p in list(roots.values())+docs+[prior,log,out/'independent_review.json',out/'source_receipt.json']]
    todo += receipts+[file_pin(p) for p in diagnosis.iterdir() if p.is_file()]
    todo += [file_pin(p) for p in (ROOT/'runs/pen_backend_results_review_diagnosis_v1').iterdir() if p.is_file()]
    allowed={p.parent for p in roots.values()}|{out};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(p) for p in allowed):todo.extend(pins(read(path)))
    write(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write(archive/'summary.json',summary)
    write(archive/'ledger.json',dict(schema='backend_response_results_ledger:1',prior_evidence=file_pin(prior),
        roots={k:file_pin(p) for k,p in roots.items()},summary=file_pin(archive/'summary.json'),
        independent_review=file_pin(out/'independent_review.json'),source_receipts=receipts+[file_pin(out/'source_receipt.json')],
        documents=[file_pin(p) for p in docs],verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),
        added_review_native_integrations=0,added_review_optimizer_steps=0,actual_response_native_integrations=43232,
        WM_optimizer_steps=12024,Actor_updates=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),
        arithmetic_checks=summary['independent_arithmetic_checks'],physical_status=summary['physical_status'])),flush=True)


if __name__=='__main__':main()
