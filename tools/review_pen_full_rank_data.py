"""Role/plant/PD/rank/repeat review;validation labels never train a model."""
import argparse,json,subprocess,sys,tomllib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_interventions import validate_full_rank_cohort
from supermimic_research.learning.pen_native_response import vector_comparison,repeated_contrast_spread


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def rank_info(d):
    d=np.asarray(d,dtype=np.float64).reshape(-1,20)
    if not len(d):return dict(rows=0,rank=0,condition=None,min_nonzero_singular_value=None)
    s=np.linalg.svd(d,compute_uv=False)
    return dict(rows=len(d),rank=int(np.linalg.matrix_rank(d)),condition=float(s[0]/s[-1]) if s[-1]>1e-14 else None,min_nonzero_singular_value=float(s[-1]))


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh full-rank data review required')
    reports={k:json.loads((cp.parent/cfg[k]).resolve().read_text()) for k in ['TRAIN','VALIDATE']};roles={'TRAIN':'TRAIN_CONTROLLED_PHYSICAL_RESPONSE','VALIDATE':'VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT'}
    configurations={};arrays={}
    for key,r in reports.items():
        if r['schema']!='controlled_pen_response_preparation:2' or r['role']!=roles[key] or r['passed']!=r['total'] or r['optimizer_steps'] or r['actual_native_integrations'] or r['CHECK_response_rows_used'] or r['validation_rows_used_for_TRAIN'] or r['scales_fitted']:raise ValueError('Complete role-preserving physical preparation required')
        configurations[key]=tomllib.loads(verify_file_pin(r['config']).read_text());validate_full_rank_cohort(configurations[key]);arrays[key]=dict(np.load(verify_file_pin(r['rows'])))
        if configurations[key]['role']!=r['role']:raise ValueError('Prepared role differs from registered configuration')
        verify_source_snapshot(json.loads(verify_file_pin(r['source_receipt']).read_text()),live=True)
    for field in ['noise_seeds','source_phases']:
        if not set(configurations['TRAIN'][field]).isdisjoint(configurations['VALIDATE'][field]):raise ValueError('Reserved validation identity overlaps TRAIN')
    oldp=(cp.parent/cfg['frozen_scale_report']).resolve();old=json.loads(oldp.read_text());scale=np.load(verify_file_pin(old['scales']))['y_scale']
    if old['scale_input_role']!='VALID_TRAIN_ONLY' or scale.shape!=(90,) or np.any(scale<=0):raise ValueError('Frozen original TRAIN-only descriptive scales required')
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    blocks=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90));checks={};cohorts={};reference=None;source_names=set();training_groups={}
    for key,r in reports.items():
        z=arrays[key];d,profiles=validate_full_rank_cohort(configurations[key]);raw=json.loads(verify_file_pin(r['probe']).read_text());metadata=r['metadata'];groups=r['response_groups'];paired={};ranks=[];stability=[];failed=Counter()
        checks[key+'_finite_masked_data']=z['x'].shape==(r['total_rows'],102) and z['y'].shape==(r['total_rows'],90) and z['heads'].shape==(r['total_rows'],90) and z['valid'].dtype==np.bool_ and int(z['valid'].sum())==r['qualified_rows'] and all(np.isfinite(z[k]).all() for k in ['x','y','heads'])
        checks[key+'_directions_exact']=np.array_equal(d,np.load(verify_file_pin(r['directions']))['directions'])
        for entry in raw['sources']:
            if entry['name'] in source_names:raise ValueError('Source identities overlap across cohorts')
            source_names.add(entry['name'])
            if not entry['available']:continue
            state=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);model=state['native']['native']['model']
            if reference is None:reference=model
            same=model['scalars']==reference['scalars'] and model['arrays'].keys()==reference['arrays'].keys() and all(v.shape==reference['arrays'][k].shape and v.dtype==reference['arrays'][k].dtype and v.tobytes()==reference['arrays'][k].tobytes() for k,v in model['arrays'].items())
            checks[entry['name']+'_full_plant']=same and len(model['arrays'])==404 and len(model['scalars'])==195
        for i,m in enumerate(metadata):
            checks[key+f'_row{i}_mask_matches']=bool(z['valid'][i])==m['valid']
            if not m['valid']:
                failed.update(k for k,v in m['checks'].items() if not v)
                if not m['geometry_legal']:failed['geometry_legal']+=1
                if not m['online_quality']:failed['online_quality']+=1
                if not m['packet_quality']:failed['packet_quality']+=1
        for gi,g in enumerate(groups):
            p,n=g['rows']['1'],g['rows']['-1'];ids=p+n;complete=len(p)==len(n)==2
            if complete:
                xp,xn=z['x'][p[0]],z['x'][n[0]]
                exact=np.array_equal(z['x'][p],np.tile(xp,(2,1))) and np.array_equal(z['x'][n],np.tile(xn,(2,1))) and np.array_equal(xp[:81],xn[:81]) and xp[-1]==xn[-1]
                actual=(xp[81:101]-xn[81:101])/(2*g['epsilon']);error=float(np.max(abs(xp[81:101]-xn[81:101]-2*g['epsilon']*d[g['direction']])))
                checks[key+f'_group{gi}_actual_PD_source']=exact and error<=4e-7
                if g['qualified']!=bool(z['valid'][ids].all()):raise ValueError('Response group mask differs')
                paired[(g['source_name'],g['direction'],g['epsilon'])]=dict(group=gi,qualified=g['qualified'],actual=actual,fd=(z['y'][p].mean(0)-z['y'][n].mean(0))/(2*g['epsilon']),spread={b:repeated_contrast_spread(z['y'][p,sl],z['y'][n,sl],scale[sl]) for b,sl in blocks.items()})
            elif g['qualified']:raise ValueError('Incomplete pair cannot become eligible')
        for source in raw['sources']:
            selected=[]
            for epsilon in [.002,.005]:
                selected_ids=[j for j in range(20) if (source['name'],j,epsilon) in paired and paired[(source['name'],j,epsilon)]['qualified']]
                selected.append(set(selected_ids));ranks.append(dict(source=source['name'],epsilon=epsilon,eligible_directions=selected_ids,declared=rank_info(d[selected_ids]),actual_PD=rank_info([paired[(source['name'],j,epsilon)]['actual'] for j in selected_ids])))
            both=sorted(selected[0]&selected[1]);ranks.append(dict(source=source['name'],epsilon='BOTH',eligible_directions=both,declared=rank_info(d[both])))
            for direction in range(20):
                small=paired.get((source['name'],direction,.002));large=paired.get((source['name'],direction,.005))
                if small is None or large is None:continue
                results={}
                for b,sl in blocks.items():
                    comparison=vector_comparison(small['fd'][sl]/scale[sl],large['fd'][sl]/scale[sl]);screen=small['qualified'] and large['qualified'] and comparison['relative_L2_error']<=cfg['native_amplitude_relative_error'] and comparison['cosine'] is not None and comparison['cosine']>=cfg['native_amplitude_cosine'] and small['spread'][b]['contrast_exceeds_observed_spread'] and large['spread'][b]['contrast_exceeds_observed_spread']
                    results[b]=dict(amplitude=comparison,small_spread=small['spread'][b],large_spread=large['spread'][b],physical_screen=bool(screen))
                stability.append(dict(source=source['name'],direction=direction,blocks=results))
        if key=='TRAIN':training_groups={f'rank{count}':[i for i,g in enumerate(groups) if g['qualified'] and g['direction']<count] for count in [4,20]}
        cohorts[key]=dict(role=r['role'],rows=r['total_rows'],qualified_rows=r['qualified_rows'],excluded_rows=r['rejected_rows'],unavailable_case_indices=r['unavailable_case_indices'],groups=len(groups),qualified_groups=sum(g['qualified'] for g in groups),failed_gate_counts=dict(failed),source_ranks=ranks,finite_amplitude_screens=stability,physical_screen_direction_counts={b:sum(row['blocks'][b]['physical_screen'] for row in stability) for b in blocks})
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='full_rank_response_data_review:1',config=file_pin(cp),protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),inputs={k:file_pin((cp.parent/cfg[k]).resolve()) for k in reports},frozen_scale_report=file_pin(oldp),frozen_descriptive_scales=old['scales'],cohorts=cohorts,TRAIN_eligible_pair_group_indices=training_groups,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,validation_rows_used_for_TRAIN=0,scope='Role-preserving full-Model/PD/rank/physical-mask review and descriptive two-amplitude/two-repeat screens. No models fit or selected;VALIDATE stays NEVER_FIT. Same original reference/frozen actor seeds,not final-test/ego/gradient/task qualification.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],counts={k:{f:v[f] for f in ['rows','qualified_rows','groups','qualified_groups','physical_screen_direction_counts']} for k,v in cohorts.items()})),flush=True)
    if not all(checks.values()):raise ValueError('Retain data-review mismatch')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
