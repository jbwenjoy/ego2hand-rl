"""Independent additive attribution and original-mask native FD arithmetic."""
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.pen_path_response import ARMS
from supermimic_research.strict_json import write_json
from review_pen_path_study import native_screens


def read(p):return json.loads(Path(p).read_text())
def close(a,b):return a is None and b is None if a is None or b is None else bool(np.isclose(a,b,rtol=1e-12,atol=1e-12))


def main():
    rp=ROOT/'runs/pen_path_decomposition_v1/report.json';r=read(rp);out=ROOT/'runs/pen_path_decomposition_review_v1'
    if out.exists() or r['passed']!=r['total'] or not all(r['checks'].values()):raise ValueError('Fresh completed decomposition required')
    evaluation=read(verify_file_pin(r['inputs']['evaluation']));fit=read(verify_file_pin(r['inputs']['fit']));common=read(verify_file_pin(fit['inputs']['common_data']));native=read(verify_file_pin(evaluation['inputs']['native_reward']))
    arrays=dict(np.load(verify_file_pin(r['arrays'])));prior=dict(np.load(verify_file_pin(evaluation['arrays'])));checks=dict(analysis73=r['passed']==r['total']==73,zero_work=r['actual_native_integrations']==r['optimizer_steps']==r['normalizer_fits']==0);counts=0;reviewed={}
    for role,cp in common['preparations'].items():
        p=read(verify_file_pin(cp));truth=np.load(verify_file_pin(native['cohorts'][role]['outputs']))['values'];dirs=np.load(verify_file_pin(p['directions']))['directions'];metadata=p['metadata']
        targets=dict(OBSERVABLE_NATIVE=np.column_stack([truth[:,4],truth[:,4]+.2*truth[:,6]]),FULL_NATIVE=truth[:,[7,8]]);screens={t:native_screens(v,p) for t,v in targets.items()};reviewed[role]={}
        for name in ARMS:
            prefix=role+'_'+name+'_';m=r['cohorts'][role]['models'][name];v=arrays[prefix+'values'];grad=arrays[prefix+'RAW'];pd=arrays[prefix+'PD_partial'];direct=arrays[prefix+'RAW_direct_partial'];diagonal=arrays[prefix+'controller_diagonal']
            checks[prefix+'value_additivity']=bool(np.max(abs(v[:,2]-v[:,0]-v[:,1]))<=1e-12)
            for key,a in [('RAW',grad),('PD',pd),('direct',direct)]:
                budget=8*np.finfo(np.float32).eps*(1+abs(a[:,2])+abs(a[:,0]+a[:,1]));checks[prefix+key+'_additivity']=bool(np.all(abs(a[:,2]-a[:,0]-a[:,1])<=budget));counts+=a[:,2].size
            checks[prefix+'RAW_chain']=np.array_equal(grad,pd*diagonal[:,None,None,:]+direct) and not bool(direct[:,1].any())
            checks[prefix+'prior_FULL']=np.array_equal(v[:,2],prior[prefix+'values']) and np.array_equal(grad[:,2],prior[prefix+'RAW'])
            attrs=[]
            for i,source in enumerate(m['base_source_order']):
                for k in range(2):
                    obs,path,full=grad[i,:,k].astype(np.float64);no,npath,nfull=map(float,map(np.linalg.norm,[obs,path,full]));cos=None if not no or not nfull else float(np.clip(np.dot(obs,full)/(no*nfull),-1,1))
                    expected=dict(observable_norm=no,path_norm=npath,full_norm=nfull,path_to_full_norm=None if not nfull else npath/nfull,full_observable_cosine=cos,
                        full_observable_angle_deg=None if cos is None else float(np.degrees(np.arccos(cos))),unit_direction_difference=None if cos is None else float(np.linalg.norm(obs/no-full/nfull)))
                    attrs.append(all(close(value,m['source_attribution'][i][k][key]) for key,value in expected.items()));counts+=len(expected)
            checks[prefix+'all_attribution']=all(attrs)
            base_map=dict(zip(m['base_source_order'],m['base_row_indices']));base_index={s:i for i,s in enumerate(m['base_source_order'])};modes={}
            for mi,mode in [(0,'OBSERVABLE'),(2,'FULL')]:
                report=m['modes'][mode];computed={};group_flags=[]
                for gi,g in enumerate(p['response_groups']):
                    source,j,e=g['source_name'],g['direction'],g['epsilon'];plus,minus=g['rows']['1'],g['rows']['-1'];done=len(plus)==len(minus)==2;eligible=bool(done and all(metadata[i]['valid'] for i in plus+minus));stored=report['groups'][gi];bi=base_map.get(source);base_valid=bi is not None and bool(metadata[bi]['valid'])
                    ok=stored['group_index']==gi and stored['source_name']==source and stored['complete']==done==g['complete'] and stored['qualified']==eligible==g['qualified'] and stored['base_row']==bi and stored['base_physical_valid']==base_valid
                    if done:
                        fd=(v[plus,mi].mean(0)-v[minus,mi].mean(0))/(8*e);ad=None if source not in base_index else grad[base_index[source],mi].astype(np.float64)@dirs[j];native_fd={t:(values[plus].mean(0)-values[minus].mean(0))/(8*e) for t,values in targets.items()}
                        computed[gi]=dict(fd=fd,ad=ad,targets=native_fd,base_valid=base_valid)
                        for target,t in native_fd.items():
                            for k in range(2):
                                q=stored['metrics'][target][k];rel=None if ad is None or not t[k] else abs(ad[k]-t[k])/abs(t[k]);same=ad is not None and ad[k]*t[k]>0
                                candidate=bool(eligible and base_valid and screens[target][source,j][k] and rel is not None and rel<1 and same)
                                ok=ok and close(q['native_FD_per_RAW'],float(t[k])) and close(q['model_FD_per_RAW'],float(fd[k])) and close(q['model_AD_per_RAW'],None if ad is None else float(ad[k])) and q['physical_scalar_screen']==screens[target][source,j][k] and q['candidate_local_AD']==candidate;counts+=5
                    else:ok=ok and stored['metrics'] is None
                    group_flags.append(bool(ok))
                aggregate_flags=[]
                for target in targets:
                    for k,score in enumerate(['residual','support']):
                        for subset in ['ALL_PHYSICAL_VALID','FROZEN_SCALAR_SCREEN']:
                            ids=[i for i,g in enumerate(p['response_groups']) if g['qualified'] and (subset=='ALL_PHYSICAL_VALID' or screens[target][g['source_name'],g['direction']][k])];ai=[i for i in ids if computed[i]['ad'] is not None]
                            den=sum(float(computed[i]['targets'][target][k]**2) for i in ids);ad_den=sum(float(computed[i]['targets'][target][k]**2) for i in ai)
                            ae=sum(float((computed[i]['ad'][k]-computed[i]['targets'][target][k])**2) for i in ai);fe=sum(float((computed[i]['fd'][k]-computed[i]['targets'][target][k])**2) for i in ids);md=sum(float(computed[i]['fd'][k]**2) for i in ai);me=sum(float((computed[i]['ad'][k]-computed[i]['fd'][k])**2) for i in ai)
                            expected=dict(pairs=len(ids),AD_pairs=len(ai),missing_AD_pairs=len(ids)-len(ai),invalid_native_base_pairs=sum(not computed[i]['base_valid'] for i in ai),AD_relative_L2=float(np.sqrt(ae/ad_den)) if ad_den else None,model_FD_relative_L2=float(np.sqrt(fe/den)) if den else None,
                                AD_vs_model_FD_relative_L2=float(np.sqrt(me/md)) if md else None,same_sign_AD_pairs=sum(bool(computed[i]['ad'][k]*computed[i]['targets'][target][k]>0) for i in ai),candidate_local_AD_pairs=sum(report['groups'][i]['metrics'][target][k]['candidate_local_AD'] for i in ids),normalized_native_squared_norm=den,AD_subset_native_squared_norm=ad_den)
                            aggregate_flags.append(all(close(value,report['aggregate'][target][score][subset][key]) for key,value in expected.items()));counts+=len(expected)
                checks[prefix+mode+'_groups']=len(group_flags)==len(p['response_groups']) and all(group_flags);checks[prefix+mode+'_aggregates']=len(aggregate_flags)==8 and all(aggregate_flags)
                modes[mode]=dict(groups=len(group_flags),all_group_checks=all(group_flags),all_aggregate_checks=all(aggregate_flags))
            reviewed[role][name]=dict(modes=modes,attribution_summary=m['attribution_summary'])
        for name in ARMS[1:]:
            base=arrays[role+'_RESPONSE_ONLY_RAW'][:,0].astype(np.float64);other=arrays[role+'_'+name+'_RAW'][:,0].astype(np.float64);den=float(np.sum(base**2));relative=float(np.sqrt(np.sum((other-base)**2)/den)) if den else None
            checks[role+'_'+name+'_representation_difference']=close(relative,r['cohorts'][role]['models'][name]['between_model_observable_relative_L2'])
    tests=ROOT/'runs/pen_path_decomposition_tests546_v1.log';checks['full546_tests']='Ran 546 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    if not all(checks.values()):raise ValueError('Decomposition review discrepancy: '+repr([k for k,v in checks.items() if not v]))
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/review_pen_path_study.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    result=dict(schema='pen_path_decomposition_independent_review:1',analysis=file_pin(rp),checks=checks,passed=sum(checks.values()),total=len(checks),arithmetic_checks=counts,cohorts=reviewed,source_receipt=file_pin(out/'source_receipt.json'),
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,scope='Independent saved derivative/norm/FD/aggregate arithmetic only. All original masks/centers and modes retained;no native evidence for unexecuted observable-only actions or multistep/Actor/ego admission.')
    write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],arithmetic_checks=counts)),flush=True)


if __name__=='__main__':main()
