"""Saved profile/timing/work review;failed speed/equivalence gates stay visible."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    inputs=dict(failed=ROOT/'runs/pen_ordinary_ppo_profile_v1/failure.json',monitored=ROOT/'runs/pen_ordinary_ppo_profile_v2/report.json',isolated=ROOT/'runs/pen_ordinary_ppo_profile_v3/report.json',comparison=ROOT/'runs/pen_empty_signal_comparison_v1/report.json');data={k:json.loads(p.read_text()) for k,p in inputs.items()};out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or any(data[k]['passed']!=data[k]['total'] for k in ['monitored','isolated','comparison']):raise ValueError('Fresh complete performance review,failed attempt retained')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};timing={};all_source=[]
    for key in ['monitored','isolated','comparison']:
        run=data[key];verify_source_snapshot(json.loads(verify_file_pin(run['source_receipt']).read_text()));all_source.append(run['source_receipt'])
        for c in run['cases']:checks[key+'_'+c['name']+'_collector']=c['collector']['passed']==c['collector']['total'] and all(c['checks'].values()) and c['actual_native_integrations']==16384 and c['optimizer_steps']==32
    comp=data['comparison'];base=[c for c in comp['cases'] if c['name'].startswith('BASE_')];fast=[c for c in comp['cases'] if c['name'].startswith('FAST_')]
    for name,cases in [('UPSTREAM',base),('EMPTY_SIGNALS',fast)]:
        timing[name]=dict(cases=[c['name'] for c in cases],wall_s={phase:[c['wall_s'][phase] for c in cases] for phase in ['collect64','consistency','update32']},reset_worlds=[c['reset_worlds'] for c in cases])
        timing[name]['median_collect_s']=float(np.median(timing[name]['wall_s']['collect64']));timing[name]['median_total_s']=float(np.median([sum(c['wall_s'].values()) for c in cases]))
    ratio=timing['UPSTREAM']['median_collect_s']/timing['EMPTY_SIGNALS']['median_collect_s']
    operators={}
    for c in comp['cases']:
        if not c['name'].startswith('TORCH_'):continue
        ops=json.loads(verify_file_pin(c['artifacts']['torch_operators']).read_text());operators[c['name']]={r['key']:r for r in ops if r['key'] in ['aten::nonzero','aten::item','aten::_local_scalar_dense','aten::_to_copy','aten::copy_']}
    checks['exactly64_fewer_nonzero_in_observed_profiles']=operators['TORCH_BASE']['aten::nonzero']['count']-operators['TORCH_FAST']['aten::nonzero']['count']==64
    isolated=data['isolated'];pc=next(c for c in isolated['cases'] if c['name']=='CPROFILE');calls=json.loads(verify_file_pin(pc['artifacts']['python_calls']).read_text())
    checks['no_background_subprocess_in_isolated_python_profile']=not any('subprocess.py' in r['file'] for r in calls)
    selected=[r for r in calls if (r['function'] in ['reference_pose_world','critic','policy','sample','validate','reset'] and any(s in r['file'] for s in ['pen_native_policy.py','owned_pen_runtime.py','pen_reference_cache.py','resident_contact_metrics.py']))]
    checks['512_resident_native_samples']=sum(r['total_calls'] for r in calls if r['function']=='sample' and r['file'].endswith('resident_contact_metrics.py'))==512
    checks['64_live_reference_world_recomputes']=sum(r['total_calls'] for r in calls if r['function']=='reference_pose_world' and r['file'].endswith('pen_native_policy.py'))==64
    memories={c['name']:torch.load(verify_file_pin(c['rollout']),map_location='cpu',weights_only=False) for c in base+fast};fields={};within=True
    for key in ['obs','critic_obs','raw_reward','reward','EMA_after','transmitted_ctrl','contact_geometry_max','contact_wrench_sum','contact_Fn6_sum','phase_source','terminated','truncated']:
        b=np.array([memories[c['name']][key].numpy() for c in base]);f=np.array([memories[c['name']][key].numpy() for c in fast]);checks[key+'_finite']=bool(np.isfinite(b).all() and np.isfinite(f).all())
        if b.dtype==np.bool_:
            inside=np.array([np.any(np.all(v==b,axis=tuple(range(1,b.ndim)))) for v in f]);row=dict(all_FAST_matches_one_BASE=bool(inside.all()),FAST_matches=inside.tolist())
            within=within and bool(inside.all())
        else:
            lo=b.min(0);hi=b.max(0);outside=(f<lo)|(f>hi);distance=np.maximum(np.maximum(lo-f,f-hi),0);row=dict(outside_elements=int(outside.sum()),all_FAST_inside_BASE=not bool(outside.any()),max_outside_distance=float(distance.max()),max_BASE_range=float((hi-lo).max()))
            within=within and not bool(outside.any())
        fields[key]=row
    resets=timing['UPSTREAM']['reset_worlds'];reset_inside=all(min(resets)<=r<=max(resets) for r in timing['EMPTY_SIGNALS']['reset_worlds']);within=within and reset_inside
    native=sum(d['actual_native_integrations'] for d in data.values());updates=sum(d['optimizer_steps'] for d in data.values());checks.update(native_work=native==434176,optimizer_work=updates==832,three_clean_each=len(base)==len(fast)==3)
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='ordinary_pen_performance_saved_review:1',inputs={k:file_pin(p) for k,p in inputs.items()},source_receipt=file_pin(out/'source_receipt.json'),timing=timing,BASE_over_FAST_median_collect_ratio=ratio,
        performance_status='NO_CLEAN_SPEEDUP_OBSERVED' if ratio<=1 else 'OBSERVED_MEDIAN_RATIO_ONLY_NOT_FULL_QUALIFICATION',operators=operators,isolated_python_calls=selected,BASE_repeat_envelope=fields,FAST_reset_counts_inside_BASE=reset_inside,
        equivalence_status='WITHIN_OBSERVED_REPEAT_ENVELOPES_ONLY' if within else 'NOT_ESTABLISHED_OUTSIDE_OBSERVED_BASE_ENVELOPES',adoption='UPSTREAM_DEFAULT_RETAINED',
        cohort_actual_native_integrations=native,cohort_optimizer_steps=updates,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Separate contract arithmetic from failed speed/empirical-equivalence qualification. Baseline repeats are not probability bounds;differences do not prove driver causality. Nonzero time includes waiting for earlier GPU work;no GPU-kernel attribution from CPU profiles. Diagnostic PPO updates are not main-policy promotion.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],ratio=ratio,performance=result['performance_status'],equivalence=result['equivalence_status'],native=native,optimizer_steps=updates)),flush=True)
    if not all(checks.values()):raise ValueError('Retain performance-review contract discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
