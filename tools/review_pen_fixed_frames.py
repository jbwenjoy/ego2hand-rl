"""Independent source-fork,feature proof and clean timing/operator review."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def differences(left,right,path=()):
    if type(left)!=type(right):return [path]
    if isinstance(left,torch.Tensor):
        same=left.shape==right.shape and left.dtype==right.dtype and left.stride()==right.stride() and left.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()==right.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return [] if same else [path]
    if isinstance(left,np.ndarray):return [] if left.shape==right.shape and left.dtype==right.dtype and left.strides==right.strides and left.tobytes()==right.tobytes() else [path]
    if isinstance(left,dict):
        if set(left)!=set(right):return [path]
        return [p for key in left for p in differences(left[key],right[key],path+(key,))]
    if isinstance(left,(list,tuple)):
        if len(left)!=len(right):return [path]
        return [p for i,(a,b) in enumerate(zip(left,right)) for p in differences(a,b,path+(i,))]
    if isinstance(left,(float,np.floating)):return [] if np.asarray(left).tobytes()==np.asarray(right).tobytes() else [path]
    return [] if left==right else [path]


def main(a):
    run=json.loads(a.comparison.read_text());qualification=json.loads(verify_file_pin(run['qualification']).read_text());gradient=json.loads(a.gradients.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or run['schema']!='ordinary_pen_PPO_fixed_frame_comparison:1' or run['passed']!=run['total'] or qualification['passed']!=qualification['total'] or gradient['passed']!=gradient['total']:raise ValueError('Fresh complete fixed-frame review')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    source=torch.load(verify_file_pin(run['fixture']),map_location='cpu',weights_only=False);mode_path=next(p for p in source['native']['python']['scalars'] if p and p[-1]==('attr','fixed_frame_cache'));checks={};fork_audits=[];timing={};operators={}
    for c in run['cases']:
        fork=torch.load(verify_file_pin(c['feature_fork']['source']),map_location='cpu',weights_only=False);diff=differences(source,fork);expected=[] if c['feature_fork']['feature_mode']=='LIVE' else [('metadata','fixed_frame_features'),('native','metadata','fixed_frame_features'),('native','python','scalars',mode_path)]
        checks[c['name']+'_only_declared_source_changes']=set(diff)==set(expected)
        checks[c['name']+'_source_receipt_unchanged']=fork['native']['sources']==source['native']['sources']
        checks[c['name']+'_original_collector_gates']=c['collector']['passed']==c['collector']['total'] and all(c['checks'].values())
        checks[c['name']+'_full_native_readback']=c['restore_audit']['native']['all_array_bytes_equal'] and c['restore_audit']['native']['all_scalar_values_equal']
        fork_audits.append(dict(name=c['name'],feature_mode=c['feature_fork']['feature_mode'],actual_changed_paths=[repr(p) for p in diff],only_declared_changes=set(diff)==set(expected)))
        if c['name'].startswith('TORCH_'):
            ops=json.loads(verify_file_pin(c['artifacts']['torch_operators']).read_text());operators[c['name']]={r['key']:r for r in ops if r['key'] in ['aten::mul','aten::stack','aten::linalg_vector_norm','aten::bmm','aten::nonzero','aten::_local_scalar_dense']}
    for mode in ['LIVE','CACHED']:
        cases=[c for c in run['cases'] if c['name'].startswith(mode+'_')];values=np.array([c['wall_s']['collect64'] for c in cases]);total=np.array([sum(c['wall_s'].values()) for c in cases]);timing[mode]=dict(cases=[c['name'] for c in cases],collect_s=values.tolist(),collect_median_s=float(np.median(values)),whole_phase_s=total.tolist(),whole_phase_median_s=float(np.median(total)),reset_worlds=[c['reset_worlds'] for c in cases])
    ratio=timing['LIVE']['collect_median_s']/timing['CACHED']['collect_median_s'];reduction=1-1/ratio
    cfg=tomllib.loads(verify_file_pin(run['config']).read_text());frozen=cfg.get('initialization')=='FROZEN_W32_TRAINED64_WEIGHTS'
    expected_native=cfg['calibration_controls']*256+(cfg['warmup_iterations']+len(cfg['cases']))*16384
    checks.update(all369_actual_phases=len([r for r in qualification['rows'] if r['name'].startswith('PHASE')])==369,all_actual_feature_rows_exact=qualification['all_feature_rows_exact'],all_feature_JVP_rows_exact=all(r['forward_exact'] and r['JVP_exact'] for r in gradient['rows']),work_native=run['actual_native_integrations']==expected_native,work_updates=run['optimizer_steps']==(cfg['warmup_iterations']+len(cfg['cases']))*32)
    if frozen:
        init=json.loads(verify_file_pin(run['initialization']).read_text());initial=torch.load(verify_file_pin(run['frozen_initial']),map_location='cpu',weights_only=False)
        checks.update(full_trained_context=init['full_model_arrays']==404 and init['full_model_scalars']==195 and init['full_model_bytes_equal'],native_RNG_not_imported=init['full_native_and_RNG_unchanged'] and not init['old_native_optimizer_RNG_counters_loaded'],warmup_discarded=not differences(initial,source),fresh_counters=source['iteration']==source['optimizer_steps']==source['training_transitions']==source['calibration_transitions']==0 and not source['optimizer']['state'])
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(json.loads(verify_file_pin(run['source_receipt']).read_text()),live=True)
    result=dict(schema='fixed_frame_cache_saved_review:1',comparison=file_pin(a.comparison),qualification=run['qualification'],gradients=file_pin(a.gradients),source_receipt=file_pin(out/'source_receipt.json'),fork_audits=fork_audits,timing=timing,operators=operators,
        collect_throughput_ratio=ratio,collect_wall_reduction_fraction=reduction,whole_phase_throughput_ratio=timing['LIVE']['whole_phase_median_s']/timing['CACHED']['whole_phase_median_s'],
        performance_status='OBSERVED_COMPONENT_TIMING_GAIN' if ratio>1 else 'NO_OBSERVED_GAIN',mechanism_status='EXACT_TESTED_FEATURES_GRADIENTS_AND_DECLARED_SOURCE_FORKS' if all(checks.values()) else 'INCOMPLETE',
        checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Current-source fixed-input mechanism/source proofs and three clean timings per mode. Fixture='+('TRAINED64_WEIGHTS_WITH_FRESH_W32_PHYSICS' if frozen else 'EARLY_TRAINING32')+'. Not universal speed,identical chaotic trajectories,policy improvement,physical WM gradients or full task/ego qualification. Default LIVE until a declared caller chooses CACHED.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],ratio=ratio,reduction=reduction,timing=timing)),flush=True)
    if not all(checks.values()):raise ValueError('Retain fixed-frame source/timing review discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--gradients',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
