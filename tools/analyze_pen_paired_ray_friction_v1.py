"""All paired queries/rays plus one fixed-input GPU friction arithmetic batch."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.pen_ray_reference_v1 import friction_point
from supermimic_research.runtime.pen_ray_trace_factory_v1 import RAY_FIELDS,INT_FIELDS
from supermimic_research.strict_json import write_json


def read(pin): return json.loads(verify_file_pin(pin).read_text())


def distribution(a):
    a=np.asarray(a,float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(n=0)


def csv_write(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def main(a):
    started=time.perf_counter(); cp=a.config.resolve(); cfg=tomllib.loads(cp.read_text()); out=ROOT/'runs/pen_paired_ray_friction_analysis_v1'
    if out.exists(): raise ValueError('Fresh complete analysis required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['registration','output','reader_output']}
    regpin=file_pin(paths['registration']); reg=read(regpin); inputs=[regpin]; roots={}; queries=[]; rays=[]; friction=[]; packet_pairs={}; nested_checks=0; reference_points=0
    for replica in ['A','B']:
        pp=file_pin(paths['output']/replica/'report.json'); rp=file_pin(paths['reader_output']/replica/'report.json'); p,v=read(pp),read(rp); roots[replica]=dict(producer=pp,reader=rp); inputs += [pp,rp]
        if p['actual_native_integrations']!=p['attempted_native_integrations'] or p['actual_native_integrations']!=960 or v['producer']!=pp or p['passed']!=p['total'] or v['passed']!=v['total'] or not all(p['checks'].values()) or not all(v['checks'].values()): raise ValueError('Both closed independently reviewed processes required')
        trace_reports={Path(pin['path']).parent.name:pin for pin in v['trace_reviews']}; reference_points+=v['reference_points']
        for slot in p['rows']:
            q=read(slot['report']); c=read(q['case']); decl=q['declaration']; inputs += [slot['report'],q['case'],q['solve_exit'],q['epoch_review'],c['trace']]
            o=dict(np.load(verify_file_pin(q['solve_exit']))); e=read(q['epoch_review']); z=dict(np.load(verify_file_pin(c['trace'])))
            key=(replica,decl['algorithm'],decl['repeat'],decl['row_index']); packet_pairs.setdefault(key,{})[decl['scope']]=(q['new_impulse_valid'],z)
            queries.append(dict(**decl,impulse_valid=q['new_impulse_valid'],physical_valid=q['new_physical_valid'],peak_impulse_Ns=e['peak_original_impulse_Ns'],
                cap_ticks=int((o['data.solver_niter']==50).sum()),query_wall_s=c['wall_s'],terminated=c['terminated'],truncated=c['truncated']))
            if decl['scope']!='SOLVE_EXIT_PLUS_RAY_TRACE': continue
            pin=trace_reports[decl['name']]; traced=read(pin); inputs.append(pin); nested_checks+=traced['total']
            if traced['passed']!=traced['total'] or not all(traced['checks'].values()): raise ValueError('All coupled replay checks required')
            for rr in traced['rays']:
                t,i=rr['tick'],rr['outer_iteration']; rv=dict(zip(RAY_FIELDS,o['trace.ray'][t,i,0])); iv=dict(zip(INT_FIELDS,o['trace.integers'][t,i,0])); ref=next(p for p in rr['references'] if p['label']=='initial_Newton')
                rows=o['trace.rows'][t,i,0,:int(iv['nefc'])]; alpha=float(rv['newton_alpha']); begin=len(friction)
                for k in range(int(iv['ne']),int(iv['ne']+iv['nf'])):
                    jar,jv,d,loss=rows[k].astype(float); value=friction_point(jar,alpha*jv,d,loss)[0]
                    friction.append([jar,jv,d,loss,alpha,value])
                if len(friction)>2000000: raise ValueError('Fixed-input arithmetic probe row cap2M exceeded')
                rays.append(dict(replica=replica,algorithm=decl['algorithm'],repeat=decl['repeat'],row_index=decl['row_index'],name=decl['original_name'],tick=t,outer=i,
                    query_impulse_valid=q['new_impulse_valid'],failed_query_peak=not q['new_impulse_valid'] and t==e['peak_tick'] and i==int(o['data.solver_niter'][t,0])-1,
                    ne=int(iv['ne']),nf=int(iv['nf']),nefc=int(iv['nefc']),ls_used=int(iv['ls_used']),initial_accepted=bool(iv['initial_converged']),
                    accepted_alpha=float(rv['accepted_alpha']),improvement=float(rv['improvement']),gtol=float(rv['gtol']),initial_gradient=float(rv['newton_gradient']),
                    GPU_initial_delta=float(rv['newton_delta']),FP64_initial_delta=float(ref['FP64'][0]),initial_delta_error=float(ref['delta_error']),
                    initial_cost_sign_disagrees=ref['negative_delta_sign_disagrees'],pre_G_dot_actual_search=rr['pre_G_dot_actual_search'],
                    GPU_p0_gradient=rr['GPU_p0_gradient'],FP64_p0_gradient=rr['FP64_p0_gradient'],friction_begin=begin,friction_end=len(friction)))
        if sum(r['replica']==replica for r in rays)!=v['ray_count']: raise ValueError('Every registered actual ray must be retained')
    tests=ROOT/'runs/pen_friction_cost_probe_v1_tests.log'; text=tests.read_text(encoding='utf-8-sig')
    if 'Ran 2 tests' not in text or '\nOK\n' not in text: raise ValueError('Standalone primitive CPU tests required')
    inputs.append(file_pin(tests))
    for pin in inputs: verify_file_pin(pin)
    out.mkdir(); receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,ROOT/'tests/test_pen_friction_cost_probe_v1.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    raw=np.asarray(friction,np.float64); batch=raw[:,:5].astype(np.float32); target=raw[:,5]
    if not np.isfinite(raw).all() or np.any(batch[:,2]<=0): raise ValueError('Finite positive-D friction payload required;no dropping unsupported rows')
    np.savez_compressed(out/'fixed_inputs.npz',inputs=batch,FP64_delta=target)
    write_json(out/'intent.json',dict(roots=roots,row_count=len(batch),row_cap=2000000,primitive_launches=1,actual_native_integrations=0,optimizer_steps=0,
        scope='Posthoc fixed-recorded-input arithmetic;no simulation or solver binding. Same-regime cost substitution is exploratory,not a physical backend.'))
    import warp as wp
    import mujoco
    import mujoco_warp as mw
    from supermimic_research.runtime.pen_friction_cost_probe_v1 import probe_friction_cost
    wp.init(); device=wp.get_device('cuda:0')
    if not device.is_cuda: raise ValueError('Actual CUDA arithmetic probe required')
    tg=time.perf_counter()
    with wp.ScopedDevice(device),patch.object(mujoco,'mj_step',side_effect=AssertionError('No integration in arithmetic probe')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward in arithmetic probe')),patch.object(mw,'step',side_effect=AssertionError('No native step in arithmetic probe')):
        arrays=[wp.array(batch[:,i].copy(),dtype=float,device=device) for i in range(5)]
        old=wp.zeros((len(batch),3),dtype=float,device=device); stable=wp.zeros_like(old); handled=wp.zeros(len(batch),dtype=int,device=device)
        wp.launch(probe_friction_cost,dim=len(batch),inputs=arrays,outputs=[old,stable,handled],device=device); wp.synchronize()
        old,stable,handled=old.numpy(),stable.numpy(),handled.numpy()
        inputs_unchanged=all(np.array_equal(a.numpy(),batch[:,i]) for i,a in enumerate(arrays))
    gpu_pipeline_wall=time.perf_counter()-tg
    np.savez_compressed(out/'primitive_outputs.npz',original=old,same_regime=stable,handled=handled)
    for row in rays:
        lo,hi=row['friction_begin'],row['friction_end']; change=float((stable[lo:hi,0].astype(float)-old[lo:hi,0].astype(float)).sum())
        row['fixed_primitive_friction_delta_change']=change; row['illustrative_aggregate_delta']=row['GPU_initial_delta']+change
        row['illustrative_cost_predicate_turns_negative']=bool(row['GPU_initial_delta']>=0 and row['illustrative_aggregate_delta']<0)
        row['primitive_fallback_rows']=int((handled[lo:hi]==0).sum())
    scope_pairs=[]
    for key,values in packet_pairs.items():
        b,bz=values['SOLVE_EXIT_ONLY']; v,vz=values['SOLVE_EXIT_PLUS_RAY_TRACE']
        scope_pairs.append(dict(replica=key[0],algorithm=key[1],repeat=key[2],row_index=key[3],masks_equal=b==v,
            raw_bytes_equal=all(bz[k].dtype==vz[k].dtype and bz[k].shape==vz[k].shape and bz[k].tobytes()==vz[k].tobytes() for k in bz)))
    failures=[r for r in rays if r['failed_query_peak']]
    checks=dict(all240=len(queries)==240,all120_scope_pairs=len(scope_pairs)==120,all_actual_rays=len(rays)==sum(read(roots[k]['reader'])['ray_count'] for k in roots),
        row_cap=len(batch)<=2000000,finite_outputs=np.isfinite(old).all() and np.isfinite(stable).all(),inputs_unchanged=inputs_unchanged,
        derivative_hessian_bytes_equal=old[:,1:].tobytes()==stable[:,1:].tobytes(),fallback_bit_exact=old[handled==0].tobytes()==stable[handled==0].tobytes())
    csv_write(out/'all240_queries.csv',queries); csv_write(out/'all_rays.csv',rays); csv_write(out/'all120_scope_pairs.csv',scope_pairs)
    result=dict(schema='paired_ray_friction_arithmetic_analysis:1',roots=roots,inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
        fixed_inputs=file_pin(out/'fixed_inputs.npz'),primitive_outputs=file_pin(out/'primitive_outputs.npz'),queries_csv=file_pin(out/'all240_queries.csv'),
        rays_csv=file_pin(out/'all_rays.csv'),scope_pairs_csv=file_pin(out/'all120_scope_pairs.csv'),checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),
        queries=240,actual_rays=len(rays),reference_points=reference_points,nested_replay_checks=nested_checks,
        invalid_queries_by_algorithm=dict(Counter(q['algorithm'] for q in queries if not q['impulse_valid'])),
        endpoint_scope_equal_masks=sum(s['masks_equal'] for s in scope_pairs),endpoint_scope_equal_bytes=sum(s['raw_bytes_equal'] for s in scope_pairs),
        initial_cost_sign_disagreements=sum(r['initial_cost_sign_disagrees'] for r in rays),failed_query_peak_rays=failures,
        primitive_rows=len(batch),same_regime_rows=int(handled.sum()),fallback_rows=int((handled==0).sum()),
        original_abs_delta_error=distribution(abs(old[:,0].astype(float)-target)),same_regime_abs_delta_error=distribution(abs(stable[:,0].astype(float)-target)),
        native_integrations_in_producers=1920,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        wall_s=time.perf_counter()-started,gpu_arithmetic_pipeline_wall_s=gpu_pipeline_wall,
        timing_scope='One CUDA primitive launch plus upload,JIT,download/input readback. Concurrent analysis/native costs elsewhere;not isolated speed or training acceleration.',
        scope='Every declared query and traced ray retained. Standalone same-regime friction-cost arithmetic preserves derivatives/Hessian and uses explicit fallback. '
              'Primitive compilation/reduction may differ from inlined native code. Aggregate delta replacement is illustrative,not a rerun of line search or a solver/backend/task fix;no native/Actor/ego admission.')
    for pin in inputs: verify_file_pin(pin)
    verify_source_snapshot(receipt,live=True); write_json(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],rays=len(rays),primitive_rows=len(batch),initial_sign_disagreements=result['initial_cost_sign_disagreements'],native=0)),flush=True)
    if not all(checks.values()): raise ValueError('Complete friction arithmetic analysis discrepancy')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); main(parser.parse_args())
