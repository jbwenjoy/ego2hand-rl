"""Independent saved-output review of guarded arithmetic and rounding controls."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    started=time.perf_counter(); out=ROOT/'runs/pen_guarded_cost_arithmetic_saved_review_v1'
    if out.exists(): raise ValueError('Fresh saved qualification review required')
    qp=file_pin(ROOT/'runs/pen_stable_friction_backend_qualification_v4/report.json'); q=json.loads(verify_file_pin(qp).read_text())
    pp=file_pin(ROOT/'runs/pen_friction_rounding_probe_v2/report.json'); p=json.loads(verify_file_pin(pp).read_text())
    inputs=[qp,pp,q['source_receipt'],p['source_receipt'],q['factory_manifest'],q['inputs']['fixed_inputs'],q['outputs'],q['distinct_outputs'],q['separate_outputs'],p['outputs']]
    for r in [q,p]: verify_source_snapshot(json.loads(verify_file_pin(r['source_receipt']).read_text()),live=True)
    raw=dict(np.load(verify_file_pin(q['inputs']['fixed_inputs']))); x=raw['inputs']; z=dict(np.load(verify_file_pin(q['outputs']))); dz=dict(np.load(verify_file_pin(q['distinct_outputs']))); sz=dict(np.load(verify_file_pin(q['separate_outputs']))); probe=dict(np.load(verify_file_pin(p['outputs'])))
    j,v,d,f,a=x.astype(float).T; r=f/d; y=j+a*v
    handled=((-r<j)&(j<r)&(-r<y)&(y<r))|((j<=-r)&(y<=-r))|((j>=r)&(y>=r)); fallback=~handled
    checks=dict(all139160=x.shape==(139160,5),producer22=q['passed']==q['total']==22 and all(q['checks'].values()),
        rounding_producer4=p['passed']==p['total']==4 and all(p['checks'].values()),classification=np.array_equal(handled,z['handled']) and np.array_equal(handled,probe['handled']),
        all_finite=all(np.isfinite(vv).all() for payload in [z,dz,sz,probe] for vv in payload.values()),zero_native=all(t['actual_native_integrations']==t['optimizer_steps']==t['normalizer_fits']==0 for t in [q,p]))
    for label,source in [('mixed_single',(z['original_single'],z['candidate_single'])),('mixed_three',(z['original_three'],z['candidate_three'])),('separate_single',(sz['original_single'],sz['candidate_single'])),('separate_triple',(sz['original_triple'],sz['candidate_triple']))]:
        old,new=source; checks[label+'_derivative_hessian']=old[...,1:].tobytes()==new[...,1:].tobytes(); checks[label+'_fallback']=old[fallback].tobytes()==new[fallback].tobytes()
    checks['distinct_alphas_exact']=dz['alphas'].tobytes()==(x[:,4,None]*np.asarray([1.,.5,1.5],np.float32)).astype(np.float32).tobytes()
    checks['distinct_derivative_hessian']=dz['original_three'][:,:,1:].tobytes()==dz['candidate_three'][:,:,1:].tobytes()
    for k in range(3):
        yy=j+dz['alphas'][:,k].astype(float)*v
        h=((-r<j)&(j<r)&(-r<yy)&(yy<r))|((j<=-r)&(yy<=-r))|((j>=r)&(yy>=r))
        checks[f'distinct{k}_fallback']=dz['original_three'][~h,k].tobytes()==dz['candidate_three'][~h,k].tobytes()
    for kind in ['single','triple']:
        old=probe['original_'+kind]; ident=probe['identity_'+kind]; guarded=probe['guarded_'+kind]
        checks['identity_'+kind]=old.tobytes()==ident.tobytes()
        checks['guarded_'+kind+'_slope']=old[:,1:].tobytes()==guarded[:,1:].tobytes()
        checks['guarded_'+kind+'_fallback']=old[fallback].tobytes()==guarded[fallback].tobytes()
    fields=probe['intermediate']; F=np.float32
    expected_sub=(fields[:,8]-fields[:,5]).astype(np.float32)
    expected_negative=(fields[:,8].astype(float)-fields[:,3].astype(float)*fields[:,4].astype(float)).astype(np.float32)
    expected_positive=(fields[:,6].astype(float)*fields[:,7].astype(float)-fields[:,5].astype(float)).astype(np.float32)
    for label,value,col in [('sub',expected_sub,9),('negative_fma',expected_negative,10),('positive_fma',expected_positive,11)]:
        checks[label+'_CPU_GPU_fallback']=value[fallback].tobytes()==fields[fallback,col].tobytes()
    checks['original_single_fma_negative']=probe['original_single'][fallback,0].tobytes()==fields[fallback,10].tobytes()
    checks['original_triple_fma_positive']=probe['original_triple'][fallback,0].tobytes()==fields[fallback,11].tobytes()
    checks['old_unconditional_stable_sub']=probe['stable_single'][fallback,0].tobytes()==fields[fallback,9].tobytes() and probe['stable_triple'][fallback,0].tobytes()==fields[fallback,9].tobytes()
    for pin in inputs: verify_file_pin(pin)
    out.mkdir(); result=dict(schema='pen_guarded_cost_arithmetic_saved_review:1',source=file_pin(__file__),inputs=inputs,checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),
        rows=len(x),fallback=int(fallback.sum()),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
        scope='Saved-output derivative/Hessian/fallback and explicit rounding reconciliation only. Confirms the observed fixed-input rounding distinction,not final driver SASS or native solver/task/Actor/ego qualification.')
    write_json(out/'report.json',result); print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']])),flush=True)
    if not all(checks.values()): raise ValueError('Saved guarded cost arithmetic discrepancy')


if __name__=='__main__': main()
