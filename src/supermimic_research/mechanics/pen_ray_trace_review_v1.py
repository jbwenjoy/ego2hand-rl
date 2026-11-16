"""Replay recorded GPU decisions; FP64 objective differences remain observations."""
import numpy as np

from ..runtime.pen_ray_trace_factory_v1 import RAY_FIELDS,INT_FIELDS,BRACKET_FIELDS,VECTOR_FIELDS
from ..runtime.pen_ray_post_kernels_v1 import POST_VECTOR_FIELDS,POST_FLOAT_FIELDS,POST_INT_FIELDS
from .pen_ray_reference_v1 import evaluate

EPS=np.finfo(np.float32).eps
TINY=np.finfo(np.float32).tiny


def near_calculation(actual,expected,magnitude):
    return bool(np.isfinite(actual).all() and np.isfinite(expected).all() and np.all(abs(actual-expected)<=16*EPS*(np.asarray(magnitude)+TINY)))


def ray_replay(ray,integers,rows,vectors,brackets):
    r=dict(zip(RAY_FIELDS,ray)); iv=dict(zip(INT_FIELDS,integers)); used=int(iv['ls_used'])
    initial=bool(abs(r['newton_gradient'])<r['gtol'] and r['newton_delta']<0)
    checks=dict(valid=int(iv['valid'])==int(iv['final_recorded'])==1,
                row_prefix=len(rows)==int(iv['nefc']) and int(iv['ne'])+int(iv['nf'])<=len(rows),
                initial_predicate=initial==bool(iv['initial_converged']),
                loop_count=used==0 if initial else 1<=used<=50)
    cache={}; references=[]
    def reference(label,alpha,gpu):
        key=float(alpha)
        if key not in cache: cache[key]=evaluate(key,int(iv['ne']),int(iv['nf']),rows,vectors)
        result=cache[key]['point']; gpu=np.asarray(gpu,np.float64)
        references.append(dict(label=label,alpha=key,gpu=gpu.tolist(),FP64=result.tolist(),
            delta_error=float(gpu[0]-result[0]),gradient_error=float(gpu[1]-result[1]),hessian_error=float(gpu[2]-result[2]),
            negative_delta_sign_disagrees=bool((gpu[0]<0)!=(result[0]<0))))
    base=evaluate(0.,int(iv['ne']),int(iv['nf']),rows,vectors)
    references.append(dict(label='p0_derivative_hessian',alpha=0.,gpu=[0.,float(r['p0_grad0']),float(r['p0_hessian0'])],FP64=base['point'].tolist(),
        delta_error=0.,gradient_error=float(r['p0_grad0']-base['point'][1]),hessian_error=float(r['p0_hessian0']-base['point'][2]),negative_delta_sign_disagrees=False))
    reference('initial_Newton',r['newton_alpha'],[r['newton_delta'],r['newton_gradient'],r['newton_hessian']])
    denominator=r['p0_hessian0'] if r['p0_hessian0']!=0 else 1e-15
    expected=-float(r['p0_grad0'])/float(denominator)
    checks['Newton_alpha_arithmetic']=near_calculation(np.asarray(r['newton_alpha']),np.asarray(expected),abs(expected))
    search=vectors[:,VECTOR_FIELDS.index('search')].astype(float); snorm=float(np.linalg.norm(search))
    checks['search_norm']=near_calculation(np.asarray(r['snorm']),np.asarray(snorm),snorm)
    gtol=max(float(r['tolerance'])*float(r['ls_tolerance'])*float(r['snorm'])*float(r['meaninertia'])*26,1e-6)
    checks['gtol_arithmetic']=near_calculation(np.asarray(r['gtol']),np.asarray(gtol),gtol)
    floor=(8*1.1920929e-7)*max(1.,(np.sqrt(2*max(float(r['rows_cost0']),0)*max(float(r['rows_hessian0']),0))+abs(float(r['gauss_grad0'])))/float(denominator))
    checks['noise_floor_arithmetic']=near_calculation(np.asarray(r['noise_floor']),np.asarray(floor),abs(floor))
    alpha=float(r['newton_alpha']) if initial else 0.; improvement=-float(r['newton_delta']) if initial else 0.
    def point(row,prefix): return np.array([row[prefix+'_'+s] for s in ['delta','gradient','hessian']],np.float32)
    def in_bracket(x,y): return (x[1]<y[1] and y[1]<0) or (x[1]>y[1] and y[1]>0)
    p0=np.array([0.,r['p0_grad0'],r['p0_hessian0']],np.float32)
    first=np.array([r['newton_delta'],r['newton_gradient'],r['newton_hessian']],np.float32)
    less=first[1]<p0[1]; lo,hi=(first,p0) if less else (p0,first)
    la,ha=(float(r['newton_alpha']),0.) if less else (0.,float(r['newton_alpha']))
    for k in range(used):
        b=dict(zip(BRACKET_FIELDS,brackets[k])); prelo,prehi=point(b,'entry_lo'),point(b,'entry_hi')
        checks[f'{k}_entry']=np.array_equal(prelo,lo) and np.array_equal(prehi,hi) and b['entry_lo_alpha']==la and b['entry_hi_alpha']==ha
        expected=[]; magnitudes=[]
        for x,a in [(lo,la),(hi,ha)]:
            ratio=float(x[1])/float(x[2] if x[2]!=0 else 1e-15)
            expected.append(a-ratio); magnitudes.append(abs(a)+abs(ratio))
        expected.append(.5*(la+ha)); magnitudes.append(abs(la)+abs(ha))
        actual=np.array([b['trial_lo_alpha'],b['trial_hi_alpha'],b['trial_mid_alpha']],float)
        checks[f'{k}_trial_alpha_arithmetic']=near_calculation(actual,np.asarray(expected),np.asarray(magnitudes))
        trials={name:point(b,name) for name in ['trial_lo','trial_hi','trial_mid']}
        for label in trials: reference(f'bracket{k}_{label}',b[label+'_alpha'],trials[label])
        swaplo=False; swaphi=False
        for label in ['trial_lo','trial_mid','trial_hi']:
            if in_bracket(lo,trials[label]): lo=trials[label]; la=float(b[label+'_alpha']); swaplo=True
        for label in ['trial_hi','trial_mid','trial_lo']:
            if in_bracket(hi,trials[label]): hi=trials[label]; ha=float(b[label+'_alpha']); swaphi=True
        done=bool((not swaplo and not swaphi) or (lo[0]<0 and lo[1]<0 and lo[1]>-r['gtol']) or (hi[0]<0 and hi[1]>0 and hi[1]<r['gtol']))
        improved=bool(lo[0]<0 or hi[0]<0); better=lo[0]<hi[0]; bestalpha=la if better else ha; bestdelta=float(lo[0] if better else hi[0])
        if improved: alpha=bestalpha; improvement=-bestdelta
        checks[f'{k}_swaps']=bool(b['swap_lo'])==swaplo and bool(b['swap_hi'])==swaphi and np.array_equal(point(b,'exit_lo'),lo) and np.array_equal(point(b,'exit_hi'),hi) and b['exit_lo_alpha']==la and b['exit_hi_alpha']==ha
        checks[f'{k}_decisions']=bool(b['ls_done'])==done and bool(b['improved'])==improved and b['best_alpha']==bestalpha and b['best_delta']==bestdelta and b['accepted_alpha']==alpha and b['improvement']==improvement
        checks[f'{k}_loop_stopping']=not done if k<used-1 else (done or used==50)
    checks['final_alpha_improvement']=r['accepted_alpha']==alpha and r['improvement']==improvement
    preG=vectors[:,VECTOR_FIELDS.index('raw_grad')].astype(float)*float(r['pre_grad_scale'])
    return dict(checks={k:bool(v) for k,v in checks.items()},references=references,
                pre_G_dot_actual_search=float(preG@search),FP64_p0_gradient=float(base['point'][1]),GPU_p0_gradient=float(r['p0_grad0']),
                scope='Decision replay uses actual recorded FP32 points;FP64 errors/sign changes are observations,not requirements for a passing physical gate.')


def review_trace(o,algorithm):
    if algorithm not in ['INSTALLED_ORIGINAL_OR','DROP_IMPROVEMENT_ONLY']: raise ValueError('Declared termination algorithm required')
    checks={}; rays=[]
    keys=['ray','vectors','rows','brackets','integers','errors','post_vectors','post_floats','post_ints']
    if any('trace.'+k not in o for k in keys): raise ValueError('Complete coupled payload required')
    tr={k:o['trace.'+k] for k in keys}
    shapes=dict(ray=(8,50,1,len(RAY_FIELDS)),vectors=(8,50,1,26,len(VECTOR_FIELDS)),rows=(8,50,1,4096,4),
        brackets=(8,50,1,50,len(BRACKET_FIELDS)),integers=(8,50,1,len(INT_FIELDS)),errors=(8,1),
        post_vectors=(8,50,1,26,len(POST_VECTOR_FIELDS)),post_floats=(8,50,1,len(POST_FLOAT_FIELDS)),post_ints=(8,50,1,len(POST_INT_FIELDS)))
    if any(v.shape!=shapes[k] or v.dtype!=np.dtype(np.int32 if k in ['integers','errors','post_ints'] else np.float32) for k,v in tr.items()):
        raise ValueError('Exact full-capacity typed trace envelope required')
    checks['finite']=all(np.isfinite(v).all() for v in tr.values()); checks['no_overflow']=not np.any(tr['errors'])
    for t in range(8):
        count=int(o['data.solver_niter'][t,0]); ints=tr['integers'][t,:,0]; post=tr['post_ints'][t,:,0]
        if not 1<=count<=50: raise ValueError('Native iteration count outside registered cap')
        expected=np.r_[np.ones(count,int),np.zeros(50-count,int)]
        checks[f'{t}_valid_prefix']=1<=count<=50 and np.array_equal(ints[:,INT_FIELDS.index('valid')],expected) and np.array_equal(post[:,POST_INT_FIELDS.index('valid')],expected)
        for i in range(count):
            iv=ints[i]; pf=tr['post_floats'][t,i,0]; pv=tr['post_vectors'][t,i,0]; r=tr['ray'][t,i,0]; v=tr['vectors'][t,i,0]
            n=int(iv[INT_FIELDS.index('nefc')]); rr=ray_replay(r,iv,tr['rows'][t,i,0,:n],v,tr['brackets'][t,i,0])
            checks[f'{t}_{i}_nefc']=n==int(o['data.nefc'][t,0]) and 0<=n<=4096
            checks.update({f'{t}_{i}_{k}':val for k,val in rr['checks'].items()})
            rr.update(tick=t,outer_iteration=i,nefc=n); rays.append(rr)
            rv=dict(zip(RAY_FIELDS,r)); pp=dict(zip(POST_INT_FIELDS,post[i])); ff=dict(zip(POST_FLOAT_FIELDS,pf))
            checks[f'{t}_{i}_post_index']=pp['solver_niter']==i+1 and bool(pp['done'])==(i==count-1)
            checks[f'{t}_{i}_post_line_values']=ff['alpha']==rv['accepted_alpha'] and ff['improvement']==rv['improvement'] and bool(pp['ls_exhausted'])==(abs(rv['accepted_alpha'])<rv['noise_floor'])
            for label,precol,postcol,direction in [('qacc',4,2,0),('Ma',2,3,1)]:
                advance=float(rv['accepted_alpha'])*v[:,direction].astype(float); before=v[:,precol].astype(float)
                checks[f'{t}_{i}_{label}_update']=near_calculation(pv[:,postcol].astype(float),before+advance,abs(before)+abs(advance))
            if i+1<count:
                nv=tr['vectors'][t,i+1,0]; nr=dict(zip(RAY_FIELDS,tr['ray'][t,i+1,0]))
                checks[f'{t}_{i}_next_phase']=all(np.array_equal(nv[:,a],pv[:,b]) for a,b in [(5,0),(0,1),(4,2),(2,3)]) and all(nr['pre_'+k]==ff[k] for k in ['grad_scale','grad_dot','newton_decrement'])
            else:
                checks[f'{t}_final_vectors']=all(np.array_equal(pv[:,j],o[key][t,0,:26]) for j,key in [(0,'context.grad'),(1,'context.search'),(2,'data.qacc'),(3,'data.efc.Ma')])
                checks[f'{t}_final_scalars']=all(ff[k]==o['context.'+k][t,0] for k in ['grad_scale','grad_dot','newton_decrement','alpha','improvement']) and ff['time']==o['data.time'][t,0]
                checks[f'{t}_final_flags']=all(pp[k]==o['context.'+k][t,0] for k in ['state_changed_count','quad_changed_count','done','ls_exhausted','search_unchanged'])
    return dict(checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),rays=rays,
        scope='Coupled phases,valid prefixes and recorded decisions. FP64 fixed-ray comparisons are not full dynamics or action-gradient truth.')
