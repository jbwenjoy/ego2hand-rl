"""Fixed native probes of frozen one-step RAW score gradients;no optimization."""
import numpy as np

GOALS=['RESIDUAL','SUPPORT']
EPSILONS=[.002,.005]


def direction_from_raw_gradient(gradient):
    g=np.asarray(gradient,np.float64)
    if g.shape!=(20,) or not np.isfinite(g).all():raise ValueError('Finite original RAW20 gradient required')
    norm=float(np.linalg.norm(g))
    return (g/norm if norm>0 else None),norm


def proposal_raw(base,direction,epsilon,sign):
    base=np.asarray(base);direction=np.asarray(direction,np.float64)
    if base.shape!=(20,) or direction.shape!=(20,) or not np.isfinite(base).all() or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-8 or epsilon not in EPSILONS or sign not in [-1,1]:
        raise ValueError('Exact unit RAW-gradient direction,PD radius and sign required')
    value=(base+4*sign*epsilon*direction).astype(np.float32)
    return value,bool(np.isfinite(value).all() and np.all(abs(value)<=1))


def profiles(models):
    result=[dict(name=f'BASE_R{r}',model=None,goal=None,goal_index=None,direction=-1,epsilon=0.,sign=0,repeat=r) for r in range(2)]
    for mi,model in enumerate(models):
        for k,goal in enumerate(GOALS):
            for epsilon in EPSILONS:
                for sign in [1,-1]:
                    for repeat in range(2):
                        name=f'{model}_{goal}_E{int(epsilon*1000):03d}_{"P" if sign>0 else "M"}_R{repeat}'
                        result.append(dict(name=name,model=model,goal=goal,goal_index=k,direction=2*mi+k,epsilon=epsilon,sign=sign,repeat=repeat))
    return result


def summarize_improvement(base,plus,minus,*,epsilon,gradient_norm,physical_valid,original_base_valid):
    """Full native score scalar;observed repeat envelopes are not confidence bounds."""
    b,p,m=[np.asarray(v,np.float64) for v in [base,plus,minus]]
    if any(v.shape!=(2,) or not np.isfinite(v).all() for v in [b,p,m]) or epsilon not in EPSILONS or not np.isfinite(gradient_norm) or gradient_norm<0:
        raise ValueError('Two repeats and fixed proposal radius/gradient norm required')
    bg=float(b.mean());pg=float(p.mean());mg=float(m.mean());gain=pg-bg;separation=pg-mg
    gain_noise=float(abs(p-pg).max()+abs(b-bg).max());separation_noise=float(abs(p-pg).max()+abs(m-mg).max())
    eligible=bool(physical_valid and original_base_valid);fd=separation/(8*epsilon)
    return dict(eligible=eligible,base_mean=bg,positive_mean=pg,negative_mean=mg,positive_gain=gain,negative_gain=mg-bg,
        positive_gain_repeat_envelope=gain_noise,separation_repeat_envelope=separation_noise,native_FD_per_RAW=fd,predicted_AD_per_RAW=float(gradient_norm),
        improvement_exceeds_observed_repeat_envelope=bool(eligible and gain>gain_noise),
        descent_exceeds_observed_repeat_envelope=bool(eligible and gain< -gain_noise),
        ascent_beats_reverse_exceeds_observed_repeat_envelope=bool(eligible and separation>separation_noise),
        AD_absolute_error=abs(float(gradient_norm)-fd),AD_relative_error=None if fd==0 else abs(float(gradient_norm)-fd)/abs(fd),
        scope='One native control at fixed source;not policy improvement,confidence interval or Actor qualification.')
