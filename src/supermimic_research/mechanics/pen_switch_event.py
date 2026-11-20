"""Source-geometry forecasts of the first same-pair presence change within seven 2.5ms gaps."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

DT = .0025
BINS = 7


def first_event(presence):
    """Presence is an 8-sample binary sequence. Return type (1:on,2:off) and destination tick."""
    bits = np.asarray(presence, dtype=bool)
    if bits.shape != (BINS + 1,):
        raise ValueError('Expected eight pre-integration contact samples')
    changes = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    if not len(changes):
        return 0, 0
    k = int(changes[0])
    return (1 if bits[k] else 2), k


def metrics(y, decision, score, observed_tick, predicted_tick):
    """Binary AP groups all tied scores, so equally rated negatives do not benefit from row order."""
    y, decision = np.asarray(y, bool), np.asarray(decision, bool)
    tp = int(np.sum(y & decision))
    count, predicted = int(y.sum()), int(decision.sum())
    order = np.argsort(-score, kind='stable')
    scores, positives = score[order], y[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]),len(y)-1]
    cumulative = np.cumsum(positives)[ends]
    ap = np.sum(np.diff(np.r_[0,cumulative]) * cumulative/(ends+1)) / count if count else None
    terr = abs(predicted_tick[y & decision]-observed_tick[y & decision])*DT*1000
    return dict(rows=len(y),positives=count,predicted=predicted,tp=tp,fp=predicted-tp,fn=count-tp,
                precision=tp/predicted if predicted else None,recall=tp/count if count else None,
                average_precision=float(ap) if ap is not None else None,
                timing_matches=tp,timing_mae_ms=float(terr.mean()) if tp else None)


def source_features(row, pair_index, num_pairs):
    """Use the source screen's signed gap/velocity only; distance-limit censored rows have no velocity."""
    censored = row is None
    g = .03 if censored else row['gap_m']
    v = 0. if censored else row['normal_speed_m_s']
    travel = v*DT*BINS
    start, end = g*1000, 1000*(g+travel)
    near = lambda x: np.exp(-min(abs(x),30.)/.5)
    vf=np.clip(1000*travel,-5.,5.)
    typ = 1 if g>0 else 2
    cross_time = -g/v if v and g*v<=0 else np.inf
    crossing = not censored and ((g>0 and end<=0) or (g<=0 and end>0))
    # Onset includes zero gap. Release requires strictly positive separation.
    destination = (np.ceil(cross_time/DT) if typ == 1 else np.floor(cross_time/DT)+1) if crossing else 0
    crossing_tick = int(np.clip(destination,1,BINS)) if crossing else 0
    # Signed analytic score for PR: smooth occupancy change on the same observed horizon.
    occ=lambda a: 1/(1+np.exp(np.clip(a/.25,-60.,60.)))
    change=float(occ(end)-occ(start))
    features=[1.,censored,g<=0,np.clip(start,-5.,5.),np.clip(end,-5.,5.),vf,abs(vf),
              near(start),near(end),change,abs(change),crossing,near(start)*vf,near(start)*abs(vf)]
    ident=np.zeros(num_pairs);ident[pair_index]=1.
    return np.r_[features,ident], typ if crossing else 0,crossing_tick,[max(change,0),max(-change,0)]


def fit(x, y, tick, reg=1.):
    """Two regularized event logits against a zero nonevent logit; timing ridge is trained only on positives."""
    n,d=x.shape
    def fg(flat):
        w=flat.reshape(d,2)
        u=np.c_[np.zeros(n), x@w]
        lu=logsumexp(u,axis=1)
        loss=np.sum(lu-u[np.arange(n),y])+.5*reg*np.square(w[1:]).sum()
        u=np.exp(u-lu[:,None]);u[np.arange(n),y]-=1
        grad=x.T@u[:,1:];grad[1:]+=reg*w[1:]
        return loss,grad.ravel()
    if len(np.unique(y))<3:
        raise ValueError(f'Not enough event types in TRAIN: {np.bincount(y,minlength=3)}')
    opt=minimize(fg,np.zeros(d*2),method='L-BFGS-B',jac=True,options={'maxiter':1200,'ftol':1e-11})
    if not opt.success:raise RuntimeError(opt.message)
    times=[]
    for kind in (1,2):
        a=x[y==kind];b=tick[y==kind];penalty=np.eye(d)*10.;penalty[0,0]=0
        times.append(np.linalg.solve(a.T@a+penalty,a.T@b))
    return opt.x.reshape(d,2),np.stack(times,axis=1)


def predict(x, weights):
    w,t=weights;u=np.c_[np.zeros(len(x)),x@w]
    return np.exp(u-logsumexp(u,axis=1)[:,None])[:,1:],np.clip(x@t,1.,BINS)


def classify(prob,threshold):
    kind=np.argmax(prob,axis=1)
    return np.where(prob.max(axis=1)>=threshold,kind+1,0)


def choose_threshold(y,prob):
    """Maximize typed micro-F1 on OOF TRAIN predictions; ties prefer fewer alerts."""
    score=prob.max(axis=1);pred=prob.argmax(axis=1)+1;order=np.argsort(-score,kind='stable')
    ends=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(y)-1]
    tp=np.cumsum((pred==y)[order])[ends];f1=2*tp/(int((y>0).sum())+ends+1)
    return float(score[order[ends[int(np.argmax(f1))]]])
