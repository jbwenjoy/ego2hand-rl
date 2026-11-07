"""Saved-array arithmetic only; no frameworks, simulator, solver or model load."""
from __future__ import annotations
import numpy as np

PATHS = ('A0', 'A1L', 'P_ONLY', 'A2')
PHASES = (0, 4, 8, 12)
EPS = (.002, .001)
ARRAYS = {
    'snapshot_x': ((16,58), 'float32'), 'snapshot_u': ((16,20), 'float32'),
    'donor_next_x': ((16,58), 'float32'), 'donor_self15': ((16,15), 'float32'),
    'goal_p': ((16,3), 'float32'), 'goal_R': ((16,3,3), 'float32'),
    'goal_p_next': ((16,3), 'float32'), 'goal_R_next': ((16,3,3), 'float32'),
    'snapshot_completed': ((16,), 'bool'), 'branch_x_next': ((560,58), 'float32'),
    'branch_u': ((560,20), 'float32'), 'branch_self15': ((560,15), 'float32'),
    'completed': ((560,), 'bool'), 'restore_qualified': ((560,), 'bool'),
    'branch_object_terms_k': ((560,2), 'float32'),
    'branch_object_terms_kplus1': ((560,2), 'float32'),
    'donor_object_terms_k': ((16,2), 'float32'),
    'donor_object_terms_kplus1': ((16,2), 'float32'),
}


def need(ok, message):
    if not bool(ok): raise ValueError(message)


def slots():
    rows=[]
    for s in range(16):
        common={'snapshot_id':s,'path':PATHS[s//4],'k':PHASES[s%4]}
        for repeat in range(3):
            rows.append(dict(common,type='baseline',baseline_repeat=repeat,j=None,
                             epsilon_index=None,epsilon_rad=None,sign=0))
        for e,eps in enumerate(EPS):
            for j in range(8):
                for sign in (-1,1):
                    rows.append(dict(common,type='direction',baseline_repeat=None,j=j,
                                     epsilon_index=e,epsilon_rad=eps,sign=sign))
    return [dict(row_id=i,**r) for i,r in enumerate(rows)]


def object_terms_f64(x, p, R):
    """Independent geometric arithmetic diagnostic, not native FP32 reward."""
    x,p,R=np.asarray(x,np.float64),np.asarray(p,np.float64),np.asarray(R,np.float64)
    pos=x[...,40:43];rot=x[...,43:52].reshape(x.shape[:-1]+(3,3))
    off=.1*rot[..., :,0];goaloff=.1*R[..., :,0]
    error=.5*(np.linalg.norm(pos+off-p-goaloff,axis=-1)+np.linalg.norm(pos-off-p+goaloff,axis=-1))
    return np.stack((2*np.exp(-(np.linalg.norm(pos-p,axis=-1)/.05)**2),8*np.exp(-(error/.04)**2)),axis=-1)


def object_reward(x,p,R):
    """Secondary FP64 geometry diagnostic only, never used for physical FD."""
    return object_terms_f64(x,p,R).sum(-1)


def native_rewards(terms):
    """Frozen physical terms summed in FP32, then promoted for differences."""
    terms=np.asarray(terms)
    need(terms.dtype==np.float32 and terms.shape[-1]==2,'Native FP32 two terms required')
    return terms.sum(axis=-1,dtype=np.float32).astype(np.float64)


def check_terms(terms,x,p,R):
    diagnostic=object_terms_f64(x,p,R)
    difference=terms.astype(float)-diagnostic
    bound=8*np.finfo(np.float32).eps*np.maximum(1,abs(diagnostic))
    need(np.isfinite(terms).all() and np.all(abs(difference)<=bound),'Saved nativeFP32 terms exceed fixed8eps arithmetic diagnostic bound')
    return difference,bound


def expected_target(row,directions):
    s=row['snapshot_id']
    if row['type']=='baseline': return directions['centers'][s]
    return directions['plus' if row['sign']==1 else 'minus'][s,row['j'],row['epsilon_index']]


def required_slot(row,directions):
    return row['type']=='baseline' or bool(directions['valid_pair'][row['snapshot_id'],row['j'],row['epsilon_index']])


def bitexact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def validate_arrays(a, rows, d):
    need(set(a)==set(ARRAYS),'Unknown/missing physical array fields')
    for k,(shape,dtype) in ARRAYS.items():
        need(a[k].shape==shape and str(a[k].dtype)==dtype,'Wrong shape/dtype '+k)
    need(type(rows) is list and len(rows)==560,'Fixed560 row index required')
    expected=slots();metadata=set(expected[0])|{'actualu','r_k','r_kplus1','restorequalified','status','reason'}
    for i,(row,wanted) in enumerate(zip(rows,expected)):
        need(type(row) is dict and set(row)==metadata,'Unknown/missing branch metadata keys')
        for k,v in wanted.items():
            need(type(row[k]) is type(v) and row[k]==v,'Branch order/type changed '+str(i)+'/'+k)
        done=bool(a['completed'][i]);restored=bool(a['restore_qualified'][i])
        need(type(row['restorequalified']) is bool and row['restorequalified']==restored,'Restore flag disagreement')
        need(type(row['status']) is str and (row['reason'] is None or type(row['reason']) is str),'Malformed branch status/reason')
        if done:
            need(row['status']=='COMPLETED' and restored and row['reason'] is None,'Completed branch must have qualified restore/no failure reason')
            need(required_slot(row,d),'Unregistered omitted-domain direction executed')
            for k in ('branch_x_next','branch_u','branch_self15','branch_object_terms_k','branch_object_terms_kplus1'):
                need(np.isfinite(a[k][i]).all(),'Nonfinite completed branch '+k)
            need(bitexact(a['branch_u'][i],expected_target(row,d)),'Changed/rounded/clipped branch target')
            need(np.all((a['branch_u'][i]>=d['lower'])&(a['branch_u'][i]<=d['upper'])),'Branch target outside fixed inner domain')
            need(type(row['actualu']) is list and bitexact(np.asarray(row['actualu'],np.float32),a['branch_u'][i]),'Actualu metadata mismatch')
            s=row['snapshot_id']
            for key,gp,gR,terms in [('r_k','goal_p','goal_R','branch_object_terms_k'),('r_kplus1','goal_p_next','goal_R_next','branch_object_terms_kplus1')]:
                need(type(row[key]) is float and np.isfinite(row[key]),'Missing completed physical reward')
                check_terms(a[terms][i],a['branch_x_next'][i],a[gp][s],a[gR][s])
                reward=float(native_rewards(a[terms][i]))
                need(row[key]==reward,'Physical reward metadata differs from exact FP32 term sum')
        else:
            need(row['actualu'] is None and row['r_k'] is None and row['r_kplus1'] is None,'Unexecuted row must have null result metadata')
            need(row['status'] in ('NOT_RUN','INVALID'),'Unknown unexecuted status')
            for k in ('branch_x_next','branch_u','branch_self15','branch_object_terms_k','branch_object_terms_kplus1'):
                need(np.isnan(a[k][i]).all(),'Unexecuted slot cannot contain fabricated values')
        need(not done or np.isin(a['branch_self15'][i],[0.,1.]).all(),'Nonbinary native branch self15')
    for s in range(16):
        complete=bool(a['snapshot_completed'][s])
        for k in ('snapshot_x','snapshot_u','donor_next_x','donor_self15','goal_p','goal_R','goal_p_next','goal_R_next','donor_object_terms_k','donor_object_terms_kplus1'):
            need(np.isfinite(a[k][s]).all() if complete else np.isnan(a[k][s]).all(),'Bad snapshot finite/NOT_RUN '+k)
        if complete:
            need(bitexact(a['snapshot_u'][s],d['centers'][s]),'Fresh baselineu differs from fixed direction center')
            need(np.isin(a['donor_self15'][s],[0.,1.]).all(),'Nonbinary donor self15')
            check_terms(a['donor_object_terms_k'][s],a['donor_next_x'][s],a['goal_p'][s],a['goal_R'][s])
            check_terms(a['donor_object_terms_kplus1'][s],a['donor_next_x'][s],a['goal_p_next'][s],a['goal_R_next'][s])
    return expected


def baseline_gate(a):
    """Independently check all48 versus fresh donor, binary self labels separate."""
    rows=[]
    for s in range(16):
        ids=np.arange(35*s,35*s+3);complete=bool(a['snapshot_completed'][s] and a['completed'][ids].all() and a['restore_qualified'][ids].all())
        row={'snapshot_id':s,'path':PATHS[s//4],'k':PHASES[s%4],'complete':complete,'passed':False}
        if complete:
            err=np.max(abs(a['branch_x_next'][ids].astype(float)-a['donor_next_x'][s].astype(float)),axis=0)
            channels={name:float(err[sl].max()) for name,sl in [('q',slice(0,20)),('qd',slice(20,40)),('p',slice(40,43)),('R_entries',slice(43,52)),('v',slice(52,55)),('omega',slice(55,58))]}
            b=native_rewards(a['branch_object_terms_k'][ids]);truth=float(native_rewards(a['donor_object_terms_k'][s]))
            reward_error=float(abs(b-truth).max());noise=float(max(np.ptp(b),abs(b-truth).max()))
            row.update(channel_maxabs=channels,reward_maxabs=reward_error,baseline3_rewards=b.tolist(),fresh_donor_reward=truth,
                       noise=noise,self15_disagreement_bits=(a['branch_self15'][ids]!=a['donor_self15'][s]).tolist(),
                       self15_disagreement_count=int((a['branch_self15'][ids]!=a['donor_self15'][s]).sum()),
                       passed=all(channels[k]<=(2e-5 if k in ('q','p','R_entries') else 2e-4) for k in channels) and reward_error<=2e-5)
        rows.append(row)
    return {'all48_completed':all(r['complete'] for r in rows),'all16_passed':all(r['passed'] for r in rows),'snapshots':rows}


def physical_secants(a,d,baseline):
    out=[]
    for s in range(16):
        for j in range(8):
            slopes=[];signals=[];complete=True;ids=[]
            for e,eps in enumerate(EPS):
                minus=35*s+3+16*e+2*j;plus=minus+1;ids.append((minus,plus))
                ok=bool(d['valid_pair'][s,j,e] and a['completed'][[minus,plus]].all())
                complete &= ok
                signal=float(native_rewards(a['branch_object_terms_k'][plus])-native_rewards(a['branch_object_terms_k'][minus])) if ok else None
                signals.append(signal);slopes.append(signal/(2*eps) if ok else None)
            base=baseline['snapshots'][s];floor=max(1e-5,10*base['noise']) if base['passed'] else None
            nonflat=complete and floor is not None and all(abs(x)>floor for x in signals)
            disagreement=abs(slopes[0]-slopes[1])/max(abs(slopes[0]),abs(slopes[1]),1e-12) if complete else None
            sign_agrees=bool(slopes[0]*slopes[1]>0) if complete else False
            stable=bool(nonflat and sign_agrees and disagreement<=.5)
            out.append({'snapshot_id':s,'path':PATHS[s//4],'k':PHASES[s%4],'j':j,'complete':complete,
                        'classification':'NOT_RUN' if not complete else 'STABLE_NONFLAT' if stable else 'NONFLAT_UNSTABLE' if nonflat else 'FLAT_OR_BELOW_NOISE',
                        'signals':signals,'slopes':slopes,'signal_floor':floor,'two_epsilon_sign_agrees':sign_agrees,
                        'relative_slope_disagreement':disagreement,'stable_nonflat':stable,'branch_minus_plus_ids':ids})
    return out


def descriptive_agreement(rows):
    """Descriptive dependent-row summaries only, no CI/bootstrap/p-values."""
    selected=[r for r in rows if r['stable_nonflat']]
    if not selected:return {'planned_rows':len(rows),'eligible_rows':0,'sign_agreement':None,'pearson_r':None}
    x=np.array([r['AD_effective_FP32'] for r in selected]);y=np.array([r['physical_slope'] for r in selected])
    return {'planned_rows':len(rows),'eligible_rows':len(selected),'sign_agreement':float(np.mean(np.sign(x)==np.sign(y))),
            'pearson_r':float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12 else None}


def self15_calibration(logits,labels):
    z=np.asarray(logits,float);y=np.asarray(labels,float)
    need(z.shape==(16,15) and y.shape==(16,15) and np.isfinite(z).all() and np.isin(y,[0,1]).all(),'Bad native self15 calibration inputs')
    p=np.exp(-np.logaddexp(0,-z));pred=p>=.5;positive=y.astype(bool)
    tp=int((pred&positive).sum());fp=int((pred&~positive).sum());fn=int((~pred&positive).sum())
    bins=[]
    for i in range(10):
        mask=(p>=i/10)&(p<((i+1)/10) if i<9 else p<=1)
        n=int(mask.sum());bins.append({'bin':i,'count':n,'mean_probability':float(p[mask].mean()) if n else None,'positive_fraction':float(y[mask].mean()) if n else None})
    return {'native_self15_only':True,'fresh_donor_truth':'fixed_primary_no_majority_vote','dependent_snapshot_count':16,'binary_slots':240,
            'BCE':float((np.logaddexp(0,z)-y*z).mean()),'Brier':float(((p-y)**2).mean()),'accuracy_at_probability_half':float((pred==positive).mean()),
            'true_positive_slots':int(y.sum()),'predicted_positive_slots':int(pred.sum()),'TP':tp,'FP':fp,'FN':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'calibration_bins':bins,'ECE10':sum(b['count']*abs(b['mean_probability']-b['positive_fraction']) for b in bins if b['count'])/240}
