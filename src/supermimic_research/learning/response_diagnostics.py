"""Read-only response metrics; finite native secants are not analytic gradients."""
import numpy as np
import torch

BLOCKS={'delta52':(0,52),'warm26':(52,78),'wrench6':(78,84),'Fn3':(84,87),'contact9':(78,87),'object12':(40,52)}


def vector_comparison(predicted,actual):
    p,a=np.asarray(predicted,float).ravel(),np.asarray(actual,float).ravel()
    aa=float(a@a);pp=float(p@p)
    return dict(versus_zero_NMSE=float((p-a)@(p-a)/aa) if aa>0 else None,
                cosine=float(p@a/np.sqrt(pp*aa)) if aa>0 and pp>0 else None,
                predicted_RMS=float(np.sqrt(pp/len(p))),native_RMS=float(np.sqrt(aa/len(a))))


def paired_indices(raw,*,include_amplitude=False):
    fields=['k','dt','joint']+(['amplitude'] if include_amplitude else [])
    values=list(zip(*(raw[f].tolist() for f in fields)));pairs=[];keys=[]
    for key in sorted(set(values)):
        matching=np.array([i for i,v in enumerate(values) if v==key]);sign=raw['sign'][matching]
        if len(matching)!=2 or sorted(sign.tolist())!=[-1,1]:raise ValueError('Exactly one negative and positive branch per declared key')
        minus=int(matching[sign==-1][0]);plus=int(matching[sign==1][0])
        if not np.array_equal(raw['x'][minus,:81],raw['x'][plus,:81]) or raw['x'][minus,-1]!=raw['x'][plus,-1]:raise ValueError('Pair must share source state/warmstart/dt')
        pairs.append((minus,plus));keys.append(key)
    return np.array(pairs,dtype=np.int64),keys


def paired_metrics(model,raw,device):
    pairs,_=paired_indices(raw);x=torch.as_tensor(raw['x'],dtype=torch.float32,device=device)
    with torch.no_grad():prediction=model.prediction(x).cpu().numpy()
    scale=model.y_scale.detach().cpu().numpy();lo,hi=pairs.T
    actual=(raw['y'][hi]-raw['y'][lo])/scale;pred=(prediction[hi]-prediction[lo])/scale
    return {name:vector_comparison(pred[:,a:b],actual[:,a:b]) for name,(a,b) in BLOCKS.items()}


def probe_metrics(model,raw,device):
    pairs,keys=paired_indices(raw,include_amplitude=True);minus,plus=pairs.T
    eps=raw['amplitude'][plus];du=raw['du_da'][plus]
    center=(raw['x'][plus]+raw['x'][minus])*.5;direction=np.zeros_like(center)
    direction[np.arange(len(pairs)),81+raw['joint'][plus]]=du
    if not np.allclose(raw['x'][plus]-raw['x'][minus],2*eps[:,None]*direction,rtol=1e-10,atol=1e-14):raise ValueError('Actual EMA/control derivative does not match central direction')
    x=torch.as_tensor(center,dtype=torch.float32,device=device);v=torch.as_tensor(direction,dtype=torch.float32,device=device)
    with torch.no_grad():
        _,ad=torch.func.jvp(model.prediction,(x,),(v,))
        predictions=model.prediction(torch.as_tensor(raw['x'],dtype=torch.float32,device=device)).cpu().numpy()
    model_ad=ad.cpu().numpy();model_secant=(predictions[plus]-predictions[minus])/(2*eps[:,None]);native=(raw['y'][plus]-raw['y'][minus])/(2*eps[:,None])
    scale=model.y_scale.detach().cpu().numpy();normalized_native=native/scale;normalized_ad=model_ad/scale;normalized_secant=model_secant/scale
    all_metrics={}
    for amplitude in sorted(set(eps.tolist())):
        selected=eps==amplitude
        all_metrics[str(amplitude)]={name:dict(AD=vector_comparison(normalized_ad[selected,a:b],normalized_native[selected,a:b]),
            model_secant=vector_comparison(normalized_secant[selected,a:b],normalized_native[selected,a:b])) for name,(a,b) in BLOCKS.items()}
    dt_metrics={}
    for amplitude in sorted(set(eps.tolist())):
        a=[];b=[]
        for k in [0,3,6]:
            for joint in [0,4,8,9]:
                ids=[i for i,key in enumerate(keys) if key[0]==k and key[2]==joint and key[3]==amplitude]
                if len(ids)!=2:raise ValueError('Exactly two dt per finite response')
                a.append(normalized_native[ids[0]]);b.append(normalized_native[ids[1]])
        a,b=np.array(a),np.array(b)
        dt_metrics[str(amplitude)]={name:vector_comparison(a[:,lo:hi],b[:,lo:hi]) for name,(lo,hi) in BLOCKS.items()}
    arrays=dict(pair_indices=pairs,pair_keys=np.array(keys),model_AD_da=model_ad,model_secant_da=model_secant,native_secant_da=native,prediction=predictions)
    return dict(pairs=len(pairs),by_amplitude=all_metrics,native_coarse_vs_fine_dt=dt_metrics,
        source='Full identical native state, central ±amplitude after actual EMA; AD evaluated at action0. Same finite-width probes never fit.',
        limitation='Central native secants include contact switches/finite solver effects; dt mismatch is not a pure discretization error bound. No assertion of smooth native gradient or imagined reward validity.'),arrays
