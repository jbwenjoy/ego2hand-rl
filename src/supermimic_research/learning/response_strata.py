"""Read-only component/joint response diagnostics; never discard zero-effect axes."""
import numpy as np
from .response_diagnostics import vector_comparison

COMPONENTS={'hand40':(0,40),'COM_position3':(40,43),'rotation3':(43,46),'COM_velocity3':(46,49),'omega3':(49,52),'force3':(78,81),'torque3':(81,84),'Fn3':(84,87)}


def response_strata(raw,arrays):
    pairs=raw['pairs'];high=pairs[:,1].astype(np.int64);joint=pairs[:,2].astype(np.int64);amplitude=raw['amplitude'][high]
    actual=arrays['native_secant_normalized'];ad=arrays['model_AD_normalized'];secant=arrays['model_secant_normalized']
    if actual.shape!=ad.shape or actual.shape!=secant.shape or actual.shape!=(len(pairs),87):raise ValueError('All original pair rows and output87, not a filtered subset')
    groups={}
    for amp in sorted(set(amplitude.tolist())):
        masks={'all':amplitude==amp}
        masks.update({f'joint{j}':(amplitude==amp)&(joint==j) for j in range(20)})
        for name,mask in masks.items():
            components={block:dict(AD=vector_comparison(ad[mask,a:b],actual[mask,a:b]),secant=vector_comparison(secant[mask,a:b],actual[mask,a:b])) for block,(a,b) in COMPONENTS.items()} if mask.any() else None
            groups[f'amplitude{amp}/{name}']=dict(pairs=int(mask.sum()),components=components)
    return dict(groups=groups,pairs=len(pairs),scope='All CHECK pairs, fixed original normalization; predeclared output components and all20 control axes. Zero native effect retains native_RMS0, predicted_RMS and undefined ratio/cosine=None. No data filtering, fitting or model selection.')
