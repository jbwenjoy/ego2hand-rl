"""Independent executed-control versus nominal-epsilon task-gradient arithmetic.

NumPy only. No scores/models are evaluated. Two physical eps form one group;
their old stability descriptor is model-independent and uses nominal eps slopes.
Actual coordinate secants and legacy parameter slopes remain separate.
"""
import numpy as np

EPS=np.asarray([.002,.001],dtype=np.float64)


def f32(a,shape,label):
    a=np.asarray(a)
    if a.dtype!=np.float32 or a.shape!=shape or not np.isfinite(a).all():raise ValueError('Original finite FP32 '+label)
    return a


def physical_classification(minus_scores,plus_scores,baseline3,donor_score):
    m=f32(minus_scores,(2,),'score minus');p=f32(plus_scores,(2,),'score plus');b=f32(baseline3,(3,),'baseline scores')
    d=np.asarray(donor_score)
    if d.dtype!=np.float32 or d.shape!=() or not np.isfinite(d):raise ValueError('Original finite FP32 donor scalar')
    bb=b.astype('f8');span=float(bb.max()-bb.min());donor_error=float(abs(bb-float(d)).max())
    floor=max(1e-5,10*max(span,donor_error));delta=p.astype('f8')-m.astype('f8');slopes=delta/(2*EPS)
    signal=abs(delta)>floor;relative=float(abs(slopes[0]-slopes[1])/max(abs(slopes[0]),abs(slopes[1]),1e-12))
    stable=bool(signal.all() and np.sign(slopes[0])==np.sign(slopes[1]) and relative<=.5)
    return {'baseline_range':span,'baseline_donor_error':donor_error,'signal_floor':floor,'delta_score':delta,
      'legacy_nominal_slopes':slopes,'signal_above_floor':signal,'relative_epsilon_disagreement':relative,'stable_nonflat':stable,
      'label':'STABLE_NONFLAT' if stable else 'FLAT_OR_BELOW_NOISE' if not signal.all() else 'NONFLAT_UNSTABLE',
      'classification_source':'physical only; shared32 groups, no model/old9 selection'}


def projection_conventions(gradient20,center20,minus2x20,plus2x20,minus_scores,plus_scores,joint_index):
    g=f32(gradient20,(20,),'model AD');c=f32(center20,(20,),'actual center');m=f32(minus2x20,(2,20),'actual minus');p=f32(plus2x20,(2,20),'actual plus')
    rminus=f32(minus_scores,(2,),'physical score minus');rplus=f32(plus_scores,(2,),'physical score plus')
    if type(joint_index) is not int or not 0<=joint_index<20:raise ValueError('Control coordinate20')
    delta=p.astype('f8')-m.astype('f8');h=delta[:,joint_index]
    if (h==0).any() or not np.isfinite(h).all():raise ValueError('Any zero actual coordinate denominator blocks whole group, no epsilon drop')
    dr=rplus.astype('f8')-rminus.astype('f8');effective=delta/(2*EPS[:,None]);offset=(p.astype('f8')+m.astype('f8'))*.5-c.astype('f8')
    ad_delta=np.asarray([np.dot(g.astype('f8'),v) for v in delta]);ad_legacy=np.asarray([np.dot(g.astype('f8'),v) for v in effective])
    return {'actual_delta_u_rad_F64':delta,'actual_du_coordinate_rad_F64':h,'physical_delta_score_F64':dr,
      'physical_actual_coordinate_secant':dr/h,'model_actual_coordinate_projection':ad_delta/h,
      'physical_legacy_nominal_slope':dr/(2*EPS),'model_legacy_effective_projection':ad_legacy,
      'model_nominal_coordinate_gradient':float(g[joint_index]),'legacy_d_eff_dimensionless':effective,'center_offset_rad':offset,
      'units':'controlled post20 goal_k score / command rad; legacy parameter and actual-coordinate conventions are distinct',
      'physical_or_wholehidden_Jacobian_qualified':False}
