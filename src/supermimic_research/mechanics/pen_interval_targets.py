"""Four interval maxima from saved ticks; no endpoint geometry substitution."""
import numpy as np

CHANNELS=['object_depth_max_m','self_depth_max_m','joint_excursion_max_rad','nonhand_normal_max_N']
TRACE_KEYS=['pre_penetration_m','pre_self_penetration_m','post_joint_limit_excursion_rad']


def interval_targets(trace,window,geom_bodyid,geom_names,object_body,cfg,*,phase_valid):
    if any(np.asarray(trace[k]).shape!=(8,) for k in TRACE_KEYS) or np.asarray(window['geometry_max']).shape!=(3,):
        raise ValueError('Exactly eight saved native ticks and three window maxima required')
    raw=np.asarray(trace['raw_contact_tick_plus23']);body=np.asarray(geom_bodyid)
    if raw.ndim!=2 or raw.shape[1]!=24 or not np.isfinite(raw).all() or len(geom_names)!=len(body):raise ValueError('Exact finite raw contact table and geometry map required')
    nonhand=np.zeros(8,np.float64)
    for row in raw:
        tick,a,b=(int(row[k]) for k in range(3))
        if not (row[0]==tick and 0<=tick<8 and row[1]==a and row[2]==b and 0<=a<len(body) and 0<=b<len(body)):
            raise ValueError('Raw contact indices outside saved interval')
        oa,ob=body[a]==object_body,body[b]==object_body
        if oa!=ob and not geom_names[b if oa else a].startswith('robot/'):
            nonhand[tick]+=max(0.,float(row[18]))
    independent=np.r_[[np.asarray(trace[k]).max() for k in TRACE_KEYS],nonhand.max()]
    authoritative=np.r_[window['geometry_max'],window['nonhand_max_sum'][0]].astype(np.float64)
    if not np.isfinite(independent).all() or not np.isfinite(authoritative).all() or (authoritative<0).any():
        raise ValueError('Finite nonnegative path maxima required')
    limits=np.array([cfg[k] for k in ['object_penetration_m','self_penetration_m','joint_excursion_rad','nonhand_normal_N']],np.float64)
    if not np.isfinite(limits).all() or (limits<=0).any():raise ValueError('Positive original physical limits required')
    errors=int(window['errors']);global_error=int(np.asarray(window['global_error']).reshape(-1)[0])
    failed=bool((authoritative>limits).any() or errors or global_error or not phase_valid)
    # Original geometry_loss casts its sum to float32 before reward arithmetic.
    components=np.log1p(np.square(authoritative/limits));loss=np.asarray(components.sum(),dtype=np.float32)
    penalty=np.asarray(-cfg['geometry_soft_weight']*loss-cfg['geometry_failure_penalty']*np.float32(failed),dtype=np.float32)
    checks=dict(eight_samples=int(window['counts'][0])==8,
        trace_geometry_matches_window=bool(np.all(abs(independent[:3]-authoritative[:3])<=1e-12)),
        nonhand_sum_then_max=bool(abs(independent[3]-authoritative[3])<=1e-9),
        original_loss=bool(abs(float(loss)-float(window['objective_geometry_loss']))<=2e-6),
        original_failure=failed==bool(window['objective_geometry_failure']))
    return dict(path4=authoritative,trace_path4=independent,log1p_squared_ratio4=components,
        geometry_loss=loss,geometry_penalty=penalty,geometry_failure=failed,errors=errors,global_error=global_error,checks=checks)
