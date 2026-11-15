"""Role-checked common response data and valid-TRAIN-only normalization."""
import numpy as np
from .pen_response import fit_pen_scales

TRAIN_ROLE='TRAIN_CONTROLLED_PHYSICAL_RESPONSE'
CHECK_ROLE='CHECK_TRAINED_ACTOR_PHYSICAL_RESPONSE_NEVER_FIT'


def common_response_data(train_report,train,check_report,check,directions):
    if train_report['role']!=TRAIN_ROLE or check_report['role']!=CHECK_ROLE or train_report['CHECK_response_rows_used']!=0:raise ValueError('Explicit disjoint TRAIN supervision and NEVER_FIT developmental CHECK')
    for report,data in [(train_report,train),(check_report,check)]:
        n=report['total_rows']
        if report['passed']!=report['total'] or data['x'].shape!=(n,102) or data['y'].shape!=(n,90) or data['heads'].shape!=(n,90) or data['valid'].shape!=(n,) or data['valid'].dtype!=np.bool_ or int(data['valid'].sum())!=report['qualified_rows'] or not all(np.isfinite(data[k]).all() for k in ['x','y','heads']):raise ValueError('Reviewed finite aligned records and explicit qualification mask')
    valid=train['valid'];scales,heads=fit_pen_scales(train['x'][valid],train['y'][valid],train['heads'][valid])
    directions=np.asarray(directions,dtype=np.float64)
    if directions.ndim!=2 or directions.shape[1]!=20 or not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-12,rtol=0):raise ValueError('Fixed finite unit PD directions20')
    pairs=[];audit=[]
    for group in train_report['response_groups']:
        plus=np.asarray(group['rows']['1'],dtype=np.int64);minus=np.asarray(group['rows']['-1'],dtype=np.int64);ids=np.r_[plus,minus]
        if plus.shape!=(2,) or minus.shape!=(2,) or len(set(ids.tolist()))!=4 or np.any(ids<0) or np.any(ids>=len(valid)):raise ValueError('Two distinct original repeats of each sign')
        eligible=bool(valid[ids].all())
        if eligible!=group['qualified']:raise ValueError('Group mask cannot override row qualification')
        audit.append(dict(source_name=group['source_name'],direction=group['direction'],epsilon=group['epsilon'],eligible=eligible))
        if not eligible:continue
        xp,xm=train['x'][plus[0]],train['x'][minus[0]];epsilon=float(group['epsilon']);direction=directions[group['direction']]
        if epsilon<=0 or not np.array_equal(train['x'][plus],np.tile(xp,(2,1))) or not np.array_equal(train['x'][minus],np.tile(xm,(2,1))) or not np.array_equal(xp[:81],xm[:81]) or xp[-1]!=xm[-1] or np.max(abs(xp[81:101]-xm[81:101]-2*epsilon*direction))>4e-7:raise ValueError('Exact source/warm/dt and actual paired PD intervention required')
        yp=train['y'][plus].mean(0);ym=train['y'][minus].mean(0)
        pairs.append(dict(x_plus=xp,x_minus=xm,y_plus=yp,y_minus=ym,finite_contrast=(yp-ym)/(2*epsilon),epsilon=epsilon,direction20=direction,TRAIN_row_plus=plus,TRAIN_row_minus=minus))
    if not pairs:raise ValueError('No qualified TRAIN finite-response pairs')
    return dict(scales=scales,head_scales=heads,pairs={k:np.array([p[k] for p in pairs]) for k in pairs[0]},pair_audit=audit)
