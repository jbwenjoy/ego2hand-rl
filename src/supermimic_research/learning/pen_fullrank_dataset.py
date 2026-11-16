"""Full20 TRAIN pairs and explicit reserved-validation isolation.

Existing CHECK/common-data schemas are not reinterpreted. The same full20
point bank supplies both rank4/rank20 response-supervision comparisons.
"""
import numpy as np
from .pen_response import fit_pen_scales

TRAIN_ROLE='TRAIN_CONTROLLED_PHYSICAL_RESPONSE'
VALIDATION_ROLE='VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT'


def full_rank_response_data(train_report,train,validation_report,validation,directions):
    for report,data,role in [(train_report,train,TRAIN_ROLE),(validation_report,validation,VALIDATION_ROLE)]:
        if (report['schema']!='controlled_pen_response_preparation:2' or report['role']!=role
                or report['CHECK_response_rows_used']!=0 or report['validation_rows_used_for_TRAIN']!=0
                or report['passed']!=report['total'] or report['scales_fitted']):
            raise ValueError('Explicit new full-rank TRAIN and reserved NEVER_FIT validation required')
        n=report['total_rows']
        if (data['x'].shape!=(n,102) or data['y'].shape!=(n,90) or data['heads'].shape!=(n,90)
                or data['valid'].shape!=(n,) or data['valid'].dtype!=np.bool_
                or int(data['valid'].sum())!=report['qualified_rows']
                or not all(np.isfinite(data[k]).all() for k in ['x','y','heads'])):
            raise ValueError('Reviewed finite aligned full-rank data and exact masks required')
    directions=np.asarray(directions,dtype=np.float64)
    if directions.shape!=(20,20) or not np.isfinite(directions).all() or not np.allclose(directions@directions.T,np.eye(20),atol=2e-15,rtol=0):
        raise ValueError('Explicit orthonormal20 TRAIN PD basis required')
    valid=train['valid'];pairs=[];audit=[];seen=set()
    for gi,group in enumerate(train_report['response_groups']):
        direction=group['direction'];epsilon=group['epsilon'];identity=(group['source_name'],direction,epsilon)
        if identity in seen or type(direction) is not int or not 0<=direction<20 or epsilon not in [.002,.005]:
            raise ValueError('Unique declared TRAIN source/direction/amplitude groups required')
        seen.add(identity);plus=np.asarray(group['rows']['1'],dtype=np.int64);minus=np.asarray(group['rows']['-1'],dtype=np.int64);ids=np.r_[plus,minus]
        if plus.ndim!=1 or minus.ndim!=1 or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(valid)):
            raise ValueError('Distinct in-range TRAIN row identities required')
        complete=plus.shape==(2,) and minus.shape==(2,)
        eligible=bool(complete and valid[ids].all())
        if group['complete']!=complete or group['qualified']!=eligible:raise ValueError('Group cannot override completeness/physical row masks')
        audit.append(dict(TRAIN_group_index=gi,source_name=group['source_name'],direction=direction,epsilon=epsilon,complete=complete,eligible=eligible))
        if not eligible:continue
        xp,xm=train['x'][plus[0]],train['x'][minus[0]]
        if (not np.array_equal(train['x'][plus],np.tile(xp,(2,1))) or not np.array_equal(train['x'][minus],np.tile(xm,(2,1)))
                or not np.array_equal(xp[:81],xm[:81]) or xp[-1]!=xm[-1]
                or np.max(abs(xp[81:101]-xm[81:101]-2*epsilon*directions[direction]))>4e-7):
            raise ValueError('Exact source/warm/dt and actual paired PD intervention required')
        yp=train['y'][plus].mean(0);ym=train['y'][minus].mean(0)
        pairs.append(dict(x_plus=xp,x_minus=xm,y_plus=yp,y_minus=ym,finite_contrast=(yp-ym)/(2*epsilon),epsilon=epsilon,
            direction20=directions[direction],direction_index=direction,TRAIN_group_index=gi,TRAIN_row_plus=plus,TRAIN_row_minus=minus))
    if not pairs:raise ValueError('No qualified full-rank TRAIN pairs')
    packed={k:np.array([p[k] for p in pairs]) for k in pairs[0]}
    subsets={f'rank{n}':np.flatnonzero(packed['direction_index']<n) for n in [4,20]}
    if not len(subsets['rank4']):raise ValueError('No matched rank4 TRAIN response subset')
    scales,heads=fit_pen_scales(train['x'][valid],train['y'][valid],train['heads'][valid])
    return dict(scales=scales,head_scales=heads,pairs=packed,pair_subsets=subsets,pair_audit=audit)
