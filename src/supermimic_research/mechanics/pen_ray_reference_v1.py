"""Independent FP64 evaluation of fixed pyramidal ray inputs, not dynamics truth."""
import numpy as np


def friction_point(jaref,advance,d,loss):
    if d<=0:
        raise ValueError('Positive friction-row D required; preserve unsupported rows')
    radius=loss/d; x=jaref+advance
    def cost(value):
        return .5*d*value*value if -radius<value<radius else loss*(abs(value)-.5*radius)
    if -radius<jaref<radius and -radius<x<radius:
        delta=advance*(d*jaref+.5*d*advance)
    elif jaref<=-radius and x<=-radius:
        delta=-loss*advance
    elif jaref>=radius and x>=radius:
        delta=loss*advance
    else:
        # Integrate the continuous clipped force over branch boundaries. This
        # avoids subtracting two large costs when the branch changes.
        left,right=sorted([jaref,x]); points=sorted(set([left,right]+[b for b in [-radius,radius] if left<b<right]))
        delta=0.
        for a,b in zip(points,points[1:]):
            midpoint=.5*(a+b)
            delta+=(b-a)*(-loss if midpoint<=-radius else loss if midpoint>=radius else .5*d*(a+b))
        if x<jaref: delta=-delta
    if -radius<x<radius: gradient=d*x; hessian=d
    elif x<=-radius: gradient=-loss; hessian=0.
    else: gradient=loss; hessian=0.
    return delta,gradient,hessian,cost(jaref)


def evaluate(alpha,ne,nf,rows,vectors):
    """Rows: Jaref,jv,D,loss; vectors: search,mv,Ma,smooth,qacc,raw_grad.

    Raw arrays are fixed observations. No regenerated geometry/J/M, forward,
    optimizer, new line-search decision, action derivative or time integration.
    """
    rows=np.asarray(rows,np.float64); vectors=np.asarray(vectors,np.float64)
    if rows.ndim!=2 or rows.shape[1]!=4 or vectors.ndim!=2 or vectors.shape[1]!=6:
        raise ValueError('Explicit recorded row/vector layout required')
    if type(ne) is not int or type(nf) is not int or min(ne,nf)<0 or ne+nf>len(rows):
        raise ValueError('Actual equality/friction counts required')
    if not np.isfinite(alpha) or not np.isfinite(rows).all() or not np.isfinite(vectors).all() or np.any(rows[:,2:]<0):
        raise ValueError('Finite ray and nonnegative D/loss required')
    search,mv,ma,smooth=vectors[:,:4].T
    q1=float(search@(ma-smooth)); hessian=float(search@mv)
    gauss=np.array([alpha*(q1+.5*alpha*hessian),q1+alpha*hessian,hessian])
    total=np.zeros(3); row_points=[]; absolute_cost0=0.
    for i,(jar,jv,d,loss) in enumerate(rows):
        advance=alpha*jv; x=jar+advance
        if ne<=i<ne+nf:
            delta,g,h,cost0=friction_point(jar,advance,d,loss); point=np.array([delta,g*jv,h*jv*jv])
        elif i<ne:
            point=np.array([advance*(d*jar+.5*d*advance),d*x*jv,d*jv*jv]); cost0=.5*d*jar*jar
        else:
            cost0=.5*d*jar*jar if jar<0 else 0.
            if x<0:
                delta=advance*(d*jar+.5*d*advance) if jar<0 else .5*d*x*x
                point=np.array([delta,d*x*jv,d*jv*jv])
            else: point=np.array([-cost0,0.,0.])
        total+=point; row_points.append(point); absolute_cost0+=cost0
    return dict(point=gauss+total,gauss=gauss,rows=total,per_row=np.asarray(row_points).reshape((-1,3)),absolute_rows_cost0=absolute_cost0,
        scope='FP64 piecewise objective at fixed recorded FP32 ray inputs;not true full-dynamics precision or an alternate accepted step.')
