"""Sampled surface error and explicit clearance gauges, not global certificates."""
import numpy as np


def nearest_surface(mesh,points,chunk=8):
    import trimesh
    closest=[];distances=[]
    for start in range(0,len(points),chunk):
        p,d,_=trimesh.proximity.closest_point_naive(mesh,np.asarray(points)[start:start+chunk]);closest.extend(p);distances.extend(d)
    return np.array(closest),np.array(distances)


def plane_lower_bound(points,halfspaces):
    """Sufficient exterior separation bound for a union of unit-normal polytopes."""
    p=np.asarray(points);values=np.full(len(p),np.inf)
    for eq in halfspaces:values=np.minimum(values,np.max(p@eq[:,:3].T+eq[:,3],axis=1))
    return values


def gauges():
    handle=np.c_[np.full(29,-.052),np.full(29,.05),np.linspace(-.07,.07,29)]
    cavity=np.c_[np.zeros(22),np.linspace(.02,.125,22),np.full(22,.0025)]
    return dict(handle=handle,cavity=cavity),np.array([[.043,.05,.0025],[-.062,.05,.0025],[0.,.004,.0025]])


def corridor_clearance(distances,points,radius):
    """If each endpoint has >r+half segment length, the whole tube is clear.

    Distances can be actual source surface distances or sufficient convex-union
    separation lower bounds; both are1-Lipschitz in exact arithmetic. Float
    calculations are not interval-certified, and source material is unknown.
    """
    steps=np.linalg.norm(np.diff(points,axis=0),axis=1)
    margin=np.minimum(np.asarray(distances)[:-1],np.asarray(distances)[1:])-radius-steps/2
    return dict(minimum_numerical_margin_m=float(margin.min()),clear=bool(np.all(margin>1e-8)))
