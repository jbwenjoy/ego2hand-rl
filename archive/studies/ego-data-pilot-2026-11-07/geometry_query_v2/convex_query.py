"""Independent Euclidean queries for ideal convex colliders (NumPy/SciPy only).

For capsule S+[radius ball] and polytope P, the forbidden capsule translations are
C=conv(P-a,P-b)+radius ball. We compute origin's signed Euclidean distance to C.
Positive is true separation distance; negative is minimum *pure translation* to
touching at the current orientation. It is not a scan SDF or rotation distance.
Qhull uses no jitter/repair; degenerate or numerically invalid hulls raise.
"""
import numpy as np
from scipy.spatial import ConvexHull


def points3(value,name):
    x=np.asarray(value,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or len(x)<4 or not np.isfinite(x).all():
        raise ValueError(name+" must contain finite full-dimensional [N,3] points")
    return x


def _hull(points):
    p=points3(points,"polytope")
    h=ConvexHull(p)  # no QJ random perturbation or failure-as-zero fallback
    if not np.isfinite(h.volume) or h.volume<=0:
        raise ValueError("Non-full-dimensional hull")
    n=h.equations[:,:3];length=np.linalg.norm(n,axis=1)
    if not np.isfinite(h.equations).all() or np.any(length<=0):
        raise ValueError("Invalid hull halfspaces")
    equations=h.equations/length[:,None]
    return p,h,equations


def _closest_triangle_points_origin(triangles):
    """Closest point on every filled triangle: interior plane or one of 3 edges."""
    a,b,c=np.asarray(triangles,dtype=np.float64).transpose(1,0,2)
    e0,e1=b-a,c-a
    n=np.cross(e0,e1);n2=np.einsum('ij,ij->i',n,n)
    if np.any(n2<=0) or not np.isfinite(n2).all():
        raise ValueError("Degenerate hull triangle; no fallback")
    plane=n*(np.einsum('ij,ij->i',a,n)/n2)[:,None]
    q=plane-a
    v=np.einsum('ij,ij->i',np.cross(q,e1),n)/n2
    w=np.einsum('ij,ij->i',np.cross(e0,q),n)/n2
    inside=(v>=0)&(w>=0)&(v+w<=1)
    candidates=[]
    for x,y in ((a,b),(b,c),(c,a)):
        d=y-x;d2=np.einsum('ij,ij->i',d,d)
        if np.any(d2<=0):raise ValueError("Zero hull edge")
        t=np.clip(-np.einsum('ij,ij->i',x,d)/d2,0.,1.)
        candidates.append(x+t[:,None]*d)
    candidates.append(plane)
    candidates=np.stack(candidates,axis=1)
    distances=np.einsum('nij,nij->ni',candidates,candidates)
    distances[~inside,3]=np.inf
    best=distances.argmin(1)
    return candidates[np.arange(len(a)),best]


def signed_origin_polytope(points):
    """Signed origin distance to a closed full-dimensional convex polytope.

    Inside uses nearest supporting face. Outside uses all filled hull triangles,
    including edge/vertex witnesses, rather than max halfspace as a fake distance.
    """
    p,h,eq=_hull(points)
    offsets=eq[:,3]
    outside=bool(offsets.max()>0.)
    if outside:
        closest=_closest_triangle_points_origin(p[h.simplices])
        i=int(np.einsum('ij,ij->i',closest,closest).argmin())
        witness=closest[i];d=float(np.linalg.norm(witness))
        if d<=0:raise ValueError("Inconsistent exterior zero distance")
        normal=-witness/d
        signed=d
    else:
        i=int(offsets.argmax())
        normal=eq[i,:3]
        witness=-offsets[i]*normal
        signed=float(offsets[i])
    support=float(np.max(p@normal))
    plane_gap=-support
    scale=max(1.,float(np.max(np.abs(p))))
    allowance=256*np.finfo(np.float64).eps*scale
    # Direct supporting-plane witness check is independent of hull-plane offsets.
    support_error=abs(plane_gap-signed)
    if support_error>allowance:
        raise ArithmeticError(f"Hull/witness support disagreement {support_error:g} > {allowance:g}")
    return {"signed_origin_distance_m":signed,"origin_outside_unrounded_hull":outside,
            "translation_normal_world":normal,"closest_unrounded_boundary_world":witness,
            "supporting_plane_gap_m":plane_gap,"support_witness_error_m":support_error,
            "roundoff_guard_m":allowance,"input_points":len(p),"hull_vertices":len(h.vertices),
            "hull_triangles":len(h.simplices),"Q_volume_m3":float(h.volume)}


class ConvexPolytope:
    def __init__(self,vertices):
        p,h,_=_hull(vertices)
        self.vertices=p[h.vertices].copy()
        self.source_vertex_count=len(p)

    def capsule(self,a,b,radius):
        a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
        r=float(radius)
        if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([*a,*b,r]).all() or r<0:
            raise ValueError("Finite capsule endpoints/radius>=0 required")
        q=np.concatenate((self.vertices-a,self.vertices-b))
        return self._finish(signed_origin_polytope(q),r,"capsule_vs_convex")

    def convex(self,other_vertices,max_pair_points=500000):
        other=ConvexPolytope(other_vertices)
        count=len(self.vertices)*len(other.vertices)
        if count>max_pair_points:
            raise ValueError(f"Declared pair-point budget exceeded: {count}")
        q=(self.vertices[:,None,:]-other.vertices[None,:,:]).reshape(-1,3)
        out=self._finish(signed_origin_polytope(q),0.,"convex_vs_convex")
        out["other_hull_vertices"]=len(other.vertices)
        return out

    @staticmethod
    def _finish(result,radius,kind):
        result=dict(result)
        margin=result["signed_origin_distance_m"]-radius
        guard=result["roundoff_guard_m"]
        result.update(query_kind=kind,radius_m=radius,signed_translation_margin_m=margin,
            positive_separation_distance_m=max(0.,margin),
            pure_translation_exit_depth_m=max(0.,-margin),
            classification="SEPARATED" if margin>guard else "INTERSECTING" if margin < -guard else "TOUCHING_OR_ROUNDOFF_BAND",
            separation_halfspace_lower_bound_m=result["supporting_plane_gap_m"]-radius-guard,
            margin_roundoff_guarded_interval_m=[margin-guard,margin+guard],
            interpretation="Ideal convex colliders at fixed relative orientation; floating-point Qhull, not interval arithmetic/formal exact predicates, scan SDF, rotation MTD, measured contact or force.")
        return result


def jsonable(result):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
