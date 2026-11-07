"""Small optimizer-facing interface over the unchanged checked convex_query core.

normal_world is the direction for translating the HAND away from the object.
Support feature sets are true extremal points of the stored ideal colliders.
They are not an arbitrarily selected Euclidean closest surface-witness pair.
"""
import numpy as np
from convex_query import ConvexPolytope


def support_features(result,object_vertices,*,hand_vertices=None,a=None,b=None,radius=None):
    p=np.asarray(object_vertices,dtype=np.float64)
    n=np.asarray(result['translation_normal_world'],dtype=np.float64)
    n=n/np.linalg.norm(n)
    guard=float(result['roundoff_guard_m'])
    po=p@n;maximum=float(po.max())
    pids=np.flatnonzero(po>=maximum-guard)
    if hand_vertices is not None:
        hand=np.asarray(hand_vertices,dtype=np.float64)
        hh=hand@n;minimum=float(hh.min())
        hids=np.flatnonzero(hh<=minimum+guard)
        hp=hand[hids]
        feature='convex_vertex_set'
    else:
        endpoints=np.stack([a,b]).astype(np.float64)
        hh=endpoints@n;minimum=float(hh.min())-float(radius)
        hids=np.flatnonzero(hh<=hh.min()+guard)
        hp=np.unique(endpoints[hids]-float(radius)*n,axis=0)
        feature='capsule_extremal_point_or_generator_endpoints'
    gap=minimum-maximum
    if abs(gap-result['signed_translation_margin_m'])>2*guard:
        raise ArithmeticError('Direct object/hand support plane disagrees with signed query')
    return {'signed_gap_m':result['signed_translation_margin_m'],'normal_world':n,
            'normal_semantics':'Translating the hand in +normal locally increases separation; orientation fixed.',
            'normal_uniqueness':'Unique outside unrounded configuration hull; NOT_ASSERTED inside/touching where nearest-face ties may exist.' if result['origin_outside_unrounded_hull'] else 'NOT_ASSERTED_INSIDE_OR_TOUCHING_CONFIGURATION_HULL',
            'supporting_plane_gap_m':gap,'object_plane_max_projection_m':maximum,'hand_plane_min_projection_m':minimum,
            'object_support_points_world_m':p[pids],'object_support_source_indices':pids,
            'hand_support_points_world_m':hp,'hand_support_source_indices':hids,
            'hand_support_feature':feature,'support_feature_nonunique':len(pids)>1 or len(hp)>1,
            'closest_surface_witness_pair':None,
            'witness_scope':'Both extremal feature sets support the separating/exit plane. Arbitrary cross-pairs need not be closest or touch tangentially; do not optimize their chosen point coordinates.'}


class ConvexQuery:
    def __init__(self,object_vertices):
        self.object_vertices=np.asarray(object_vertices,dtype=np.float64).copy()
        self.poly=ConvexPolytope(self.object_vertices)

    def capsule(self,a,b,radius):
        result=self.poly.capsule(a,b,radius)
        return {**result,**support_features(result,self.object_vertices,a=a,b=b,radius=radius)}

    def convex(self,hand_vertices):
        result=self.poly.convex(hand_vertices)
        return {**result,**support_features(result,self.object_vertices,hand_vertices=hand_vertices)}


def capsule_convex(a,b,radius,object_vertices):
    return ConvexQuery(object_vertices).capsule(a,b,radius)


def convex_convex(hand_vertices,object_vertices):
    return ConvexQuery(object_vertices).convex(hand_vertices)


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value
