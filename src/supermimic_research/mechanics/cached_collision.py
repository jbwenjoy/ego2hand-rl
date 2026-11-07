"""Bounded exact-input cache for CPU ideal-convex pair queries.

Keys contain raw world-shape bytes and exact-query mode. No state, normal,
distance or bound is reused across changed operands; no approximate hashing.
"""
from collections import OrderedDict
import numpy as np
from .collision import CollisionAudit


class CachedCollisionAudit(CollisionAudit):
    def __init__(self,model,capacity=2048):
        super().__init__(model)
        if type(capacity) is not int or capacity<1:raise ValueError('Positive pair cache capacity')
        self.capacity=capacity;self.cache=OrderedDict();self.query_counts=dict(calls=0,hits=0,misses=0,evictions=0)

    @staticmethod
    def key(shape):
        values=np.r_[shape['a'],shape['b'],shape['r']] if shape['kind']=='capsule' else shape['vertices']
        values=np.asarray(values)
        return shape['kind'],values.dtype.str,values.shape,values.tobytes()

    def query(self,a,b,force_exact=False):
        self.query_counts['calls']+=1;key=(self.key(a),self.key(b),force_exact)
        if key in self.cache:
            self.query_counts['hits']+=1;value=self.cache.pop(key);self.cache[key]=value
            return value.copy()
        self.query_counts['misses']+=1;value=super().query(a,b,force_exact=force_exact)
        self.cache[key]=value.copy()
        if len(self.cache)>self.capacity:self.cache.popitem(last=False);self.query_counts['evictions']+=1
        return value
