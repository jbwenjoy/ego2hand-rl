"""Current compiled convex geometry audit, independent of native CCD distances.

Float Qhull/support-witness checks, not exact predicates or scan geometry.
An unknown query cannot pass. Construction only reads compiled collider data.
"""
from itertools import product
import numpy as np
from .convex import ConvexPolytope


def segment_distance(a,b,c,d):
    u=b-a;v=d-c;r=a-c;A=float(u@u);E=float(v@v);F=float(v@r)
    if A<=1e-30 and E<=1e-30:return float(np.linalg.norm(a-c))
    if A<=1e-30:s=0.;t=float(np.clip(F/E,0,1))
    else:
        C=float(u@r)
        if E<=1e-30:t=0.;s=float(np.clip(-C/A,0,1))
        else:
            B=float(u@v);den=A*E-B*B
            s=float(np.clip((B*F-C*E)/den,0,1)) if den>1e-30 else 0.
            t=(B*s+F)/E
            if t<0:t=0.;s=float(np.clip(-C/A,0,1))
            elif t>1:t=1.;s=float(np.clip((B-C)/A,0,1))
    return float(np.linalg.norm(a+s*u-c-t*v))


def self_pairs(model):
    """Original compiled mask/weld-parent/exclude/explicit-pair eligibility."""
    import mujoco
    hand=[g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g]) and g not in (model.geom('phone24_collision_convex').id,model.geom('stage1_single_table').id)]
    explicit={tuple(sorted((int(a),int(b)))) for a,b in zip(model.pair_geom1,model.pair_geom2)}
    excluded=set(map(int,model.exclude_signature));disable=bool(int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT));pairs=[]
    for i,g in enumerate(hand):
        for h in hand[i+1:]:
            b1,b2=int(model.geom_bodyid[g]),int(model.geom_bodyid[h]);w1,w2=int(model.body_weldid[b1]),int(model.body_weldid[b2])
            pw1=int(model.body_weldid[model.body_parentid[w1]]);pw2=int(model.body_weldid[model.body_parentid[w2]])
            if (g,h) not in explicit:
                if b1==b2 or w1==w2:continue
                if not disable and w1 and w2 and (w1==pw2 or w2==pw1):continue
                if ((min(b1,b2)<<16)|max(b1,b2)) in excluded:continue
                if not ((int(model.geom_contype[g])&int(model.geom_conaffinity[h])) or (int(model.geom_contype[h])&int(model.geom_conaffinity[g]))):continue
            pairs.append((g,h))
    return hand,pairs


class CollisionAudit:
    def __init__(self,model):
        self.model=model;self.phone=model.geom('phone24_collision_convex').id;self.table=model.geom('stage1_single_table').id
        self.hand,self.self=self_pairs(model);self.locals={}
        if len(self.hand)+1!=22 or len(self.self)!=189:raise ValueError('Original22/189 collision set required')
        for gid in self.hand+[self.phone,self.table]:
            kind=int(model.geom_type[gid])
            if kind==7:
                mesh=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh])
                self.locals[gid]=ConvexPolytope(model.mesh_vert[start:start+n].astype(float)).vertices
            elif kind==6:self.locals[gid]=np.array(list(product(*[(-s,s) for s in model.geom_size[gid]])))
            elif kind!=3:raise ValueError('Unsupported compiled collider type')

    def shape(self,data,gid):
        R=data.geom_xmat[gid].reshape(3,3);p=data.geom_xpos[gid]
        if int(self.model.geom_type[gid])==3:
            delta=self.model.geom_size[gid,1]*R[:,2]
            a,b=p-delta,p+delta;r=float(self.model.geom_size[gid,0])
            return dict(kind='capsule',a=a,b=b,r=r,bounds=np.array([np.minimum(a,b)-r,np.maximum(a,b)+r]))
        vertices=self.locals[gid]@R.T+p
        return dict(kind='convex',vertices=vertices,bounds=np.array([vertices.min(0),vertices.max(0)]))

    @staticmethod
    def query(a,b,force_exact=False):
        A,B=a['bounds'],b['bounds'];sep=float(np.linalg.norm(np.maximum(np.maximum(A[0]-B[1],B[0]-A[1]),0)))
        guard=256*np.finfo(float).eps*max(1.,float(abs(A).max()),float(abs(B).max()))
        if not force_exact and sep>.0005:return dict(status='DISJOINT_AABB_LOWER_BOUND',lower_bound_m=sep-guard,upper_bound_m=None,gap_m=None)
        if a['kind']==b['kind']=='capsule':gap=segment_distance(a['a'],a['b'],b['a'],b['b'])-a['r']-b['r']
        else:
            if a['kind']!='convex':a,b=b,a
            poly=ConvexPolytope(a['vertices'])
            q=poly.capsule(b['a'],b['b'],b['r']) if b['kind']=='capsule' else poly.convex(b['vertices'],max_pair_points=4_000_000)
            gap=q['signed_translation_margin_m'];guard=q['roundoff_guard_m']
        return dict(status='NUMERICAL_IDEAL_CONVEX',lower_bound_m=float(gap-guard),upper_bound_m=float(gap+guard),gap_m=float(gap))

    def evaluate(self,data):
        shapes={g:self.shape(data,g) for g in self.hand+[self.phone,self.table]};rows=[]
        for kind,pairs in [('phone',[(self.phone,g) for g in self.hand+[self.table]]),('self',self.self)]:
            for g,h in pairs:
                pad=kind=='phone' and h in (15,28,41)
                try:
                    q=self.query(shapes[g],shapes[h],force_exact=pad)
                    passed=q['lower_bound_m']>=(-.0001 if pad else 0.) and (not pad or q['upper_bound_m']<=.0002)
                    rows.append(dict(kind=kind,geom_ids=[g,h],pad=pad,passed=bool(passed),**q))
                except (ValueError,ArithmeticError,RuntimeError) as error:
                    rows.append(dict(kind=kind,geom_ids=[g,h],pad=pad,passed=False,status='UNKNOWN',error=repr(error)))
        return dict(rows=rows,phone_pairs=22,self_pairs=189,passed=all(r['passed'] for r in rows),unknown=sum(r['status']=='UNKNOWN' for r in rows),
            scope='Fixed-orientation ideal convex geometry with float Qhull/support witnesses or conservative disjoint AABBs. Original pad[-100,+200]um, nonpad/self>=0. No formal exact predicates, native contact force or future-state certificate.')
