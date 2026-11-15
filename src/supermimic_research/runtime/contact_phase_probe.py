"""UNQUALIFIED separate GJK/EPA diagnostic; target-native equivalence failed.

Analytic calibration does not qualify the complex-mesh wrapper. Retain its
outputs only as diagnostic evidence; never use them for task forces/control.
"""
import numpy as np
import warp as wp

from mujoco_warp._src.collision_core import geom_collision_pair,contact_margin_gap
from mujoco_warp._src.collision_gjk import gjk,gjk_phase,epa_phase
from mujoco_warp._src.types import MJ_MAX_EPAFACES,MJ_MAX_EPAHORIZON,GeomType


@wp.struct
class PairArrays:
    geom_type:wp.array(dtype=int)
    geom_dataid:wp.array2d(dtype=int)
    geom_size:wp.array2d(dtype=wp.vec3)
    geom_margin:wp.array2d(dtype=float)
    geom_gap:wp.array2d(dtype=float)
    pair_margin:wp.array2d(dtype=float)
    pair_gap:wp.array2d(dtype=float)
    mesh_vertadr:wp.array(dtype=int)
    mesh_vertnum:wp.array(dtype=int)
    mesh_graphadr:wp.array(dtype=int)
    mesh_vert:wp.array(dtype=wp.vec3)
    mesh_graph:wp.array(dtype=int)
    mesh_polynum:wp.array(dtype=int)
    mesh_polyadr:wp.array(dtype=int)
    mesh_polynormal:wp.array(dtype=wp.vec3)
    mesh_polyvertadr:wp.array(dtype=int)
    mesh_polyvertnum:wp.array(dtype=int)
    mesh_polyvert:wp.array(dtype=int)
    mesh_polymapadr:wp.array(dtype=int)
    mesh_polymapnum:wp.array(dtype=int)
    mesh_polymap:wp.array(dtype=int)
    geom_xpos:wp.array2d(dtype=wp.vec3)
    geom_xmat:wp.array2d(dtype=wp.mat33)
    tolerance:wp.array(dtype=float)


@wp.kernel(enable_backward=False)
def _phases(p:PairArrays,g:int,h:int,pairid:int,gjk_iterations:int,epa_iterations:int,
            vert:wp.array(dtype=wp.vec3),vert_index:wp.array(dtype=int),face:wp.array(dtype=int),
            face_pr:wp.array(dtype=wp.vec3),face_norm:wp.array(dtype=float),horizon:wp.array(dtype=int),
            overflow:wp.array(dtype=int),floats:wp.array(dtype=float),integers:wp.array(dtype=int),
            simplices:wp.array2d(dtype=wp.vec3),indices:wp.array2d(dtype=int),witness:wp.array(dtype=wp.vec3)):
    pair=wp.vec2i(g,h);t1=p.geom_type[g];t2=p.geom_type[h]
    a,b=geom_collision_pair(p.geom_type,p.geom_dataid,p.geom_size,p.mesh_vertadr,p.mesh_vertnum,p.mesh_graphadr,
        p.mesh_vert,p.mesh_graph,p.mesh_polynum,p.mesh_polyadr,p.mesh_polynormal,p.mesh_polyvertadr,p.mesh_polyvertnum,
        p.mesh_polyvert,p.mesh_polymapadr,p.mesh_polymapnum,p.mesh_polymap,p.geom_xpos,p.geom_xmat,pair,0)
    margin,gap=contact_margin_gap(p.geom_margin,p.geom_gap,p.pair_margin,p.pair_gap,pair,pairid,0)
    a.margin=margin;b.margin=margin
    integers[9]=a.index;integers[10]=b.index
    # Match installed _discrete_geoms for supported cylinder/box/mesh inputs.
    discrete=(t1==GeomType.BOX or t1==GeomType.MESH) and (t2==GeomType.BOX or t2==GeomType.MESH) and margin==0.0
    raw=gjk(p.tolerance[0],gjk_iterations,a,b,a.pos,b.pos,t1,t2,gap,discrete)
    integers[11]=a.index;integers[12]=b.index
    c,d=geom_collision_pair(p.geom_type,p.geom_dataid,p.geom_size,p.mesh_vertadr,p.mesh_vertnum,p.mesh_graphadr,
        p.mesh_vert,p.mesh_graph,p.mesh_polynum,p.mesh_polyadr,p.mesh_polynormal,p.mesh_polyvertadr,p.mesh_polyvertnum,
        p.mesh_polyvert,p.mesh_polymapadr,p.mesh_polymapnum,p.mesh_polymap,p.geom_xpos,p.geom_xmat,pair,0)
    c.margin=margin;d.margin=margin;integers[13]=c.index;integers[14]=d.index
    need,dist,ncontact,x1,x2,result,a2,b2=gjk_phase(p.tolerance[0],gap,gjk_iterations,c,d,t1,t2,c.pos,d.pos)
    floats[0]=margin;floats[1]=gap;floats[2]=p.tolerance[0];floats[3]=raw.dist;floats[4]=dist
    integers[0]=raw.dim;integers[1]=int(need);integers[2]=result.dim
    integers[3]=raw.index1;integers[4]=raw.index2
    for j in range(4):
        simplices[0,j]=raw.simplex[j];simplices[1,j]=raw.simplex1[j];simplices[2,j]=raw.simplex2[j]
        indices[0,j]=raw.simplex_index1[j];indices[1,j]=raw.simplex_index2[j]
    idx=int(-1)
    if need:
        dist,ncontact,x1,x2,idx=epa_phase(p.tolerance[0],epa_iterations,result,a2,b2,t1,t2,
            vert,vert_index,face,face_pr,face_norm,horizon,False,0,overflow)
    floats[5]=dist;floats[6]=dist+margin
    integers[5]=ncontact;integers[6]=idx;integers[7]=int(dist>=gap);integers[8]=int(dist<gap and ncontact>0)
    witness[0]=x1;witness[1]=x2


class ContactPhaseProbe:
    def __init__(self,model,data,*,device='cuda:0'):
        # Kernels use native functions verbatim with explicit separate diagnostic
        # scratch. CYLINDER/BOX/MESH; no sphere/capsule shrink or sensor cutoff.
        self.p=PairArrays()
        for name in PairArrays.vars:
            if name=='tolerance':value=model.opt.ccd_tolerance
            elif name in ['geom_xpos','geom_xmat']:value=getattr(data,name)
            else:value=getattr(model,name)
            setattr(self.p,name,value)
        self.gjk_iterations=int(model.opt.ccd_iterations);self.epa_iterations=int(model.opt.ccd_iterations)
        self.types=model.geom_type.numpy();self.device=device
        self.arrays=dict(vert=wp.zeros(10+2*self.epa_iterations,dtype=wp.vec3,device=device),
            vert_index=wp.zeros(10+2*self.epa_iterations,dtype=int,device=device),
            face=wp.zeros(6+MJ_MAX_EPAFACES*self.epa_iterations,dtype=int,device=device),
            face_pr=wp.zeros(6+MJ_MAX_EPAFACES*self.epa_iterations,dtype=wp.vec3,device=device),
            face_norm=wp.zeros(6+MJ_MAX_EPAFACES*self.epa_iterations,dtype=float,device=device),
            horizon=wp.zeros(MJ_MAX_EPAHORIZON,dtype=int,device=device),overflow=wp.zeros(1,dtype=int,device=device),
            floats=wp.zeros(7,dtype=float,device=device),integers=wp.zeros(15,dtype=int,device=device),
            simplices=wp.zeros((3,4),dtype=wp.vec3,device=device),indices=wp.zeros((2,4),dtype=int,device=device),witness=wp.zeros(2,dtype=wp.vec3,device=device))

    def query(self,g,h,pairid=-1):
        if any(int(self.types[x]) not in [5,6,7] for x in [g,h]):raise ValueError('Diagnostic restricted to CYLINDER/BOX/MESH')
        for value in self.arrays.values():value.zero_()
        wp.launch(_phases,1,inputs=[self.p,g,h,pairid,self.gjk_iterations,self.epa_iterations,*self.arrays.values()],device=self.device)
        values={key:value.numpy().copy() for key,value in self.arrays.items()};f,i=values['floats'],values['integers']
        result=dict(margin_m=float(f[0]),gap_m=float(f[1]),ccd_tolerance_m=float(f[2]),raw_gjk_distance_m=float(f[3]),
            phase_gjk_distance_m=float(f[4]),final_pre_margin_distance_m=float(f[5]),final_surface_distance_m=float(f[6]),
            raw_simplex_dim=int(i[0]),needs_epa=bool(i[1]),phase_simplex_dim=int(i[2]),support_cache_indices=[int(i[3]),int(i[4])],
            geometry_cache_before_raw=[int(i[9]),int(i[10])],geometry_cache_after_raw=[int(i[11]),int(i[12])],geometry_cache_fresh_phase=[int(i[13]),int(i[14])],
            pre_multiccd_contacts=int(i[5]),multiccd_index=int(i[6]),native_gap_reject=bool(i[7]),pre_multiccd_emit=bool(i[8]),
            overflow=bool(values['overflow'][0]),geometry_types=[int(self.types[g]),int(self.types[h])],configured_gjk_iterations=self.gjk_iterations,configured_epa_iterations=self.epa_iterations,
            native_internal_equivalence_qualified=False,scratch_policy='ZERO_BEFORE_EACH_QUERY_SEPARATE_FROM_NATIVE_PIPELINE',
            scope='Installed GJK/EPA function outputs, no original iteration counter/termination label fabricated. Pre-multiccd is not the final native contact manifold.')
        return result,values
