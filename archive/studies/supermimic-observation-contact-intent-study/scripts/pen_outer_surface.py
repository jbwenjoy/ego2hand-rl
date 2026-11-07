"""Exact nearest exterior patches for the known coaxial250mm pen union.

Specialized to one cylinder plus abutting axis-aligned markerboxes whose
cross-sections enclose the cylinder. Hidden primitive endcaps are excluded.
Not a contact/force oracle, and never calls mj_geomDistance or physics.
"""
import hashlib
from pathlib import Path
import numpy as np
import mujoco

def quat_matrix(q):
    q=np.asarray(q,dtype=float);norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or (norm<1e-12).any():raise ValueError('Invalid quaternion')
    w,x,y,z=np.moveaxis(q/norm,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))

class PenExterior:
    def __init__(self,model_path,context_path):
        m=mujoco.MjModel.from_binary_path(str(model_path));nominal={k:getattr(m,k).copy() for k in ['geom_size','geom_pos']}
        with np.load(context_path,allow_pickle=False) as z:
            for k in ['geom_size','geom_pos']:
                a=z[k]
                if a.shape[1:]!=getattr(m,k).shape or not np.isfinite(a).all() or not np.array_equal(a,np.broadcast_to(a[:1],a.shape)):raise ValueError('Fixed actual geometry context required')
                getattr(m,k)[:]=a[0]
        ids=[m.geom(n).id for n in ['object/cube','object/A_bottom','object/A_top']]
        if [int(m.geom_type[i]) for i in ids]!=[5,6,6]:raise ValueError('Expected cylinder+two markerboxes')
        R=quat_matrix(m.geom_quat[ids]);p=m.geom_pos[ids].copy();size=m.geom_size[ids].copy()
        if np.max(np.abs(np.abs(R[0,:,2])-[1,0,0]))>1e-12 or np.max(np.abs(p[0]))>1e-10:raise ValueError('Cylinder must be atbodyorigin alongbodyx')
        self.radius,self.half_length=map(float,size[0,:2]);self.boxes=[];self.patches=[{'component':'object/cube','feature':'cylinder_lateral_external','kind':'cylinder_side'}]
        signs=[]
        for j in [1,2]:
            if np.max(np.abs(np.abs(R[j]).max(axis=0)-1))>1e-12 or np.max(np.abs(np.abs(R[j]).max(axis=1)-1))>1e-12:raise ValueError('Marker rotation must be signed permutation ofbodyaxes')
            half=np.abs(R[j])@size[j];center=p[j];sign=1 if center[0]>0 else -1;signs.append(sign)
            if np.max(np.abs(center[1:]))>1e-10 or abs(abs(center[0])-half[0]-self.half_length)>1e-8 or min(half[1:])<self.radius:raise ValueError('Reject noncoaxial/gapped/overlapping or too narrow marker layout')
            self.boxes.append((center,half,sign))
            for axis in range(3):
                for face_sign in [-1,1]:
                    inner=axis==0 and face_sign==-sign
                    self.patches.append({'component':m.geom(ids[j]).name,'feature':f'box_axis{axis}_face{face_sign}'+('_circle_cutout' if inner else ''),'kind':'marker_face','box':j-1,'axis':axis,'sign':face_sign,'inner_circle_cutout':inner})
        if sorted(signs)!=[-1,1]:raise ValueError('One marker at each cylinder end required')
        self.metadata={'scope':'True nearest visible exterior patch of this validated pen union, signed by closed union membership. Not min primitive SDF. No margins/force/contactlabels.','model_sha256':hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),'context_sha256':hashlib.sha256(Path(context_path).read_bytes()).hexdigest(),'radius_m':self.radius,'cylinder_half_length_bodyx_m':self.half_length,'box_specs':[{'center_body_m':c.tolist(),'halfsize_body_m':h.tolist(),'end_sign':s} for c,h,s in self.boxes],'object_geoms':[{'id':int(i),'name':m.geom(i).name,'type':int(m.geom_type[i]),'size':m.geom_size[i].tolist(),'position_body':m.geom_pos[i].tolist(),'quaternion_wxyz':m.geom_quat[i].tolist()} for i in ids],'validated_signed_permutation_rotations':True,'seams_abut_and_box_crosssection_encloses_cylinder':True,'nominal_vs_actual_geometry_max_abs':{k:float(np.max(np.abs(getattr(m,k)-v))) for k,v in nominal.items()},'patches':self.patches,'tie_tolerance_m':1e-10,'geometry_scale_calibrated_for_observations':False}

    def query(self,points):
        p=np.asarray(points,dtype=float);shape=p.shape[:-1]
        if p.shape[-1]!=3 or not np.isfinite(p).all():raise ValueError('Finite points(...,3) required')
        p=p.reshape(-1,3);rad=np.linalg.norm(p[:,1:],axis=1);unit=np.zeros((len(p),2));unit[:,0]=1;np.divide(p[:,1:],rad[:,None],out=unit,where=rad[:,None]>1e-14)
        side=np.c_[np.clip(p[:,0],-self.half_length,self.half_length),self.radius*unit];candidates=[side];normals=[np.c_[np.zeros(len(p)),unit]]
        for patch in self.patches[1:]:
            center,half,_=self.boxes[patch['box']];q=np.clip(p,center-half,center+half);axis,sign=patch['axis'],patch['sign'];q[:,axis]=center[axis]+sign*half[axis]
            if patch['inner_circle_cutout']:
                inside=np.linalg.norm(q[:,1:],axis=1)<self.radius;q[inside,1:]=self.radius*unit[inside]
            candidates.append(q);n=np.zeros_like(p);n[:,axis]=sign;normals.append(n)
        surfaces=np.stack(candidates,axis=1);patchnorm=np.stack(normals,axis=1);d2=np.sum((p[:,None]-surfaces)**2,axis=2);choice=d2.argmin(1);unsigned=np.sqrt(d2[np.arange(len(p)),choice]);closest=surfaces[np.arange(len(p)),choice]
        eps=8*np.finfo(float).eps*(self.half_length+self.radius);inside=(np.abs(p[:,0])<=self.half_length+eps)&(rad<=self.radius+eps)
        for center,half,_ in self.boxes:inside|=(np.abs(p-center)<=half+eps).all(1)
        signed=np.where(inside,-unsigned,unsigned);normal=patchnorm[np.arange(len(p)),choice].copy();nonzero=unsigned>1e-13;normal[nonzero]=(p[nonzero]-closest[nonzero])/signed[nonzero,None]
        ties=np.abs(np.sqrt(d2)-unsigned[:,None])<=1e-10
        result={'signed_distance_m':signed.reshape(shape),'unsigned_distance_m':unsigned.reshape(shape),'closest_o_m':closest.reshape(shape+(3,)),'outward_normal_o':normal.reshape(shape+(3,)),'feature_index':choice.reshape(shape),'tie_mask':ties.reshape(shape+(len(self.patches),)),'tie_count':ties.sum(1).reshape(shape),'axis_radial_normal_ambiguity':((rad<=1e-14)&ties[:,0]).reshape(shape)}
        return result
