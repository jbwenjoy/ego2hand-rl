"""Finite-direction current-pose convex descriptors; no native or learned model.

The ideal real-arithmetic value is a lower bound on signed translational
configuration-space separation. The floating implementation is NOT an exact
SDF, closest normal, interval certificate, contact label, or future-H claim.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import torch

SCHEMA='phone_directional_gap_features:1'
from .geometry import PhoneGeometryProvider
from ._artifacts import checked_data_pin
from .geometry_context import CurrentGeometry, ImmutableTensorCache

def checked(p, override=None):
    return checked_data_pin(p, override=override)


def vertex_plane_gaps(phone_vertices,other_vertices,directions):
    """All (...,D) plane gaps, with actual direction norm divided out."""
    pn=torch.amax(phone_vertices@directions.transpose(-1,-2),dim=-2)
    hn=torch.amin(other_vertices@directions.transpose(-1,-2),dim=-2)
    return (hn-pn)/torch.linalg.vector_norm(directions,dim=-1)

def capsule_plane_gaps(phone_vertices,segment_endpoints,radius,directions):
    norms=torch.linalg.vector_norm(directions,dim=-1)
    pn=torch.amax(phone_vertices@directions.transpose(-1,-2),dim=-2)
    hn=torch.amin(segment_endpoints@directions.transpose(-1,-2),dim=-2)-radius*norms
    return (hn-pn)/norms

class DirectionalGapProvider:
    """Zero trainable parameters; current state58 only, canonical22 pairs.

    Phone vertices and all other supports share BODY axes and a COM origin.
    Batch shapes are inherited from the frozen differentiable FK provider.
    Chunks bound intermediate support-projection memory, not direction count.
    """
    def __init__(self,catalog_path,direction_chunk=64,*,metadata_path=None,geometry_arrays_path=None,direction_arrays_path=None):
        self.catalog_path=Path(catalog_path).resolve(strict=True)
        self.catalog=json.loads(self.catalog_path.read_text(encoding='utf8'))
        if self.catalog['schema']!='phone_directional_gap_geometry_catalog:1':
            raise ValueError('Directional catalog schema')
        self.source_provenance={'provider_source':self.catalog.get('provider_source')}
        self.provider=PhoneGeometryProvider(checked(self.catalog['provider_metadata'],metadata_path),geometry_arrays_path=geometry_arrays_path)
        with np.load(checked(self.catalog['direction_arrays'],direction_arrays_path),allow_pickle=False) as z:
            self.directions=torch.from_numpy(z['directions_BODY_F64'].copy())
        if self.directions.ndim!=2 or self.directions.shape[1]!=3 or not torch.isfinite(self.directions).all():
            raise ValueError('Finite nonzero BODY plane directions')
        if torch.any(torch.linalg.vector_norm(self.directions,dim=-1)<.99):
            raise ValueError('Unit-scale directions required')
        if not isinstance(direction_chunk,int) or direction_chunk<1:
            raise ValueError('Positive projection chunk size')
        self.chunk=direction_chunk
        self.counts={'descriptor_calls':0,'pure_Torch_FK_calls':0,'native_calls':0,'NN_calls':0}
        constant_sources={'directions':self.directions}
        for gid,b in self.provider.geometry_by_id.items():
            if b['kind']=='box':constant_sources[f'box_size_{gid}']=torch.tensor(b['size_m'],dtype=torch.float64)
        constant_sources['box_signs']=torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],dtype=torch.float64)
        self._constant_cache=ImmutableTensorCache(constant_sources)
        self._support_cache={}

    def bundle(self,state58,*,retain_all_directions=False):
        """Share one exact current-state graph; caller owns its short lifetime."""
        return CurrentGeometry(self,state58,retain_all_directions=retain_all_directions)

    def _supports(self,x):
        """State-independent supports; same direction chunks and reduction order.

        No cached tensor depends on a state or an AD graph. Derived buffers are
        ordinary no-grad tensors even when first requested in inference mode.
        """
        pc=self.provider
        c=self._constant_cache.get(x);pconst=pc._constant_cache.get(x);arrays=pc._array_cache.get(x)
        key=(x.device.type,x.device.index,x.dtype)
        if key not in self._support_cache:
            with torch.inference_mode(False),torch.no_grad():
                D=c['directions'];pid=pc.meta['phone_geom_id'];pb=pc.geometry_by_id[pid]
                rawP=arrays[pb['vertices_key']]
                # Raw compiled geom vertices -> BODY once -> COM origin.
                P=rawP@pconst['geom_R'][pid].transpose(-1,-2)+pconst['geom_pos'][pid]-pconst['phone_body_ipos']
                pieces=[]
                for start in range(0,len(D),self.chunk):
                    dc=D[start:start+self.chunk]
                    pieces.append((dc,torch.amax(P@dc.transpose(-1,-2),dim=-2),torch.linalg.vector_norm(dc,dim=-1)))
                shapes={}
                for gid in pc.pair_geom_ids:
                    b=pc.geometry_by_id[gid]
                    if b['kind']=='mesh':
                        V=arrays[b['vertices_key']]
                        shapes[gid]=(V,torch.amax(torch.linalg.vector_norm(V,dim=-1)))
                    elif b['kind']=='box':
                        size=c[f'box_size_{gid}']
                        shapes[gid]=(c['box_signs']*size,torch.linalg.vector_norm(size))
                    elif b['kind']=='capsule':
                        radius,half=b['size_m'][:2]
                        shapes[gid]=(None,torch.as_tensor(radius+half,dtype=x.dtype,device=x.device))
                self._support_cache[key]=dict(D=D,P=P,pieces=pieces,shapes=shapes,phone_radius=torch.amax(torch.linalg.vector_norm(P,dim=-1)))
        return self._support_cache[key]

    def features(self,state58,return_all_directions=False,*,context=None):
        if context is not None:
            context.require(self,state58)
            return context.directional(return_all_directions=return_all_directions)
        poses=self.provider.poses(state58);self.counts['pure_Torch_FK_calls']+=1
        return self._features_from_poses(state58,poses,return_all_directions=return_all_directions)

    def _features_from_poses(self,state58,poses,return_all_directions=False):
        if state58.dtype not in (torch.float64,torch.float32):
            raise ValueError('F64 reference or explicitly approximate FP32 descriptor')
        self.counts['descriptor_calls']+=1
        x=state58;lead=x.shape[:-1];dtype=x.dtype;device=x.device
        constants=self._supports(x);D=constants['D'];pc=self.provider
        relp=poses['pair_relative_translation_BODY_COM_m']
        relR=poses['pair_relative_rotation_BODY9'].reshape(lead+(22,3,3))
        values=[];argidx=[];guards=[];known=[];all_planes=[]
        for j,gid in enumerate(pc.pair_geom_ids):
            b=pc.geometry_by_id[gid];R=relR[...,j,:,:];p=relp[...,j,:]
            if b['kind']=='mesh':
                V,radius_bound=constants['shapes'][gid]
                H=V@R.transpose(-1,-2)+p[...,None,:]
            elif b['kind']=='box':
                V,radius_bound=constants['shapes'][gid];H=V@R.transpose(-1,-2)+p[...,None,:]
            elif b['kind']=='capsule':
                radius,half=b['size_m'][:2]
                offset=half*R[..., :,2]
                H=torch.stack((p-offset,p+offset),dim=-2)
                radius_bound=constants['shapes'][gid][1]
            else:
                nan=torch.full(lead,float('nan'),dtype=dtype,device=device)
                values.append(nan);argidx.append(torch.full(lead,-1,dtype=torch.int64,device=device));guards.append(nan);known.append(torch.zeros(lead,dtype=torch.bool,device=device))
                if return_all_directions:all_planes.append(nan[...,None].expand(lead+(len(D),)))
                continue
            pieces=[]
            for dc,phone_support,dnorm in constants['pieces']:
                hand_support=torch.amin(H@dc.transpose(-1,-2),dim=-2)
                if b['kind']=='capsule':hand_support=hand_support-radius*dnorm
                pieces.append((hand_support-phone_support)/dnorm)
            planes=torch.cat(pieces,dim=-1)
            val,idx=torch.max(planes,dim=-1)
            valid=torch.isfinite(planes).all(dim=-1)
            values.append(torch.where(valid,val,torch.full_like(val,float('nan'))))
            argidx.append(torch.where(valid,idx,torch.full_like(idx,-1)))
            known.append(valid)
            # Explicit heuristic scale allowance, not interval-certified FK error.
            scale=1.+torch.linalg.vector_norm(p,dim=-1)+radius_bound+constants['phone_radius']
            guards.append(256*torch.finfo(dtype).eps*scale)
            if return_all_directions:all_planes.append(planes)
        value=torch.stack(values,-1);km=torch.stack(known,-1);idx=torch.stack(argidx,-1);guard=torch.stack(guards,-1)
        chosen=D[torch.clamp(idx,min=0)]
        chosen=chosen/torch.linalg.vector_norm(chosen,dim=-1,keepdim=True)
        chosen=torch.where(km[...,None],chosen,torch.full_like(chosen,float('nan')))
        no=torch.zeros(lead+(22,),dtype=torch.bool,device=device)
        out=dict(schema=SCHEMA,source_state_schema='phone_COM_world_state58:1',model_sha256=pc.meta['model']['sha256'],
            pair_geom_ids=pc.pair_geom_ids,pair_names=pc.pair_names,
            directional_gap_m=value,directional_gap_known=km,
            numeric_heuristic_guard_m=guard,guarded_directional_descriptor_m=value-guard,
            maximizing_plane_direction_BODY3=chosen,direction_argmax_index=idx,
            pair_pose_known=torch.ones_like(no),pair_geometry_known=torch.ones_like(no),
            pair_exact_signed_gap_known=no,closest_contact_normal_known=no.clone(),normal_unique_known=no.clone(),
            formal_floating_certificate_known=no.clone(),full_H_zero_contact_certificate_known=no.clone(),
            direction_count=len(D),dtype_scope='F64 numerical reference' if dtype==torch.float64 else 'FP32 approximate descriptor; no strict certificate',
            mathematical_scope='Finite unit-direction support-plane maximum for ideal convex supports; signed translation lower-bound in real arithmetic, not exact SDF/scan/force/normal or future-contact truth',
            numerical_scope='Guard is a declared scale heuristic, not directed rounding or a proven total FK error envelope')
        if return_all_directions:out['all_directional_plane_gaps_m']=torch.stack(all_planes,-2)
        return out

def validate_features(out):
    if out['schema']!=SCHEMA or len(out['pair_geom_ids'])!=22 or len(set(out['pair_geom_ids']))!=22:
        raise ValueError('All canonical22 pair identities required')
    s=out['directional_gap_m'].shape
    for name in ['directional_gap_known','pair_pose_known','pair_geometry_known','pair_exact_signed_gap_known','closest_contact_normal_known','normal_unique_known','formal_floating_certificate_known','full_H_zero_contact_certificate_known']:
        if out[name].shape!=s or out[name].dtype!=torch.bool:raise ValueError('Separate explicit knowledge masks')
    for name in ['pair_exact_signed_gap_known','closest_contact_normal_known','normal_unique_known','formal_floating_certificate_known','full_H_zero_contact_certificate_known']:
        if out[name].any():raise ValueError('Directional descriptors do not certify exact gaps/normals/fullH')
    k=out['directional_gap_known']
    if not torch.isfinite(out['directional_gap_m'][k]).all() or not torch.isnan(out['directional_gap_m'][~k]).all():
        raise ValueError('Known values finite; unknown values NaN, never fabricated zero')
    return {'schema':SCHEMA,'pairs':22,'trainable_parameters':0,'exact_gap_or_future_certificates':0}
