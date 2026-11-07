"""UNFROZEN: isolated WM boundary-impulse wrapper/loss; no trainer or I/O.

The caller first validates the old152 dataset and strict initial native WM.
This module never constructs a native WorldModel or computes native LogR loss.
Only same-call hidden features are observed; labels enter losses, not forward.
"""
from __future__ import annotations
import copy,hashlib,math,random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
ARMS=('shared_public_losses','shared_public_plus_cross_target_consistency')
ARTIFACT_SCHEMA='full151105_boundary_impulse_WM_component:1'
RECIPE=ROOT/'docs/full151_boundary_impulse_matched_loss_recipe_draft_v1.md'
RECIPE_SHA='5926e869f69e1ab6a5ea7aac7d5dee4f0b9adc4b2d2686dfc6b6a6a488f785b2'
OWN={'component':ROOT/'src/full151_gradient/boundary_impulse_component105_v1.py','checker':ROOT/'scripts/check_full151_boundary_impulse_component_cpu_v1.py'}


def require(value,message):
    if not value:raise ValueError(message)


def bit_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def tensor_check(value,shape,dtype,name):
    require(isinstance(value,torch.Tensor) and value.shape==shape and value.dtype==dtype and value.device.type=='cpu' and bool(torch.isfinite(value).all()),'Finite CPU '+name)
    return value


def _frozen_tensor(value,shape,dtype,name):
    tensor_check(value,shape,dtype,name);require(not value.requires_grad,'Physical/target value must not learn: '+name)
    return value.clone().contiguous()


def tensor_digest(value):
    """Evidence only on parameters/buffers, never used to build prediction loss."""
    with torch.no_grad():
        a=value.cpu().contiguous();payload=bytes(a.reshape(-1).view(torch.uint8).tolist())
    return hashlib.sha256(str((str(a.dtype),tuple(a.shape))).encode()+payload).hexdigest()


def state_digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):h.update(name.encode());h.update(tensor_digest(value).encode())
    return h.hexdigest()


class FixedPhysicalContext(nn.Module):
    def __init__(self,*,mass,Ib,g,dt,P,s52,xi,row_count,row_guards):
        super().__init__();require(row_count>0 and row_guards and all(row_guards.values()),'Caller full-row physical context enrollment')
        for name,value in (('mass_actual_FP32',mass),('I_body_F64',Ib),('gravity_palm_F64',g),('DeltaT_F64',dt),('palm_rot_FP32',P),('effective_s52_FP32',s52),('physical_xi46_FP32',xi)):
            self.register_buffer(name,value.clone().contiguous())
        self.register_buffer('I_body_antisymmetry_maxabs_F64',(Ib-Ib.T).abs().max())
        self.row_count=row_count;self.row_guards=dict(row_guards)
        self._reference={name:value.clone() for name,value in self.named_buffers()}

    def assert_frozen(self):
        require(all(bit_equal(value,self._reference[name]) and not value.requires_grad for name,value in self.named_buffers()),'Physical context buffer changed')


def fixed_context_from_rows(mass_actual_FP32,I_body_F64,gravity_palm_rows_F64,DeltaT_rows_F64,palm_rot_rows_FP32,effective_s52_FP32,physical_xi46_FP32):
    """Strictly enroll ALL known context rows before extracting fixed values."""
    n=len(DeltaT_rows_F64);require(n>0,'Nonempty explicit context rows')
    mass=_frozen_tensor(mass_actual_FP32,(),torch.float32,'actual mass');Ib=_frozen_tensor(I_body_F64,(3,3),torch.float64,'actual inertia observer')
    g=_frozen_tensor(gravity_palm_rows_F64,(n,3),torch.float64,'all gravity rows');dt=_frozen_tensor(DeltaT_rows_F64,(n,),torch.float64,'all duration rows')
    P=_frozen_tensor(palm_rot_rows_FP32,(n,3,3),torch.float32,'all raw palm rotations');s=_frozen_tensor(effective_s52_FP32,(52,),torch.float32,'effective frozen scale52')
    xi=_frozen_tensor(physical_xi46_FP32,(46,),torch.float32,'physical xi46')
    require(bool((mass>0).all()) and bool((dt>0).all()) and bool((s>0).all()),'Positive actual mass/duration/effective scale')
    # Keep the source observer bytes; tiny F64 antisymmetry is a descriptor,
    # never a new equality threshold, projection or symmetrization.
    torch.linalg.cholesky(Ib)
    guards={'all_gravity_rows_same_bytes':bit_equal(g,g[:1].expand_as(g).contiguous()),'all_duration_rows_same_bytes':bit_equal(dt,dt[:1].expand_as(dt).contiguous()),'all_raw_palm_rows_same_bytes':bit_equal(P,P[:1].expand_as(P).contiguous())}
    require(all(guards.values()),'No bid0 context assumption, averaging, or mixed physical contexts')
    return FixedPhysicalContext(mass=mass,Ib=Ib,g=g[0],dt=dt[0],P=P[0],s52=s,xi=xi,row_count=n,row_guards=guards)


class ZeroAuxHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Direct zeros avoid any random parameter initialization altogether.
        self.weight=nn.Parameter(torch.zeros((14,512),dtype=torch.float32))
        self.bias=nn.Parameter(torch.zeros(14,dtype=torch.float32))

    def forward(self,h):return F.linear(h,self.weight,self.bias)


def _normalizer_snapshot(wm):
    buffers={}
    for owner in ('norm','delta_norm'):
        module=getattr(wm,owner)
        for name,value in module.named_buffers():buffers[owner+'.'+name]=value.clone()
    buffers['error_floor']=wm.error_floor.clone()
    host={owner:{name:getattr(getattr(wm,owner),name) for name in ('eps','max_count','_host_count') if hasattr(getattr(wm,owner),name)} for owner in ('norm','delta_norm')}
    return buffers,host


def _validate_raw_wm(wm,ctx):
    require(isinstance(wm,nn.Module) and not isinstance(wm,BoundaryImpulseWM),'Pass RAW WM, never an already wrapped module')
    require(not wm.training and getattr(wm,'amp',None) is False,'Explicit native eval/FP32 CPU shadow')
    require(isinstance(wm.net[-1],nn.Linear) and wm.net[-1].in_features==512 and wm.net[-1].out_features==67,'Native final Linear512->67')
    require(all(not module._forward_hooks and not module._forward_pre_hooks for module in wm.modules()),'Raw WM must be hook-free before deepcopy')
    require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in wm.parameters()),'Native storage FP32 CPU only')
    require(all(not b.requires_grad for b in wm.buffers()),'Native normalizer buffers must not learn')
    effective=torch.maximum(wm.delta_norm.std(),wm.error_floor)
    require(bit_equal(effective,ctx.effective_s52_FP32),'Same initial effective max(std,error_floor), not raw std/new labels')


class BoundaryImpulseWM(nn.Module):
    def __init__(self,wm,context,head,arm):
        super().__init__();require(arm in ARMS,'Explicit matched arm');_validate_raw_wm(wm,context)
        self.wm=wm;self.context=context;self.aux_head=head;self.arm=arm
        self._cache=None;self._active=False;self._closed=False;self._this_hook_calls=0
        self.counts={'forward_attempted':0,'native_forward_returned':0,'hook_calls':0,'head_forward_returned':0,'forward_returned':0}
        self._norm_reference,self._norm_host_reference=_normalizer_snapshot(wm)
        self._hook_handle=self.wm.net[-1].register_forward_pre_hook(self._capture)
        for parameter in self.wm.parameters():parameter.requires_grad_(True)
        self.wm.eval()

    def __deepcopy__(self,memo):raise TypeError('Never deepcopy a wrapped WM/hook closure; copy RAW WM first')
    def __getstate__(self):raise TypeError('Serialize own state_dict payload, never module/hooks/cached autograd graph')

    def assert_frozen(self):
        self.context.assert_frozen();current,host=_normalizer_snapshot(self.wm)
        require(host==self._norm_host_reference and set(current)==set(self._norm_reference) and all(bit_equal(v,self._norm_reference[k]) for k,v in current.items()),'Native norm/hostconfig/error_floor changed')

    def _capture(self,module,args):
        require(self._active and self._cache is None and len(args)==1,'One same-call native feature capture, no reentrance/shared cache')
        h=args[0];require(h.ndim==2 and h.shape[1]==512 and h.dtype==torch.float32 and h.device.type=='cpu','Same-call FP32 hidden[N,512]')
        self._cache=h;self._this_hook_calls+=1;self.counts['hook_calls']+=1
        return None

    def forward(self,x,u,dynamics):
        require(not self._closed and not self._active and self._cache is None,'Open independent empty capture cache')
        n=len(x);tensor_check(x,(n,58),torch.float32,'state58');tensor_check(u,(n,20),torch.float32,'actual u20');tensor_check(dynamics,(n,278),torch.float32,'dynamics278')
        require(bit_equal(dynamics[:,:58],x),'Actual input state also fills dynamics[:58]')
        require(bit_equal(dynamics[:,58:104],self.context.physical_xi46_FP32[None].expand(n,-1).contiguous()) and bool((dynamics[:,104:]==0).all()),'Fixed xi46/zero174 only; no label input oracle')
        self.assert_frozen();self._active=True;self._this_hook_calls=0;self.counts['forward_attempted']+=1
        try:
            prediction,self15=self.wm(x,u,dynamics);self.counts['native_forward_returned']+=1
            require(self._this_hook_calls==1 and self._cache is not None,'Exactly one native final-layer feature capture')
            tensor_check(prediction,(n,58),torch.float32,'native next58');tensor_check(self15,(n,15),torch.float32,'native self15')
            auxiliary=self.aux_head(self._cache);self.counts['head_forward_returned']+=1
            tensor_check(auxiliary,(n,14),torch.float32,'ungated auxiliary14');self.assert_frozen();self.counts['forward_returned']+=1
            return prediction,self15,auxiliary[:,:12].reshape(n,2,6),auxiliary[:,12:]
        finally:self._cache=None;self._active=False

    def close(self):
        if not self._closed:self._hook_handle.remove();self._closed=True
        self._cache=None;self._active=False

    def train(self,mode=True):
        super().train(mode);self.wm.eval();return self


def _storage_keys(module):return {(str(t.device),t.data_ptr()) for t in list(module.parameters())+list(module.buffers()) if t.numel()}


def make_matched_components(raw_wm,fixed_context):
    require(isinstance(fixed_context,FixedPhysicalContext),'Enrolled full-row physical context required');fixed_context.assert_frozen();_validate_raw_wm(raw_wm,fixed_context)
    rng_before=torch.get_rng_state().clone();python_before=random.getstate();parent_digest=state_digest(raw_wm)
    # All RAW copies exist before the first wrapper installs any bound hook.
    raw_copies={name:copy.deepcopy(raw_wm) for name in ARMS}
    with torch.random.fork_rng(devices=[]):template=ZeroAuxHead()
    components={name:BoundaryImpulseWM(raw_copies[name],copy.deepcopy(fixed_context),copy.deepcopy(template),name) for name in ARMS}
    parent_unchanged=state_digest(raw_wm)==parent_digest
    raw_identity={name:state_digest(module.wm)==parent_digest for name,module in components.items()}
    heads={name:state_digest(module.aux_head) for name,module in components.items()}
    a,b=(components[name] for name in ARMS);disjoint=not(_storage_keys(a)&_storage_keys(b)) and not(_storage_keys(a)&_storage_keys(raw_wm)) and not(_storage_keys(b)&_storage_keys(raw_wm))
    rng_unchanged=torch.equal(rng_before,torch.get_rng_state()) and python_before==random.getstate()
    require(parent_unchanged and all(raw_identity.values()) and len(set(heads.values()))==1 and disjoint and rng_unchanged,'Raw/zero-head/storage/RNG construction identity')
    return components,{'schema':'full151105_boundary_component_construction:1','raw_WM_parent_digest':parent_digest,'raw_WM_unchanged':parent_unchanged,
        'raw_clone_state_identity':raw_identity,'zero_head_digests':heads,'disjoint_model_head_buffer_storage':disjoint,'constructor_Python_TorchCPU_RNG_unchanged':rng_unchanged,
        'head_initialization':'single direct-zero Parameter template in explicit CPU constructor fork_rng, independently copied',
        'feature_hooks_independent':a._capture.__self__ is a and b._capture.__self__ is b,'samebatch_native_function_parity':'PENDING_CALLER_VALIDATION_FORWARD',
        'native_WM_constructors':0,'label_oracle_in_forward':False}


def component_losses(component,x,prediction,z,mode_logits,native_point_loss,labels,*,lambda_mode,lambda_imp,lambda_cons):
    """Caller supplies the unchanged native FP32 LogR/chart point loss.

All prediction-dependent physics below stays in Torch autograd via double();
no NumPy, detach, projection, pseudoinverse, mode gating or target-R oracle.
"""
    require(isinstance(component,BoundaryImpulseWM),'Own component only');component.assert_frozen();n=len(x)
    tensor_check(x,(n,58),torch.float32,'known pre58');tensor_check(prediction,(n,58),torch.float32,'predicted58')
    tensor_check(z,(n,2,6),torch.float32,'predicted dimensionless z12');tensor_check(mode_logits,(n,2),torch.float32,'predicted mode logits')
    tensor_check(native_point_loss,(),torch.float32,'caller native point loss')
    require(set(labels)=={'J_palm_F64','K_palm_F64','boundary_constraint_witness','r_v_saved_m_s_F64','r_w_saved_rad_s_F64'},'Explicit supervised-only label fields')
    J=labels['J_palm_F64'];K=labels['K_palm_F64'];mode=labels['boundary_constraint_witness'];rv_saved=labels['r_v_saved_m_s_F64'];rw_saved=labels['r_w_saved_rad_s_F64']
    for value,shape,dtype,name in ((J,(n,2,3),torch.float64,'saved J'),(K,(n,2,3),torch.float64,'saved K'),(mode,(n,2),torch.bool,'constraint witness'),(rv_saved,(n,3),torch.float64,'saved rv'),(rw_saved,(n,3),torch.float64,'saved rw')):
        tensor_check(value,shape,dtype,name);require(not value.requires_grad,'Targets do not learn')
    for value in (lambda_mode,lambda_imp,lambda_cons):require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Caller explicit finite nonnegative coefficients')
    require(component.arm!=ARMS[0] or lambda_cons==0,'Control has exactly zero consistency coefficient')
    ctx=component.context;m=ctx.mass_actual_FP32.double();Ib=ctx.I_body_F64;sv=ctx.effective_s52_FP32[46:49].double();sw=ctx.effective_s52_FP32[49:52].double()
    R0=x[:,43:52].double().reshape(n,3,3);R1=prediction[:,43:52].double().reshape(n,3,3)
    I0=R0@Ib@R0.transpose(-1,-2);I1=R1@Ib@R1.transpose(-1,-2)
    zj=z[:,:,:3].double();zk=z[:,:,3:].double();Jhat=m*(sv*zj);Khat=(I0[:,None]@(sw*zk)[...,None])[...,0]
    zJ_target=(J/m)/sv;zK_target=torch.linalg.solve(I0[:,None],K[...,None])[...,0]/sw
    ztarget=torch.cat((zJ_target,zK_target),dim=-1)
    L_mode=F.binary_cross_entropy_with_logits(mode_logits,mode.float());L_imp=(z.double()-ztarget).square().mean()
    rv=prediction[:,52:55].double()-x[:,52:55].double()-(Jhat.sum(dim=1)+m*ctx.gravity_palm_F64*ctx.DeltaT_F64)/m
    rhs=(I0@x[:,55:58].double()[...,None])[...,0]+Khat.sum(dim=1)
    omega_from_impulse=torch.linalg.solve(I1,rhs[...,None])[...,0]
    rw=prediction[:,55:58].double()-omega_from_impulse
    normalized=torch.cat(((rv-rv_saved)/sv,(rw-rw_saved)/sw),dim=-1);L_cons=normalized.square().mean()
    public=native_point_loss+lambda_mode*L_mode+lambda_imp*L_imp;total=public+lambda_cons*L_cons
    require(all(bool(torch.isfinite(value).all()) for value in (Jhat,Khat,ztarget,rv,rw,L_mode,L_imp,L_cons,public,total)),'Finite all-row auxiliary/solve/loss; no repair/filter')
    component.assert_frozen()
    return {'L_state_FP32':native_point_loss,'L_mode_FP32':L_mode,'L_imp_F64':L_imp,'L_cons_F64':L_cons,'L_public':public,'L_total':total,
        'J_hat_palm_F64':Jhat,'K_hat_palm_F64':Khat,'z_target_F64':ztarget,'r_v_hat_F64':rv,'r_w_hat_F64':rw,'I_palm0_F64':I0,'I_palm1_hat_F64':I1,
        'angular_RHS_F64':rhs,'omega_from_predicted_impulses_F64':omega_from_impulse,'solve_residual_F64':(I1@omega_from_impulse[...,None])[...,0]-rhs}
