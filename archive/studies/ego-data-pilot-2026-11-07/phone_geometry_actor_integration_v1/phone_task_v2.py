"""Fixed-wrist phone task and engineered CHORD prior, not an Actor trainer.

Import is NumPy/stdlib only. Torch/geometry calls occur only in explicit API
calls. No native simulator, pretrained Actor or optimizer is constructed here.
"""
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np

TASK_SCHEMA='phone_fixed_wrist_ego61_tripod_task:1'
OBS_DIM=222
MEMORY_DIM=83

def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):
    p=Path(v['path']);assert pin(p)==v,'Frozen source/operand changed';return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load_source(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def tensor(value,like):
    import torch
    return torch.as_tensor(value,dtype=like.dtype,device=like.device)

def validate_memory_clock(m):
    import torch
    step,h,n=m['step'],m['native_h_s'],m['n_substeps']
    if step.dtype not in (torch.int32,torch.int64) or n.dtype not in (torch.int32,torch.int64):raise ValueError('Integer step and native substeps, no truncation')
    if h.shape!=step.shape or n.shape!=step.shape or not h.is_floating_point():raise ValueError('Matching explicit memory clock shapes')
    if not torch.isfinite(h).all() or torch.any(h<=0) or torch.any((n!=64)&(n!=128)) or torch.any((step<0)|(step>100)):raise ValueError('Finite task clock,64/128 ticks,step0..100')
    if torch.any(abs(h*n-.02)>4*torch.finfo(h.dtype).eps*.02):raise ValueError('Every transition is exactly declared H20ms')
    for key in ['a1','a2','c1','last_u']:
        if m[key].shape!=step.shape+(20,) or not torch.isfinite(m[key]).all():raise ValueError('Finite memory20 and shared leading shape')
    return True

class EngineeredMaterialSupportProxy:
    """Hypothetical same-material/same-normal prior, NEVER a contact estimator.

    Template points/normals originate from one pinned static native probe.
    Hand-side points are transported by current differentiable compiled FK.
    BODY force normals stay at their template values by declared assumption.
    No directional-descriptor maximizing plane is used as a contact normal.
    """
    def __init__(self,bundle,geometry,initial58):
        import torch
        self.geometry=geometry
        self.b=bundle
        self.weighted=load_source('frozen_phone_CHORD_weighted',checked(bundle['sources']['weighted_support'])).weighted_wrench_support
        self.chord=load_source('frozen_phone_CHORD_contacts',checked(bundle['sources']['chord_contacts']))
        template=bundle['contact_template'];self.gids=[v['hand_geom_id'] for v in template]
        self.phone_points=tensor([v['point_BODY_COM_m'] for v in template],initial58)
        self.normals=tensor([v['force_on_object_unit_normal_BODY'] for v in template],initial58)
        with np.load(checked(bundle['basis']),allow_pickle=False) as z:self.basis=tensor(z['basis512_unit6_F64'],initial58)
        # One pure Tensor FK at construction; no native call or model load.
        gp=geometry.provider.poses(initial58)
        points=tensor([v['source_world_contact_position_m'] for v in template],initial58)
        R=gp['geom_rotation_world'][self.gids];p=gp['geom_position_world_m'][self.gids]
        self.local=(R.transpose(-2,-1)@(points-p)[...,None])[...,0].detach()
        self.phone_points=self.phone_points.detach();self.normals=self.normals.detach();self.basis=self.basis.detach()
        self.reference=self.weighted(self.phone_points,self.normals,torch.ones(len(template),dtype=initial58.dtype,device=initial58.device),self.basis,
            bundle['support']['radius_m'],mu=bundle['support']['mu_design'],edges=8).detach()
        self.counts=dict(initial_pure_Tensor_FK=1,current_pure_Tensor_FK=0,native_calls=0,NN_calls=0)

    def __call__(self,current58):
        import torch
        poses=self.geometry.provider.poses(current58);self.counts['current_pure_Tensor_FK']+=1
        Rg=poses['geom_rotation_world'][...,self.gids,:,:];pg=poses['geom_position_world_m'][...,self.gids,:]
        local=self.local.to(current58);world=(Rg@local[...,None])[...,0]+pg
        Rphone=current58[...,43:52].reshape(current58.shape[:-1]+(3,3))
        body=(Rphone.transpose(-2,-1)[...,None,:,:]@(world-current58[...,None,40:43])[...,None])[...,0]
        target=self.phone_points.to(current58);n=self.normals.to(current58);basis=self.basis.to(current58)
        distance=torch.linalg.vector_norm(body-target,dim=-1)
        weights=torch.exp(-(distance/self.b['support']['proximity_bandwidth_m']).square())
        support=self.weighted(body,n.expand_as(body),weights,basis,self.b['support']['radius_m'],mu=self.b['support']['mu_design'],edges=8)
        reference=self.reference.to(current58).expand_as(support)
        score=self.chord.support_reward(support[...,None,None,:],reference[...,None,None,:],tolerance=.1,var=.1)
        known=torch.isfinite(score)&torch.isfinite(support).all(-1)
        return dict(schema='phone_engineered_material_support_proxy:1',support_BODY_COM512=support,reference_support_BODY_COM512=reference,
            score=score,proxy_known=known,actual_contact_known=torch.zeros_like(known),actual_force_known=torch.zeros_like(known),
            material_distance_m=distance,soft_proximity_weights=weights,points_BODY_COM_m=body,normals_template_BODY=n.expand_as(body),
            scope='Hypothesis of fixed material correspondence and template BODY normals; not closest geometry, observed contact, force, native friction certificate or physical support truth')

class PhoneTask:
    """Task-memory/observation/control/reward functions for injected Actor code.

    Mode A1 is object tracking. Mode A2 adds the engineered material-support
    prior. Actual-contact knowledge stays independent in both. Old PenMDP and
    DexSuperMimicRunner are not silently made compatible by sharing dim58.
    """
    def __init__(self,bundle_path,*,mode='A1_tracking',geometry=None,torch_dtype=None):
        self.bundle_path=Path(bundle_path);self.b=read(bundle_path)
        assert self.b['schema']==TASK_SCHEMA and mode in ['A1_tracking','A2_engineered_material_proxy']
        for v in self.b['sources'].values():checked(v)
        with np.load(checked(self.b['reference_arrays']),allow_pickle=False) as z:
            self.reference_pose=z['control_T_world_from_phone_F64'].copy();self.reference_COM=z['control_COM_world_m'].copy();self.reference_time=z['control_time_s'].copy()
        assert self.reference_pose.shape==(101,4,4) and self.reference_COM.shape==(101,3)
        assert np.allclose(self.reference_time,np.arange(101)*.02,rtol=0,atol=1e-12)
        self.mode=mode;self.geometry=geometry;self.proxy=None
        if mode=='A2_engineered_material_proxy':
            if geometry is None:raise ValueError('A2 requires explicitly injected pinned current-only geometry provider')
            import torch
            dtype=torch.float64 if torch_dtype is None else torch_dtype
            initial=torch.tensor(self.b['initial']['native_PRE_state58_F64'],dtype=dtype)
            self.proxy=EngineeredMaterialSupportProxy(self.b,geometry,initial)

    def initial_memory(self,like,*,native_h_s,n_substeps):
        import torch
        if not isinstance(n_substeps,(int,np.integer)) or isinstance(n_substeps,bool) or n_substeps not in [64,128] or not math.isfinite(float(native_h_s)) or abs(float(native_h_s)*n_substeps-.02)>4*np.finfo(np.float64).eps*.02:raise ValueError('Explicit integer64/128 and h*n=.02, no coercion')
        lead=like.shape[:-1];u0=tensor(self.b['initial']['ctrl20_F64'],like).expand(lead+(20,))
        zero=torch.zeros(lead+(20,),dtype=like.dtype,device=like.device)
        result=dict(step=torch.zeros(lead,dtype=torch.int64,device=like.device),a1=zero,a2=zero.clone(),c1=zero.clone(),last_u=u0,
            native_h_s=torch.full(lead,float(native_h_s),dtype=like.dtype,device=like.device),n_substeps=torch.full(lead,int(n_substeps),dtype=torch.int64,device=like.device))
        validate_memory_clock(result);return result

    def reference(self,memory,offset=1):
        import torch
        validate_memory_clock(memory)
        step=memory['step']+int(offset)
        # Command-only padding at the known finite reference end, never a physics target.
        idx=step.clamp(0,100)
        x=memory['last_u'];R=tensor(self.reference_pose[:,:3,:3],x)[idx];COM=tensor(self.reference_COM,x)[idx]
        return dict(COM_world_m=COM,R_BODY_world=R,frame=idx,time_s=tensor(self.reference_time,x)[idx],command_padded=step>100)

    def control_target(self,action,memory):
        import torch
        validate_memory_clock(memory)
        if action.shape!=memory['a1'].shape or not torch.isfinite(action).all() or torch.any(abs(action)>1):raise ValueError('Finite tanh normalized20, no action clipping')
        a=action.to(memory['last_u'])  # Preserve F64 absolute-control baseline; AD crosses the cast.
        scale=tensor(self.b['action']['symmetric_residual_scale20_rad'],a)
        residual=.5*scale*a+.5*memory['c1']
        u=tensor(self.b['initial']['ctrl20_F64'],a)+residual
        lo=tensor(self.b['action']['target_lower20'],a);hi=tensor(self.b['action']['target_upper20'],a)
        if torch.any(u<lo) or torch.any(u>hi):raise ValueError('Declared bounded mapping failed; no clamp')
        return residual,u

    def observation(self,x,memory,geometry):
        import torch
        f=geometry.features(x);gap=f['directional_gap_m'];known=f['directional_gap_known']
        refs=[self.reference(memory,k) for k in [1,5,10]]
        command=torch.cat([torch.cat([v['COM_world_m'],v['R_BODY_world'].flatten(-2)],-1) for v in refs],-1)
        phase=torch.stack([memory['step'].to(x.dtype)/100,(100-memory['step']).to(x.dtype)/100],-1)
        numerical_gap=torch.where(known,gap,torch.zeros_like(gap))
        obs=torch.cat([x,memory['a1'],memory['a2'],memory['c1'],memory['last_u'],command,phase,numerical_gap,known.to(x.dtype),memory['native_h_s'][...,None],memory['n_substeps'].to(x.dtype)[...,None]],-1)
        assert obs.shape[-1]==OBS_DIM
        return obs.to(torch.float32)

    def flatten_memory(self,m):
        import torch
        return torch.cat([m['step'].to(m['last_u'].dtype)[...,None],m['a1'],m['a2'],m['c1'],m['last_u'],m['native_h_s'][...,None],m['n_substeps'].to(m['last_u'].dtype)[...,None]],-1)

    def unflatten_memory(self,value):
        import torch
        if value.shape[-1]!=MEMORY_DIM:raise ValueError('New phone memory83 only')
        if not torch.isfinite(value).all() or torch.any(value[...,0]!=torch.round(value[...,0])) or torch.any(value[...,82]!=torch.round(value[...,82])):raise ValueError('Encoded step/n must be exact integers before cast')
        result=dict(step=value[...,0].to(torch.int64),a1=value[...,1:21],a2=value[...,21:41],c1=value[...,41:61],last_u=value[...,61:81],native_h_s=value[...,81],n_substeps=value[...,82].to(torch.int64))
        validate_memory_clock(result);return result

    def advance_memory(self,memory,action,residual,executed_u):
        return dict(memory,step=memory['step']+1,a2=memory['a1'],a1=action,c1=residual,last_u=executed_u)

    def rewards(self,x1,memory,action):
        import torch
        ref=self.reference(memory,1);R=x1[...,43:52].reshape(x1.shape[:-1]+(3,3))
        epos=(x1[...,40:43]-ref['COM_world_m']).square().sum(-1)
        # Smooth full-orientation chordal penalty; not the pen's two-axis score.
        erot=(R-ref['R_BODY_world']).square().sum((-2,-1))
        rate=((action-memory['a1']).square()+(action-2*memory['a1']+memory['a2']).square()).sum(-1)
        qr=tensor(self.b['joint_range20x2'],x1);q=x1[...,:20]
        excess=torch.relu(qr[:,0]-q)+torch.relu(q-qr[:,1])
        terms=dict(object_COM=2*torch.exp(-epos/.01**2),object_SO3_chordal=8*torch.exp(-erot/(8*math.sin(math.radians(10)/2)**2)),
            action_rate=-.1*rate,joint_limit=-2*(excess/.001).square().sum(-1))
        base=.02*sum(terms.values());proxy=None
        if self.mode=='A2_engineered_material_proxy':
            proxy=self.proxy(x1);known=proxy['proxy_known'];raw=base+.02*proxy['score']
            total=torch.where(known,raw,torch.full_like(raw,float('nan')))
        else:known=torch.isfinite(base);total=base
        return dict(total=total,tracking_only=base,terms_unscaled=terms,objective_known=known,engineered_support_proxy=proxy,
            actual_contact_reward_known=torch.zeros_like(known),target_frame=ref['frame'],reward_phase='POST state with pre-advance memory, reference k+1; no future observed force')

    def termination(self,x1,memory):
        import torch
        initial=tensor(self.b['initial']['native_PRE_state58_F64'],x1);R=x1[...,43:52].reshape(x1.shape[:-1]+(3,3));R0=initial[43:52].reshape(3,3)
        drift=torch.linalg.vector_norm(x1[...,40:43]-initial[40:43],dim=-1)
        relative=R@R0.T;cos=(relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        skew=torch.stack([relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]],-1)/2
        angle=torch.atan2(torch.linalg.vector_norm(skew,dim=-1),cos)
        return dict(predicted_loss_of_hold=(drift>.05)|(angle>math.pi/4),reference_timeout=memory['step']+1>=100,
            soft_loss_probability=1-(1-torch.sigmoid((drift-.05)/.002))*(1-torch.sigmoid((angle-math.pi/4)/.02)),
            scope='State-only imagination stop descriptor; native warnings/penetration/contact/tail gates require actual evaluation')

def actor_critic_constructor_contract():
    return dict(Actor=dict(obs_dim=OBS_DIM),Critic=dict(dim=OBS_DIM,amp=False,empirical_norm=True),action_dim=20,
        ReplayBuffer=dict(holdout_fraction=0.),new_phone_runner_required=True,old_PenMDP_or_Runner_drop_in_compatible=False)

def wm_step_current_only(x,u,h,n,*,model,norms,physical,codec,geometry):
    """Injected frozen new WM only; no checkpoint load or future-label argument."""
    import torch
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Actor imagination requires eval/frozen WM; input AD remains enabled')
    raw=model.forward_from_raw(x,u,h,n,norms,geometry)
    v=raw[...,:178].to(torch.float64)*norms['head_std']+norms['head_mean']
    heads=dict(joint_delta40=v[...,:40],COM_ballistic_residual3_m=v[...,40:43],world_rotation_delta3_rad=v[...,43:46],pair_impulse_source_BODY_moving_COM22x6=v[...,46:].reshape(v.shape[:-1]+(22,6)))
    y=codec.decode_physical_heads(x.to(torch.float64),heads,(h*n).to(torch.float64),physical['mass'],physical['I'],physical['g'])
    return dict(next_state58=y,predicted_pair_meanFn22_N=torch.nn.functional.softplus(raw[...,178:200]),predicted_pair_coverage22=torch.sigmoid(raw[...,200:222]),physical_heads=heads,
        taskcontact_roles_are_external=True,actual_contact_known=False,scope='Current inputs only; predicted pair labels are not actual contact/force or tripod task qualification')
