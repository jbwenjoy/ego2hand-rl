"""Geometry-gated nominal support reward; no body force or physics changes."""
from pathlib import Path
import copy
import tomllib
import numpy as np
import torch
from mjlab.utils.lab_api.math import combine_frame_transforms
from .owned_pen_runtime import OwnedPenRuntime
from .pen_reference_cache import tensor_digest
from supermimic_research.artifacts import file_pin
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench
from supermimic_research.mechanics.pen_support_objective import objective_terms,select_interval

CONTRACT='pen_geometry_and_nominal_source_BODY_support:1'


class ReferenceMeanSupport:
    def __init__(self,runtime):
        self.body=runtime.features.body;self.worlds=runtime.num_envs;self.dt=.02
        f=runtime.motion._frames;w=f['wrist_pose_w'].detach().cpu().double();p=f['object_pose'].detach().cpu().double()
        position,quat=combine_frame_transforms(w[:,:3],w[:,3:],p[:,:3],p[:,3:])
        self.pose=torch.cat((position,quat),-1).numpy();self.frames=len(self.pose)
        self.mass,self.inertia,self.gravity=self.physics(runtime)
        targets=[];valid=None
        for i in range(self.worlds):
            result=mean_reference_wrench(self.pose[:,:3],self.pose[:,3:],self.mass[i],np.diag(self.inertia[i]),self.gravity[i],self.dt)
            targets.append(result['wrench_world']);valid=result['valid_interval']
        self.targets=torch.as_tensor(np.stack(targets),device=runtime.device,dtype=torch.float64)
        self.valid=torch.as_tensor(valid,device=runtime.device,dtype=torch.bool)
        self.spec=dict(schema='pen_nominal_mean_moving_COM_world:1',frames=self.frames,intervals=self.frames-1,valid_intervals=int(valid.sum()),
            used_phase_interval='[k-1,k], table index k-1',invalid_tail_intervals=3,dt=self.dt,
            target_sha256=tensor_digest(self.targets),valid_sha256=tensor_digest(self.valid),
            canonical_origin='FP64 original wrist/object composition, no per-world translation derivative',
            comparison='Hand mean in actual SOURCE BODY axes, MOVING COM; window slot5, no fixed-COM shift.',
            source='Nominal reference momentum prior, not measured ego forces or feedback/applied force.')

    def physics(self,runtime):
        m=runtime.env.sim.wp_model;b=self.body;W=self.worlds
        if np.any(m.body_ipos.numpy()[:,b]) or not np.all(m.body_iquat.numpy()[:,b]==[1,0,0,0]):raise ValueError('Declared COM origin and aligned principal inertia required')
        mass=np.broadcast_to(m.body_mass.numpy()[:,b],(W,)).astype(np.float64).copy()
        inertia=np.broadcast_to(m.body_inertia.numpy()[:,b],(W,3)).astype(np.float64).copy()
        gravity=np.broadcast_to(m.opt.gravity.numpy(),(W,3)).astype(np.float64).copy()
        return mass,inertia,gravity

    def validate_physics(self,runtime):
        if any(not np.array_equal(a,b) for a,b in zip(self.physics(runtime),(self.mass,self.inertia,self.gravity))):raise ValueError('Support targets stale after native physical parameter change')


class SupportPenRuntime(OwnedPenRuntime):
    def __init__(self,objective_config,num_envs):
        self.objective_config_path=Path(objective_config).resolve();self.objective_cfg=tomllib.loads(self.objective_config_path.read_text());self.support_weight=0.
        variant=self.objective_cfg.get('support_score','ABSOLUTE');schema=self.objective_cfg.get('checkpoint_schema','pen_geometry_support_PPO_training:3')
        if (variant,schema) not in [('ABSOLUTE','pen_geometry_support_PPO_training:3'),('CENTERED','pen_geometry_support_PPO_training:4')]:raise ValueError('Explicit score/training-schema pairing required')
        super().__init__((self.objective_config_path.parent/self.objective_cfg['adapter_config']).resolve(),num_envs)
        self.support_table=ReferenceMeanSupport(self)
        self.objective_constants={'geometry_limits':torch.tensor([self.objective_cfg[k] for k in ['object_penetration_m','self_penetration_m','joint_excursion_rad']],device=self.device,dtype=torch.float64)}
        shapes=dict(target_world=(6,),target_source=(6,),measured_source=(6,),support_loss=(),support_bonus=(),support_eligible=(),tracking_reward=(),geometry_loss=(),geometry_failure=(),phase_valid=(),target_valid=(),native_terminated=())
        doubles={'target_world','target_source','measured_source','support_loss'};booleans={'support_eligible','geometry_failure','phase_valid','target_valid','native_terminated'}
        self.objective_views={k:torch.zeros((num_envs,)+shape,device=self.device,dtype=torch.float64 if k in doubles else torch.bool if k in booleans else torch.float32) for k,shape in shapes.items()}
        self.contact_views.update({'objective_'+k:v for k,v in self.objective_views.items()});self._binding_resources=self._pointers()

    def _pointers(self):
        result=super()._pointers()
        if hasattr(self,'support_table'):
            for k in ['targets','valid']:
                v=getattr(self.support_table,k);result['support.'+k]=(v.data_ptr(),tuple(v.shape),v.stride(),str(v.dtype),str(v.device))
        if hasattr(self,'objective_constants'):
            for k,v in self.objective_constants.items():result['objective_constant.'+k]=(v.data_ptr(),tuple(v.shape),v.stride(),str(v.dtype),str(v.device))
        return result

    def begin_collection(self):
        self.support_table.validate_physics(self);super().begin_collection()

    def end_collection(self):
        self.support_table.validate_physics(self);super().end_collection()

    def _reset_impl(self,ids):
        ob=super()._reset_impl(ids)
        if hasattr(self,'objective_views'):
            for value in self.objective_views.values():value[ids]=0
        return ob

    def _step_impl(self,sampled_raw):
        ob,tracking,native_term,trunc,info=super()._step_impl(sampled_raw);c=self.contact_views;phase=info['reference_phase']
        target_world,target_valid,phase_valid=select_interval(self.support_table.targets,self.support_table.valid,phase,self.reference_cache.world_ids)
        terms=objective_terms(tracking,native_term,c['geometry_max'],c['nonhand_max_sum'],c['errors'],c['global_error'],c['wrench_sum'],c['counts'],c['source_R'],target_world,target_valid,phase_valid,self.objective_cfg,self.support_weight,self.objective_constants['geometry_limits'])
        values=dict(target_world=target_world,target_valid=target_valid,phase_valid=phase_valid,tracking_reward=tracking,native_terminated=native_term,
            **{k:v for k,v in terms.items() if k!='reward'})
        for k,v in values.items():self.objective_views[k].copy_(v)
        term=native_term|terms['geometry_failure'];trunc=trunc&~term;self.pending.copy_(term|trunc)
        info['termination_reasons']['geometry']=terms['geometry_failure'];info['nominal_support_prior']=True
        return ob,terms['reward'],term,trunc,info

    def metadata(self):
        contract=CONTRACT if self.objective_cfg.get('support_score','ABSOLUTE')=='ABSOLUTE' else 'pen_geometry_and_zero_wrench_centered_source_BODY_support:2'
        result=super().metadata();result.update(objective_contract=contract,objective_config=file_pin(self.objective_config_path),support_table_spec=copy.deepcopy(self.support_table.spec),
            support_weight_state_path=(('attr','support_weight'),),
            contact_output='Borrowed native window plus objective fields; geometry/native termination at20ms control boundary; support reward only, no applied force.')
        return result
