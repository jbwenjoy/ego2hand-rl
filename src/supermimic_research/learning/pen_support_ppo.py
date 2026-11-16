"""Training3 pilot metadata/initialization on the shared training2 machinery."""
import copy
import hashlib
import tomllib
import numpy as np
import torch
from .owned_pen_ppo import OwnedPenPPO,SCHEMA as FORMAT2
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.portable_inventory import path_value
from supermimic_research.runtime.pen_reference_cache import tensor_digest

SCHEMA='pen_geometry_support_PPO_training:3'
CENTERED_SCHEMA='pen_geometry_support_PPO_training:4'


def pilot_settings(path):
    cfg=tomllib.loads(path.read_text());base=tomllib.loads((path.parent/cfg['ppo_config']).resolve().read_text())['ppo']
    base.update(seed=cfg['seed'],ppo_rollout_steps=cfg['rollout_controls'],ppo_init_std=cfg['initial_std'],ppo_std_min=cfg['std_min'],ppo_std_max=cfg['std_max'])
    return cfg,base


def fork_initial_state(common,expected_metadata):
    if common['schema'] not in [SCHEMA,CENTERED_SCHEMA] or common['schema']!=expected_metadata['checkpoint_schema'] or any(common[k]!=0 for k in ['iteration','optimizer_steps','training_transitions']) or common['memory'] is not None:raise ValueError('Fork only a calibrated untrained common source of the same explicit schema')
    metadata=copy.deepcopy(common['metadata']);changes=[]
    for key in ['arm','support_weight']:
        if metadata[key]!=expected_metadata[key]:changes.append(dict(path='metadata.'+key,before=metadata[key],after=expected_metadata[key]))
        metadata[key]=expected_metadata[key]
    if metadata!=expected_metadata:raise ValueError('Fork changes more than registered arm/weight')
    result=copy.deepcopy(common);result['metadata']=metadata;path=(('attr','support_weight'),)
    before=result['native']['python']['scalars'][path];after=float(expected_metadata['support_weight'])
    if before!=after:changes.append(dict(path='native.python.scalars.support_weight',before=before,after=after))
    result['native']['python']['scalars'][path]=after
    return result,changes


def fork_fixed_frame_mode(state,expected_metadata,expected_native_metadata):
    """Current-source performance fork:only feature mode may change.

    The ordinary loader still validates sources/layout/full state. No historical
    source receipt,config,physics,NN/Adam/RNG or observation is rebound/repaired.
    Both paths must already agree at the saved observation source.
    """
    field='fixed_frame_features';before=state['metadata'].get(field);after=expected_metadata.get(field)
    if state['schema'] not in [SCHEMA,CENTERED_SCHEMA] or state['schema']!=expected_metadata['checkpoint_schema'] or state['native']['schema']!=native.SCHEMA or before not in ['LIVE','CACHED'] or after not in ['LIVE','CACHED']:raise ValueError('Explicit current owned feature-mode source required')
    metadata=copy.deepcopy(state['metadata']);metadata[field]=after
    native_metadata=copy.deepcopy(state['native']['metadata']);native_metadata[field]=after
    if metadata!=expected_metadata or native_metadata!=expected_native_metadata:raise ValueError('Feature-mode fork changes more than its declared mode')
    paths=[p for p,v in state['native']['python']['scalars'].items() if p and p[-1]==('attr','fixed_frame_cache')]
    if len(paths)!=1 or type(state['native']['python']['scalars'][paths[0]]) is not bool or state['native']['python']['scalars'][paths[0]]!=(before=='CACHED') or paths[0] in state['native']['immutable_scalars']:raise ValueError('One mutable feature-mode owner with matching metadata required')
    result=copy.deepcopy(state);result['metadata']=metadata;result['native']['metadata']=native_metadata;result['native']['python']['scalars'][paths[0]]=after=='CACHED'
    changes=[] if before==after else [dict(path='metadata.'+field,before=before,after=after),dict(path='native.metadata.'+field,before=before,after=after),dict(path=repr(paths[0]),before=before=='CACHED',after=after=='CACHED')]
    return result,changes


def static_contract(runtime,inv):
    roots=[p for p in inv['objects'] if path_value(runtime,p) is runtime.support_table]
    cfg_path=(('attr','objective_cfg'),);path_path=(('attr','objective_config_path'),);constants=(('attr','objective_constants'),)
    def selected(p):return any(p[:len(root)]==root for root in roots) or p[:1] in [cfg_path,path_path,constants]
    return dict(tensors={p:tensor_digest(v) for p,v in inv['tensors'].items() if selected(p)},
        arrays={p:hashlib.sha256(str((v.shape,v.dtype)).encode()+v.tobytes()).hexdigest() for p,v in inv['arrays'].items() if selected(p)},
        scalars={p:copy.deepcopy(v) for p,v in inv['scalars'].items() if selected(p)})


class SupportPenPPO(OwnedPenPPO):
    def __init__(self,adapter,config,arm):
        if not isinstance(adapter,SupportPenRuntime) or arm not in adapter.objective_cfg['support_weights']:raise ValueError('Preregistered objective runtime/arm required')
        self.arm=arm;self.checkpoint_schema=adapter.objective_cfg.get('checkpoint_schema',SCHEMA);adapter.support_weight=float(adapter.objective_cfg['support_weights'][arm]);super().__init__(adapter,config)
        with torch.no_grad():
            self.actor.net[-1].weight[:20].zero_();self.actor.net[-1].bias[:20].copy_(adapter.initial_raw)

    @torch.no_grad()
    def calibrate(self,steps):
        if 'step' in self.adapter.__dict__:raise ValueError('Exclusive calibration hook')
        original=self.adapter.step
        self.adapter.step=lambda zero:original(self.adapter.initial_raw[None].expand(self.adapter.num_envs,-1))
        try:return super().calibrate(steps)
        finally:del self.adapter.step

    def metadata(self):
        contract='geometry_hard_nominal_support_mean:1' if self.checkpoint_schema==SCHEMA else 'geometry_hard_zero_wrench_centered_support_mean:2'
        result=super().metadata();result.update(checkpoint_schema=self.checkpoint_schema,task_reward_contract=contract,role='CONTROLLED_SINGLE_SEED_FEASIBILITY_PILOT',
            arm=self.arm,support_weight=float(self.adapter.support_weight),actor_initialization='Mean head weights0/bias initialized raw; same total-raw action/logprob contract.')
        return result

    def checkpoint(self,sources):
        result=super().checkpoint(sources);result['schema']=self.checkpoint_schema;result['task_static']=static_contract(self.adapter,native.inventory(self.adapter))
        result['scope']='Geometry/support pilot complete state. Original strict full task, repeated-seed benefit, physical WM gradients and ego generalization unproven.'
        return result

    def preflight(self,state):
        expected_schema=getattr(self,'checkpoint_schema',SCHEMA)
        if state.get('schema')!=expected_schema:raise ValueError('Explicit training3/training4 objective checkpoint required, without cross-loading')
        expected=float(self.adapter.objective_cfg['support_weights'][self.arm]);path=(('attr','support_weight'),)
        if self.adapter.support_weight!=expected or state['native']['python']['scalars'][path]!=expected:raise ValueError('Objective arm/weight state mismatch')
        inv=native.inventory(self.adapter);contract=static_contract(self.adapter,inv)
        if contract!=state['task_static']:raise ValueError('Immutable task contract mismatch')
        for p,digest in contract['tensors'].items():
            if tensor_digest(state['native']['python']['tensors'][p])!=digest:raise ValueError('Saved support table tensor changed')
        for p,digest in contract['arrays'].items():
            v=state['native']['python']['arrays'][p]
            if hashlib.sha256(str((v.shape,v.dtype)).encode()+v.tobytes()).hexdigest()!=digest:raise ValueError('Saved support table array changed')
        for p,value in contract['scalars'].items():
            if state['native']['python']['scalars'][p]!=value:raise ValueError('Saved task scalar changed')
        # Explicit private STRUCTURE reuse after the public training3 guard;
        # public training2 payloads are never accepted or rewritten on disk.
        bridge=dict(state,schema=FORMAT2)
        return super().preflight(bridge)

    @torch.no_grad()
    def consistency(self):
        result=super().consistency();m=self.memory;c=self.adapter.objective_cfg
        expected=m['contact_objective_tracking_reward']-c['geometry_soft_weight']*m['contact_objective_geometry_loss']-c['geometry_failure_penalty']*m['contact_objective_geometry_failure'].float()+self.adapter.support_weight*m['contact_objective_support_bonus']
        checks=dict(result['checks'],objective_reward=torch.equal(expected,m['raw_reward']),geometry_failure_terminated=bool((~m['contact_objective_geometry_failure']|m['terminated']).all()),
            no_support_on_invalid=bool((m['contact_objective_support_bonus'][~m['contact_objective_support_eligible']]==0).all()))
        return dict(result,checks=checks,passed=sum(checks.values()),total=len(checks),scope='Geometry/support collector consistency, not strict task success.')
