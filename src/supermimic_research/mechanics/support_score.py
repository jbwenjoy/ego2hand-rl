"""Dimensionless mean-wrench scores, separate from physics/contact feasibility."""
import torch

VARIANTS=('ABSOLUTE','CONTACT_ABSOLUTE','CENTERED','CONTACT_CENTERED')


def support_score(measured,target,eligible,force_scale_N,torque_scale_Nm,variant,*,normal_mean_N=None,contact_threshold_N=.1,loss=None):
    if variant not in VARIANTS:raise ValueError('Explicit registered support-score variant required')
    if force_scale_N<=0 or torque_scale_Nm<=0 or contact_threshold_N<0:raise ValueError('Positive force/torque scales and nonnegative contact threshold required')
    if loss is None:
        loss=((measured[...,:3]-target[...,:3])/force_scale_N).square().mean(-1)+((measured[...,3:]-target[...,3:])/torque_scale_Nm).square().mean(-1)
    baseline_loss=(target[...,:3]/force_scale_N).square().mean(-1)+(target[...,3:]/torque_scale_Nm).square().mean(-1)
    absolute=torch.exp(-.5*loss);baseline=torch.exp(-.5*baseline_loss)
    value=absolute-baseline if variant in ['CENTERED','CONTACT_CENTERED'] else absolute
    active=eligible
    if variant in ['CONTACT_ABSOLUTE','CONTACT_CENTERED']:
        if normal_mean_N is None:raise ValueError('Explicit mean normal load required for contact-gated candidate')
        active=active&(normal_mean_N>contact_threshold_N)
    return dict(score=torch.where(active,value,torch.zeros_like(value)),absolute=absolute,zero_baseline=baseline,loss=loss,active=active)
