"""Differentiable unit-force contact-wrench support, inspired by CHORD.

This geometric fingerprint is NOT a contact force estimator or dynamics proof.
Normals point in the force direction exerted on the object (inward).
Torque is divided by radius exactly once; the shared basis is a unit 6-sphere.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def sample_basis(count=512, seed=0, *, dtype=torch.float32, device=None):
    if count < 1:
        raise ValueError("basis must be non-empty")
    raw = np.random.default_rng(seed).standard_normal((count, 6))
    raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
    return torch.as_tensor(raw, dtype=dtype, device=device)


def wrench_support(points_o, normals_o, active, basis, radius_m, *, mu=0.1, edges=8):
    """(...,K,3), (...,K,3), (...,K) -> (...,N).

    Binary contact masks are mandatory for observed contacts. Differentiation
    flows through positions/normals, never through a fabricated contact mask.
    An imagined-support head must be separately calibrated before use in reward.
    """
    if points_o.shape != normals_o.shape or points_o.shape[-1] != 3:
        raise ValueError("contact points and normals must have equal (...,K,3) shapes")
    if active.shape != points_o.shape[:-1] or basis.ndim != 2 or basis.shape[-1] != 6:
        raise ValueError("invalid contact mask or wrench basis shape")
    if not math.isfinite(mu) or mu < 0 or edges < 3:
        raise ValueError("invalid friction coefficient/cone resolution")
    radius = torch.as_tensor(radius_m, dtype=points_o.dtype, device=points_o.device)
    if torch.any(~torch.isfinite(radius)) or torch.any(radius <= 0):
        raise ValueError("radius must be positive")
    if torch.any((active != 0) & (active != 1)):
        raise ValueError("observed contact masks must be binary")
    if not all(torch.isfinite(x).all() for x in (points_o, normals_o, basis)):
        raise ValueError("non-finite contact geometry")
    if basis.shape[0] == 0 or not torch.allclose(torch.linalg.vector_norm(basis,dim=-1), torch.ones_like(basis[:,0]), atol=1e-5):
        raise ValueError("wrench basis must contain unit directions")
    if points_o.shape[-2] == 0:
        return points_o.new_zeros(points_o.shape[:-2]+(basis.shape[0],))
    norm = torch.linalg.vector_norm(normals_o, dim=-1)
    if torch.any((active != 0) & (norm <= 1e-3)):
        raise ValueError("active contact has no normal")
    n = F.normalize(normals_o, dim=-1, eps=1e-8)
    # Frisvad tangent frame, same cone phase as FlashCHORD.
    sign = torch.where(n[..., 2] >= 0, 1.0, -1.0)
    a = -1 / (sign + n[..., 2])
    b = n[..., 0]*n[..., 1]*a
    t1 = F.normalize(torch.stack((1+sign*n[..., 0]**2*a, sign*b, -sign*n[..., 0]), -1), dim=-1)
    t2 = F.normalize(torch.stack((b, sign+n[..., 1]**2*a, -n[..., 1]), -1), dim=-1)
    theta = torch.arange(edges, device=n.device, dtype=n.dtype)*(2*math.pi/edges)
    rays = F.normalize(n[..., None, :]+mu*(theta.cos()[:, None]*t1[..., None, :]
                                          +theta.sin()[:, None]*t2[..., None, :]), dim=-1)
    rays = torch.cat((rays, n[..., None, :]), -2)
    torque = torch.linalg.cross(points_o[..., None, :].expand_as(rays), rays, dim=-1)
    torque = torque / radius[..., None, None, None]
    w = torch.cat((rays, torque), -1)
    values = torch.einsum("...keq,nq->...ken", w, basis)
    values = torch.where(active[..., None, None].bool(), values, -torch.inf)
    return values.amax(dim=(-3, -2)).clamp_min(0)


def support_reward(sim_support, reference_support, *, tolerance=0.1, var=0.1, active_threshold=1e-3):
    """FlashCHORD tolerance-band score for (..., hands, bodies, basis).

    Average over commanded-active basis, bodies, hands, in that order. Missing
    contact labels must be masked outside this function. Empty reference scores
    zero; release/unexpected contacts need their own penalty. This score is not
    task success. It uses the same hard active gates as the simulator objective.
    """
    if sim_support.shape != reference_support.shape:
        raise ValueError("support fingerprints must share shape and basis")
    if sim_support.ndim < 3 or not math.isfinite(tolerance) or tolerance < 0 or not math.isfinite(var) or var <= 0:
        raise ValueError("expected (...,hands,bodies,basis), valid tolerance and var")
    if not all(torch.isfinite(x).all() and torch.all(x>=0) for x in (sim_support,reference_support)):
        raise ValueError("support fingerprints must be finite and non-negative")
    target = reference_support.detach()
    commanded = target > active_threshold
    current = sim_support > active_threshold
    loss = ((1-tolerance)*target-sim_support).clamp_min(0).square() + (sim_support-(1+tolerance)*target).clamp_min(0).square()
    score = torch.where(commanded & current, torch.exp(-loss/var), 0.)
    body_active = commanded.any(-1)
    body_score = score.sum(-1)/commanded.sum(-1).clamp_min(1)
    hand_active = body_active.any(-1)
    hand_score = body_score.sum(-1)/body_active.sum(-1).clamp_min(1)
    return hand_score.sum(-1)/hand_active.sum(-1).clamp_min(1)
