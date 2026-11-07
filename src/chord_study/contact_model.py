"""Small CPU contact surrogate for independently supervised diagnostic probes.

Physical state follows Dex-SuperMimic's 58D palm-frame layout. This predicts
hand-object contact groups and COM wrench support, never self-collision labels.
The model has no authority to redefine the frozen reference contact target.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STATE_DIM = 58
ACTION_DIM = 20
# Preserve the model's numbered fingers; no anatomical naming is assumed.
CONTACT_GROUPS = ("finger1", "finger2", "finger3", "finger4", "finger5", "palm")
STATE_LAYOUT = {"q_rad": [0, 20], "qd_rad_s": [20, 40], "object_pos_palm_m": [40, 43],
                "object_rotation_palm_matrix_rowmajor": [43, 52],
                "object_linear_velocity_palm_m_s": [52, 55], "object_angular_velocity_palm_rad_s": [55, 58]}


def pack_palm_state(q, qd, object_position_w, object_rotation_w, object_linear_velocity_w,
                    object_angular_velocity_local, palm_position_w, palm_rotation_w):
    """Match upstream SimReader.state, including MuJoCo's local angular qvel."""
    inverse = np.asarray(palm_rotation_w).T
    rotation = np.asarray(object_rotation_w)
    return np.concatenate((q, qd, inverse @ (object_position_w-palm_position_w),
                           (inverse @ rotation).reshape(-1), inverse @ object_linear_velocity_w,
                           inverse @ rotation @ object_angular_velocity_local)).astype(np.float32)


def grouped_split(groups, *, validation_groups=6, test_groups=6):
    """Reserve whole independent reset trajectories; adjacent samples never cross."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if validation_groups < 1 or test_groups < 1 or len(unique) <= validation_groups+test_groups:
        raise ValueError("need nonempty disjoint train/validation/test groups")
    return {"train": np.isin(groups, unique[:-(validation_groups+test_groups)]),
            "validation": np.isin(groups, unique[-(validation_groups+test_groups):-test_groups]),
            "test": np.isin(groups, unique[-test_groups:])}


def frozen_support_objective(support, fixed_target):
    """Negative support MSE diagnostic, NOT the CHORD reward or task success.

    A teacher contact target is established before data collection and remains
    detached. This objective deliberately avoids hard activity gates so their
    discontinuities do not hide the surrogate's action-gradient failures.
    """
    return -(support-fixed_target.detach()).square().mean(-1)


def secant_response_loss(predicted_plus, predicted_minus, observed_plus, observed_minus, epsilon, scale):
    """Supervise causal endpoint response, including zero-response interventions.

    All pairs must share an exact simulator snapshot and remain in the same
    data split. Multiple epsilons are retained as secants; this does not assert
    a unique Jacobian at a contact-mode discontinuity.
    """
    epsilon = torch.as_tensor(epsilon, dtype=predicted_plus.dtype, device=predicted_plus.device)
    scale = torch.as_tensor(scale, dtype=predicted_plus.dtype, device=predicted_plus.device)
    if torch.any(epsilon <= 0) or torch.any(scale <= 0):
        raise ValueError("positive intervention epsilon and derivative scale required")
    predicted = (predicted_plus-predicted_minus)/(2*epsilon[..., None])
    observed = ((observed_plus-observed_minus)/(2*epsilon[..., None])).detach()
    return F.smooth_l1_loss(predicted/scale, observed/scale, beta=.5)


def qualify_local_proxy(metrics, probe):
    """Preregistered v4 local gradient gates, never a global CHORD success test."""
    correlation = probe.get("pearson_epsilon_stable")
    agreement = probe.get("sign_agreement_epsilon_stable")
    ghost_rmse = metrics.get("support_rmse_known_no_contact")
    ghost_max = metrics.get("support_max_known_no_contact")
    checks = {
        "at_least_16_stable_nonflat_directions": probe.get("epsilon_stable_nonflat_direction_count", 0) >= 16,
        "stable_nonflat_sign_at_least_075": agreement is not None and agreement >= .75,
        "stable_nonflat_pearson_at_least_05": correlation is not None and correlation >= .5,
        "at_least_8_test_states": probe.get("states", 0) >= 8,
        "actual_improvement_fraction_above_half": probe.get("improved_fraction", 0) > .5,
        "known_contact_free_test_examples_present": metrics.get("known_no_contact_samples", 0) > 0,
        "ghost_support_rmse_at_most_010": ghost_rmse is not None and ghost_rmse <= .10,
        "ghost_support_max_at_most_025": ghost_max is not None and ghost_max <= .25,
    }
    return {"qualified_as_local_candidate": all(checks.values()), "checks": checks,
            "scope": "local fixed-context gradient candidate only; no automatic actor integration or task-success claim"}


class ContactSurrogate(nn.Module):
    def __init__(self, context_dim, support_dim=64, hidden_dim=128):
        super().__init__()
        self.context_dim, self.support_dim, self.hidden_dim = context_dim, support_dim, hidden_dim
        dimension = STATE_DIM+ACTION_DIM+context_dim
        self.register_buffer("input_mean", torch.zeros(dimension))
        self.register_buffer("input_scale", torch.ones(dimension))
        self.network = nn.Sequential(nn.Linear(dimension, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, support_dim+len(CONTACT_GROUPS)))

    def features(self, state, executed_target, context):
        if state.shape[-1] != STATE_DIM or executed_target.shape[-1] != ACTION_DIM:
            raise ValueError("expected physical state58 and actual executed joint target20")
        if context.shape[-1] != self.context_dim:
            raise ValueError("physical context does not match checkpoint contract")
        return torch.cat((state, executed_target, context), -1)

    @torch.no_grad()
    def fit_normalizer(self, state, executed_target, context):
        features = self.features(state, executed_target, context)
        self.input_mean.copy_(features.mean(0))
        scale = features.std(0, unbiased=False)
        # Fixed context is constant by design: do not invent sensitivity to an
        # unobserved context dimension by dividing through a tiny variance.
        self.input_scale.copy_(torch.where(scale < 1e-6, torch.ones_like(scale), scale.clamp_min(1e-3)))

    def forward(self, state, executed_target, context):
        features = (self.features(state, executed_target, context)-self.input_mean)/self.input_scale
        raw = self.network(features)
        support = F.softplus(raw[..., :self.support_dim])
        logits = raw[..., self.support_dim:]
        return support, logits


def calibration_metrics(logits, labels, *, temperature=1.0, bins=10):
    """Micro calibration plus per-group base rates; undefined F1 stays explicit."""
    probabilities = torch.sigmoid(torch.as_tensor(logits)/temperature).detach().cpu().numpy()
    labels = np.asarray(labels, dtype=float)
    prediction = probabilities >= .5
    true = labels.astype(bool)
    tp, fp, fn = int((prediction & true).sum()), int((prediction & ~true).sum()), int((~prediction & true).sum())
    ece, bin_rows = 0.0, []
    for index in range(bins):
        low, high = index/bins, (index+1)/bins
        mask = (probabilities >= low) & ((probabilities <= high) if index == bins-1 else (probabilities < high))
        count = int(mask.sum())
        confidence = float(probabilities[mask].mean()) if count else None
        observed = float(labels[mask].mean()) if count else None
        if count:
            ece += count/labels.size * abs(confidence-observed)
        bin_rows.append({"low": low, "high": high, "count": count, "probability_mean": confidence, "positive_rate": observed})
    per_group = []
    for group in range(labels.shape[-1]):
        group_positive = int(true[:, group].sum())
        predicted_positive = int(prediction[:, group].sum())
        group_tp = int((true[:, group] & prediction[:, group]).sum())
        per_group.append({"name": CONTACT_GROUPS[group], "positive_count": group_positive,
                          "predicted_positive_count": predicted_positive,
                          "brier": float(np.mean((probabilities[:, group]-labels[:, group])**2)),
                          "precision": group_tp/predicted_positive if predicted_positive else None,
                          "recall": group_tp/group_positive if group_positive else None,
                          "positive_case_coverage": "unvalidated_no_positive_labels" if not group_positive else "observed"})
    return {"brier": float(np.mean((probabilities-labels)**2)), "ece": float(ece),
            "micro_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "accuracy": float(np.mean(prediction == true)),
            "per_group_positive_rate": labels.mean(0).tolist(), "per_group": per_group, "probability_bins": bin_rows}
