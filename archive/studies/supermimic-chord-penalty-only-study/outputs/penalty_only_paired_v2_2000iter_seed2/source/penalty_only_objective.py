"""CPU-validated penalty-only follow-up; no GPU training validation yet.

The independent P_ONLY entry uses the hash-locked, unchanged A2 objective.
CPU value, gradient and isolation checks passed. No dt is applied here.
"""
from __future__ import annotations

import copy
import torch
from torch import nn


class PenaltyOnlyPrior(nn.Module):
    """Replace only the support coefficient1 with0, retaining A2's exact P."""

    def __init__(self, frozen_a2_objective):
        super().__init__()
        self.base = frozen_a2_objective
        self.frame_count = self.base.frame_count
        self.metadata = copy.deepcopy(self.base.metadata)
        self.metadata.update(
            schema="supermimic_chord_penalty_only_prior:1",
            scientific_role="exploratory penalty-only control, not a qualified contact/force model",
            original_A2_formula="phase_weight * support_match - .1*penetration_prior",
            formula="-.1*penetration_prior",
            support_coefficient=0.,
            penetration_coefficient=.1,
            penalty_source="exact geometry_intent_penetration_prior returned by frozen A2 objective",
            implementation="A2 full forward retained; support output is not used in the reward graph",
            dt_applied_inside_module=False,
            caller_dt_s=.02,
        )
        self.metadata["base_A2_scoring"] = copy.deepcopy(self.metadata["scoring"])
        self.metadata["scoring"]["support_coefficient"] = 0.
        self.metadata["support_score_is_diagnostic_only"] = True

    def forward(self, x1_state58, memory):
        _, base_terms = self.base(x1_state58, memory)
        # Directly consume the exact shared P. Do not subtract attraction from
        # A2's combined bonus: cancellation could mask value/gradient errors.
        penalty = base_terms["geometry_intent_penetration_prior"]
        bonus = -.1 * penalty
        terms = dict(base_terms)
        terms["geometry_intent_bonus_unscaled"] = bonus
        terms["geometry_intent_support_coefficient"] = torch.zeros_like(bonus)
        terms["geometry_intent_penetration_coefficient"] = torch.full_like(bonus, .1)
        return bonus, terms
