"""Causal next-control geometry-capacity forecast for the fixed BODY directions."""
import torch
from torch import nn
from torch.nn import functional as F


CAPACITY_WIDTH = 72
SCHEMA = 'pen_body72_capacity_forecast:1'


def capacity_features(state, pd, *, use_history, previous_capacity=None, history_present=None):
    """Build features before applying a PD target. Missing and measured zero differ."""
    batch = state.shape[:-1]
    if state.shape[-1:] != (55,) or pd.shape != (*batch, 20):
        raise ValueError('Expected state55 and actual PD20 with matching batch dimensions')
    parts = [state, pd - state[..., :20]]
    if use_history:
        if previous_capacity is None:
            if history_present is not None:
                raise ValueError('A per-row history mask requires previous capacity values')
            previous_capacity = state.new_zeros((*batch, CAPACITY_WIDTH))
            present = state.new_zeros((*batch, 1))
        else:
            if previous_capacity.shape != (*batch, CAPACITY_WIDTH):
                raise ValueError('Expected previous completed-control capacity72')
            if history_present is None:
                present = state.new_ones((*batch, 1))
            else:
                if history_present.shape != batch or history_present.dtype != torch.bool:
                    raise ValueError('Expected one Boolean history flag for each state')
                mask = history_present.unsqueeze(-1)
                previous_capacity = torch.where(mask, previous_capacity, 0.)
                present = mask.to(state.dtype)
        parts.extend((previous_capacity, present))
    elif previous_capacity is not None or history_present is not None:
        raise ValueError('The state-only capacity model does not consume history')
    return torch.cat(parts, dim=-1)


def capacity_mismatch(prediction, reference):
    """Per-sample mean absolute discrepancy on the 72 BODY probe values."""
    if prediction.shape[-1:] != (CAPACITY_WIDTH,) or reference.shape[-1:] != (CAPACITY_WIDTH,):
        raise ValueError('Expected 72 directional capacities')
    return (prediction - reference).abs().mean(dim=-1)


class CapacityHead(nn.Module):
    """Smooth nonnegative directional predictions from state, target and optional history.

    A history value contains the previous control's mean support evaluated in the
    moving BODY axes at each of its eight ticks. Directions are identified by
    index in that local frame. The head does not reconstruct a contact hull.
    """
    def __init__(self, use_history):
        super().__init__()
        self.use_history = use_history
        width = 75 + (CAPACITY_WIDTH + 1 if use_history else 0)
        self.register_buffer('center', torch.zeros(width))
        self.register_buffer('scale', torch.ones(width))
        self.register_buffer('varying', torch.ones(width, dtype=torch.bool))
        self.register_buffer('target_scale', torch.ones(CAPACITY_WIDTH))
        self.register_buffer('offset', torch.zeros(CAPACITY_WIDTH))
        self.net = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, CAPACITY_WIDTH))

    @torch.no_grad()
    def set_training_stats(self, x, y):
        self.center.copy_(x.mean(0))
        self.scale.copy_(x.std(0, unbiased=False).clamp_min(.001))
        self.varying.copy_(x.amax(0) != x.amin(0))
        self.target_scale.copy_(y.std(0, unbiased=False).clamp_min(.05))
        # The inverse softplus gives the TRAIN mean at zero residual output.
        mean = y.mean(0).clamp_min(.025)
        self.offset.copy_(mean + torch.log(-torch.expm1(-4 * mean)) / 4)

    def forward_features(self, x):
        normalized = ((x - self.center) / self.scale) * self.varying
        return F.softplus(self.offset + self.net(normalized) * self.target_scale, beta=4.)

    def forward(self, state, pd, previous_capacity=None, history_present=None):
        return self.forward_features(capacity_features(
            state, pd, use_history=self.use_history, previous_capacity=previous_capacity,
            history_present=history_present))


def load_capacity_head(checkpoint, dtype=None, device=None):
    """Restore one TRAIN-fitted capacity head. Input is a loaded checkpoint dict."""
    if checkpoint['schema'] != SCHEMA or type(checkpoint['use_history']) is not bool:
        raise ValueError('Unsupported capacity checkpoint')
    model = CapacityHead(checkpoint['use_history'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(dtype=dtype, device=device).eval()
