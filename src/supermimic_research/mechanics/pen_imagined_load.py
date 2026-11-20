"""One imagined control step with source-BODY mean contact load about the COM."""
import torch

from .impulse_codec import exp_world_unchecked, mv, rotation55_torch as pen_rotation


def pen_load_step(state, pd, history, net, mass, inertia_body, gravity, dt=0.02):
    """Advance the 55-state pen chart and rotate the predicted load to the new frame.

    The first two outputs can be fed directly into the next step. The third
    output is the current network prediction in the *source* frame, suitable
    for comparison with the registered force/torque and six digit-normal labels.
    Leading batch dimensions are shared by state, pd, and optional history.
    The caller sets the network and input tensors to the working dtype/device.
    Mass, the full inertia matrix expressed in BODY, and WORLD gravity
    are shared across the batch.
    """
    if state.shape[-1:] != (55,) or pd.shape != (*state.shape[:-1], 20):
        raise ValueError("Expected state55 and PD20 with matching batch dimensions")
    if history is not None and history.shape != (*state.shape[:-1], 12):
        raise ValueError("Previous load must have twelve source-BODY components")
    args = dict(dtype=state.dtype, device=state.device)
    m = torch.as_tensor(mass, **args)
    inertia = torch.as_tensor(inertia_body, **args)
    g = torch.as_tensor(gravity, **args)
    if m.ndim != 0 or inertia.shape != (3, 3) or g.shape != (3,):
        raise ValueError("Expected one mass, BODY inertia matrix and WORLD gravity vector")

    parts = [state, pd - state[..., :20]]
    if history is not None:
        parts.append(history)
    estimate = net(torch.cat(parts, dim=-1))
    if estimate.shape != (*state.shape[:-1], 52):
        raise ValueError("Expected twelve load components and forty hand increments")
    load = estimate[..., :12]
    rotation = pen_rotation(state)
    world_force = mv(rotation, load[..., :3])
    world_torque = mv(rotation, load[..., 3:6])
    v, omega = state[..., 49:52], state[..., 52:55]
    v_next = v + dt * (world_force / m + g)
    pos_next = state[..., 40:43] + (0.5 * dt) * (v + v_next)

    # Update world angular momentum about the moving COM. Rotate the BODY
    # inertia to the estimated midpoint for the attitude update, then use
    # the final attitude for the end velocity.
    momentum = mv(rotation @ inertia @ rotation.mT, omega)
    mid_rotation = exp_world_unchecked((0.5 * dt) * omega) @ rotation
    mid_momentum = momentum + (0.5 * dt) * world_torque
    mid_omega = torch.linalg.solve(
        mid_rotation @ inertia @ mid_rotation.mT, mid_momentum.unsqueeze(-1)
    ).squeeze(-1)
    next_rotation = exp_world_unchecked(dt * mid_omega) @ rotation
    next_momentum = momentum + dt * world_torque
    next_omega = torch.linalg.solve(
        next_rotation @ inertia @ next_rotation.mT, next_momentum.unsqueeze(-1)
    ).squeeze(-1)

    next_state = torch.cat((
        state[..., :40] + estimate[..., 12:], pos_next, next_rotation[..., :, 0],
        next_rotation[..., :, 1], v_next, next_omega,
    ), dim=-1)
    next_history = torch.cat((
        mv(next_rotation.mT, world_force), mv(next_rotation.mT, world_torque),
        load[..., 6:12],
    ), dim=-1)
    return next_state, next_history, load
