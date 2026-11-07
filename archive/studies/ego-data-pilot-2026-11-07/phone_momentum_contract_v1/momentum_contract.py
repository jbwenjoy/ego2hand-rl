"""Differentiable momentum closure for phone state58; no neural parameters.

Contact impulse6 is expressed in the FROZEN source body axes. Its force
components integrate world force rotated by the constant R_source.T; torque
components integrate moments about the MOVING COM in those same fixed axes.
An instantaneous rotating-body integral or torque about a fixed world point is
not interchangeable with this quantity. Include all actual boundary contacts.

These equations constrain momentum only. They do not specify COM position or
orientation integration, contact causality, or a complete Markov state. Nominal
simulator parameters and small integration defects remain explicit assumptions.
"""
import torch

SCHEMA = 'phone_source_BODY_moving_COM_momentum_closure:1'


def _mv(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def rotation_from_state58(state):
    if state.shape[-1] != 58:
        raise ValueError('state58 required: q20,qd20,COM3,Rbody_world9,vCOM3,omega_world3')
    return state[..., 43:52].reshape(state.shape[:-1] + (3, 3))


def inertia_world(rotation_body_world, inertia_body):
    """I_body includes the actual inertial-principal-axis rotation already."""
    return rotation_body_world @ inertia_body @ rotation_body_world.transpose(-1, -2)


def angular_momentum_world(state58, inertia_body):
    return _mv(inertia_world(rotation_from_state58(state58), inertia_body), state58[..., 55:58])


def source_moving_COM_impulse_to_world(source58, impulse6_source):
    if impulse6_source.shape[-1] != 6:
        raise ValueError('contact force impulse3 [Ns], COM torque impulse3 [Nms]')
    r0 = rotation_from_state58(source58)
    return torch.cat((_mv(r0, impulse6_source[..., :3]),
                      _mv(r0, impulse6_source[..., 3:])), dim=-1)


def contact_impulse_required_source(source58, target58, mass_kg, inertia_body,
                                    gravity_world, duration_s):
    """Algebraic endpoint requirement, not a predicted/measured contact label."""
    r0t = rotation_from_state58(source58).transpose(-1, -2)
    linear = mass_kg * (target58[..., 52:55] - source58[..., 52:55]
                        - gravity_world * duration_s[..., None])
    angular = angular_momentum_world(target58, inertia_body) - angular_momentum_world(source58, inertia_body)
    return torch.cat((_mv(r0t, linear), _mv(r0t, angular)), dim=-1)


def momentum_residual_world(source58, target58, contact_impulse6_source,
                            mass_kg, inertia_body, gravity_world, duration_s):
    """Separate world residuals [Ns] and [Nms]; no mixed-unit scalar norm."""
    world = source_moving_COM_impulse_to_world(source58, contact_impulse6_source)
    linear = mass_kg * (target58[..., 52:55] - source58[..., 52:55]
                        - gravity_world * duration_s[..., None]) - world[..., :3]
    angular = (angular_momentum_world(target58, inertia_body)
               - angular_momentum_world(source58, inertia_body) - world[..., 3:])
    return {'linear_world_Ns': linear, 'angular_world_Nms': angular}


def velocities_from_source_impulse(source58, target_rotation_body_world,
                                    contact_impulse6_source, mass_kg, inertia_body,
                                    gravity_world, duration_s):
    """Conditional velocity closure given a separate target orientation.

    No orientation/position is predicted here. Supplying a target R does not
    make it physically achievable. No clipping/projection or momentum repair.
    """
    world = source_moving_COM_impulse_to_world(source58, contact_impulse6_source)
    velocity = (source58[..., 52:55] + gravity_world * duration_s[..., None]
                + world[..., :3] / mass_kg)
    momentum = angular_momentum_world(source58, inertia_body) + world[..., 3:]
    omega = torch.linalg.solve(inertia_world(target_rotation_body_world, inertia_body),
                               momentum.unsqueeze(-1)).squeeze(-1)
    return torch.cat((velocity, omega), dim=-1)
