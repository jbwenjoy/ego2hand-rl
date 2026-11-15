"""State-error correction to a nominal WORLD COM wrench objective, not an applied force."""
import torch


def feedback_support_target(nominal, mass, source_position, source_velocity,
                            reference_position, reference_velocity,
                            position_gain=100., velocity_gain=20.):
    """Gains have units s^-2/s^-1. All arguments use the same WORLD frame.

    Preserve nominal torque and return explicit diagnostic errors/correction.
    The caller owns derivative-valid masks. No clamp, state repair, device
    readback, simulator mutation or force application takes place here.
    """
    position_error = reference_position - source_position
    velocity_error = reference_velocity - source_velocity
    correction = mass * (position_gain * position_error + velocity_gain * velocity_error)
    target = torch.cat((nominal[:3] + correction, nominal[3:]))
    return dict(target_world=target, correction_force_world=correction,
                position_error_world_m=position_error,
                velocity_error_world_m_s=velocity_error)
