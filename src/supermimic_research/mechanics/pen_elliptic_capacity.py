"""Ideal unit-normal elliptic friction capacity for comparison to native pyramids."""
import numpy as np

from .pen_wrench_capacity import body_wrench_rays


def elliptic_support(contacts, signs, com_world, world_from_body, length_scale_m, directions):
    """Support along BODY probes of [force, COM moment / length] for all contacts.

    The normal budget shared by the entire contact set is <= 1 N. On each
    single-contact slice at unit normal force the admissible nonnormal load
    is diag(friction) times the Euclidean unit ball in (dim - 1) dimensions.
    This includes the torsional/rolling moments when specified by condim.
    """
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 6 or not len(directions) or not np.isfinite(directions).all():
        raise ValueError('Expected a nonempty array of finite six-component directions')
    contacts = list(contacts)
    rays = body_wrench_rays(contacts, signs, com_world=com_world,
                            world_from_body=world_from_body, length_scale_m=length_scale_m)
    projections = rays @ directions.T
    start = 0
    support = np.zeros(len(directions))
    for contact in contacts:
        if contact.dim == 1:
            value = projections[start]
            start += 1
        else:
            width = 2 * (contact.dim - 1)
            pairs = projections[start:start+width].reshape(contact.dim - 1, 2, -1)
            # At unit normal load, each signed native ray is w0 +/- mu_j*w_j.
            # For an elliptic slice the support adds the L2 norm of the signed
            # directional components to w0, rather than the pyramid's maximum.
            normal = pairs[0].mean(axis=0)
            tangent = (pairs[:, 1] - pairs[:, 0]) * .5
            value = normal + np.linalg.norm(tangent, axis=0)
            start += width
        np.maximum(support, value, out=support)
    return support
