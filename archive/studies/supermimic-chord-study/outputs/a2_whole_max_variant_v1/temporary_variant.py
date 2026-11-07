def weighted_wrench_support(points_com_o, inward_normals_o, weights, basis, radius, *, mu=.1, edges=8):
    """Same unit-ray/max/basis/COM convention as CHORD, with soft weights.

    Binary weights recover the observed-contact support kernel. Continuous
    weights are a geometric proxy; they must not be interpreted as force N.
    Algebra avoids materializing the points × rays × basis tensor.
    """
    n = F.normalize(inward_normals_o, dim=-1, eps=1e-12)
    sign = torch.where(n[..., 2] >= 0, 1., -1.)
    a = -1/(sign+n[..., 2])
    b = n[..., 0]*n[..., 1]*a
    t1 = F.normalize(torch.stack((1+sign*n[..., 0].square()*a, sign*b, -sign*n[..., 0]), -1), dim=-1)
    t2 = F.normalize(torch.stack((b, sign+n[..., 1].square()*a, -n[..., 1]), -1), dim=-1)

    def project(direction):
        return (torch.einsum("...pj,nj->...pn", direction, basis[:, :3])
                +torch.einsum("...pj,nj->...pn", torch.linalg.cross(points_com_o, direction, dim=-1), basis[:, 3:])/radius)

    normal, tangent1, tangent2 = project(n), project(t1), project(t2)
    tangent_best = None
    for edge in range(edges):
        angle = 2*math.pi*edge/edges
        value = math.cos(angle)*tangent1+math.sin(angle)*tangent2
        tangent_best = value if tangent_best is None else torch.maximum(tangent_best, value)
    support = torch.maximum(normal, (normal+mu*tangent_best)/math.sqrt(1+mu*mu))
    return (support*weights[..., None]).max(dim=-2).values.clamp_min(0)
