"""GPU-resident contact diagnostics for bounded, derivative-free proposals."""
import numpy as np


class GPUContactMetrics:
    def __init__(self, oracle):
        import mujoco
        import torch
        import warp as wp
        self.o, self.torch = oracle, torch
        self.fields = {k: wp.to_torch(getattr(oracle.data.contact, k)) for k in
                       ['worldid', 'geom', 'dist', 'efc_address']}
        m = oracle.model
        self.object_geom = torch.as_tensor(m.geom_bodyid == oracle.body, device='cuda')
        hand = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '' for g in range(m.ngeom)]
        self.hand_geom = torch.as_tensor([name.startswith('robot/') for name in hand], device='cuda')
        self.index = torch.arange(oracle.data.naconmax, device='cuda')

    def read(self):
        """Columns: object penetration, self penetration, nonhand count, active count, capacity flag."""
        torch, o, z = self.torch, self.o, self.fields
        valid = self.index < o.views['nacon'][0]
        valid_world = (z['worldid'] >= 0) & (z['worldid'] < o.nworld)
        valid_geom = ((z['geom'] >= 0) & (z['geom'] < o.model.ngeom)).all(dim=1)
        live = valid & valid_world & valid_geom
        world = z['worldid'].clamp(0, o.nworld - 1).long()
        geom = z['geom'].clamp(0, o.model.ngeom - 1).long()
        obj = self.object_geom[geom]
        hand = self.hand_geom[geom]
        object_pair = obj[:, 0] ^ obj[:, 1]
        self_pair = hand[:, 0] & hand[:, 1]
        nonhand = (obj[:, 0] & ~hand[:, 1]) | (obj[:, 1] & ~hand[:, 0])
        depth = (-z['dist']).clamp(min=0)
        cols = []
        for mask in [object_pair, self_pair]:
            out = torch.zeros(o.nworld, device='cuda')
            out.scatter_reduce_(0, world, torch.where(live & mask, depth, 0.), reduce='amax')
            cols.append(out)
        for mask in [nonhand, object_pair & (z['efc_address'][:, 0] >= 0)]:
            out = torch.zeros(o.nworld, device='cuda')
            out.scatter_add_(0, world, (live & mask).float())
            cols.append(out)
        capacity = ((o.views['nacon'][0] <= o.data.naconmax)
                    & ((~valid) | (valid_world & valid_geom)).all()
                    & (o.views['nefc'] <= o.data.njmax).all())
        cols.append(capacity.float().expand(o.nworld))
        return torch.stack(cols, dim=1)


def proposal_score(wrench, hand_accel, metrics, target_wrench, target_accel, joint_delta, config):
    """Declared selection score only; independent gates decide qualification."""
    import torch
    scales = torch.as_tensor([config['force_scale_N']] * 3 + [config['torque_scale_Nm']] * 3,
                             device=wrench.device)
    cost = ((wrench - target_wrench) / scales).square().sum(dim=1)
    cost += config['hand_accel_weight'] * ((hand_accel - target_accel) / config['hand_accel_scale_rad_s2']).square().mean(dim=1)
    cost += config['joint_regularization'] * (joint_delta / config['joint_bound_rad']).square().sum(dim=1)
    cost += ((metrics[:, 0] - config['object_penetration_m']).clamp(min=0) / config['collision_scale_m']).square()
    cost += ((metrics[:, 1] - config['self_penetration_m']).clamp(min=0) / config['self_collision_scale_m']).square()
    cost += config['nonhand_contact_penalty'] * metrics[:, 2]
    valid = torch.isfinite(wrench).all(dim=1) & torch.isfinite(hand_accel).all(dim=1) & (metrics[:, 4] == 1)
    return torch.where(valid, cost, torch.full_like(cost, float('inf')))
