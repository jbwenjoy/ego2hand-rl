"""Resident, forward-only GPU queries of bounded hand pose and PD targets."""
import numpy as np


class GPUContactOracle:
    def __init__(self, model, native_data, nworld, *, nconmax=128, njmax=512):
        import mujoco
        import mujoco_warp as mw
        import torch
        import warp as wp
        self.mw, self.torch, self.wp = mw, torch, wp
        self.model, self.native_data, self.nworld = model, native_data, nworld
        if nworld < 1 or model.neq or np.any(model.body_gravcomp):
            raise ValueError('Independent forward worlds without equality/gravity aid')
        if (not np.all(model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
                or np.any(native_data.qfrc_applied) or np.any(native_data.xfrc_applied)):
            raise ValueError('Only joint actuation, no applied force')
        self.body = model.body('object/cube').id
        jid = model.body_jntadr[self.body]
        if model.jnt_type[jid] != int(mujoco.mjtJoint.mjJNT_FREE) or np.any(model.body_ipos[self.body]):
            raise ValueError('Explicit free object with COM at its BODY origin')
        self.da = int(model.jnt_dofadr[jid])
        names = [f'robot/right_finger{i}_joint{j}' for i in range(1, 6) for j in range(1, 5)]
        jids = np.array([model.joint(name).id for name in names])
        self.qids = model.jnt_qposadr[jids].copy()
        self.dofs = model.jnt_dofadr[jids].copy()
        self.aids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j)[0] for j in jids])
        if len(set(self.aids)) != 20 or model.nu != 20:
            raise ValueError('Exactly twenty uniquely indexed hand actuators')
        wp.init()
        self.torch_stream = torch.cuda.Stream()
        self.stream = wp.stream_from_torch(self.torch_stream)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            self.wp_model = mw.put_model(model)
            self.data = mw.put_data(model, native_data, nworld=nworld, nconmax=nconmax, njmax=njmax)
            self.views = {key: wp.to_torch(getattr(self.data, key)) for key in [
                'qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'qacc', 'qfrc_constraint', 'xmat', 'time', 'nefc', 'nacon']}
            self.base_qpos = torch.as_tensor(native_data.qpos, device='cuda', dtype=torch.float32)
            self.base_qvel = torch.as_tensor(native_data.qvel, device='cuda', dtype=torch.float32)
            self.base_ctrl = torch.as_tensor(native_data.ctrl, device='cuda', dtype=torch.float32)
            self.qindex = torch.as_tensor(self.qids, device='cuda')
            self.dindex = torch.as_tensor(self.dofs, device='cuda')
            self.aindex = torch.as_tensor(self.aids, device='cuda')
            # Forward warmup compiles kernels before graph capture; graph arrays
            # are never replaced after capture.
            self._set_state(self.base_qpos[self.qindex].expand(nworld, -1),
                            self.base_ctrl[self.aindex].expand(nworld, -1))
            mw.forward(self.wp_model, self.data)
            wp.synchronize_stream(self.stream)
            with wp.ScopedCapture(stream=self.stream) as capture:
                mw.forward(self.wp_model, self.data)
            self.graph = capture.graph
        self.query_count = 0

    def _set_state(self, q20, ctrl20):
        self.views['qpos'].copy_(self.base_qpos.expand(self.nworld, -1))
        self.views['qpos'][:, self.qindex] = q20
        self.views['qvel'].copy_(self.base_qvel.expand(self.nworld, -1))
        self.views['ctrl'].copy_(self.base_ctrl.expand(self.nworld, -1))
        self.views['ctrl'][:, self.aindex] = ctrl20
        self.views['qacc_warmstart'].zero_()

    def query(self, q20, ctrl20):
        """CUDA float32 matrices in, CUDA result tensors out; no host readback.

        Shape/device/type are metadata checks. Numeric bounds/finite values and
        capacity are deliberately audited at batch boundaries by the caller.
        """
        torch, wp = self.torch, self.wp
        for value in [q20, ctrl20]:
            if value.shape != (self.nworld, 20) or value.dtype != torch.float32 or not value.is_cuda:
                raise ValueError('Resident CUDA float32 world-by-20 inputs')
        caller_stream = torch.cuda.current_stream()
        self.torch_stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            self._set_state(q20, ctrl20)
            wp.capture_launch(self.graph, stream=self.stream)
            R = self.views['xmat'][:, self.body].reshape(self.nworld, 3, 3)
            force = self.views['qfrc_constraint'][:, self.da:self.da + 3]
            torque = self.views['qfrc_constraint'][:, self.da + 3:self.da + 6]
            wrench = torch.cat((torch.bmm(R.transpose(1, 2), force[:, :, None])[:, :, 0], torque), dim=1)
            qacc = self.views['qacc'][:, self.dindex]
        caller_stream.wait_stream(self.torch_stream)
        self.query_count += self.nworld
        return dict(wrench_BODY_COM=wrench, hand_qacc=qacc)

    def readback(self, result):
        """One packed CPU transfer for query outputs and full state receipts."""
        keys = ['wrench_BODY_COM', 'hand_qacc', 'qpos', 'qvel', 'ctrl', 'qacc', 'qfrc_constraint', 'time', 'nefc']
        blocks = [result[k] if k in result else self.views[k] for k in keys]
        sizes = [value.numel() // self.nworld for value in blocks]
        packed = self.torch.cat([v.reshape(self.nworld, -1).to(self.torch.float64) for v in blocks], dim=1).cpu().numpy()
        values = np.split(packed, np.cumsum(sizes)[:-1], axis=1)
        return dict(zip(keys, values))
