"""Resident physical planning clones with original pen residual EMA and phase."""
import numpy as np
from .gpu_contact_oracle import GPUContactOracle


class GPUPenRollout(GPUContactOracle):
    def __init__(self, model, native_data, nworld, initial_ema20, **kwargs):
        import mujoco
        if model.na or model.nhistory or model.nplugin or model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
            raise ValueError('Explicit activation/history/plugin/sleep-free controlled pen')
        if not np.all(model.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_NONE)):
            raise ValueError('No hidden actuator dynamics')
        super().__init__(model, native_data, nworld, **kwargs)
        torch, wp = self.torch, self.wp
        self.decimation = round(.02 / model.opt.timestep)
        if abs(self.decimation * model.opt.timestep - .02) > 1e-15:
            raise ValueError('Exact original50Hz control interval')
        self.qa = int(model.jnt_qposadr[model.body_jntadr[self.body]])
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            for key in ['qfrc_applied', 'xfrc_applied', 'mocap_pos', 'mocap_quat', 'overflow']:
                self.views[key] = wp.to_torch(getattr(self.data, key))
            self.initial_warm = torch.as_tensor(native_data.qacc_warmstart, device='cuda', dtype=torch.float32)
            self.initial_time = torch.full((nworld,), native_data.time, device='cuda')
            self.initial_ema = torch.as_tensor(initial_ema20, device='cuda', dtype=torch.float32).clone()
            if self.initial_ema.shape != (20,):
                raise ValueError('Explicit initial physical residual20')
            self.ema = self.initial_ema.expand(nworld, -1).clone()
            jids = model.actuator_trnid[self.aids, 0]
            self.lower = torch.as_tensor(np.maximum(model.jnt_range[jids, 0], model.actuator_ctrlrange[self.aids, 0]), device='cuda', dtype=torch.float32)
            self.upper = torch.as_tensor(np.minimum(model.jnt_range[jids, 1], model.actuator_ctrlrange[self.aids, 1]), device='cuda', dtype=torch.float32)
            self.mw.step(self.wp_model, self.data)  # Explicit setup integration, not an episode.
            wp.synchronize_stream(self.stream)
            with wp.ScopedCapture(stream=self.stream) as capture:
                self.mw.step(self.wp_model, self.data)
            self.step_graph = capture.graph
        self.physics_ticks = 0
        self.reset_clone()

    def _enter(self):
        caller = self.torch.cuda.current_stream()
        self.torch_stream.wait_stream(caller)
        return caller

    def reset_clone(self, state=None):
        """Restore independent planning worlds; never mutates an external env.

        A supplied state consists of qpos,qvel,ctrl,warmstart,time,ema tensors
        with one source world or the exact batch. Fixed mocap/zero aid persist.
        """
        torch, wp = self.torch, self.wp
        caller = self._enter()
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            if state is None:
                state = dict(qpos=self.base_qpos, qvel=self.base_qvel, ctrl=self.base_ctrl,
                             qacc_warmstart=self.initial_warm, time=self.initial_time, ema=self.initial_ema)
            for key in ['qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'time']:
                self.views[key].copy_(state[key].expand_as(self.views[key]))
            self.ema.copy_(state['ema'].expand_as(self.ema))
            self.views['overflow'].zero_()
            wp.capture_launch(self.graph, stream=self.stream)
        caller.wait_stream(self.torch_stream)

    def set_action(self, raw20, reference20):
        torch, wp = self.torch, self.wp
        if raw20.shape != (self.nworld, 20) or not raw20.is_cuda or raw20.dtype != torch.float32:
            raise ValueError('Resident world-by20 original raw actions')
        caller = self._enter()
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            self.ema.mul_(.5).add_((.5 * raw20).clamp(-.6, .6), alpha=.5)
            target = (reference20 + self.ema).clamp(self.lower, self.upper)
            self.views['ctrl'][:, self.aindex] = target
        caller.wait_stream(self.torch_stream)

    def physics_step(self):
        caller = self._enter()
        with self.wp.ScopedStream(self.stream):
            self.wp.capture_launch(self.step_graph, stream=self.stream)
        caller.wait_stream(self.torch_stream)
        self.physics_ticks += self.nworld

    def refresh_boundary(self):
        """Same nonintegrating forward refresh as the task after decimation."""
        caller = self._enter()
        with self.wp.ScopedStream(self.stream):
            self.wp.capture_launch(self.graph, stream=self.stream)
        caller.wait_stream(self.torch_stream)

    def output(self):
        torch = self.torch
        R = self.views['xmat'][:, self.body].reshape(self.nworld, 3, 3)
        force = self.views['qfrc_constraint'][:, self.da:self.da + 3]
        torque = self.views['qfrc_constraint'][:, self.da + 3:self.da + 6]
        return dict(wrench_BODY_COM=torch.cat((torch.bmm(R.transpose(1, 2), force[:, :, None])[:, :, 0], torque), dim=1),
                    hand_qacc=self.views['qacc'][:, self.dindex])

    def state(self):
        return {**{key: self.views[key].clone() for key in
                   ['qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'time']}, 'ema': self.ema.clone()}
