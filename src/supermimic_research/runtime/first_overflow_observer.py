"""External read-only first-event latch; no simulator/collector state attachment.

All live tensors are borrowed. Storage and conditional copies remain on the
caller's device/stream; only packet() crosses to CPU at a collection boundary.
Geometry poses after step belong to its latest collision forward, while qpos
may already be integrated. Neither this packet nor pre-state is a full resume.
"""
import torch


class FirstOverflowObserver:
    def __init__(self, views):
        required = {'overflow', 'phase', 'qpos', 'qvel', 'ctrl', 'time',
                    'qacc_warmstart', 'geom_xpos', 'geom_xmat'}
        if set(views) != required:
            raise ValueError('Explicit physical first-event views required')
        flag = views['overflow']
        if flag.ndim != 1 or flag.dtype != torch.int32:
            raise ValueError('Raw int32 native overflow mask required, never bool')
        if any(v.device != flag.device or v.shape[0] != len(flag) for v in views.values()):
            raise ValueError('World-aligned, same-device views required')
        self.views = views
        self.scratch = {k: torch.empty_like(v) for k, v in views.items()}
        self.pre = {k: torch.zeros_like(v) for k, v in views.items()}
        self.post = {k: torch.zeros_like(v) for k, v in views.items()}
        self.raw_mask = torch.zeros_like(flag)
        self.event = torch.zeros((len(flag), 4), device=flag.device, dtype=torch.int64)
        self.active = False

    @torch.no_grad()
    def before(self):
        if self.active:
            raise ValueError('Nested native operations require separate observer scratch')
        self.active = True
        for k, v in self.views.items():
            self.scratch[k].copy_(v)

    @torch.no_grad()
    def after(self, *, iteration, kind, step_call, forward_call):
        if not self.active or kind not in [1, 2]:
            raise ValueError('Matched STEP=1/FORWARD=2 observation required')
        raw = self.views['overflow']
        first = (self.raw_mask == 0) & (raw != 0)
        for k, v in self.views.items():
            mask = first.reshape((len(first),) + (1,) * (v.ndim - 1))
            self.pre[k].copy_(torch.where(mask, self.scratch[k], self.pre[k]))
            self.post[k].copy_(torch.where(mask, v, self.post[k]))
        for col, value in enumerate([iteration, kind, step_call, forward_call]):
            self.event[:, col].copy_(torch.where(first, value, self.event[:, col]))
        self.raw_mask.copy_(torch.where(first, raw, self.raw_mask))
        self.active = False

    def packet(self):
        if self.active:
            raise ValueError('Incomplete operation cannot be called a captured event')
        return dict(schema='first_native_overflow_diagnostic_NOT_RESUMABLE:1',
            raw_mask=self.raw_mask.detach().cpu().clone(), event=self.event.detach().cpu().clone(),
            pre={k: v.detach().cpu().clone() for k, v in self.pre.items()},
            post={k: v.detach().cpu().clone() for k, v in self.post.items()},
            event_columns=['iteration', 'kind_STEP1_FORWARD2', 'step_call_1based', 'forward_call_1based'],
            scope='First observed nonzero raw mask per world at operation return, before later reset. Exact borrowed field bytes; no collision-local geom-pair attribution, full solver/graph scratch or resumable state. post geom transforms describe collision forward; post qpos can describe integrated state.')
