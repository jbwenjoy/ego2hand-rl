"""Explicit same-state geometry graph and immutable device constants.

Contexts are short-lived: one exact Tensor, provider and autograd mode. They
never detach geometry or memoize across predicted states/optimizer updates.
Only state-independent constants use the persistent dtype/device cache.
"""
import torch


def _version(value):
    # Inference tensors deliberately do not have a version counter. Callers
    # must keep those tensors immutable for their explicit context lifetime.
    try:
        return value._version
    except RuntimeError:
        return None


def _stamp(value):
    return (id(value), _version(value), value.shape, value.dtype, value.device,
            value.requires_grad)


class ImmutableTensorCache:
    """Read-only constant copies, not an input/state or detached-graph cache.

    A first request inside inference_mode still builds ordinary tensors, so
    later input AD can save these constants for backward. Source and returned
    cache tensors must not be mutated; versioned mutations are rejected.
    """
    def __init__(self, sources):
        self.sources = dict(sources)
        if any(not isinstance(t, torch.Tensor) or t.requires_grad for t in self.sources.values()):
            raise ValueError('Only non-trainable constant tensors may be cached')
        self._source_stamps = {k: _stamp(v) for k, v in self.sources.items()}
        self._buckets = {}

    def require_unchanged(self):
        if any(_stamp(v) != self._source_stamps[k] for k, v in self.sources.items()):
            raise ValueError('Immutable geometry constants changed; construct a new provider')

    def get(self, like):
        self.require_unchanged()
        key = (like.device.type, like.device.index, like.dtype)
        if key not in self._buckets:
            with torch.inference_mode(False), torch.no_grad():
                values = {k: v.to(device=like.device, dtype=like.dtype).clone()
                          for k, v in self.sources.items()}
            self._buckets[key] = (values, {k: _stamp(v) for k, v in values.items()})
        values, stamps = self._buckets[key]
        if any(_stamp(v) != stamps[k] for k, v in values.items()):
            raise ValueError('A cached immutable geometry constant was mutated')
        return values


class CurrentGeometry:
    """Lazy complete FK/descriptor graph owned by one exact current state.

    This is not a global value cache. Equal-valued clones require new contexts.
    Ordinary Tensor in-place mutation and mode changes are rejected. Inference
    tensors have no version counter and must be treated as read-only by caller.
    A retained all-direction debug table must be requested at construction.
    """
    def __init__(self, owner, state58, *, retain_all_directions=False):
        if not isinstance(state58, torch.Tensor):
            raise ValueError('Geometry context requires a Torch state Tensor')
        self.owner = owner
        self.state = state58
        self._state_stamp = _stamp(state58)
        self._mode = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        self._retain_all = bool(retain_all_directions)
        self._poses = None
        self._directional = None

    def require(self, owner, state58):
        if owner is not self.owner or state58 is not self.state or _stamp(state58) != self._state_stamp:
            raise ValueError('Geometry context belongs to one unchanged exact state and owner')
        if self._mode != (torch.is_grad_enabled(), torch.is_inference_mode_enabled()):
            raise ValueError('Geometry context cannot cross grad/inference lifecycles')
        return self

    def require_provider(self, provider, state58):
        self.require(self.owner, state58)
        if provider is not getattr(self.owner, 'provider', self.owner):
            raise ValueError('Geometry context has a different compiled FK provider')
        return self

    def poses(self):
        self.require(self.owner, self.state)
        if self._poses is None:
            self._poses = getattr(self.owner, 'provider', self.owner).poses(self.state)
            if hasattr(self.owner, 'provider'):
                self.owner.counts['pure_Torch_FK_calls'] += 1
        return self._poses

    def directional(self, return_all_directions=False):
        self.require(self.owner, self.state)
        if return_all_directions and not self._retain_all:
            raise ValueError('Request retained all-direction debug planes at bundle construction')
        if self._directional is None:
            if not hasattr(self.owner, '_features_from_poses'):
                raise ValueError('This context owner has no directional descriptor')
            self._directional = self.owner._features_from_poses(
                self.state, self.poses(), return_all_directions=self._retain_all)
        if return_all_directions or not self._retain_all:
            return self._directional
        return {k: v for k, v in self._directional.items() if k != 'all_directional_plane_gaps_m'}
