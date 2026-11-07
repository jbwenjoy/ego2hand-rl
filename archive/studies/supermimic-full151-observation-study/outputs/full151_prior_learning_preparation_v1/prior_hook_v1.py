"""Scoped imagination-only geometric prior; no factory or Torch import at top.

This hook does not change the native phase<150 mask, actor/WM/critic formula,
reference, real collector, optimizer order, random draws or controller state.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
from .io import jsonable, pin

DT = .02
METHODS = ('A1L', 'P_ONLY', 'A2')
COUNTERS = ('policy_update_calls', 'objective_forward_calls', 'active_rows',
            'inactive_rows', 'dt_applications')


class ImaginationPrior:
    def __init__(self, method, objective, spec_pin, output):
        if method not in METHODS:
            raise ValueError('Unknown full151 prior method')
        if (method == 'A1L' and (objective is not None or spec_pin is not None)) or (method != 'A1L' and (objective is None or spec_pin is None)):
            raise ValueError('A1L must have no objective/spec; P_ONLY/A2 require both')
        self.method = method; self.objective = objective
        self.spec_pin = copy.deepcopy(spec_pin); self.output = Path(output)
        self.objective_metadata = None if objective is None else jsonable(copy.deepcopy(objective.metadata))
        self.counts = {name: 0 for name in COUNTERS}
        self.last_metrics = {}; self.logging_errors = []
        self.build_rng = {'status': 'NOT_YET_RECORDED'}

    def evidence(self):
        return {'schema': 'full151_imagination_prior_checkpoint_evidence:1',
            'method': self.method, 'status': 'NONE' if self.objective is None else 'IMAGINATION_ONLY',
            'spec_pin': copy.deepcopy(self.spec_pin),
            'objective_metadata': copy.deepcopy(self.objective_metadata), **self.counts,
            'dt_s': DT, 'imagined_soft_required': True,
            'real_collect_shaped': False, 'WMfit_shaped': False,
            'dead_end_rule': 'unchanged native full151 phase<150 hardmask after reward',
            'source_pins': {'prior_hook': pin(__file__)},
            'build_rng': copy.deepcopy(self.build_rng),
            'logging_errors': copy.deepcopy(self.logging_errors)}

    @contextmanager
    def policy_scope(self, runner):
        if self.objective is None:
            self.last_metrics = {}
            yield
            return
        import torch
        if getattr(runner, '_full151_prior_scope_active', False):
            raise ValueError('Nested imagination prior scope rejected')
        original = runner.reward_step; owned = 'reward_step' in vars(runner)
        before = dict(self.counts); self.counts['policy_update_calls'] += 1
        sums = {}; active_counts = []; inactive_counts = []; body_error = None
        self.last_metrics = {}; runner._full151_prior_scope_active = True

        def shaped(x1, memory, action, residual, contact, torque, soft=False):
            if soft is not True:
                raise ValueError('Prior is imagination-only: soft=True required')
            reward, cage, native_terms = original(x1, memory, action, residual, contact, torque, soft=soft)
            self.counts['objective_forward_calls'] += 1
            bonus, extra = self.objective(x1, memory)
            if (not isinstance(bonus, torch.Tensor) or bonus.shape != reward.shape
                    or bonus.ndim != 1 or not bool(torch.isfinite(bonus).all())):
                raise ValueError('Finite per-row unscaled prior bonus required')
            phase = memory['time'].squeeze(-1)
            if phase.shape != reward.shape or not bool(torch.isfinite(phase).all()) or bool((phase < 0).any()):
                raise ValueError('Full151 prior phase must be finite, nonnegative and batch-aligned')
            active = phase < 150
            active_counts.append(active.detach().sum()); inactive_counts.append((~active).detach().sum())
            for name, value in {'bonus_unscaled': bonus, **extra}.items():
                if not isinstance(value, torch.Tensor) or value.shape != reward.shape:
                    raise ValueError('Prior log terms must be per-row tensors: ' + name)
                masked = torch.where(active, value.detach().float(), torch.zeros_like(value, dtype=torch.float32))
                if not bool(torch.isfinite(masked).all()):
                    raise FloatingPointError('Nonfinite active prior log term: ' + name)
                sums.setdefault(name, []).append(masked.sum())
            self.counts['dt_applications'] += 1
            # Preserve bonus autograd and all original native return fields.
            return reward + DT * bonus, cage, native_terms

        runner.reward_step = shaped
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            if owned: runner.reward_step = original
            else: delattr(runner, 'reward_step')
            delattr(runner, '_full151_prior_scope_active')
            # Only detached logging is materialized here, after the H8 scope.
            active_total = torch.stack(active_counts).sum() if active_counts else torch.tensor(0)
            inactive_total = torch.stack(inactive_counts).sum() if inactive_counts else torch.tensor(0)
            self.counts['active_rows'] += int(active_total)
            self.counts['inactive_rows'] += int(inactive_total)
            denominator = active_total.clamp_min(1)
            self.last_metrics = {'prior/' + name: torch.stack(values).sum() / denominator
                                 for name, values in sums.items()}
            row = {'schema': 'full151_imagination_prior_policy_log:1',
                'method': self.method, 'policy_update_call': self.counts['policy_update_calls'],
                'iteration': int(runner.iteration),
                **{name: self.counts[name] - before[name] for name in COUNTERS if name != 'policy_update_calls'},
                'complete': body_error is None, 'dt_s': DT,
                'active_means': jsonable(self.last_metrics), 'cumulative': self.evidence()}
            try:
                with (self.output / 'full151_prior_evidence.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
            except BaseException as exc:
                self.logging_errors.append(repr(exc))
                if body_error is None: raise
