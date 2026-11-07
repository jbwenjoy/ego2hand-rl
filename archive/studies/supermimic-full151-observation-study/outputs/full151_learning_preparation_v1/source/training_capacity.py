"""Reset-aware capacity evidence for the separate full151 format105 learner.

No model, optimizer, physical step, reset, RNG or native save formula is added.
This process-wide scope is deliberately single-threaded and single-Env.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path

import importlib.util

CAPACITY_SOURCE = Path('D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/capacity.py')

def capacity_observer(*args, **kwargs):
    if _pin(CAPACITY_SOURCE)['sha256'] != CAPACITY_SHA:
        raise ValueError('Frozen reset-aware capacity observer changed')
    spec = importlib.util.spec_from_file_location('_full151_training_capacity7380', CAPACITY_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.capacity_observer(*args, **kwargs)
from .metadata import write

CAPACITY_SHA = '7380bef76a4c6af5c65204d0bf9f5921d68826072a71a77656f1b4b2fda3844a'
ALLOWED_ROLES = ('engineering_smoke', 'primary_matched')


def _pin(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _pins():
    capacity = _pin(CAPACITY_SOURCE)
    if capacity['sha256'] != CAPACITY_SHA:
        raise ValueError('Qualified capacity observer changed')
    return {'training_capacity': _pin(__file__), 'capacity_observer': capacity,
            'learning': _pin(Path(__file__).with_name('learning.py')),
            'controller': _pin(Path(__file__).with_name('controller.py')),
            'runtime': _pin(Path(__file__).with_name('runtime.py'))}


class TrainingCapacityScope:
    """Install before Runner construction, finalize before Env.close.

    The two private frozen-observer calls at checkpoint are intentionally pinned:
    _read('checkpoint') reads once, _raise_if_unqualified rejects latched failures.
    They do not finalize, reset flags, attribute flags to worlds or clear history.
    """
    def __init__(self, manager_class, torch_module, output, *, stage, source_pins=None, runner_class=None):
        self.manager_class = manager_class
        self.torch = torch_module
        self.output = Path(output)
        self.stage = stage
        self.runner_class = runner_class
        self.source_pins = copy.deepcopy(source_pins if source_pins is not None else _pins())
        self.entries = []
        self.finalization_errors = []

    def _snapshot(self, entry, *, kind):
        result = entry['observer'].snapshot()
        result.update(evidence_kind=kind,
                      entire_run_finalized=result['finalized'],
                      prefix_at_checkpoint_capacity_qualified=(
                          kind == 'checkpoint_prefix' and result['status'] == 'OK'
                          and result['unknown_read_count'] == 0 and not result['finalized']),
                      source_pins=copy.deepcopy(self.source_pins))
        return result

    def _finalize(self, entry):
        failure = None
        try:
            entry['observer'].final_read()
        except BaseException as exc:
            failure = exc
            self.finalization_errors.append({'stage': 'final_read', 'error': repr(exc)})
        try:
            write(self.output / 'capacity_observation.json', self._snapshot(entry, kind='entire_run_final'))
        except BaseException as exc:
            self.finalization_errors.append({'stage': 'capacity_snapshot_write', 'error': repr(exc)})
            if failure is None:
                failure = exc
        if failure is not None:
            raise failure

    def _install(self, env, cfg):
        if self.entries:
            raise ValueError('Training capacity scope permits exactly one Env instance')
        fixed = {'nconmax': int(cfg.sim.nconmax), 'njmax': int(cfg.sim.njmax)}
        if fixed != {'nconmax': 192, 'njmax': 4096}:
            raise ValueError('Frozen full151 capacity configuration differs')
        cm = capacity_observer(env, stage=self.stage, fixed_capacities=fixed)
        observer = cm.__enter__()
        entry = {'env': env, 'observer': observer, 'context': cm,
                 'close_owned': 'close' in vars(env), 'close_previous': vars(env).get('close'),
                 'native_close': env.close, 'closed': False, 'context_exited': False}
        self.entries.append(entry)

        def close(*args, **kwargs):
            # Keep cleanup possible even if final observation/write fails.
            first_error = None
            try:
                self._finalize(entry)
            except BaseException as exc:
                first_error = exc
            try:
                result = entry['native_close'](*args, **kwargs)
            except BaseException as exc:
                self.finalization_errors.append({'stage': 'env.close', 'error': repr(exc)})
                if first_error is None:
                    first_error = exc
                result = None
            finally:
                entry['closed'] = True
            if first_error is not None:
                raise first_error
            return result
        env.close = close
        record = observer._read('after_constructor_before_actual8')
        observer._raise_if_unqualified(record)

    def checkpoint(self, state, *args, **kwargs):
        # Initial CPU tensor artifacts and any other non105 save retain their
        # exact payload. Controller/native guards own checkpoint-format refusal.
        if not isinstance(state, dict) or state.get('format_version') != 105:
            return self.original_save(state, *args, **kwargs)
        if type(state['format_version']) is not int:
            raise ValueError('Format105 must be a strict integer')
        if 'full151_controller' not in state:
            raise ValueError('Capacity checkpoint evidence requires the final format105 serialization boundary')
        if 'full151_runtime_evidence' not in state or 'full151_initial_parameter_fingerprints' not in state:
            raise ValueError('Full151 runtime/fingerprint serialization must precede capacity evidence')
        if 'full151_capacity_evidence' in state:
            raise ValueError('Duplicate capacity checkpoint evidence')
        if len(self.entries) != 1 or self.entries[0]['observer'].finalized:
            raise ValueError('Checkpoint requires exactly one active, non-finalized Env observer')
        observer = self.entries[0]['observer']
        record = observer._read('checkpoint')
        observer._raise_if_unqualified(record)
        evidence = self._snapshot(self.entries[0], kind='checkpoint_prefix')
        evidence['checkpoint_iteration'] = state.get('iteration')
        evidence['checkpoint_sim_steps'] = state.get('sim_steps')
        return self.original_save({**state, 'full151_capacity_evidence': evidence}, *args, **kwargs)

    def __enter__(self):
        if getattr(self.manager_class, '_full151_training_capacity_scope', False):
            raise ValueError('Duplicate training capacity class scope')
        self.original_init = self.manager_class.__init__
        self.original_save = self.torch.save
        self.original_runner_init = self.runner_class.__init__ if self.runner_class is not None else None
        scope = self
        def initialize(env, cfg, *args, **kwargs):
            scope.original_init(env, cfg, *args, **kwargs)
            scope._install(env, cfg)
        self.manager_class._full151_training_capacity_scope = True
        self.manager_class.__init__ = initialize
        self.torch.save = self.checkpoint
        if self.runner_class is not None:
            def runner_initialize(runner, *args, **kwargs):
                scope.original_runner_init(runner, *args, **kwargs)
                modules = {}
                if set(runner.modules()) != {'wm', 'fail_head', 'actor', 'critic'}:
                    raise ValueError('Exactly four native from-scratch modules required')
                for name, module in runner.modules().items():
                    named = list(module.named_parameters())
                    modules[name] = {'parameter_names': [key for key, _ in named],
                                     'parameters': [value.detach().cpu().clone() for _, value in named],
                                     'shapes': [list(value.shape) for _, value in named],
                                     'dtypes': [str(value.dtype) for _, value in named]}
                scope.torch.save({'schema': 'full151_native_initial_parameters:1', 'modules': modules,
                                 'capture_point': 'after_native_DexRunner_init_before_first_collect_or_optimizer',
                                 'source_pins': copy.deepcopy(scope.source_pins)},
                                scope.output / 'initial_parameters.pt')
            self.runner_class.__init__ = runner_initialize
        return self

    def __exit__(self, exc_type, exc, tb):
        cleanup_error = None
        try:
            for entry in self.entries:
                try:
                    if not entry['closed']:
                        entry['env'].close()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
                finally:
                    env = entry['env']
                    if entry['close_owned']:
                        env.close = entry['close_previous']
                    else:
                        delattr(env, 'close')
                    try:
                        entry['context'].__exit__(exc_type, exc, tb)
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = error
                    finally:
                        entry['context_exited'] = True
        finally:
            self.manager_class.__init__ = self.original_init
            delattr(self.manager_class, '_full151_training_capacity_scope')
            self.torch.save = self.original_save
            if self.runner_class is not None:
                self.runner_class.__init__ = self.original_runner_init
        if exc is None and cleanup_error is not None:
            raise cleanup_error
        return False

    def qualified_final(self):
        return (len(self.entries) == 1 and self.entries[0]['closed']
                and self.entries[0]['observer'].finalized
                and self.entries[0]['observer'].status == 'OK'
                and not self.finalization_errors)

