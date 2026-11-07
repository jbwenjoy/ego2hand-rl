"""Real compiled geometry: inference cache warm-up followed by CPU input AD.

No simulator, neural module, optimizer, training or CUDA calls. The two saved
contexts are mechanical baseline IDs 0 and 3, not selected by their values.
"""
from pathlib import Path
import json
import os
import sys
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import ArtifactRegistry
from supermimic_research.runtime.directional import DirectionalGapProvider, validate_features as validate_directional
from supermimic_research.runtime.geometry import validate_features as validate_geometry
from supermimic_research.runtime.phone_task import PhoneTask


def cached_tensors(provider):
    """Only persistent device/dtype buffers, excluding source or state tensors."""
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values(): yield from tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value: yield from tensors(item)
    for cache in (provider._constant_cache, provider.provider._constant_cache,
                  provider.provider._array_cache):
        for values, _ in cache._buckets.values(): yield from tensors(values)
    yield from tensors(provider._support_cache)


class GeometryInferenceTests(unittest.TestCase):
    measurements = []

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.registry = ArtifactRegistry(ROOT / 'configs/artifacts.toml')
        cls.catalog_path = cls.registry.resolve('geometry_catalog')
        catalog = json.loads(cls.catalog_path.read_text(encoding='utf8'))
        cls.metadata_path = cls.registry.resolve_pin(catalog['provider_metadata'])
        metadata = json.loads(cls.metadata_path.read_text(encoding='utf8'))
        cls.geometry_arrays_path = cls.registry.resolve_pin(metadata['geometry_arrays'])
        cls.direction_arrays_path = cls.registry.resolve_pin(catalog['direction_arrays'])
        report = json.loads(cls.registry.resolve('action_response').read_text(encoding='utf8'))
        rows = json.loads(cls.registry.resolve_pin(report['branches']).read_text(encoding='utf8'))
        selected = [rows[i] for i in (0, 3)]
        if not all(row['kind'] == 'baseline' and row['status'] == 'EXECUTED' and row['labels_known'] for row in selected):
            raise ValueError('Fixed saved baseline IDs 0 and 3 required')
        xs, us = [], []
        for row in selected:
            with np.load(cls.registry.resolve_pin(row['output']), allow_pickle=False) as z:
                xs.append(z['source_state58_F64'].copy())
                us.append(z['actual_control20_F64'].copy())
        cls.states = np.stack(xs)
        cls.controls = np.stack(us)
        cls.hs = np.array([row['dt_s'] for row in selected])
        cls.ns = np.array([row['n_substeps'] for row in selected])

    def check_cache_then_AD(self, dtype):
        self.assertFalse(torch.cuda.is_initialized())
        with torch.inference_mode():
            geometry = DirectionalGapProvider(self.catalog_path,
                metadata_path=self.metadata_path,
                geometry_arrays_path=self.geometry_arrays_path,
                direction_arrays_path=self.direction_arrays_path)
            warm_state = torch.tensor(self.states, dtype=dtype, device='cpu')
            warm = geometry.bundle(warm_state, retain_all_directions=True)
            warm_poses = warm.poses()
            warm_descriptor = warm.directional(return_all_directions=True)
            self.assertTrue(warm_poses['geom_position_world_m'].is_inference())
            self.assertTrue(warm_descriptor['directional_gap_m'].is_inference())
            buffers = list(cached_tensors(geometry))
            self.assertGreater(len(buffers), 0)
            self.assertTrue(all(not t.is_inference() and not t.requires_grad for t in buffers))
            identities = [id(t) for t in buffers]

        # The graph itself is never shared across inference/grad lifecycles.
        with self.assertRaises(ValueError): warm.poses()
        state = torch.tensor(self.states, dtype=dtype, device='cpu', requires_grad=True)
        control = torch.tensor(self.controls, dtype=dtype, device='cpu', requires_grad=True)
        h = torch.tensor(self.hs, dtype=dtype, device='cpu')
        n = torch.tensor(self.ns, dtype=torch.int64, device='cpu')
        context = geometry.bundle(state)
        poses = context.poses()
        feature = geometry.provider.features(state, control, h, n, context=context)
        descriptor = geometry.features(state, context=context)
        validate_geometry(feature); validate_directional(descriptor)
        self.assertIs(poses, context.poses())
        self.assertIs(descriptor, context.directional())
        self.assertEqual(identities, [id(t) for t in cached_tensors(geometry)])
        self.assertTrue(all(not t.is_inference() and not t.requires_grad for t in cached_tensors(geometry)))
        weights = torch.linspace(.1, 1., 22, dtype=dtype)
        scalar = (poses['geom_position_world_m'].square().sum()
            + feature['pair_relative_translation_BODY_COM_m'].square().sum()
            + (descriptor['directional_gap_m'] * weights).sum()
            + .01 * feature['absolute_u20'].square().sum())
        state_gradient, control_gradient = torch.autograd.grad(scalar, (state, control))
        self.assertTrue(torch.isfinite(state_gradient).all())
        self.assertTrue(torch.isfinite(control_gradient).all())
        self.assertGreater(float(state_gradient[..., :20].abs().max()), 0.)
        self.assertTrue(torch.equal(control_gradient, .02 * control.detach()))
        self.assertEqual(geometry.provider.counts['pure_Torch_FK_calls'], 2)
        self.assertEqual(geometry.counts['descriptor_calls'], 2)
        self.assertFalse(torch.cuda.is_initialized())
        self.measurements.append(dict(dtype=str(dtype), contexts=2,
            cached_normal_buffers=len(buffers), cached_tensor_ids_unchanged=True,
            inference_FK_calls=1, grad_FK_calls=1, input_AD_calls=1,
            q_gradient_max_abs=float(state_gradient[..., :20].abs().max()),
            direction_count=descriptor['direction_count'], all22_known=bool(descriptor['directional_gap_known'].all()),
            no_cached_inference_tensor=True, finite_state_and_control_input_AD=True))

    def test_F64_inference_initialization_and_cache_then_input_AD(self):
        self.check_cache_then_AD(torch.float64)

    def test_F32_inference_initialization_and_cache_then_input_AD(self):
        self.check_cache_then_AD(torch.float32)

    def test_tracking_task_without_owned_geometry_accepts_current_context(self):
        path = self.registry.resolve('phone_task')
        bundle = json.loads(path.read_text(encoding='utf8'))
        task = PhoneTask(path, mode='A1_tracking', geometry=None,
            reference_arrays_path=self.registry.resolve_pin(bundle['reference_arrays']))
        geometry = DirectionalGapProvider(self.catalog_path,
            metadata_path=self.metadata_path,
            geometry_arrays_path=self.geometry_arrays_path,
            direction_arrays_path=self.direction_arrays_path)
        state = torch.tensor(task.b['initial']['native_PRE_state58_F64'], dtype=torch.float64)
        memory = task.initial_memory(state, native_h_s=.0003125, n_substeps=64)
        action = torch.zeros(20, dtype=torch.float64)
        expected = task.rewards(state, memory, action)
        result = task.rewards(state, memory, action, context=geometry.bundle(state))
        self.assertTrue(torch.equal(expected['total'], result['total']))
        self.assertTrue(torch.equal(expected['objective_known'], result['objective_known']))
        self.assertEqual(geometry.provider.counts['pure_Torch_FK_calls'], 0)
        self.assertEqual(geometry.counts['descriptor_calls'], 0)


if __name__ == '__main__':
    unittest.main()
