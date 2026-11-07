"""Saved-only tracking and frozen-math checks, no model or native execution."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry
from supermimic_research.runtime.phone_task import PhoneTask
from supermimic_research.runtime import state_codec


def load_tool():
    spec=importlib.util.spec_from_file_location('online_eval_test',ROOT/'tools/evaluate_online.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class TrackingTests(unittest.TestCase):
    def test_original_tracking_math_AST(self):
        old=ast.parse((ROOT/'archive/studies/ego-data-pilot-2026-11-07/phone_tripod_policy_evaluation_v1/evaluate_learned_actors_v2.py').read_text('utf8'))
        new=ast.parse((ROOT/'tools/evaluate_online.py').read_text('utf8'))
        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Name) and node.value.id=='C' and node.attr=='need':return ast.Name(id='need',ctx=ast.Load())
                return self.generic_visit(node)
            def visit_FunctionDef(self,node):
                node=self.generic_visit(node)
                if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
                return node
        for name in ('angle_rad','tracking_from_saved'):
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(Normalize().visit(a)),ast.dump(Normalize().visit(b)))

    def test_actual_saved_failed_case_retains_integer_POST_masks_and_NaNs(self):
        registry=ArtifactRegistry(ROOT/'configs/artifacts.toml')
        old=registry.roots['ego']/'phone_tripod_policy_evaluation_v1/learned_fixed1000_run2/report.json'
        if not old.exists():self.skipTest('Requires the saved fixed1000 native evaluation fixture')
        report=json.loads(old.read_text('utf8'));case=report['cases'][0]
        bundle_path=registry.resolve('phone_task');bundle=json.loads(bundle_path.read_text('utf8'))
        task=PhoneTask(bundle_path,mode='A1_tracking',reference_arrays_path=registry.resolve_pin(bundle['reference_arrays']))
        geometry_catalog=json.loads(registry.resolve('geometry_catalog').read_text('utf8'))
        metadata=json.loads(registry.resolve_pin(geometry_catalog['provider_metadata']).read_text('utf8'))
        arrays,statistics=load_tool().tracking_from_saved(registry.resolve_pin(case['native_summary']['trace']),task,state_codec,np.array(metadata['phone_body_ipos']),case['dt_s'])
        with np.load(registry.resolve_pin(case['tracking_arrays']),allow_pickle=False) as expected:
            self.assertEqual(set(arrays),set(expected.files))
            for key,value in arrays.items():
                self.assertEqual(value.dtype,expected[key].dtype)
                self.assertEqual(value.shape,expected[key].shape)
                self.assertEqual(value.tobytes(),expected[key].tobytes(),key)
        self.assertEqual(statistics,case['tracking'])
        self.assertEqual(statistics['available_POST_command_frames'],6)
        self.assertEqual(statistics['missing_POST_command_frames'],94)
        self.assertTrue(np.isnan(arrays['state58_F64'][~arrays['known101']]).all())


if __name__=='__main__':unittest.main()
