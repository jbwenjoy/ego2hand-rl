"""Saved NumPy/AST collection-port parity; no Actor or native execution."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry
from supermimic_research.runtime import native_collection as N,native_labels as D,persistence as C


def legacy_module(name,path):
    """Only this test may file-import frozen historical implementations."""
    spec=importlib.util.spec_from_file_location('collection_parity_'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module);return module


def function_AST(path,name):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    return ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)


class NativeCollectionPortTests(unittest.TestCase):
    measurements=[]

    @classmethod
    def setUpClass(cls):
        cls.registry=ArtifactRegistry(ROOT/'configs/artifacts.toml')
        cls.dataset=json.loads(cls.registry.resolve('tripod_dataset').read_text())
        cls.old_adapter=legacy_module('adapter',cls.registry.resolve_pin(cls.dataset['sources']['adapter']))
        cls.old_exporter=legacy_module('exporter',cls.registry.resolve_pin(cls.dataset['sources']['exporter']))
        cls.old_decode=legacy_module('decode',cls.registry.resolve_pin(cls.dataset['sources']['frozen_decode_window']))
        cls.old_online=legacy_module('online',cls.registry.roots['ego']/'phone_tripod_online_collection_v1/collect_online_v3.py')
        cls.producer=json.loads(cls.registry.resolve_pin(cls.dataset['collector_report']).read_text())
        protocol=json.loads(cls.registry.resolve_pin(cls.producer['protocol']).read_text())
        cls.old_raw=legacy_module('raw_collector',cls.registry.resolve_pin(protocol['sources']['collector']))
        cls.bundle=json.loads(cls.registry.resolve_pin(cls.dataset['task_bundle']).read_text())
        cls.physical=json.loads(cls.registry.resolve_pin(cls.dataset['physical_parameters']).read_text())
        cls.identities=json.loads(cls.registry.resolve_pin(cls.producer['sample_identities']).read_text())
        cls.episodes=json.loads(cls.registry.resolve_pin(cls.producer['episode_records']).read_text())
        cls.old_math=cls.old_adapter.load_math();cls.new_math=D.load_math()
        cls.byid={r['sample_id']:r for r in cls.identities}

    def equal(self,a,b):
        if isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for key in a:self.equal(a[key],b[key])
        elif isinstance(a,np.ndarray):
            self.assertEqual(a.shape,b.shape);self.assertEqual(a.dtype,b.dtype)
            self.assertTrue(np.array_equal(a,b,equal_nan=True))
        elif isinstance(a,(list,tuple)):
            self.assertEqual(type(a),type(b));self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.equal(x,y)
        else:self.assertEqual(a,b)

    def row(self,ep,sid):
        identity=self.byid[sid]
        with np.load(self.registry.resolve_pin(identity['trace']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        with np.load(self.registry.resolve_pin(identity['decision']),allow_pickle=False) as z:decision={k:z[k].copy() for k in z.files}
        a.update(decision);a['complete_H_bool']=np.asarray(identity['complete_H20ms']);a['reward_scalar_F64']=a['reward_F64']
        a['task_done']=a['task_done_bool'];a['reader_done']=a['reader_done_bool']
        return a,identity

    def test_frozen_math_and_persistence_AST(self):
        checks=[]
        for name in ['need','bits','close','require_array','receipt','observed_window','canonical_row','analytic_tripod_semantic_fixture']:
            self.assertEqual(function_AST(self.old_adapter.__file__,name),function_AST(D.__file__,name));checks.append(name)
        self.assertEqual(function_AST(self.old_decode.__file__,'decode_window'),function_AST(D.__file__,'decode_window'));checks.append('decode_window')
        for name in ['need','pin','sha','same','clean','write','save','tensor_arrays','receipt_meta','PersistentRawSink']:
            self.assertEqual(function_AST(self.old_raw.__file__,name),function_AST(C.__file__,name));checks.append(name)
        self.assertEqual(function_AST(self.old_exporter.__file__,'stack_rows'),function_AST(D.__file__,'stack_rows'));checks.append('stack_rows')
        self.measurements.append(dict(test='frozen_math_and_persistence_AST',equal_functions_or_classes=len(checks),names=checks))

    def test_fixed_profile_rules_and_field_map(self):
        self.equal(N.profile_plan(),self.old_online.profile_plan())
        self.equal(N.RULES,self.old_online.RULES);self.equal(D.FIELD_MAP,self.old_exporter.FIELD_MAP)
        for name in ['profile_plan','tensor_tree_digest']:
            self.assertEqual(function_AST(N.__file__,name),function_AST(self.old_online.__file__,name))
        self.measurements.append(dict(test='fixed_plan_and_ABI',profiles=4,axes=20,intervals=20,planned_episodes=16,maximum_native_steps=30720))

    def test_fixed_TRAIN_complete_and_terminal_rows(self):
        cases=[(self.episodes[0],self.episodes[0]['interval_row_ids'][0]),
               (self.episodes[0],self.episodes[0]['interval_row_ids'][-1]),
               (self.episodes[1],self.episodes[1]['interval_row_ids'][0])]
        for ep,sid in cases:
            with self.subTest(sample_id=sid):
                self.assertEqual(ep['split'],'TRIPOD_TRAIN')
                a,identity=self.row(ep,sid)
                old=self.old_adapter.canonical_row(a,identity,ep,self.bundle,self.physical,self.old_math,pre_hold_alive=True)
                new=D.canonical_row(a,identity,ep,self.bundle,self.physical,self.new_math,pre_hold_alive=True)
                self.equal(old,new)
                online=dict(identity,split='ONLINE_TRAIN',data_role='ONLINE_TRAIN_ONLY_NEW_STAGE')
                r,note,alias=D.canonical_online_row(a,online,ep,self.bundle,self.physical,pre_hold_alive=True,maths=self.new_math)
                expected=dict(old[0],acquisition_done_bool=bool(identity['done']))
                self.equal(r,expected);self.equal(alias,old[2]);self.assertEqual(note['split'],'ONLINE_TRAIN')
                self.measurements.append(dict(test='saved_TRAIN_canonical_row',sample_id=sid,dt_s=identity['dt_s'],complete=r['complete_H_bool'],boundary_task_loss=r['task_hold_failed_boundary_POST_bool'],all_fields_exact=True,fields=len(r)))

    def test_saved_prefix_and_joint_coverage_semantics(self):
        self.equal(self.old_adapter.analytic_tripod_semantic_fixture(self.old_math),D.analytic_tripod_semantic_fixture(self.new_math))
        ep=self.episodes[0];a,identity=self.row(ep,ep['interval_row_ids'][0]);N=len(a['pre_tick'])-1
        prefix=dict(a);prefix['pre_tick']=a['pre_tick'][:N];prefix['post_state58']=a['post_state58'][:N]
        prefix['contact_rows24']=a['contact_rows24'][a['contact_rows24'][:,0]<=a['pre_tick'][N-1]]
        prefix['pre_contact_wrench_about_COM_world6']=a['pre_contact_wrench_about_COM_world6'][:N]
        self.equal(self.old_adapter.observed_window(prefix,a['source_state58'],identity['dt_s'],self.old_math),D.observed_window(prefix,a['source_state58'],identity['dt_s'],self.new_math))
        self.measurements.append(dict(test='saved_prefix_only_no_new_H_receipt_claim',actual_saved_prefix_ticks=N,simultaneous_coverage_not_marginal_product=True))

    def test_exclusive_fsync_writer_retains_partial_NaNs(self):
        parent=ROOT/'runs/migration_native_collection';parent.mkdir(parents=True,exist_ok=True)
        # Temporary cleanup is restricted to a generated directory inside this
        # explicit test workspace; no existing run is moved or removed.
        with tempfile.TemporaryDirectory(prefix='writer_',dir=parent) as folder:
            folder=Path(folder);self.assertTrue(folder.resolve().is_relative_to(parent.resolve()))
            a=dict(target_at_H_state58=np.full(58,np.nan),observed_prefix_target58=np.arange(58,dtype=np.float64),complete_H_bool=np.array(False))
            C.save(folder/'partial.npz',a)
            with np.load(folder/'partial.npz',allow_pickle=False) as z:self.equal(a,{k:z[k].copy() for k in z.files})
            with self.assertRaises(FileExistsError):C.save(folder/'partial.npz',a)
            C.write(folder/'partial.json',dict(complete=False,unknown=float('nan')))
            self.assertEqual(json.loads((folder/'partial.json').read_text()),dict(complete=False,unknown=None))
            with self.assertRaises(FileExistsError):C.write(folder/'partial.json',{})
        self.measurements.append(dict(test='exclusive_fsync_saved_writer',partial_NaNs_preserved=True,no_overwrite=True))


if __name__=='__main__':unittest.main()
