import copy
from enum import Enum
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from supermimic_research.runtime.shared_snapshot_transport import SharedImmutableTransport,load_shared,tree_digest
from supermimic_research.learning.world_model import raw_tree_equal


class ExampleMode(Enum):
    LIVE='LIVE'


class SharedSnapshotTests(unittest.TestCase):
    def fixture(self):
        return dict(schema='controlled_pen_owned_state:4',worlds=1,metadata=dict(fixed_frame_features='LIVE'),native=dict(model=dict(arrays=dict(mass=np.array([.03],np.float32))),data=dict(qpos=np.arange(4))),
            python=dict(tensors={('static',):torch.arange(4,dtype=torch.float32),('dynamic',):torch.ones(3)},arrays={('tag',):np.arange(3)},scalars={('ready',):True,('phase',):1},description={}),immutable={('static',):'fixed'},immutable_arrays={('tag',):'fixed'},immutable_scalars={('ready',):True},config=dict(path=Path('example/config.toml'),mode=ExampleMode.LIVE,joints=slice(1,4),tags={'qpos','qvel'}))

    def test_two_states_share_only_static_fields_and_reconstruct_all_types(self):
        a=self.fixture();b=copy.deepcopy(a);b['python']['scalars'][('phase',)]=2;b['native']['data']['qpos']+=2
        with tempfile.TemporaryDirectory() as tmp:
            store=SharedImmutableTransport(Path(tmp)/'static.gz');x=store.save(Path(tmp)/'a.gz',a);y=store.save(Path(tmp)/'b.gz',b)
            self.assertEqual(x['shared'],y['shared']);self.assertTrue(raw_tree_equal(a,load_shared(x)));self.assertTrue(raw_tree_equal(b,load_shared(y)));self.assertEqual(tree_digest(a),x['full_tree_sha256'])

    def test_immutable_change_and_corrupt_shared_payload_fail_before_deserialization(self):
        a=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            store=SharedImmutableTransport(Path(tmp)/'static.gz');record=store.save(Path(tmp)/'a.gz',a);bad=copy.deepcopy(a);bad['native']['model']['arrays']['mass']+=1
            with self.assertRaises(ValueError):store.save(Path(tmp)/'bad.gz',bad)
            for part in ['shared','payload']:
                r=copy.deepcopy(record);r[part]['uncompressed_sha256']='0'*64
                with patch('supermimic_research.runtime.shared_snapshot_transport.torch.load',side_effect=AssertionError('deserialized')):
                    with self.assertRaises(ValueError):load_shared(r)


if __name__=='__main__':unittest.main()
