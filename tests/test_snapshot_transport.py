import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from supermimic_research.runtime.snapshot_transport import pack_bytes,unpack_bytes,save_owned,load_owned
from supermimic_research.learning.world_model import raw_tree_equal


class SnapshotTransportTests(unittest.TestCase):
    def test_byte_exact_deterministic_transport_and_no_overwrite(self):
        payload=b'snapshot\x00\xff'*1000
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.gz',Path(tmp)/'b.gz';r=pack_bytes(a,payload);pack_bytes(b,payload)
            self.assertEqual(unpack_bytes(r),payload);self.assertEqual(a.read_bytes(),b.read_bytes())
            with self.assertRaises(ValueError):pack_bytes(a,b'new')

    def test_bad_size_hash_or_compressed_bytes_reject_before_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.gz';r=pack_bytes(p,b'x'*100)
            for kind in ['size','hash','compressed']:
                bad=copy.deepcopy(r)
                if kind=='size':bad['uncompressed_bytes']=99
                elif kind=='hash':bad['uncompressed_sha256']='0'*64
                else:p.write_bytes(p.read_bytes()+b'changed')
                with patch('supermimic_research.runtime.snapshot_transport.torch.load',side_effect=AssertionError('deserialized')):
                    with self.assertRaises(ValueError):load_owned(bad)

    def test_native_schema_tensor_array_and_tuple_types_roundtrip(self):
        state=dict(schema='controlled_pen_owned_state:4',worlds=1,metadata=dict(fixed_frame_features='LIVE'),data={('array',1):np.arange(12,dtype=np.float32).reshape(3,4),('tensor',2):torch.arange(8,dtype=torch.int64)[::2]})
        with tempfile.TemporaryDirectory() as tmp:
            record=save_owned(Path(tmp)/'source.gz',state);after=load_owned(record);self.assertTrue(raw_tree_equal(state,after));self.assertEqual(after['data'][('tensor',2)].stride(),state['data'][('tensor',2)].stride())
        bad=copy.deepcopy(state);bad['schema']='owned_backend_actor_probe_source:1'
        with self.assertRaises(ValueError):save_owned(Path('unused.gz'),bad)


if __name__=='__main__':unittest.main()
