import dataclasses
import unittest

import numpy as np
import warp as wp

from supermimic_research.runtime.gpu_data_snapshot import ResidentDataSnapshot


@dataclasses.dataclass
class Scratch:
    values:object
    count:int


@dataclasses.dataclass
class DynamicData:
    qpos:object
    scratch:Scratch
    optional:object=None


class ResidentDataTests(unittest.TestCase):
    def test_all_nested_bytes_including_unused_nan_and_empty_fields_restore(self):
        wp.init()
        values=np.array([1,np.nan,-0.],dtype=np.float32)
        data=DynamicData(wp.array(values,device='cpu'),Scratch(wp.empty(0,dtype=wp.int32,device='cpu'),1))
        pointers=(data.qpos.ptr,data.scratch.values.ptr);snap=ResidentDataSnapshot(data)
        data.qpos.zero_();self.assertFalse(snap.audit(data)['all_array_bytes_equal'])
        snap.restore(data);audit=snap.audit(data)
        self.assertTrue(audit['all_array_bytes_equal']);self.assertTrue(audit['scalar_equal'])
        self.assertEqual(pointers,(data.qpos.ptr,data.scratch.values.ptr))
        self.assertEqual(data.qpos.numpy().tobytes(),values.tobytes())

    def test_metadata_failure_cannot_partially_overwrite_state(self):
        data=DynamicData(wp.array([1.,2.],device='cpu'),Scratch(wp.array([3.],device='cpu'),1))
        snap=ResidentDataSnapshot(data);data.qpos.zero_();data.scratch.count=2
        with self.assertRaises(ValueError):snap.restore(data)
        np.testing.assert_array_equal(data.qpos.numpy(),[0,0])
        data.scratch.count=1
        data.scratch.values=wp.array([3.],device='cpu')
        with self.assertRaises(ValueError):snap.restore(data)
        np.testing.assert_array_equal(data.qpos.numpy(),[0,0])
        data.scratch.count=1;data.scratch.values=wp.array([3.,4.],device='cpu')
        with self.assertRaises(ValueError):snap.restore(data)
        np.testing.assert_array_equal(data.qpos.numpy(),[0,0])
