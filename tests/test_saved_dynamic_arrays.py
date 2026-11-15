import dataclasses
import unittest
import numpy as np
import warp as wp
from supermimic_research.runtime.saved_dynamic_arrays import restore_dynamic_arrays


@dataclasses.dataclass
class DataFixture:
    qpos:object
    scratch:object
    capacity:int=3


class SavedDynamicArraysTests(unittest.TestCase):
    def test_exact_nan_payload_and_pointers_without_invented_scalar_observation(self):
        wp.init();bits=np.array([0x7fc00001,0x80000000,0x7fc00100],dtype=np.uint32).view(np.float32)
        data=DataFixture(wp.zeros(3,device='cpu'),wp.empty(0,dtype=wp.int32,device='cpu'))
        pointer=data.qpos.ptr;r=restore_dynamic_arrays(data,dict(qpos=bits,scratch=np.empty(0,dtype=np.int32)))
        self.assertEqual(data.qpos.ptr,pointer);self.assertEqual(data.qpos.numpy().tobytes(),bits.tobytes())
        self.assertEqual(r['fresh_static_scalars'],dict(capacity=3));self.assertNotIn('original_scalars_exact',r)

    def test_late_dtype_or_shape_failure_cannot_partially_mutate(self):
        data=DataFixture(wp.array([1.,2.],device='cpu'),wp.array([3.],device='cpu'))
        for bad in [np.zeros(2,dtype=np.float32),np.zeros(1,dtype=np.float64)]:
            with self.assertRaises(ValueError):restore_dynamic_arrays(data,dict(qpos=np.array([9.,9.],np.float32),scratch=bad))
            np.testing.assert_array_equal(data.qpos.numpy(),[1.,2.])

    def test_missing_or_extra_field_is_rejected_before_write(self):
        data=DataFixture(wp.array([1.],device='cpu'),wp.array([2.],device='cpu'))
        for values in [dict(qpos=np.array([9.],np.float32)),dict(qpos=np.array([9.],np.float32),scratch=np.array([9.],np.float32),extra=np.array([0]))]:
            with self.assertRaises(ValueError):restore_dynamic_arrays(data,values)
            np.testing.assert_array_equal(data.qpos.numpy(),[1.])
