"""Failed preflight never changes aliases, including expanded/vector views."""
import unittest
import numpy as np
import torch
import warp as wp
from supermimic_research.runtime.owned_array_writes import prepare_writes


class OwnedWriteTests(unittest.TestCase):
    def test_disjoint_cpu_tensor_numpy_and_scalar_write(self):
        a=torch.zeros(2,3);b=np.zeros(2,np.int32);c=np.array(0,np.int64)
        plan=prepare_writes([('a',a,torch.ones_like(a)),('b',b,np.array([4,5],np.int32)),('c',c,np.array(7,np.int64))])
        self.assertFalse(a.any())
        for p in plan:p.apply();self.assertTrue(p.equal())
        self.assertEqual(int(c),7)

    def test_dtype_shape_failure_does_not_apply_earlier_good_entry(self):
        a=torch.zeros(2);b=torch.zeros(2)
        for bad in [torch.ones(3),torch.ones(2,dtype=torch.float64)]:
            with self.assertRaisesRegex(ValueError,'Saved layout mismatch'):prepare_writes([('a',a,torch.ones(2)),('b',b,bad)])
            self.assertFalse(a.any());self.assertFalse(b.any())

    def test_conflicting_overlapping_slices_are_rejected_before_copy(self):
        a=torch.zeros(5);left=a[:4];right=a[1:]
        with self.assertRaisesRegex(ValueError,'Conflicting aliased'):prepare_writes([('left',left,torch.ones(4)),('right',right,torch.ones(4)*2)])
        self.assertFalse(a.any())
        plan=prepare_writes([('left',left,torch.tensor([1.,2,3,4])),('right',right,torch.tensor([2.,3,4,5]))])
        for p in plan:p.apply()
        torch.testing.assert_close(a,torch.arange(1.,6))

    def test_expanded_zero_stride_payload_requires_identical_bytes(self):
        a=torch.zeros(1,2);expanded=a.expand(3,2)
        values=torch.ones(3,2);values[1,0]=2
        with self.assertRaisesRegex(ValueError,'Conflicting expanded'):prepare_writes([('expanded',expanded,values)])
        self.assertFalse(a.any())
        plan=prepare_writes([('expanded',expanded,torch.ones(3,2))])
        plan[0].apply();self.assertTrue(plan[0].equal())

    def test_noncontiguous_transpose_and_numpy_aliases(self):
        a=np.zeros((2,3),np.float32);t=torch.from_numpy(a);value=np.arange(6,dtype=np.float32).reshape(2,3)
        plan=prepare_writes([('numpy',a,value),('transpose',t.T,torch.from_numpy(value.T.copy()))])
        for p in plan:p.apply();self.assertTrue(p.equal())
        np.testing.assert_array_equal(a,value)

    def test_nan_aliases_use_exact_payload_bytes(self):
        bits=np.array([0x7fc00001],np.uint32);a=np.zeros(1,np.uint32).view(np.float32);t=torch.from_numpy(a)
        values=bits.view(np.float32);prepare_writes([('np',a,values),('torch',t,torch.from_numpy(values.copy()))])
        with self.assertRaisesRegex(ValueError,'Conflicting aliased'):prepare_writes([('np',a,values),('torch',t,torch.tensor([float('nan')]))])
        self.assertEqual(int(a.view(np.uint32)[0]),0)

    def test_Warp_vector_and_Torch_view_share_checked_byte_addresses(self):
        wp.init();a=wp.zeros(2,dtype=wp.vec3,device='cpu');t=wp.to_torch(a);v=np.arange(6,dtype=np.float32).reshape(2,3)
        plan=prepare_writes([('warp',a,v),('torch',t,torch.from_numpy(v.copy()))])
        for p in plan:p.apply();self.assertTrue(p.equal())
        bad=v.copy();bad[1,2]+=1
        with self.assertRaisesRegex(ValueError,'Conflicting aliased'):prepare_writes([('warp',a,v),('torch',t,torch.from_numpy(bad))])
        np.testing.assert_array_equal(a.numpy(),v)

    def test_expanded_Warp_world_axis_collapses_before_any_copy(self):
        wp.init();base=wp.zeros(2,dtype=wp.vec3,device='cpu')
        expanded=wp.array(ptr=base.ptr,dtype=wp.vec3,shape=(4,2),strides=(0,12),device='cpu',copy=False)
        value=np.broadcast_to(np.arange(6,dtype=np.float32).reshape(1,2,3),(4,2,3)).copy()
        plan=prepare_writes([('expanded_Warp',expanded,value),('base',base,value[0])])
        self.assertFalse(base.numpy().any())
        for p in plan:p.apply();self.assertTrue(p.equal())
        bad=value.copy();bad[1,0,0]+=1
        with self.assertRaisesRegex(ValueError,'Conflicting expanded'):prepare_writes([('expanded_Warp',expanded,bad)])
        np.testing.assert_array_equal(base.numpy(),value[0])

    def test_fully_expanded_Warp_scalar_has_one_owned_copy_element(self):
        wp.init();base=wp.zeros(1,dtype=float,device='cpu')
        expanded=wp.array(ptr=base.ptr,dtype=float,shape=(4,),strides=(0,),device='cpu',copy=False)
        plan=prepare_writes([('scalar_Warp',expanded,np.full(4,3,np.float32))]);plan[0].apply()
        self.assertTrue(plan[0].equal());self.assertEqual(base.numpy()[0],3)

    def test_readonly_destination_rejected_before_other_writes(self):
        a=np.zeros(2,np.float32);b=np.zeros(2,np.float32);b.flags.writeable=False
        with self.assertRaisesRegex(ValueError,'Readonly array'):prepare_writes([('good',a,np.ones(2,np.float32)),('readonly',b,np.ones(2,np.float32))])
        self.assertFalse(a.any());self.assertFalse(b.any())


if __name__=='__main__':unittest.main()
