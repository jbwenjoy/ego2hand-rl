import unittest
import numpy as np
import warp as wp
from supermimic_research.runtime.native_pen_planner import _clear_response,_response_heads


class ResponseKernelTests(unittest.TestCase):
    def test_named_finger_palm_normal_groups_and_actual_warmstart_are_recorded(self):
        wp.init();device='cpu';a=lambda v,dtype:wp.array(v,dtype=dtype,device=device)
        slot=a([1],wp.int32);warm=a(np.arange(26,dtype=np.float32)[None],wp.float32)
        geom=a(np.array([[0,1],[2,0],[0,3],[0,4],[1,2]],np.int32),wp.vec2i)
        address=np.full((5,10),-1,np.int32);address[:,0]=np.arange(5)
        Fn=a(np.full((2,1,6),9,np.float32),wp.float32);Warm=wp.zeros((2,1,26),device=device);bad=wp.zeros((2,1),dtype=wp.int32,device=device)
        wp.launch(_clear_response,6,inputs=[slot,0,Fn],device=device)
        wp.launch(_response_heads,26,inputs=[slot,0,warm,a([5],wp.int32),geom,a(np.ones(5,np.int32),wp.int32),a(address,wp.int32),
            a(np.array([[1,2,3,4,5]],np.float32),wp.float32),a([5],wp.int32),a([2,1,1,1,0],wp.int32),a([-1,0,4,5,-1],wp.int32),Warm,Fn,bad],device=device)
        np.testing.assert_array_equal(Fn.numpy()[1,0],[1,0,0,0,2,3]);np.testing.assert_array_equal(Fn.numpy()[0,0],9)
        np.testing.assert_array_equal(Warm.numpy()[1,0],np.arange(26));self.assertFalse(bad.numpy().any())
