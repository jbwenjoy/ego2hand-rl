import unittest
import numpy as np
from supermimic_research.ego.camera_projection import fisheye624,decode_hot3d_mask,rasterize,compare_masks
from supermimic_research.ego.vendor_camera.camera import OVR624CameraModel


class EgoCameraProjectionTests(unittest.TestCase):
    def test_matches_pinned_reference_radial_tangent_prism(self):
        rng=np.random.default_rng(443);xyz=rng.normal(size=(1000,3));xyz[:,2]=abs(xyz[:,2])+1.;xyz=np.r_[xyz,[[0.,0.,1.]]]
        params=np.r_[[610.,704.,701.],rng.normal(0,.001,12)];reference=OVR624CameraModel(1408,1408,params[0],params[1:3],params[3:])
        np.testing.assert_allclose(fisheye624(xyz,params),reference.eye_to_window(xyz),atol=1e-10,rtol=0)

    def test_positive_depth_and_explicit_parameters_required(self):
        for p,k in [(np.array([[0,0,0]]),np.ones(15)),(np.array([[0,0,-1]]),np.ones(15)),(np.array([[0,0,1]]),np.ones(16))]:
            with self.assertRaises(ValueError):fisheye624(p,k)

    def test_one_based_row_major_mask_and_empty(self):
        a=decode_hot3d_mask(dict(height=2,width=3,rle=[1,2,5,2]));np.testing.assert_array_equal(a,[[1,1,0],[0,1,1]])
        self.assertFalse(decode_hot3d_mask(dict(height=2,width=3,rle=[])).any())

    def test_invalid_rle_rejected_instead_of_clipped(self):
        for rle in [[0,1],[1,0],[6,2],[1,3,2,1],[1],[1.,2.]]:
            with self.subTest(rle=rle),self.assertRaises(ValueError):decode_hot3d_mask(dict(height=2,width=3,rle=rle))

    def test_masks_measure_actual_missing_and_extra_pixels(self):
        a=np.zeros((12,12),bool);a[2:6,3:7]=True;b=np.roll(a,1,axis=1);r=compare_masks(a,b)
        self.assertEqual(r['missed_pixels'],4);self.assertEqual(r['extra_pixels'],4);self.assertEqual(r['IoU'],12/20);self.assertEqual(r['symmetric_boundary_max_px'],1.)

    def test_rasterization_union_invariant_to_triangle_order(self):
        uv=np.array([[1.,1.],[4.,1.],[1.,4.],[4.,4.]]);f=np.array([[0,1,2],[1,2,3]])
        a=rasterize(uv,f,(6,6));np.testing.assert_array_equal(a,rasterize(uv,f[::-1],(6,6)));self.assertTrue(a[2,2]);self.assertFalse(a[0,0])
