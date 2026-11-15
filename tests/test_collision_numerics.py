import unittest
from types import SimpleNamespace
import numpy as np
import warp as wp

from supermimic_research.runtime.collision_numerics import validate_collision_numerics,set_collision_numerics,configure_collision_numerics


class CollisionNumericsTests(unittest.TestCase):
    def test_configuration_hook_precedes_device_creation_and_is_instance_local(self):
        class Config:
            ccd_iterations=50
            def apply(self,m):
                m.opt.ccd_iterations=self.ccd_iterations;m.opt.ccd_tolerance=1e-6;m.opt.iterations=50
        cfg=Config();other=Config();model=SimpleNamespace(opt=SimpleNamespace())
        configure_collision_numerics(cfg,{'ccd_iterations':50,'ccd_tolerance':1e-7})
        cfg.apply(model)
        self.assertEqual(model.opt.ccd_tolerance,1e-7);self.assertEqual(model.opt.iterations,50)
        other.apply(model);self.assertEqual(model.opt.ccd_tolerance,1e-6)
        with self.assertRaises(ValueError):configure_collision_numerics(other,{'ccd_iterations':50})
        other.apply(model);self.assertEqual(model.opt.ccd_tolerance,1e-6)

    def test_invalid_schema_or_values_cannot_partially_change_model(self):
        wp.init();host=SimpleNamespace(opt=SimpleNamespace(ccd_iterations=50,ccd_tolerance=1e-6))
        device=SimpleNamespace(opt=SimpleNamespace(ccd_iterations=50,ccd_tolerance=wp.array([1e-6],dtype=wp.float32,device='cpu')))
        for opts in [{},{'ccd_tolerance':1e-7},{'ccd_iterations':50,'ccd_tolerance':1e-7,'geom_friction':.8},
                     {'ccd_iterations':2.5,'ccd_tolerance':1e-7},{'ccd_iterations':True,'ccd_tolerance':1e-7},
                     {'ccd_iterations':50,'ccd_tolerance':float('nan')},{'ccd_iterations':50,'ccd_tolerance':0}]:
            with self.assertRaises(ValueError):set_collision_numerics(host,device,opts)
            self.assertEqual(host.opt.ccd_iterations,50);self.assertEqual(host.opt.ccd_tolerance,1e-6)
            self.assertEqual(device.opt.ccd_iterations,50)
            np.testing.assert_array_equal(device.opt.ccd_tolerance.numpy(),np.array([1e-6],np.float32))

    def test_explicit_observed_float32_receipt_leaves_other_parameters_unchanged(self):
        host=SimpleNamespace(opt=SimpleNamespace(ccd_iterations=50,ccd_tolerance=1e-6,iterations=50),body_mass=np.array([.03]))
        device=SimpleNamespace(opt=SimpleNamespace(ccd_iterations=50,ccd_tolerance=wp.array([1e-6],dtype=wp.float32,device='cpu'),iterations=50),body_mass=np.array([.03]))
        r=set_collision_numerics(host,device,{'ccd_iterations':50,'ccd_tolerance':1e-7})
        self.assertEqual(r['before']['host']['ccd_tolerance'],1e-6)
        self.assertEqual(r['after']['host']['ccd_tolerance'],1e-7)
        self.assertEqual(r['after']['GPU']['ccd_tolerance'],float(np.float32(1e-7)))
        self.assertEqual(host.opt.iterations,50);self.assertEqual(device.opt.iterations,50)
        np.testing.assert_array_equal(host.body_mass,[.03]);np.testing.assert_array_equal(device.body_mass,[.03])
