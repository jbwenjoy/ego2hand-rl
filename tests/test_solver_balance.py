import unittest
import numpy as np
from supermimic_research.mechanics.solver_balance import linear_solver_balance


class SolverBalance(unittest.TestCase):
    def example(self,acc):
        mass=.03;dt=.0025;gravity=np.array([0.,0.,-9.81]);a=np.tile(acc,(8,1));zero=np.zeros_like(a)
        fields={k:zero.copy() for k in ['qacc','qacc_smooth','qfrc_smooth','qfrc_constraint','qfrc_bias','qfrc_passive','qfrc_actuator','efc.Ma']}
        fields['qacc']=a;fields['efc.Ma']=mass*a;fields['qfrc_smooth'][:]=mass*gravity;fields['qfrc_bias'][:]=-mass*gravity
        packet=dict(pre_qvel=zero.copy(),post_qvel=dt*a)
        return packet,fields,mass,gravity,0,dt

    def test_balanced_free_fall(self):
        result=linear_solver_balance(*self.example([0.,0.,-9.81]))
        for value in result.values():np.testing.assert_allclose(value,0,atol=1e-16)

    def test_acceleration_force_defect_is_distinct_from_velocity_update(self):
        result=linear_solver_balance(*self.example([.03,0.,-9.81]))
        np.testing.assert_allclose(result['velocity_update_residual_m_s'],0,atol=1e-16)
        np.testing.assert_allclose(result['mass_qacc_minus_smooth_constraint_N'][:,0],.0009)
        np.testing.assert_array_equal(result['mass_qacc_minus_smooth_constraint_N'],result['public_Ma_balance_proxy_N'])

    def test_Ma_proxy_does_not_assume_equal_mass_qacc(self):
        args=self.example([0.,0.,-9.81]);args[1]['efc.Ma'][:,0]=.001
        result=linear_solver_balance(*args)
        np.testing.assert_allclose(result['Ma_minus_mass_qacc_N'][:,0],.001)
        np.testing.assert_allclose(result['public_Ma_balance_proxy_N'][:,0],.001)
        np.testing.assert_allclose(result['mass_qacc_minus_smooth_constraint_N'],0,atol=1e-16)

    def test_invalid_or_misaligned_fields_reject(self):
        for bad in [np.zeros((7,3)),np.full((8,3),np.nan)]:
            args=self.example([0.,0.,-9.81]);args[1]['qfrc_smooth']=bad
            with self.assertRaises(ValueError):linear_solver_balance(*args)
