import unittest
import numpy as np
from supermimic_research.mechanics.pen_solver_tick_balance import mass_product, partition, terminal_predicates


class SolverTickBalance(unittest.TestCase):
    def example(self):
        a=np.array([[1.,0.,0.]])
        fields=dict(qacc=a,**{'efc.Ma':.03*a},qfrc_smooth=np.zeros_like(a),qfrc_constraint=.02*a)
        ctx=dict(grad=.002*a,grad_scale=np.array([5.]))
        return .03,.0025,0,np.zeros(3),fields,ctx,.03*a

    def test_scaled_context_gradient_explains_force_residual(self):
        r=partition(*self.example())
        np.testing.assert_allclose(r['actual_scaled_gradient'],[[.000025,0,0]])
        np.testing.assert_allclose(r['total_acceleration_force_Ns'],r['actual_scaled_gradient'])
        self.assertTrue(r['recomposition_within_float64'])

    def test_zero_gradient_can_coexist_with_Ma_accumulation_defect(self):
        args=self.example();args[4]['efc.Ma']+=np.array([[.1,0,0]])
        args[4]['qfrc_constraint']=args[4]['efc.Ma'].copy();args[5]['grad']*=0
        r=partition(*args)
        np.testing.assert_allclose(r['stored_Ma_balance_N'],0)
        np.testing.assert_allclose(r['actual_scaled_gradient'],0)
        np.testing.assert_allclose(r['Ma_accumulation'],[[-.00025,0,0]])
        np.testing.assert_allclose(r['total_acceleration_force_Ns'],r['Ma_accumulation'])

    def test_all_five_terms_remain_distinct_and_recompose(self):
        rng=np.random.default_rng(47);shape=(8,26)
        fields={k:rng.normal(size=shape) for k in ['qacc','efc.Ma','qfrc_smooth','qfrc_constraint']}
        ctx=dict(grad=rng.normal(size=shape),grad_scale=rng.normal(size=8))
        r=partition(.03,.0025,20,[0,0,-9.81],fields,ctx,rng.normal(size=shape))
        self.assertTrue(r['recomposition_within_float64'])

    def test_dense_and_packed_mass_products_match(self):
        m=np.array([[[2.,.5],[.5,3.]]]);a=np.array([[1.,2.]])
        packed=np.array([[2.,.5,3.]])
        np.testing.assert_array_equal(mass_product(packed,a,[0,2,4],[0,1,0,1],[0,1,1,2]),mass_product(m,a))

    def test_invalid_mapping_and_context_reject(self):
        with self.assertRaises(ValueError):mass_product([[1.,2.]],[[1.,2.]], [0,1,2],[0,2],[0,1])
        args=self.example();args[5]['grad_scale']=np.ones((1,1))
        with self.assertRaises(ValueError):partition(*args)
        args=self.example();args[4]['qacc'][0,0]=np.nan
        with self.assertRaises(ValueError):partition(*args)

    def test_terminal_OR_predicates_do_not_select_one_cause(self):
        fields=dict(grad_dot=[0.,1.],newton_decrement=[0.,1.],improvement=[1.,0.],solver_niter=[2,50])
        r=terminal_predicates(26,.0036,1e-12,50,fields)
        self.assertTrue(r['gradient_below_tolerance'][0])
        self.assertTrue(r['newton_decrement_below_tolerance'][0])
        self.assertTrue(r['improvement_below_tolerance'][1])
        self.assertTrue(r['at_iteration_cap'][1])


if __name__=='__main__':unittest.main()
