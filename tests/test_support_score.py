"""Physical-frame, zero-net and anti-squeeze score counterexamples."""
import unittest
import torch
from supermimic_research.mechanics.support_score import support_score
from supermimic_research.mechanics.pen_support_objective import objective_terms
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,CENTERED_SCHEMA,SCHEMA
from test_pen_support_objective import fixture


class CenteredSupportTests(unittest.TestCase):
    def test_zero_net_balanced_squeeze_is_zero_and_opposite_is_negative(self):
        t=torch.tensor([[0.,0,.3,0,.003,0]],dtype=torch.float64);valid=torch.tensor([True])
        for load in [0.,.05,.5,50.]:
            r=support_score(torch.zeros_like(t),t,valid,.3,.003,'CENTERED',normal_mean_N=torch.tensor([load]))
            self.assertEqual(float(r['score'][0]),0.)
        self.assertLess(float(support_score(-t,t,valid,.3,.003,'CENTERED')['score'][0]),0.)

    def test_contact_gating_has_expected_discontinuity_but_centering_alone_does_not(self):
        t=torch.tensor([[0.,0,.3,0,0,0]],dtype=torch.float64);valid=torch.tensor([True])
        def value(v,load):return support_score(t,t,valid,.3,.003,v,normal_mean_N=torch.tensor([load]))['score']
        self.assertTrue(torch.equal(value('CENTERED',.09999),value('CENTERED',.10001)))
        self.assertEqual(float(value('CONTACT_CENTERED',.09999)[0]),0.)
        self.assertGreater(float(value('CONTACT_CENTERED',.10001)[0]),0.)

    def test_torque_error_cannot_be_hidden_by_perfect_force(self):
        t=torch.tensor([[0.,0,.3,0,.003,0]],dtype=torch.float64);valid=torch.tensor([True]);bad=t.clone();bad[:,4]*=-1
        good=support_score(t,t,valid,.3,.003,'CENTERED')['score'];wrong=support_score(bad,t,valid,.3,.003,'CENTERED')['score']
        self.assertTrue(bool((good>wrong).all()))

    def test_centered_production_reuses_original_geometry_and_invalid_target_gates(self):
        a=fixture();a['cfg']=dict(a['cfg'],support_score='CENTERED');a['wrench_sum'].zero_();r=objective_terms(**a)
        self.assertTrue(torch.equal(r['support_bonus'],torch.zeros(2)))
        a['geometry'][0,0]=.00201;a['target_valid'][1]=False;r=objective_terms(**a)
        self.assertTrue(r['geometry_failure'][0]);self.assertFalse(r['geometry_failure'][1]);self.assertTrue(torch.equal(r['support_bonus'],torch.zeros(2)))

    def test_public_centered_schema_rejects_legacy_even_before_runtime_fields(self):
        learner=SupportPenPPO.__new__(SupportPenPPO);learner.checkpoint_schema=CENTERED_SCHEMA
        with self.assertRaisesRegex(ValueError,'cross-loading'):learner.preflight({'schema':SCHEMA})


if __name__=='__main__':unittest.main()
