import unittest
from unittest.mock import patch
from types import SimpleNamespace
from supermimic_research.runtime.pen_frozen_cost_long_design_v1 import schedule
from supermimic_research.learning.pen_frozen_cost_actor_v1 import initialize


class FrozenCostDesignTests(unittest.TestCase):
    def test_all18_exact_counterbalanced_with_full_reference_budget(self):
        rows=schedule()
        self.assertEqual(len(rows),18); self.assertEqual(len({r['name'] for r in rows}),18)
        self.assertEqual(len({(r['repetition'],r['algorithm'],r['policy']) for r in rows}),18)
        self.assertEqual(rows[0]['algorithm'],'INSTALLED_ORIGINAL_OR'); self.assertEqual(rows[6]['algorithm'],'STABLE_SAME_REGIME_FRICTION_OR')
        self.assertEqual(18*368*8,52992)

    def test_wrong_training_counters_or_numeric_identity_rejected_before_weight_load(self):
        profile=SimpleNamespace(require=lambda r:None); contract=dict(backend_id='x')
        state=dict(iteration=2000,optimizer_steps=64000,training_transitions=4096000,metadata=dict(arm='RESIDUAL',collision_backend=contract),native=dict(metadata=dict(collision_backend=contract)))
        with patch('supermimic_research.learning.pen_frozen_cost_actor_v1.active_backend',return_value=SimpleNamespace(contract=contract)),patch('supermimic_research.learning.pen_frozen_cost_actor_v1.actor_for_fresh_live_W1') as load:
            state['optimizer_steps']=1
            with self.assertRaises(ValueError): initialize(None,state,None,None,'RESIDUAL',profile)
            state['optimizer_steps']=64000; state['metadata']['termination_backend']={}
            with self.assertRaises(ValueError): initialize(None,state,None,None,'RESIDUAL',profile)
            load.assert_not_called()


if __name__=='__main__': unittest.main()
