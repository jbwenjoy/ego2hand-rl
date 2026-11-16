import copy
import unittest

import numpy as np

from supermimic_research.runtime.pen_termination_design import schedule
from supermimic_research.runtime.pen_termination_source_import import import_historical_state
from supermimic_research.runtime.shared_snapshot_transport import tree_digest


class TerminationImportTests(unittest.TestCase):
    def fixture(self):
        return dict(schema='controlled_pen_owned_state:4', worlds=1,
                    metadata=dict(fixed_frame_features='LIVE', collision_backend='old'),
                    native=dict(model=dict(a=np.array([1.], np.float32)), data=dict(b=np.array([2.], np.float32))),
                    python=dict(description={'root': 'unchanged'}, scalars={('x',): 2.}, arrays={'x': np.array([3])}),
                    rng={'seed': 9}, sources={'original': True})

    def test_only_metadata_added_no_numerical_or_inventory_change(self):
        state = self.fixture(); old = tree_digest(state); contract = {'backend_id': 'new'}
        target = dict(state['metadata'], termination_backend=contract)
        imported, audit = import_historical_state(state, target, contract)
        self.assertEqual(tree_digest(state), old)
        self.assertIs(imported['native'], state['native'])
        self.assertIs(imported['python'], state['python'])
        self.assertTrue(all(audit['unchanged_top_level_fields'].values()))
        self.assertEqual(audit['original_tree_sha256'], old)
        self.assertNotEqual(audit['imported_tree_sha256'], old)
        self.assertNotIn('termination_backend', state['metadata'])

    def test_reject_other_metadata_change_and_reimport(self):
        state = self.fixture(); contract = {'backend_id': 'new'}
        target = dict(state['metadata'], termination_backend=contract, collision_backend='changed')
        with self.assertRaisesRegex(ValueError, 'Only the declared'):
            import_historical_state(state, target, contract)
        target['collision_backend'] = 'old'
        imported, _ = import_historical_state(state, target, contract)
        with self.assertRaises(ValueError):
            import_historical_state(imported, target, contract)

    def test_target_contract_must_match(self):
        state = self.fixture()
        with self.assertRaises(ValueError):
            import_historical_state(state, dict(state['metadata'], termination_backend={'x': 1}), {'x': 2})

    def test_all240_exact_and_counterbalanced_algorithm_order(self):
        rows = [dict(name=('SUPPORT_' if i < 2 else 'RESIDUAL_') + str(i), pair=i // 2,
                     role='failure' if i % 2 == 0 else 'control') for i in range(10)]
        plans = [schedule(rows, replica) for replica in ['A', 'B']]
        all_items = [d for p in plans for d in p]
        keys = {(d['replica'], d['row_index'], d['repeat'], d['scope'], d['algorithm']) for d in all_items}
        self.assertEqual(len(keys), 240)
        self.assertEqual(plans[0][0]['algorithm'], 'INSTALLED_ORIGINAL_OR')
        self.assertEqual(plans[1][0]['algorithm'], 'DROP_IMPROVEMENT_ONLY')
        self.assertEqual(len({d['name'] for d in all_items}), 240)
        for i in range(10):
            self.assertEqual(sum(d['row_index'] == i for d in all_items), 24)
        bad = copy.deepcopy(rows); bad[1]['role'] = 'failure'
        with self.assertRaises(ValueError):
            schedule(bad, 'A')


if __name__ == '__main__':
    unittest.main()
