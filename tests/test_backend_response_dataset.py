import unittest
import numpy as np
from supermimic_research.learning.backend_response_dataset import backend_response_data, ROLES
from supermimic_research.mechanics.probe_row_review import response_groups


class BackendDatasetTests(unittest.TestCase):
    def example(self):
        reports = {}; data = {}; directions = {k: np.eye(20) for k in ROLES}
        contract = dict(backend_id='numeric-fixture-only')
        for role in ROLES:
            name = role+'_source'; meta = []; xs = []; ys = []
            for j in [0, 4]:
                for e in [.002, .005]:
                    for sign in [1, -1]:
                        for repeat in [0, 1]:
                            x = np.zeros(102); x[-1] = .0025; x[81+j] = sign*e
                            y = np.zeros(90); y[40] = (j+1)*sign*e; y[84] = 2+sign*e
                            meta.append(dict(row_index=len(meta), source_name=name, valid=True,
                                             profile=dict(direction=j, epsilon=e, sign=sign, repeat=repeat)))
                            xs.append(x); ys.append(y)
            mask = np.ones(len(xs), dtype=bool); mask[-1] = False; meta[-1]['valid'] = False
            arrays = dict(x=np.asarray(xs), y=np.asarray(ys), heads=np.asarray(ys).copy(), valid=mask)
            groups = response_groups(meta, [dict(name=name), dict(name=role+'_excluded')],
                                     dict(directions=20, PD_eps_rad=[.002, .005], repeats=2))
            reports[role] = dict(schema='backend_pen_response_preparation:1', role=ROLES[role],
                collision_backend=contract, checks=dict(ok=True), passed=1, total=1, total_rows=len(xs),
                qualified_rows=int(mask.sum()), rejected_rows=np.flatnonzero(~mask).tolist(),
                qualified_response_groups=sum(g['qualified'] for g in groups),
                scales_fitted=False, CHECK_response_rows_used=0, validation_rows_used_for_TRAIN=0,
                actual_native_integrations=0, optimizer_steps=0, metadata=meta, response_groups=groups)
            data[role] = arrays
        return reports, data, directions, contract

    def test_empty_and_masked_groups_remain_with_original_row_indices(self):
        args = self.example(); result = backend_response_data(*args)
        self.assertEqual(len(result['pair_audit']), 80)
        self.assertEqual(result['pair_subsets']['rank4'].tolist(), [0, 1])
        self.assertEqual(result['pair_subsets']['rank20'].tolist(), [0, 1, 2])
        self.assertEqual(result['pairs']['TRAIN_group_index'].tolist(), [0, 1, 8])
        np.testing.assert_array_equal(result['pairs']['TRAIN_row_plus'][0], [0, 1])
        np.testing.assert_array_equal(result['pairs']['TRAIN_row_minus'][0], [2, 3])
        expected = (args[1]['TRAIN']['y'][[0, 1]].mean(0)-args[1]['TRAIN']['y'][[2, 3]].mean(0))/.004
        np.testing.assert_array_equal(result['pairs']['finite_contrast'][0], expected)

    def test_all_validation_labels_and_invalid_train_rows_cannot_affect_products(self):
        args = self.example(); before = backend_response_data(*args)
        for key in ['x', 'y', 'heads']:
            args[1]['VALIDATE'][key] += 1e6
            args[1]['TRAIN'][key][~args[1]['TRAIN']['valid']] += 1e7
        after = backend_response_data(*args)
        for key in ['scales', 'head_scales', 'pairs', 'pair_subsets']:
            for field in before[key]:
                np.testing.assert_array_equal(before[key][field], after[key][field])

    def test_old_schema_backend_role_or_mask_promotion_rejected(self):
        for change in ['schema', 'backend', 'role', 'mask', 'checks', 'excluded_denominator']:
            args = self.example(); tr = args[0]['TRAIN']
            if change == 'schema': tr['schema'] = 'controlled_pen_response_preparation:2'
            elif change == 'backend': tr['collision_backend'] = dict(backend_id='old')
            elif change == 'role': tr['role'] = ROLES['VALIDATE']
            elif change == 'mask': tr['response_groups'][9]['qualified'] = True
            elif change == 'checks': tr['checks']['ok'] = False
            else: tr['response_groups'].pop()
            with self.subTest(change=change), self.assertRaises(ValueError): backend_response_data(*args)

    def test_signed_repeat_reuse_and_wrong_actual_PD_rejected(self):
        for change in ['repeat', 'PD', 'source']:
            args = self.example()
            if change == 'repeat': args[0]['TRAIN']['metadata'][0]['profile']['repeat'] = 1
            elif change == 'PD': args[1]['TRAIN']['x'][0, 81] += .01
            else: args[0]['VALIDATE']['response_groups'][0]['source_name'] = 'TRAIN_source'
            with self.subTest(change=change), self.assertRaises(ValueError): backend_response_data(*args)

    def test_whole_validation_source_cannot_be_reused_as_train(self):
        args = self.example(); report = args[0]['VALIDATE']
        for row in report['metadata']: row['source_name'] = row['source_name'].replace('VALIDATE', 'TRAIN')
        for group in report['response_groups']: group['source_name'] = group['source_name'].replace('VALIDATE', 'TRAIN')
        with self.assertRaisesRegex(ValueError, 'Disjoint'): backend_response_data(*args)

    def test_empty_train_is_not_promoted_to_fittable_dataset(self):
        args = self.example(); r = args[0]['TRAIN']; a = args[1]['TRAIN']
        r['metadata'] = []; r['total_rows'] = r['qualified_rows'] = r['qualified_response_groups'] = 0; r['rejected_rows'] = []
        for key in a: a[key] = a[key][:0]
        for group in r['response_groups']:
            group.update(rows={'1': [], '-1': []}, complete=False, qualified=False)
        with self.assertRaisesRegex(ValueError, 'No eligible'): backend_response_data(*args)
