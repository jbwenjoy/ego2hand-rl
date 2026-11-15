import unittest
import numpy as np
from supermimic_research.learning.pen_interventions import seeded_training_profiles,prefix_noise_schedule,intervention_profiles


class TrainingInterventions(unittest.TestCase):
    def test_declared_coverage_and_balanced_repeats(self):
        d,rows=seeded_training_profiles(9301)
        self.assertEqual((d.shape,len(rows)),((4,20),34))
        np.testing.assert_allclose(np.linalg.norm(d,axis=1),1,atol=1e-14)
        self.assertEqual(len({r['name'] for r in rows}),34)
        self.assertEqual(sum(r['direction']<0 for r in rows),2)
        for direction in range(4):
            for epsilon in (.002,.005):
                for sign in (-1,1):
                    self.assertEqual(sorted(r['repeat'] for r in rows if r['direction']==direction and r['epsilon']==epsilon and r['sign']==sign),[0,1])

    def test_training_and_existing_CHECK_directions_are_distinct(self):
        d,rows=seeded_training_profiles(9301)
        same,again=seeded_training_profiles(9301)
        self.assertTrue(np.array_equal(d,same));self.assertEqual(rows,again)
        old,check_rows=intervention_profiles()
        self.assertEqual(len(check_rows),27)
        self.assertLess(np.max(abs(d@old.T)),.99)
        self.assertFalse(np.array_equal(d,seeded_training_profiles(9302)[0]))

    def test_materialized_noise_prefix_and_actor_pairing(self):
        noise=prefix_noise_schedule(8801,71,.01)
        self.assertEqual(noise.shape,(71,20));self.assertEqual(noise.dtype,np.float32)
        self.assertTrue(np.array_equal(noise,prefix_noise_schedule(8801,71,.01)))
        self.assertTrue(np.array_equal(noise[:11],prefix_noise_schedule(8801,11,.01)))
        self.assertFalse(np.array_equal(noise,prefix_noise_schedule(8802,71,.01)))

    def test_invalid_schedules_reject(self):
        for args in [(9301,0,2),(9301,21,2),(9301,4,1),(1.2,4,2)]:
            with self.assertRaises(ValueError):seeded_training_profiles(*args)
        for args in [(8801,0,.01),(8801,71,0),(8801,71,np.nan),(1.2,71,.01)]:
            with self.assertRaises(ValueError):prefix_noise_schedule(*args)
