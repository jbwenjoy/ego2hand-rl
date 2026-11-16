import copy
from types import SimpleNamespace
import unittest
import numpy as np
from supermimic_research.learning.frozen_pen_weights import validate_full_model,initialize_frozen_W32_weights
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,CENTERED_SCHEMA,FROZEN_WEIGHTS_SCHEMA


class FrozenW32Weights(unittest.TestCase):
    def fixture(self):
        arrays={k:SimpleNamespace(numpy=lambda value=v:value) for k,v in {
            'body_mass':np.ones((32,3),np.float32),
            'derived_solver_field':np.arange(64,dtype=np.int32).reshape(32,2)}.items()}
        saved=dict(arrays={k:v.numpy().copy() for k,v in arrays.items()},scalars={'solver':2})
        return arrays,saved

    def test_unlisted_derived_field_cannot_be_skipped(self):
        arrays,saved=self.fixture();validate_full_model(arrays,{'solver':2},saved)
        saved['arrays']['derived_solver_field'][3,1]+=1
        with self.assertRaisesRegex(ValueError,'derived_solver_field'):validate_full_model(arrays,{'solver':2},saved)

    def test_shape_dtype_keys_and_scalar_changes_all_reject(self):
        arrays,saved=self.fixture()
        bad=[]
        x=copy.deepcopy(saved);x['arrays']['body_mass']=x['arrays']['body_mass'][0];bad.append(x)
        x=copy.deepcopy(saved);x['arrays']['body_mass']=x['arrays']['body_mass'].astype(np.float64);bad.append(x)
        x=copy.deepcopy(saved);x['arrays'].pop('derived_solver_field');bad.append(x)
        x=copy.deepcopy(saved);x['scalars']['solver']=3;bad.append(x)
        for state in bad:
            with self.assertRaises(ValueError):validate_full_model(arrays,{'solver':2},state)

    def test_wrong_world_count_rejects_before_opening_checkpoint(self):
        learner=SimpleNamespace(adapter=SimpleNamespace(num_envs=1))
        with self.assertRaisesRegex(ValueError,'Fresh uncalibrated'):initialize_frozen_W32_weights(learner,{'missing':'must not be read'})

    def test_existing_training_rejects_before_opening_checkpoint(self):
        learner=SimpleNamespace(adapter=SimpleNamespace(num_envs=32),iteration=1)
        with self.assertRaisesRegex(ValueError,'Fresh uncalibrated'):initialize_frozen_W32_weights(learner,{'missing':'must not be read'})

    def test_seeded_norm_history_does_not_relabel_new_work(self):
        learner=SupportPenPPO.__new__(SupportPenPPO);learner.checkpoint_schema=FROZEN_WEIGHTS_SCHEMA;learner.adapter=SimpleNamespace(num_envs=32)
        learner.normalizer_seed=dict(source={'pinned':'source'},actor_count=129312,critic_count=129312)
        state={'metadata':{'normalizer_seed':copy.deepcopy(learner.normalizer_seed)}}
        self.assertEqual(learner.normalization_seed_offset(state),129280)
        state['metadata']['normalizer_seed']['actor_count']+=1
        with self.assertRaisesRegex(ValueError,'seed history'):learner.normalization_seed_offset(state)
        learner.checkpoint_schema=CENTERED_SCHEMA
        self.assertEqual(learner.normalization_seed_offset(state),0)
