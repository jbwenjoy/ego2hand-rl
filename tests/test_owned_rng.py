import copy
import random
import unittest
import numpy as np
import torch
from supermimic_research.runtime.owned_rng import validate_rng


def state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[])


def exact(a,b):
    return a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['torch'],b['torch'])


class OwnedRNGTests(unittest.TestCase):
    def test_valid_private_generator_validation_preserves_all_global_states(self):
        before=state();validate_rng(copy.deepcopy(before),cuda_count=0);self.assertTrue(exact(before,state()))

    def test_invalid_Torch_state_preserves_global_states(self):
        before=state();bad=copy.deepcopy(before);bad['torch']=torch.zeros(1,dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError,'Invalid checkpoint RNG'):validate_rng(bad,cuda_count=0)
        self.assertTrue(exact(before,state()))

    def test_invalid_python_numpy_and_device_layout_rejected_without_installing(self):
        before=state()
        for key,value in [('python',(1,())),('numpy',('invalid',)),('cuda',[torch.zeros(1,dtype=torch.uint8)])]:
            bad=copy.deepcopy(before);bad[key]=value
            with self.assertRaisesRegex(ValueError,'Invalid checkpoint RNG'):validate_rng(bad,cuda_count=0)
            self.assertTrue(exact(before,state()))
