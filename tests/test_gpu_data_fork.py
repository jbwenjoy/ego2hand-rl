import dataclasses
import unittest

import numpy as np
import torch
import warp as wp

from supermimic_research.runtime.gpu_data_fork import fork_dynamic_data


@dataclasses.dataclass
class Data:
    q:object
    cache:object
    count:int=3


class ForkTests(unittest.TestCase):
    def test_exact_alias_preserved_without_sharing_live_storage(self):
        source=wp.array(np.array([1,np.nan,-0.],np.float32),device='cpu')
        data=Data(source,source);fork,receipt=fork_dynamic_data(data)
        self.assertIs(fork.q,fork.cache);self.assertNotEqual(fork.q.ptr,source.ptr)
        self.assertEqual(source.numpy().tobytes(),fork.q.numpy().tobytes())
        fork.q.zero_();self.assertEqual(source.numpy().tobytes(),np.array([1,np.nan,-0.],np.float32).tobytes())
        self.assertEqual(receipt['distinct_nonempty_allocations'],1)

    def test_partial_alias_and_noncontiguous_views_rejected(self):
        tensor=torch.arange(8,dtype=torch.float32)
        with self.assertRaises(ValueError):fork_dynamic_data(Data(wp.from_torch(tensor),wp.from_torch(tensor[1:])))
        with self.assertRaises(ValueError):fork_dynamic_data(Data(wp.from_torch(tensor),wp.from_torch(tensor[::2])))
