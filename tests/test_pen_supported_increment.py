import numpy as np
import torch
import unittest
from supermimic_research.mechanics.pen_supported_increment import SupportedIncrement
from supermimic_research.mechanics.pen_load_increment import LoadIncrement


class TestTrainConstantLoads(unittest.TestCase):
    def test_mask_only_blocks_unlearnable_directions_in_correction(self):
        torch.manual_seed(7411)
        x=torch.randn(24,87);x[:,85:]=0
        y=torch.randn(24,52)
        a=LoadIncrement(x,y);b=SupportedIncrement(x,y)
        b.load_state_dict(dict(**a.state_dict(),learnable_input=b.learnable_input.clone()))
        torch.testing.assert_close(a(x),b(x))
        sample=x[:1].clone();sample[:,85]=.6
        # Keep the observed fifth digit load in the skip connection, but a
        # never-excited input cannot change the learned force/torque correction.
        change=b(sample)-b(x[:1]);expected=torch.zeros_like(change);expected[:,10]=.6
        torch.testing.assert_close(change,expected)
        self.assertEqual(torch.where(~b.learnable_input)[0].tolist(),[85,86])

if __name__=='__main__':unittest.main()
