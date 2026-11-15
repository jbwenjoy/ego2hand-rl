import unittest
import numpy as np
from supermimic_research.learning.pen_branch_evaluation import classify_pair


class BranchEvaluationTests(unittest.TestCase):
    def test_confounded_clipped_or_invalid_pairs_never_claim_action_derivatives(self):
        center=np.linspace(0,.5,20);direction=np.ones(20)/np.sqrt(20);eps=.005
        xp,xm=np.zeros(102),np.zeros(102);xp[81:101]=center+eps*direction;xm[81:101]=center-eps*direction;xp[-1]=xm[-1]=.005
        self.assertTrue(classify_pair(xp,xm,True,center,direction,eps)['eligible'])
        changed=xp.copy();changed[55]=1e-12
        self.assertFalse(classify_pair(changed,xm,True,center,direction,eps)['eligible'])
        clipped=xp.copy();clipped[81]=center[0]
        self.assertFalse(classify_pair(clipped,xm,True,center,direction,eps)['eligible'])
        self.assertFalse(classify_pair(xp,xm,False,center,direction,eps)['eligible'])
        changed=xp.copy();changed[-1]=.0025
        self.assertFalse(classify_pair(changed,xm,True,center,direction,eps)['eligible'])
