import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.control_alias import public_action_aliases


class ControlAliasTests(unittest.TestCase):
    def test_separate_baseline_difference_from_same_teacher_ambiguity(self):
        x=np.zeros((4,144));actions=np.zeros((4,20));actions[2,0]=.5;actions[3,0]=-.25
        d=public_action_aliases(x,actions,['HOLD/ZERO','HOLD/ZERO','HOLD/MPC','HOLD/MPC'])
        self.assertEqual(len(d['mixed_controller_context_groups']),1);self.assertEqual(d['same_context_alias_groups'],2);self.assertEqual(len(d['same_context_conflicting_actions']),1)
        item=d['same_context_conflicting_actions'][0];self.assertEqual(item['context'],'HOLD/MPC');self.assertEqual(item['rows'],[2,3]);self.assertEqual(item['max_pairwise_action_difference'],.75)


if __name__=='__main__':unittest.main()
