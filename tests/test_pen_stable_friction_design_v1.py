import unittest
from supermimic_research.runtime.pen_stable_friction_design_v1 import schedule


class StableFrictionDesignTests(unittest.TestCase):
    def test_exact_all240_counterbalanced_algorithms_sources_scopes(self):
        rows=[dict(name=('SUPPORT_' if i<2 else 'RESIDUAL_')+str(i),pair=i//2,role='failure' if i%2==0 else 'control') for i in range(10)]
        plans=[schedule(rows,r) for r in ['A','B']]; allrows=[d for p in plans for d in p]
        self.assertEqual(len({(d['replica'],d['row_index'],d['repeat'],d['scope'],d['algorithm']) for d in allrows}),240)
        self.assertEqual(plans[0][0]['algorithm'],'INSTALLED_ORIGINAL_OR'); self.assertEqual(plans[1][0]['algorithm'],'STABLE_SAME_REGIME_FRICTION_OR')
        self.assertEqual({d['scope'] for d in allrows},{'PACKET_BASELINE','PACKET_PLUS_SOLVE_EXIT'})
        for i in range(10): self.assertEqual(sum(d['row_index']==i for d in allrows),24)
        with self.assertRaises(ValueError): schedule(rows,'C')


if __name__=='__main__': unittest.main()
