import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import test_pen_ray_trace_factory_v1 as factory_fixture
from supermimic_research.mechanics.pen_ray_trace_review_v1 import ray_replay
from supermimic_research.runtime.pen_ray_trace_factory_v1 import INT_FIELDS,BRACKET_FIELDS
from supermimic_research.runtime.pen_paired_ray_design_v1 import schedule,SCOPES


class RayReplayTests(unittest.TestCase):
    def test_actual_cpu_recorded_bracket_decisions_and_reference(self):
        real=wp.launch_tiled; captured=[]
        def record(*args,**kwargs):
            result=real(*args,**kwargs); outputs=kwargs.get('outputs',[])
            if len(outputs)==14:
                ray,vec,rows,brackets,integers,errors=[x.numpy().copy() for x in outputs[8:]]
                if integers[0,0,INT_FIELDS.index('valid')]:
                    captured.append((ray[0,0],integers[0,0],rows[0,0],vec[0,0],brackets[0,0]))
            return result
        fixture=factory_fixture.RayFactoryTests()
        try:
            with patch.object(wp,'launch_tiled',side_effect=record):
                fixture.test_actual_cpu_kernels_preserve_outputs_and_record_both_acceptance_paths()
        finally: fixture.doCleanups()
        self.assertEqual(len(captured),5)
        for values in captured:
            review=ray_replay(*values)
            self.assertTrue(all(review['checks'].values()),review['checks'])
            self.assertGreater(len(review['references']),1)
        values=list(captured[1]); values[4]=values[4].copy(); values[4][0,BRACKET_FIELDS.index('swap_lo')]=1-values[4][0,BRACKET_FIELDS.index('swap_lo')]
        self.assertFalse(all(ray_replay(*values)['checks'].values()))

    def test_all240_queries_and_paired_endpoint_scopes(self):
        rows=[dict(name=('SUPPORT_' if i<2 else 'RESIDUAL_')+str(i),pair=i//2,role='failure' if i%2==0 else 'control') for i in range(10)]
        plans=[schedule(rows,r) for r in ['A','B']]; allrows=[x for p in plans for x in p]
        self.assertEqual(len({x['name'] for x in allrows}),240)
        self.assertEqual({x['scope'] for x in allrows},set(SCOPES))
        self.assertEqual(plans[0][0]['algorithm'],'INSTALLED_ORIGINAL_OR'); self.assertEqual(plans[1][0]['algorithm'],'DROP_IMPROVEMENT_ONLY')
        self.assertEqual(sum(x['scope']=='SOLVE_EXIT_PLUS_RAY_TRACE' for x in allrows),120)


if __name__=='__main__': unittest.main()
