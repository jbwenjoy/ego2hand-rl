import copy
import unittest
from supermimic_research.learning.pen_branch_fit import validate_fit_contract
from supermimic_research.learning.pen_response import INPUT_SCHEMA,OUTPUT_SCHEMA


class BranchFitTests(unittest.TestCase):
    def test_augmented_roles_schema_review_and_fixed_budget_are_mandatory(self):
        cfg=dict(first_checkpoint_update=100,planned_updates=2000,continue_after_first_healthy_checkpoint=True)
        prep=dict(schema='pen_branch_response_preparation:1',checks={'healthy':True},
            outputs={k:{} for k in ['TRAIN','CHECK','NEW_TRAIN','SEQUENCE_CHECK']},input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA)
        review=dict(schema='pen_branch_preparation_saved_review:1',passed=10,total=10)
        validate_fit_contract(cfg,prep,review)
        for change in [lambda c,p,r:p.update(schema='old87'),lambda c,p,r:p['checks'].update(healthy=False),
                lambda c,p,r:r.update(passed=9),lambda c,p,r:p['outputs'].pop('SEQUENCE_CHECK'),
                lambda c,p,r:c.update(planned_updates=3000)]:
            c,p,r=copy.deepcopy((cfg,prep,review));change(c,p,r)
            with self.assertRaises(ValueError):validate_fit_contract(c,p,r)
