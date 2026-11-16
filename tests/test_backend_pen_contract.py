import copy
import unittest
from supermimic_research.learning.backend_pen_ppo import flatten_contract,require_saved_backend


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.contract=dict(backend_id='abc',sources=[dict(path='p',bytes=3,sha256='a')],settings=dict(ccd_tolerance=1e-6))
        self.state=dict(metadata=dict(collision_backend=copy.deepcopy(self.contract)),native=dict(metadata=dict(collision_backend=copy.deepcopy(self.contract)),python=dict(scalars=flatten_contract(self.contract))))

    def test_matching_metadata_and_owned_fields(self):require_saved_backend(self.state,self.contract)

    def test_old_or_rebound_metadata_rejected(self):
        del self.state['metadata']['collision_backend']
        with self.assertRaises(ValueError):require_saved_backend(self.state,self.contract)

    def test_hidden_owned_backend_change_rejected(self):
        path=next(p for p in self.state['native']['python']['scalars'] if p[-1]==('item','backend_id'))
        self.state['native']['python']['scalars'][path]='other'
        with self.assertRaises(ValueError):require_saved_backend(self.state,self.contract)

    def test_extra_owned_field_and_type_change_rejected(self):
        path=(('attr','_collision_backend_contract'),('item','unknown'));self.state['native']['python']['scalars'][path]=1
        with self.assertRaises(ValueError):require_saved_backend(self.state,self.contract)
        self.state['native']['python']['scalars'].pop(path)
        path=next(p for p in self.state['native']['python']['scalars'] if p[-1]==('item','bytes'))
        self.state['native']['python']['scalars'][path]=3.
        with self.assertRaises(ValueError):require_saved_backend(self.state,self.contract)
