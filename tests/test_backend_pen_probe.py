import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from supermimic_research.learning.backend_pen_ppo import flatten_contract
from supermimic_research.runtime.backend_pen_probe import SCHEMA,require_backend_probe,restore_backend_probe
from supermimic_research.learning.backend_response_study import response_spec,require_final_response_actor,require_evaluated_cohort


class BackendProbeTests(unittest.TestCase):
    def setUp(self):
        self.contract=dict(backend_id='new',sources=[dict(path='p',bytes=3,sha256='a')])
        scalars=flatten_contract(self.contract);scalars[(('attr','support_weight'),)]=0.
        native=dict(worlds=1,metadata=dict(collision_backend=copy.deepcopy(self.contract),fixed_frame_features='LIVE'),python=dict(scalars=scalars))
        self.source=dict(schema=SCHEMA,role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE',collision_backend=copy.deepcopy(self.contract),actor_budget=(2000,64000,4096000),actor='RESIDUAL',native=native)

    def test_exact_source_admitted(self):require_backend_probe(self.source,self.contract)

    def test_old_envelope_rejected(self):
        self.source['schema']='owned_trained_actor_probe_source:1'
        with self.assertRaises(ValueError):require_backend_probe(self.source,self.contract)

    def test_hidden_rebinding_rejected_before_loader_import(self):
        path=(('attr','_collision_backend_contract'),('item','backend_id'))
        self.source['native']['python']['scalars'][path]='old'
        runtime=SimpleNamespace(_collision_backend_contract=self.contract);backend=SimpleNamespace(verify=lambda **k:True)
        with patch('supermimic_research.runtime.backend_pen_probe.active_backend',return_value=backend),patch('builtins.__import__',side_effect=AssertionError('No downstream loader may be imported')):
            with self.assertRaises(ValueError):restore_backend_probe(runtime,self.source)

    def test_other_backend_and_owned_type_rejected(self):
        for mutation in ['metadata','owned_type','extra']:
            source=copy.deepcopy(self.source)
            if mutation=='metadata':source['native']['metadata']['collision_backend']['backend_id']='old'
            elif mutation=='owned_type':source['native']['python']['scalars'][(('attr','_collision_backend_contract'),('item','sources'),('item',0),('item','bytes'))]=3.
            else:source['native']['python']['scalars'][(('attr','_collision_backend_contract'),('item','extra'))]=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):require_backend_probe(source,self.contract)

    def test_W32_cached_wrong_arm_weight_and_budget_rejected(self):
        for key in ['worlds','features','weight','budget','role']:
            source=copy.deepcopy(self.source)
            if key=='worlds':source['native']['worlds']=32
            elif key=='features':source['native']['metadata']['fixed_frame_features']='CACHED'
            elif key=='weight':source['native']['python']['scalars'][(('attr','support_weight'),)]=.2
            elif key=='budget':source['actor_budget']=(1,32,2048)
            else:source['role']='CHECK'
            with self.subTest(key=key),self.assertRaises(ValueError):require_backend_probe(source,self.contract)

    def test_partial_actor_cannot_enter_response_bank(self):
        state=dict(schema='pen_geometry_support_PPO_training:4',metadata=dict(collision_backend=self.contract),native=self.source['native'],iteration=1,optimizer_steps=32,training_transitions=2048)
        with self.assertRaises(ValueError):require_final_response_actor(state,self.contract,'RESIDUAL')

    def test_exact_causal_budget_and_distinct_cohorts(self):
        from pathlib import Path
        import tomllib
        root=Path(__file__).resolve().parents[1];specs=[]
        for name,total in [('train',33376),('validation',17632)]:
            cfg=tomllib.loads((root/f'configs/pen_backend_response_{name}_v1.toml').read_text());directions,profiles,budget=response_spec(cfg)
            self.assertEqual(directions.shape,(20,20));self.assertEqual(len(profiles),162);self.assertEqual(budget['max_total_native'],total);specs.append(cfg)
        self.assertFalse(set(specs[0]['noise_seeds'])&set(specs[1]['noise_seeds']))
        self.assertFalse(set(specs[0]['source_phases'])&set(specs[1]['source_phases']))
        specs[0]['object_penetration_m']=.003
        with self.assertRaises(ValueError):response_spec(specs[0])

    def final_cohort(self):
        training=dict(schema='fresh_backend_long_control_training:1',config='cfg',registration='reg',backend_id='new',actual_native_integrations=65536000,optimizer_steps=128000,
            passed=1,total=1,checks=dict(exact=True),cohort_native_before_evaluation=65538048,preparation_native_integrations=2048,
            arms=[dict(arm=arm,iterations=2000,actual_native_integrations=32768000,optimizer_steps=64000,training_transitions=4096000,checkpoint=arm) for arm in ['RESIDUAL','SUPPORT']])
        cases=[dict(repetition=r,policy=label,checkpoint=label,native_integrations=8,summary=dict(qualified=False),
            actor_initialization=dict(backend_binding_verified=True,collision_backend_id='new',old_native_backend_imported=False)) for r in range(3) for label in ['INITIAL','RESIDUAL','SUPPORT']]
        evaluation=dict(schema='fresh_backend_long_control_evaluation:1',config='cfg',registration='reg',backend_id='new',training='training',cases=cases,optimizer_steps=0,actual_native_integrations=72)
        return training,evaluation

    def test_failed_strict_task_is_retained_after_complete_evaluation(self):
        t,e=self.final_cohort();require_evaluated_cohort(t,e,'cfg','new','reg','training')

    def test_incomplete_or_wrong_evaluation_cannot_supply_actors(self):
        for key in ['missing_case','other_actor','backend','native_count','partial_training']:
            t,e=self.final_cohort()
            if key=='missing_case':e['cases'].pop()
            elif key=='other_actor':e['cases'][1]['checkpoint']='other'
            elif key=='backend':e['cases'][0]['actor_initialization']['collision_backend_id']='old'
            elif key=='native_count':e['actual_native_integrations']=73
            else:t['arms'][1]['iterations']=1999
            with self.subTest(key=key),self.assertRaises(ValueError):require_evaluated_cohort(t,e,'cfg','new','reg','training')

    def test_final_actor_requires_W32_normalized_same_arm_source(self):
        native=copy.deepcopy(self.source['native']);native['worlds']=32
        state=dict(schema='pen_geometry_support_PPO_training:4',metadata=dict(collision_backend=self.contract,arm='RESIDUAL',support_weight=0.),native=native,iteration=2000,optimizer_steps=64000,training_transitions=4096000,normalizers_ready=True)
        require_final_response_actor(state,self.contract,'RESIDUAL')
        for key in ['worlds','normalizers','arm']:
            bad=copy.deepcopy(state)
            if key=='worlds':bad['native']['worlds']=1
            elif key=='normalizers':bad['normalizers_ready']=False
            else:bad['metadata']['arm']='SUPPORT'
            with self.subTest(key=key),self.assertRaises(ValueError):require_final_response_actor(bad,self.contract,'RESIDUAL')
