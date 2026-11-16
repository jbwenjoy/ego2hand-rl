import copy
from pathlib import Path
import tomllib
import unittest
import numpy as np
from supermimic_research.learning.backend_probe_review import require_response_probe
from supermimic_research.learning.backend_response_study import response_spec
from supermimic_research.mechanics.probe_row_review import aligned_rows,response_groups


class BackendReviewTests(unittest.TestCase):
    def example(self):
        cfg=tomllib.loads((Path(__file__).resolve().parents[1]/'configs/pen_backend_response_validation_v1.toml').read_text());_,profiles,_=response_spec(cfg)
        sources=[dict(name=f'{a}_{n}_{p}',actor=a,noise_seed=n,phase=p) for a in cfg['actors'] for n in cfg['noise_seeds'] for p in cfg['source_phases']]
        cases=[dict(source_name=s['name'],role=cfg['role'],collision_backend_id='new',profile=p,available=False,native_integrations=0) for s in sources for p in profiles]
        run=dict(schema='backend_controlled_response_probe:1',config='cfg',registration='reg',collision_backend=dict(backend_id='new'),role=cfg['role'],passed=1,total=1,checks=dict(ok=True),
            restore_mode='STRICT',fixed_frame_features='LIVE',scales_fitted=False,CHECK_response_rows_used=0,validation_rows_used_for_TRAIN=0,sources=sources,cases=cases,
            producer_native_integrations=2080,actual_native_integrations=2080,attempted_native_integrations=2080,optimizer_steps=0,attempted_optimizer_steps=0,completed_call_accounting_exact=True)
        return cfg,run

    def test_excluded_sources_remain_in_complete_declared_cohort(self):
        cfg,run=self.example();require_response_probe(run,cfg,dict(backend_id='new'),'reg','cfg')

    def test_old_mixed_missing_and_miscounted_cohort_rejected(self):
        for key in ['old','backend','role','missing_profile','missing_source','native','normalization']:
            cfg,run=self.example()
            if key=='old':run['schema']='trained_pen_same_source_response_probe:3'
            elif key=='backend':run['cases'][0]['collision_backend_id']='old'
            elif key=='role':run['role']='TRAIN_CONTROLLED_PHYSICAL_RESPONSE'
            elif key=='missing_profile':run['cases'].pop()
            elif key=='missing_source':run['sources'].pop()
            elif key=='native':run['actual_native_integrations']+=8
            else:run['scales_fitted']=True
            with self.subTest(key=key),self.assertRaises(ValueError):require_response_probe(run,cfg,dict(backend_id='new'),'reg','cfg')

    def test_empty_bank_has_explicit_aligned_shapes(self):
        data=aligned_rows([],[])
        self.assertEqual(data['x'].shape,(0,102));self.assertEqual(data['y'].shape,(0,90));self.assertEqual(data['valid'].dtype,np.bool_)
        cfg=dict(directions=20,PD_eps_rad=[.002,.005],repeats=2)
        groups=response_groups([], [dict(name='missing')],cfg)
        self.assertEqual(len(groups),40);self.assertFalse(any(g['complete'] or g['qualified'] for g in groups))

    def test_invalid_signed_repeat_masks_whole_pair_and_preserves_empty_groups(self):
        cfg=dict(directions=2,PD_eps_rad=[.002,.005],repeats=2)
        rows=[dict(source_name='s',valid=(i!=3),profile=dict(direction=0,epsilon=.002,sign=sign)) for i,sign in enumerate([1,1,-1,-1])]
        groups=response_groups(rows,[dict(name='s'),dict(name='missing')],cfg)
        self.assertEqual(len(groups),8);self.assertTrue(groups[0]['complete']);self.assertFalse(groups[0]['qualified'])
        self.assertEqual(groups[0]['rows'],{'1':[0,1],'-1':[2,3]})
        self.assertFalse(any(g['qualified'] for g in groups))
