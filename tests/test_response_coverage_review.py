import copy
import unittest
import numpy as np
from supermimic_research.learning.response_coverage_review import review_coverage
from supermimic_research.mechanics.probe_row_review import response_groups


class CoverageReviewTests(unittest.TestCase):
    def example(self,*,noisy=False,reversed_large=False):
        rows=[];x=[];y=[]
        for j in range(2):
            for e in [.002,.005]:
                for sign in [1,-1]:
                    for repeat in [0,1]:
                        value=np.zeros(102);value[-1]=.0025;value[81+j]=sign*e;target=np.zeros(90)
                        target[40]=sign*e*(-1 if reversed_large and e==.005 else 1)+(0.02*(2*repeat-1) if noisy else 0)
                        rows.append(dict(row_index=len(rows),source_name='s',valid=True,profile=dict(direction=j,epsilon=e,sign=sign,repeat=repeat)));x.append(value);y.append(target)
        cfg=dict(directions=20,PD_eps_rad=[.002,.005],repeats=2);groups=response_groups(rows,[dict(name='s'),dict(name='missing')],cfg)
        return dict(x=np.array(x),y=np.array(y),valid=np.ones(len(rows),bool)),rows,groups,np.eye(20),['s','missing'],np.ones(90)

    def test_input_coverage_rank_is_not_output_response_rank(self):
        r=review_coverage(*self.example());source=next(x for x in r['source_spectra'] if x['source']=='s' and x['epsilon']==.002)
        self.assertEqual(source['declared_input']['numerical_rank'],2);self.assertEqual(source['normalized_finite_output']['object12']['numerical_rank'],1)
        self.assertEqual(r['physical_screen_direction_counts']['object12'],2)
        self.assertEqual(next(x for x in r['source_spectra'] if x['source']=='missing')['declared_input']['numerical_rank'],0)

    def test_amplitude_disagreement_does_not_select_training_pairs(self):
        reference=review_coverage(*self.example());bad=review_coverage(*self.example(reversed_large=True))
        self.assertEqual(bad['physical_screen_direction_counts']['object12'],0);self.assertEqual(reference['eligible_pair_groups'],bad['eligible_pair_groups'])

    def test_observed_repeat_noise_not_claimed_reliable_gradient(self):
        r=review_coverage(*self.example(noisy=True));self.assertEqual(r['physical_screen_direction_counts']['object12'],0)
        self.assertEqual(len(r['eligible_pair_groups']['rank20']),4)

    def test_masked_group_stays_audited_and_cannot_be_promoted(self):
        args=self.example();args[0]['valid'][0]=False;args[1][0]['valid']=False;args[2][0]['qualified']=False
        r=review_coverage(*args);self.assertEqual(len(r['group_audit']),80);self.assertEqual(len(r['eligible_pair_groups']['rank20']),3)
        args[2][0]['qualified']=True
        with self.assertRaises(ValueError):review_coverage(*args)

    def test_signed_row_relabel_or_actual_PD_corruption_rejected(self):
        for key in ['repeat','source','PD','missing_group']:
            args=self.example()
            if key=='repeat':args[1][0]['profile']['repeat']=1
            elif key=='source':args[1][0]['source_name']='missing'
            elif key=='PD':args[0]['x'][0,81]+=.01
            else:args[2].pop()
            with self.subTest(key=key),self.assertRaises(ValueError):review_coverage(*args)
