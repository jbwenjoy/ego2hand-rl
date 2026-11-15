import unittest
import numpy as np
from supermimic_research.runtime.saved_native_plant import observed_data_builder
from supermimic_research.mechanics.native_inertia import expand_sparse_inertia


class ObservedDataTests(unittest.TestCase):
    def observation(self,source):
        return dict(schema='observed_live_dynamic_Data_layout:1',arrays={k:{} for k in source},scalars=dict(nworld=1,naconmax=192,naccdmax=192,njmax=4096,njmax_nnz=106496,nvmax=26,nvmax_pad=32,njmax_pad=4096))

    def test_all_recorded_capacities_preserved_without_accidentally_enabling_compaction(self):
        source={'cM':np.empty((1,0,0))};o=self.observation(source);b=observed_data_builder(o,source,26)
        self.assertEqual(b,dict(nworld=1,naconmax=192,naccdmax=192,njmax=4096,njmax_nnz=106496))
        source['cM']=np.zeros((1,32,32));self.assertEqual(observed_data_builder(o,source,26)['nvmax'],26)

    def test_missing_or_inconsistent_observation_is_rejected(self):
        source={'cM':np.empty((1,0,0))};o=self.observation(source)
        for bad in [{**o,'arrays':{}},{**o,'scalars':{**o['scalars'],'nworld':True}},{**o,'scalars':{**o['scalars'],'nvmax':25}},
                    {**o,'scalars':{k:v for k,v in o['scalars'].items() if k!='naccdmax'}}]:
            with self.assertRaises(ValueError):observed_data_builder(bad,source,26)

    def test_sparse_mass_decode_has_correct_coupled_momentum_and_rejects_bad_layout(self):
        # Independent physical fixture: coupled first two axes, third uncoupled.
        M=expand_sparse_inertia(np.array([2.,.5,3.,4.]),[1,2,1],[0,1,3],[0,0,1,2],3)
        np.testing.assert_array_equal(M,np.array([[2.,.5,0],[.5,3.,0],[0,0,4.]]))
        np.testing.assert_array_equal(M@np.array([1.,2.,3.]),[3.,6.5,12.])
        with self.assertRaises(ValueError):expand_sparse_inertia([2.,.5,3.,4.],[1,2,1],[0,1,3],[0,1,1,2],3)
