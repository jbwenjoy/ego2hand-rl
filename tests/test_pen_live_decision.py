import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from supermimic_research.learning.pen_live_decision import from_owned_snapshot


class LiveDecisionTests(unittest.TestCase):
    def fixture(self):
        q=np.zeros((1,27),np.float32);q[0,:3]=[.1,.2,.3];q[0,3]=1;v=np.arange(26,dtype=np.float32)[None]
        snapshot=dict(schema='controlled_pen_owned_state:4',worlds=1,metadata=dict(fixed_frame_features='LIVE'),native=dict(data=dict(arrays=dict(qpos=q,qvel=v,qacc_warmstart=v.copy())),model=dict(arrays={'opt.timestep':np.array([.0025],np.float32)})))
        case=dict(qids=list(range(7,27)),dofs=list(range(6,26)),dt=.0025,model_layout=dict(object_qposadr=0,object_dofadr=0))
        context=dict(EMA=torch.zeros(20),reference_joints=torch.zeros(20),encoder_bias=torch.zeros(20),lower=torch.full((20,),-2.),upper=torch.full((20,),2.),frames=369)
        return snapshot,case,context

    def test_state_warm_actual_pd_chart_and_no_source_alias(self):
        snapshot,case,context=self.fixture();before=copy.deepcopy(snapshot)
        with patch('supermimic_research.learning.pen_live_decision.source_context',return_value=context):r=from_owned_snapshot(snapshot,1,case,np.ones(20,np.float32)*.4)
        self.assertEqual(r['input102'].shape,(102,));np.testing.assert_array_equal(r['input102'][55:81],np.arange(26));np.testing.assert_array_equal(r['actual_PD20'],np.full(20,.1,np.float32));np.testing.assert_array_equal(r['input102'][81:101],r['actual_PD20'])
        r['input102'][:]=0;np.testing.assert_array_equal(snapshot['native']['data']['arrays']['qacc_warmstart'],before['native']['data']['arrays']['qacc_warmstart'])

    def test_sampled_and_applied_raw_are_distinct_at_clipping(self):
        snapshot,case,context=self.fixture();raw=np.zeros(20,np.float32);raw[0]=2.
        with patch('supermimic_research.learning.pen_live_decision.source_context',return_value=context):r=from_owned_snapshot(snapshot,1,case,raw)
        self.assertEqual(r['sampled_RAW20'][0],2.);self.assertEqual(r['applied_RAW20'][0],1.);self.assertEqual(r['actual_PD20'][0],.25);self.assertEqual(r['clipped_components'],1)

    def test_wrong_mode_memory_clock_or_phase_rejects(self):
        snapshot,case,context=self.fixture()
        for defect in ['mode','warm','clock','phase']:
            bad=copy.deepcopy(snapshot);phase=1
            if defect=='mode':bad['metadata']['fixed_frame_features']='CACHED'
            elif defect=='warm':bad['native']['data']['arrays']['qacc_warmstart']=np.zeros((1,25))
            elif defect=='clock':bad['native']['model']['arrays']['opt.timestep']=np.array([.005],np.float32)
            else:phase=0
            with self.subTest(defect=defect),patch('supermimic_research.learning.pen_live_decision.source_context',return_value=context),self.assertRaises(ValueError):from_owned_snapshot(bad,phase,case,np.zeros(20))


if __name__=='__main__':unittest.main()
