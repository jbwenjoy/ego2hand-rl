import types
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.pen_response_targets import response_xy
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.response_decode import decode_response


class ResponseTargetTests(unittest.TestCase):
    def test_source_frame_fn6_named_indices_and_world_omega(self):
        R=Rotation.from_rotvec([0,0,np.pi/2]).as_matrix();quat=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
        q=np.zeros(27);q[3:7]=quat;v=np.zeros(26)
        source=dict(qpos=q[None],qvel=v[None],qacc_warmstart=np.arange(26)[None])
        end=q.copy();end[0]=.004;end[7]=.002;ev=v.copy();ev[0]=.2;ev[3]=1
        raw=[]
        for tick in range(4):
            for geom,force in [(1,.4),(2,.2)]:
                row=np.zeros(24);row[:3]=[tick,0,geom];row[18]=force;raw.append(row)
        packet=dict(post_qpos=np.tile(end,(4,1)),post_qvel=np.tile(ev,(4,1)),post_qacc=np.full((4,26),2.),
            post_ctrl=np.tile(np.arange(20)*.01,(4,1)),raw_contact_tick_plus23=np.array(raw))
        names={0:'object/pen',1:'robot/right_finger1_tip',2:'robot/right_finger5_tip'}
        model=types.SimpleNamespace(geom_bodyid=np.array([2,3,4]),geom=lambda g:types.SimpleNamespace(name=names[g]))
        wrench=np.tile(np.r_[R@np.array([1.,0,0]),R@np.array([0,.02,0])],(4,1))
        qids,dofs=np.arange(7,27),np.arange(6,26)
        x,y=response_xy(packet,wrench,model,source,qids,dofs,np.arange(20),2,0,0,.005)
        np.testing.assert_allclose(y[78:84],[1,0,0,0,.02,0],atol=1e-15)
        np.testing.assert_allclose(y[84:],[.4,0,0,0,.2,0],atol=1e-15)
        np.testing.assert_array_equal(x[55:81],np.arange(26))
        np.testing.assert_allclose(y[49:52],[0,1,0],atol=1e-15)
        np.testing.assert_allclose(decode_response(x[:55],y[:87]),native_state55(end,ev,qids,dofs,0,0),atol=1e-12)
