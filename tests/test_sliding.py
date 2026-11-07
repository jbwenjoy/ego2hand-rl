"""Patch migration axes, global reference composition and bounded restoration."""
from pathlib import Path
from itertools import product
import json,sys,tempfile,unittest
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry
from supermimic_research.mechanics.collision import CollisionAudit
from supermimic_research.mechanics.cached_collision import CachedCollisionAudit
from supermimic_research.mechanics.sliding import collider_axes,compose_deviation,GeometryRestoration,OFFSETS_M


class SlidingTests(unittest.TestCase):
    def test_compiled_long_axis_is_deterministic_and_rotates_with_geom(self):
        vertices=np.array(list(product((-.1,.1),(-.04,.04),(-.004,.004))))
        R=Rotation.from_rotvec([.4,-.1,.3]).as_matrix();a=collider_axes(vertices,R);b=collider_axes(vertices,R)
        self.assertEqual(a,b);np.testing.assert_allclose(a['axes']['long']['world'],R[:,0],atol=1e-14)
        self.assertAlmostEqual(a['axes']['long']['extent_m'],.2)
        with self.assertRaises(ValueError):collider_axes(np.array(list(product((-1.,1.),repeat=3))),np.eye(3))
        self.assertEqual(OFFSETS_M,(-.08,-.06,-.04,-.02,.02,.04,.06,.08))

    def test_global_reference_composes_left_rotation_of_seed_shift(self):
        q0=np.r_[np.zeros(20),[.1,.2,.3],[1.,0.,0.,0.]];restored=q0.copy();restored[0]=.1
        shift=np.array([.02,-.03,.01]);R=Rotation.from_rotvec([.1,.2,-.1]).as_matrix();t=np.array([-.01,.03,0.])
        local=dict(q20_delta_rad=np.zeros(20).tolist(),phone_root_delta_world_m=(R@(q0[20:23]+shift)+t-(q0[20:23]+shift)).tolist(),
            proposed_reference_left_transform_R=R.tolist(),proposed_reference_left_transform_t_m=t.tolist())
        out=compose_deviation(local,restored,q0,shift)
        expected=R@(q0[20:23]+shift)+t
        np.testing.assert_allclose(R@q0[20:23]+out['proposed_reference_left_transform_t_m'],expected,atol=1e-14)
        np.testing.assert_allclose(q0[20:23]+out['phone_root_delta_world_m'],expected,atol=1e-14)
        self.assertAlmostEqual(out['q20_delta_rad'][0],.1)

    def test_actual_geometry_restore_budget_has_fresh_final_and_no_forward(self):
        import mujoco
        registry=ArtifactRegistry(ROOT/'configs/artifacts.toml');model=mujoco.MjModel.from_binary_path(str(registry.resolve('phone_model')))
        q=np.array(json.loads(registry.resolve('phone_task').read_text())['initial']['qpos27_F64']);audit=CollisionAudit(model)
        engine=GeometryRestoration(model,audit,q)
        with tempfile.TemporaryDirectory() as tmp:result=engine.restore(Path(tmp)/'proposals.jsonl',maximum_calls=1)
        self.assertEqual(result['counts']['proposal_calls'],1);self.assertEqual(result['counts']['kinematics'],2)
        self.assertEqual(len(result['geometry']['rows']),211);self.assertEqual(result['geometry']['unknown'],0)
        self.assertEqual(result['counts']['forward'],0);self.assertEqual(result['counts']['integration_steps'],0)
        self.assertEqual(result['qpos27'].tobytes(),q.tobytes())

    def test_exact_pair_cache_matches_cold_geometry_across_ring_joint_change(self):
        import mujoco
        registry=ArtifactRegistry(ROOT/'configs/artifacts.toml');model=mujoco.MjModel.from_binary_path(str(registry.resolve('phone_model')))
        q=np.array(json.loads(registry.resolve('phone_task').read_text())['initial']['qpos27_F64']);cold=CollisionAudit(model);cached=CachedCollisionAudit(model)
        data=mujoco.MjData(model)
        for delta in (0.,0.,.002,0.):
            data.qpos[:]=q;data.qpos[16]+=delta;mujoco.mj_kinematics(model,data)
            self.assertEqual(cold.evaluate(data),cached.evaluate(data))
        self.assertGreaterEqual(cached.query_counts['hits'],211)
        self.assertLess(cached.query_counts['misses'],cached.query_counts['calls'])
        engine=GeometryRestoration(model,cached,q,variable_joints=20)
        with tempfile.TemporaryDirectory() as tmp:result=engine.restore(Path(tmp)/'proposals.jsonl',maximum_calls=1)
        self.assertEqual(len(result['best']['z_joint']),20)


if __name__=='__main__':unittest.main()
