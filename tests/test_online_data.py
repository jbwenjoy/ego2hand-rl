"""Online identity/mask/mix boundaries using only saved NumPy fixtures."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supermimic_research.artifacts import ArtifactRegistry, file_pin
from supermimic_research.learning import online_data as O


class OnlineDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        manifest = self.root / "artifacts.toml"; manifest.write_text('schema=1\n[roots]\ndata="."\n', encoding="utf8")
        self.registry = ArtifactRegistry(manifest)

    def tearDown(self):
        self.temp.cleanup()

    def pool(self, name, *, reward_offset=0., N=3, bad_reward=False, action_dtype=np.float64):
        n = 3; x = np.zeros((n, 58)); x[:,43:52] = np.eye(3).ravel()
        complete = np.array([True, True, False]); y = x.copy(); y[2] = np.nan
        c = dict(x58=x, u20=np.zeros((n,20), dtype=action_dtype), y58=y, memory83=np.zeros((n,83)), next_memory83=np.zeros((n,83)), obs222=np.zeros((n,222),np.float32), next_obs222=np.zeros((n,222),np.float32), action20=np.zeros((n,20)), h=np.full(n,.00015625), n=np.full(n,128,np.int64), executed_n=np.array([128,128,64]), complete_H=complete, WM_eligible=complete.copy(), Actor_start_eligible=complete.copy(), actor_absorbing_done=np.array([False,True,False]), reference_timeout=np.zeros(n,np.bool_), row_id=np.arange(n,dtype=np.int64), family_id=np.zeros(n,np.int64), episode_id=np.zeros(n,np.int64), step_k=np.array([0,99,1]), impulse22x6=np.zeros((n,22,6)), meanFn22=np.zeros((n,22)), coverage22=np.zeros((n,22)), H2_start_eligible=np.zeros(n,np.bool_))
        for key in ["impulse22x6","meanFn22","coverage22"]:c[key][2]=np.nan
        arrays={k:v[:N].copy() for k,v in c.items()}; arrays["reward_A1"]=(np.array([1.,2.,np.nan])+reward_offset)[:N]; arrays["reward_A2"]=(np.array([10.,20.,np.nan])+reward_offset)[:N]
        if bad_reward:arrays["reward_A2"][0]=np.nan
        npz=self.root/(name+".npz"); np.savez_compressed(npz,**arrays)
        report=dict(schema=O.SCHEMA, data_training_ready=True, common_pool=file_pin(npz), field_map={k:k for k in c}, mode_reward_keys=dict(zip(O.MODES,["reward_A1","reward_A2"])))
        path=self.root/(name+".json"); path.write_text(json.dumps(report),encoding="utf8")
        return file_pin(path)

    def test_repeated_local_ids_partial_masks_and_mode_rewards_survive(self):
        p0=self.pool("cycle1"); p1=self.pool("cycle2",reward_offset=100.)
        merged=O.merge_pools([p0,p1],self.registry); c=merged["canonical"]
        np.testing.assert_array_equal(c["row_id"],np.arange(6)); np.testing.assert_array_equal(c["episode_id"],[0,0,0,1,1,1])
        np.testing.assert_array_equal(c["_source_row_id"],[0,1,2,0,1,2])
        self.assertEqual([m["pool"]["sha256"] for m in merged["mapping"]],[p0["sha256"]]*3+[p1["sha256"]]*3)
        self.assertTrue(np.isnan(c["y58"][[2,5]]).all()); np.testing.assert_array_equal(c["WM_eligible"],[True,True,False]*2)
        np.testing.assert_equal(merged["rewards"][O.MODES[0]],[1.,2.,np.nan,101.,102.,np.nan])
        np.testing.assert_equal(merged["rewards"][O.MODES[1]],[10.,20.,np.nan,110.,120.,np.nan])
        # Imagined H2 start does not require the deliberately false native-H2 flag.
        np.testing.assert_array_equal(O.actor_indices(c,2),[0,3])
        rows,receipt=O.physical_rows(c); self.assertEqual(len(rows["x"]),4)
        self.assertEqual(receipt["included_global_row_ids"],[0,1,3,4])
        self.assertEqual([r["original_row_id"] for r in receipt["source_ids"]],[0,1,0,1])
        self.assertEqual([r["pool_index"] for r in receipt["source_ids"]],[0,0,1,1])

    def test_empty_duplicate_wrong_schema_and_unknown_reward_rejected(self):
        p=self.pool("one")
        with self.assertRaisesRegex(ValueError,"At least one"):O.merge_pools([],self.registry)
        with self.assertRaisesRegex(ValueError,"Duplicate"):O.merge_pools([p,p],self.registry)
        with self.assertRaisesRegex(ValueError,"Empty"):O.merge_pools([self.pool("empty",N=0)],self.registry)
        with self.assertRaisesRegex(ValueError,"per-mode reward"):O.merge_pools([self.pool("nan",bad_reward=True)],self.registry)
        wrong=self.root/"wrong.json"; wrong.write_text(json.dumps(dict(schema="heldout_action_response:1")))
        with self.assertRaisesRegex(ValueError,"Only online"):O.merge_pools([file_pin(wrong)],self.registry)

    def test_no_silent_cross_pool_dtype_promotion(self):
        with self.assertRaisesRegex(ValueError,"dtype/shape differs"):
            O.merge_pools([self.pool("f64"),self.pool("f32",action_dtype=np.float32)],self.registry)

    def test_prepare_pool_delegates_only_explicit_WM_rows(self):
        merged=O.merge_pools([self.pool("one")],self.registry)
        with patch.object(O.D,"prepare_physical_rows",return_value=({"cached":True},{"geometry":True})) as prepare:
            out=O.prepare_pool(merged,"geometry","physical",device="cpu",chunk_rows=4)
        self.assertEqual(len(prepare.call_args.args[0]["x"]),2)
        self.assertEqual(out["WM_selection"]["excluded_global_row_ids"],[2])
        self.assertEqual(out["WM_cache"],{"cached":True})
        np.testing.assert_array_equal(out["Actor_indices"],[0])

    def test_mixed_sampling_preserves_three_CPU_draws_and_block_order(self):
        class Index:
            def __init__(self,v):self.v=v
            def to(self,device):return self.v
            def tolist(self):return self.v.tolist()
        class NumpyAPI:
            calls=[]
            @classmethod
            def randint(cls,N,shape,*,generator,device):
                self.assertEqual(device,"cpu"); cls.calls.append((N,shape)); return Index(generator.integers(N,size=shape))
            @staticmethod
            def cat(values):return np.concatenate(values)
        def cache(N,base):
            x=(base+np.arange(N))[:,None].astype(np.float64)
            return dict(x=x,head=x*10)
        online,old,tripod=cache(3,100),cache(5,200),cache(7,300)
        expected=np.random.default_rng(42); indices=[expected.integers(N,size=n) for N,n in [(3,4),(5,2),(7,2)]]
        batch,receipt=O.mixed_WM_batch(online,old,tripod,np.random.default_rng(42),NumpyAPI)
        self.assertEqual(NumpyAPI.calls,[(3,(4,)),(5,(2,)),(7,(2,))])
        np.testing.assert_array_equal(batch["x"][:,0],np.r_[100+indices[0],200+indices[1],300+indices[2]])
        for name,ix,span in zip(["online","old_original","old_tripod"],indices,[[0,4],[4,6],[6,8]]):
            self.assertEqual(receipt[name]["local_indices"],ix.tolist()); self.assertEqual(receipt[name]["output_range"],span)


if __name__ == "__main__":
    unittest.main()
