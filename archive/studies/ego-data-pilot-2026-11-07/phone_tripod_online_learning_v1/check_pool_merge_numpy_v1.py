"""One bounded pure NumPy identity/mask/reward contract, no learning."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
H=Path(__file__).resolve().parent
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
source=H/'train_online_matched_v1.py'
s=importlib.util.spec_from_file_location('online_actual_merge_contract',source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
pools=[dict(path='synthetic_pool_A',bytes=1,sha256='a'*64),dict(path='synthetic_pool_B',bytes=1,sha256='b'*64)]
yA=np.arange(3*58,dtype=np.float64).reshape(3,58);yA[2]=np.nan
data=[dict(row_id=np.array([0,1,2],dtype=np.int64),episode_id=np.array([0,0,1],dtype=np.int64),complete_H=np.array([True,True,False]),WM_eligible=np.array([True,True,False]),Actor_start_eligible=np.array([True,True,False]),y58=yA),
      dict(row_id=np.array([0,1],dtype=np.int64),episode_id=np.array([0,1],dtype=np.int64),complete_H=np.array([True,True]),WM_eligible=np.array([True,True]),Actor_start_eligible=np.array([True,True]),y58=np.full((2,58),9.,dtype=np.float64))]
rewards=[dict(rA=np.array([11.,12.,np.nan]),rB=np.array([21.,22.,np.nan])),dict(rA=np.array([31.,32.]),rB=np.array([41.,42.]))]
class SavedOnlyViews:
    calls=0
    def load_common_pool(self,pin,*,reward_mode):
        i=pools.index(pin);self.calls+=1
        return dict(canonical=data[i],arrays=rewards[i],raw_source=dict(mode_reward_keys={m.MODES[0]:'rA',m.MODES[1]:'rB'}))
reader=SavedOnlyViews();merged,mode_rewards,mapping=m.merge_pool_reports(pools,reader)
checks=dict(global_rows=np.array_equal(merged['row_id'],[0,1,2,3,4]),
    repeated_local_episodes_are_not_crosscycle_episodes=np.array_equal(merged['episode_id'],[0,0,1,2,3]),
    source_identity_mapping_exact=[(a['pool_index'],a['original_row_id'],a['original_episode_id']) for a in mapping]==[(0,0,0),(0,1,0),(0,2,1),(1,0,0),(1,1,1)],
    masks_unchanged=all(np.array_equal(merged[k],np.concatenate([d[k] for d in data])) for k in ['complete_H','WM_eligible','Actor_start_eligible']),
    partial_NaN_and_targets_unchanged=np.array_equal(merged['y58'],np.concatenate([d['y58'] for d in data]),equal_nan=True) and np.isnan(merged['y58'][2]).all(),
    both_rewards_preserved_and_distinct=np.array_equal(mode_rewards[m.MODES[0]],[11,12,np.nan,31,32],equal_nan=True) and np.array_equal(mode_rewards[m.MODES[1]],[21,22,np.nan,41,42],equal_nan=True),
    original_local_ID_arrays_not_mutated=np.array_equal(data[0]['row_id'],[0,1,2]) and np.array_equal(data[1]['episode_id'],[0,1]),
    no_backend_import='torch' not in sys.modules and 'mujoco' not in sys.modules)
report=dict(schema='phone_tripod_online_pool_merge_numpy_contract:1',source=pin(source),checker=pin(__file__),checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),synthetic_pool_views=reader.calls,NPZ_native_NN_optimizer_calls=0,scope='One identity/mask/reward merge contract; no learning, sampling or physics qualification')
with (H/'pool_merge_numpy_proof_v1.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(passed=report['passed'],total=report['total'],report=pin(H/'pool_merge_numpy_proof_v1.json'))))
assert all(checks.values())
