"""TRAIN-mean validation baseline, no Torch/model/optimizer or TEST array read."""
from pathlib import Path
import importlib.util,json,hashlib,sys
import numpy as np

H=Path(__file__).resolve().parent;P=H.parent
spec=importlib.util.spec_from_file_location('phone_fresh_training_source',H/'train_phone_wm.py')
t=importlib.util.module_from_spec(spec);sys.modules[spec.name]=t;spec.loader.exec_module(t)
report_path=P/'phone_episode_acquisition_v1/acquired_v1/report.json';report=t.read(report_path)
plan=t.read(t.checked(report['family_plan']))
parts={s:t.load_partition(report['outputs'][s],[f['family_id'] for f in plan['families'] if f['split']==s]) for s in ['train','validation']}
train,val=parts['train'],parts['validation'];A=t._adapter()
delta=np.broadcast_to(train['physical_world_delta52'].mean(0),val['physical_world_delta52'].shape).copy()
decoded=A.decode_numpy(val['state58'],delta,state_schema=A.STATE_SCHEMA,delta_schema=A.DELTA_SCHEMA)
e=decoded-val['next_state58'];coverage=train['pad_coverage2'].mean(0)
force=train['logforce5'].mean(0)
j=dict(schema='phone_TRAIN_mean_validation_baseline:1',dataset_report=t.pin(report_path),source=t.pin(__file__),
    trainer_source=t.pin(H/'train_phone_wm.py'),train_rows=len(train['x80']),validation_rows=len(val['x80']),
    decoder_source=t.pin(A.__file__),baseline='Constant TRAIN mean physical delta52, coverage2 and log1p(force5); input/context ignored',
    COM_vector_RMS_m=float(np.sqrt(np.mean(np.sum(e[:,40:43]**2,axis=1)))),
    Rbody_raw9_Frobenius_RMS=float(np.sqrt(np.mean(np.sum(e[:,43:52]**2,axis=1)))),
    coverage_Brier_mean=float(np.mean((coverage-val['pad_coverage2'])**2)),
    logforce5_RMSE=np.sqrt(np.mean((force-val['logforce5'])**2,axis=0)).tolist(),
    physical_delta_RMSE_per_feature=np.sqrt(np.mean((delta-val['physical_world_delta52'])**2,axis=0)).tolist(),
    TRAIN_mean_delta52=train['physical_world_delta52'].mean(0).tolist(),
    dtype='F64 baseline; training model usesF32, no bit-equivalence claim',
    counts=dict(train_validation_NPZ_reads=2,TEST_NPZ_reads=0,NN_training_native_GPU=0),
    scope='Local held-family validation comparator. Not independent ego, policy, rollout, or true physics Jacobian qualification.')
out=H/'constant_mean_baseline_v1';out.mkdir()
with (out/'report.json').open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False)
print(json.dumps({k:j[k] for k in ['train_rows','validation_rows','COM_vector_RMS_m','Rbody_raw9_Frobenius_RMS','coverage_Brier_mean']}))
