"""Actual22-pair contact labels for original TRAIN only; preserve partials."""
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
out=H/'completed_v1';protocol=H/'protocol_v1.json';assert not out.exists() and not protocol.exists()
pr=dict(schema='original_TRAIN_pair22_saved_contact_export_protocol:1',source=pin(__file__),
    decode_source=pin(P/'phone_causal_impulse_labels_v1/export_saved_labels_v2.py'),
    inputs={'original_acquisition_report':pin(P/'phone_episode_acquisition_v1/acquired_v1/report.json'),
        'geometry_metadata':pin(P/'phone_geometry_features_v1/completed_v1/compiled_constants.json')},
    scope='Only original split=train720 rows. Preserve36actual prefixes but H-supervision remains NaN. No VAL/TEST traces, models, native or learned update.')
write(protocol,pr);out.mkdir();start=time.perf_counter()
sp=importlib.util.spec_from_file_location('frozen_saved_pair22_decoder',checked(pr['decode_source']));helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
acq=read(checked(pr['inputs']['original_acquisition_report']));meta=read(checked(pr['inputs']['geometry_metadata']))
rows=[r for r in read(checked(acq['sample_identities'])) if r['split']=='train'];assert len(rows)==720
pairids=[int(v['other_geom_id']) for v in meta['pairs']];assert len(pairids)==22
fields=['impulse_source_BODY_moving_COM22x6','mean_positive_Fn22_N','min_positive_Fn22_N','max_positive_Fn22_N','last_PRE_positive_Fn22_N','Fn_gt_0p1_coverage22','generated_contact_coverage22']
labels={k:np.full((720,22,6) if k==fields[0] else (720,22),np.nan) for k in fields};prefixJ=np.zeros((720,22,6));prefixFn=np.zeros((720,22))
known=np.zeros(720,dtype=np.bool_);duration=np.zeros(720);ticks=contacts=0;wmatch=0.
for i,r in enumerate(rows):
    with np.load(checked(r['trace']),allow_pickle=False) as z:
        a={k:z[k].copy() for k in ['source_state58','post_state58','pre_tick','contact_rows24','pre_contact_wrench_about_COM_world6']}
    n=len(a['pre_tick']);assert n==r['executed_substeps'] and n>0
    s=a['source_state58'];R=s[43:52].reshape(3,3);preCOM=np.vstack([s[None,40:43],a['post_state58'][:-1,40:43]])
    v=helper.decode_window(a['contact_rows24'],a['pre_tick'],preCOM,R,r['dt_s'],pairids,phone_gid=meta['phone_geom_id'])
    prefixJ[i]=v[fields[0]];prefixFn[i]=v['mean_positive_Fn22_N'];duration[i]=n*r['dt_s'];ticks+=n;contacts+=v['phone_contact_rows']
    wmatch=max(wmatch,float(np.max(abs(v['world_tick_wrench6']-a['pre_contact_wrench_about_COM_world6']))))
    if r['training_eligible']:
        assert r['complete_H20ms'] and duration[i]==.02;known[i]=True
        for key in fields:labels[key][i]=v[key]
checks=dict(original_TRAIN_only720=len(rows)==720,eligible684_partial36=int(known.sum())==684 and int((~known).sum())==36,
    partial_H_targets_NaN=bool(all(np.isnan(a[~known]).all() for a in labels.values())),prefix_observations_finite=bool(np.isfinite(prefixJ).all() and np.isfinite(prefixFn).all()),
    complete_H20ms=bool(np.all(duration[known]==.02)),decoded_all_pair_sum_matches_reader=wmatch<1e-12)
dest=out/'original_TRAIN720_pair22_labels.npz'
np.savez_compressed(dest,**labels,sample_id_i64=np.array([v['sample_id'] for v in rows],dtype=np.int64),physical_H_labels_known=known,
    actual_observed_duration_s=duration,observed_prefix_impulse_source_BODY_moving_COM22x6=prefixJ,observed_prefix_mean_Fn22_N=prefixFn,pair_other_geom_ids=np.array(pairids,dtype=np.int64))
write(out/'row_identities.json',[{k:r[k] for k in ['sample_id','family_id','episode_id','interval_index','split','training_eligible','complete_H20ms']} for r in rows])
report=dict(schema='original_TRAIN_pair22_saved_contact_export_result:1',status='PASS' if all(checks.values()) else 'FAILED_CHECKS',protocol=pin(protocol),
    checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),rows=720,known_H_rows=684,partial_H_unknown_rows=36,
    counts=dict(original_TRAIN_trace_NPZ_reads=720,original_VAL_TEST_trace_NPZ_reads=0,observed_PRE_ticks=ticks,phone_contact_rows=contacts,new_native_NN_AD_training_calls=0),
    decoded_wrench_max_abs_difference=wmatch,outputs={'labels':pin(dest),'identities':pin(out/'row_identities.json')},elapsed_s=time.perf_counter()-start,
    scope='Independent actual contact targets, sourceBODY axes/movingCOM, same canonical22 as geometryprovider. Partial prefixes are actual observations but never fake fullH targets. OriginalTRAIN includes states also used in new action-check baselines; only new response targets remain held out for training.')
write(out/'report.json',report);print(json.dumps(dict(report=pin(out/'report.json'),passed=report['passed'],total=report['total'])));assert all(checks.values())
