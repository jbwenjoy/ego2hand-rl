"""Package immutable targets by declared response role; no new target math."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
out=H/'partitioned_v1';out.mkdir(exist_ok=False)
report=read(H/'completed_v2/report.json');rows=read(report['outputs']['row_identities']['path']);path=Path(report['outputs']['all_labels']['path'])
assert pin(path)==report['outputs']['all_labels']
with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
outputs={};summaries={}
for role in sorted({v['partition'] for v in rows}):
    ids=np.array([v['branch_id'] for v in rows if v['partition']==role],dtype=np.int64)
    assert len(ids)==1476
    d={k:(a[ids] if a.ndim and a.shape[0]==2952 else a.copy()) for k,a in data.items()}
    assert np.array_equal(d['branch_id'],ids)
    file=out/(role+'.npz');np.savez_compressed(file,**d);outputs[role]=pin(file)
    summaries[role]=dict(rows=1476,known=int(d['physical_labels_known'].sum()),unknown=int((~d['physical_labels_known']).sum()))
receipt=dict(schema='causal_pair22_target_role_packaging:1',source=pin(__file__),producer_report=pin(H/'completed_v2/report.json'),source_labels=pin(path),
    summaries=summaries,outputs=outputs,actual_counts=dict(combined_saved_label_NPZ_reads=1,new_native_NN_AD_training_calls=0),
    scope='Saved target slicing only. Engineering process physically read combined targets; subsequent trainer can open FIT-only archive and hash CHECK/combined. This does not erase prior target exposure or make original TRAIN source states unseen.')
with (out/'report.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False)
print(json.dumps(dict(report=pin(out/'report.json'),summaries=summaries)))
