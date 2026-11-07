"""Exact saved integration-receipt/control aliases; no data modification."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
D=H/'dataset_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(p.read_text())
r=read(D/'report.json')
for record in r['outputs'].values():assert pin(Path(record['path']))==record
ids=read(D/'sample_identities322.json')
with np.load(D/'transitions322_F64.npz',allow_pickle=False) as z:U=z['control20'].copy();cohort=z['source_boundary_index'].copy()
assert U.shape==(322,20) and U.dtype==np.dtype('float64') and len(ids)==322
groups=defaultdict(list)
for i,s in enumerate(ids):
    assert s['sample_id']==i and s['source_boundary_index']==cohort[i]
    key=(s['input_integration_sha256'],s['target_integration_sha256'],U.dtype.str,U[i].shape,U[i].tobytes())
    groups[key].append(i)
parent=list(range(7))
def find(a):
    while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
    return a
def union(a,b):
    a,b=find(a),find(b)
    if a!=b:parent[max(a,b)]=min(a,b)
aliases=[]
for key,samples in groups.items():
    cs=sorted(set(int(cohort[i]) for i in samples))
    if len(cs)<2:continue
    for c in cs[1:]:union(cs[0],c)
    aliases.append(dict(input_integration_sha256=key[0],target_integration_sha256=key[1],control_dtype=key[2],control_shape=list(key[3]),control_raw_bytes_sha256=hashlib.sha256(key[4]).hexdigest(),sample_ids=samples,source_boundary_indices=cs,source_samples=[{k:ids[i][k] for k in ['sample_id','source_boundary_cohort','branch_id','H2_step','source_evaluation_index','selected_branch']} for i in samples]))
aliases.sort(key=lambda x:x['sample_ids'])
connected=defaultdict(list)
for i in range(7):connected[find(i)].append(i)
unions=[]
for gid,members in enumerate(sorted(connected.values())):
    samples=[i for i in range(322) if int(cohort[i]) in members]
    unions.append(dict(union_group_id=f'ALIAS_CONNECTED_PREFIX_{gid:03d}',source_boundary_indices=members,sample_count=len(samples),sample_ids=samples))
report=dict(schema='phone_transition_cross_cohort_receipt_alias_audit:1',source=pin(Path(__file__)),inputs=dict(dataset_report=pin(D/'report.json'),sample_identities=pin(D/'sample_identities322.json'),NPZ=pin(D/'transitions322_F64.npz')),alias_rule='Exact tuple(input integration SHA256, target integration SHA256, control20 dtype+shape+rawbytes); integration signatures source-bound in original identities. No state58 rounding or approximate matching.',cross_cohort_alias_group_count=len(aliases),cross_cohort_aliased_sample_count=len({i for a in aliases for i in a['sample_ids']}),aliases=aliases,cohort_union_group_count=len(unions),cohort_union_groups=unions,original_source_boundary_cohort_count=7,original_sample_count=322,all322_samples_retained=True,split_assigned=False,deduplication_applied=False,frozen_dataset_changed=False,new_data_model_FK_LP_physics_NN_training=0,scope='Saved producer integration-state receipts and raw command bytes only; actual full300 states not independently reconstructed. Alias-connected boundaries must not be treated as independent train/test groups. Even disconnected cohorts would remain one temporally related failed-hold/planner prefix, not independent trajectories or a state58 Markov certificate.')
out=H/'overlap_audit_v1.json'
with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({'report':pin(out),'cross_alias_groups':len(aliases),'aliases':[a['sample_ids'] for a in aliases],'union_groups':[g['source_boundary_indices'] for g in unions]}))
