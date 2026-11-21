"""Read-only CPU audit of every FlashSAC checkpoint tensor using the tagged learner's tree layout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from safetensors import safe_open

from flash_chord.configuration import instantiate_typed
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.learner import create_learner
from flash_chord.training.flash_sac.checkpoint import actor_inference_state,training_checkpoint


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def schema(directory,last):
    cfg=checkpoint_config(read_checkpoint_metadata(directory/f'policy_{last}.safetensors'))
    training=instantiate_typed(OmegaConf.create(cfg['training']),TrainingConfig)
    pol=json.loads((directory.parent/'policy_schema.json').read_bytes());crit=json.loads((directory.parent/'critic_schema.json').read_bytes())
    obs=pol['observation']['dimension'];ad=pol['action']['dimension'];ct=crit['context']['dimension']
    with safe_open(directory/f'replay_{last}.safetensors',framework='np') as f:
        m=json.loads(f.metadata()['manifest']);objectives=m[2]['shape'][-1]
    def build():
        state,_=create_learner(training,jnp.zeros((training.world_count,obs)),jnp.zeros((training.world_count,ct)),
            action_dim=ad,objective_term_count=objectives,critic_context_names=tuple(x['name'] for x in crit['context']['blocks']))
        return actor_inference_state(state),training_checkpoint(state),state.replay
    t=jax.eval_shape(build)
    paths={}
    for mode,tree in zip(['policy','state','replay'],t):
        leaves,_=jax.tree_util.tree_flatten_with_path(tree)
        paths[mode]=[(jax.tree_util.keystr(path),list(value.shape)) for path,value in leaves]
    return paths,pol['observation']['blocks']


def summarize(path,layout,blocks):
    bad={};groups={};bad_count=0;scalars={}
    with safe_open(path,framework='np') as f:
        md=f.metadata(); manifest=json.loads(md['manifest'])
        if len(manifest)!=len(layout):raise ValueError('Checkpoint schema/leaf count changed')
        for entry,(name,shape) in zip(manifest,layout):
            if entry['shape']!=shape:raise ValueError(f'Unexpected leaf shape {name}: {entry["shape"]} != {shape}')
            a=f.get_tensor(entry['key'])
            group='.'.join(name.lstrip('.').split('.')[:2]).split("['")[0]
            info=groups.setdefault(group,dict(leaves=0,bad_leaves=0,nonfinite_values=0))
            info['leaves']+=1
            if not shape and a.size==1:scalars[name]=float(a.item()) if np.isfinite(a.item()) else 'nonfinite'
            if not np.issubdtype(a.dtype,np.inexact):continue
            invalid=~np.isfinite(a)
            if not invalid.any():continue
            axes=list(a.shape)
            entry_meta=dict(flat_key=entry['key'],shape=shape,nan=int(np.isnan(a).sum()),posinf=int(np.isposinf(a).sum()),neginf=int(np.isneginf(a).sum()))
            n=int(invalid.sum());bad_count+=n;info['nonfinite_values']+=n;info['bad_leaves']+=1
            if len(axes)>1:
                rowmask=np.any(invalid,axis=tuple(range(1,invalid.ndim)))
                entry_meta['bad_rows']=int(rowmask.sum())
                entry_meta['example_row_ids']=np.flatnonzero(rowmask)[:5].tolist()
            if a.shape[-1]==blocks[-1]['end']:
                flat=invalid.reshape(-1,a.shape[-1])
                mask=flat.any(axis=0)
                dims=np.flatnonzero(mask)
                entry_meta['bad_observation_dims']=dims.tolist()
                entry_meta['bad_observation_blocks']={b['name']:int(mask[b['start']:b['end']].sum()) for b in blocks if mask[b['start']:b['end']].any()}
            bad[name]=entry_meta
        sortedkeys=sorted(bad)
        return dict(file=path.name,sha256=sha(path),size_bytes=path.stat().st_size,
                    checkpoint_set_id=md.get('user.checkpoint_set_id'),environment_steps=int(md['user.environment_steps']),
                    optimizer_steps=int(md['user.global_update_step']),leaf_count=len(layout),bad_leaf_count=len(bad),
                    total_nonfinite=bad_count,groups=groups,bad_leaves={k:bad[k] for k in sortedkeys},scalar_counters=scalars)


def main(a):
    a.out.mkdir(exist_ok=False,parents=True)
    directory=a.checkpoints
    steps=sorted(int(x.stem.split('_')[-1]) for x in directory.glob('policy_*.safetensors'))
    paths,blocks=schema(directory,steps[-1])
    result=[]
    for step in steps:
        files={x:summarize(directory/f'{x}_{step}.safetensors',paths[x],blocks) for x in ('policy','state','replay')}
        assert len(set(v['checkpoint_set_id'] for v in files.values()))==1
        result.append(dict(step=step,files=files))
        print(step,{k:v['bad_leaf_count'] for k,v in files.items()},flush=True)
    report=dict(device='CPU only',method='Tagged create_learner abstract templates with shape validation, read-only numeric safetensors scans',
                observation_layout=blocks,snapshots=result)
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    log=directory.parent/'learner_metrics.jsonl'
    if log.exists():
        abnormal=[];first_by_key={}
        for line in log.read_text().splitlines():
            row=json.loads(line)
            bad=[k for k,v in row['metrics'].items() if v is None or isinstance(v,(int,float)) and not np.isfinite(v)]
            if bad:
                abnormal.append(dict(step=row['step'],null_or_nonfinite=bad))
                for k in bad: first_by_key.setdefault(k,row['step'])
        (a.out/'metric_anomalies.json').write_text(json.dumps(dict(source_sha256=sha(log),rows=abnormal,
                 first_by_key=first_by_key,scope='LocalMetrics writes nonfinite upstream scalars as JSON null; individual nulls can also represent empty episode means.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
