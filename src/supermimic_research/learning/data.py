"""Registry-backed canonical replay reader, no legacy script imports.

All raw rows/masks are returned. Training selection is explicit and records
excluded IDs; finite values never silently upgrade unknown or partial labels.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np

REQUIRED={'x58','u20','y58','memory83','next_memory83','obs222','next_obs222','action20','h','n','executed_n','complete_H','WM_eligible','Actor_start_eligible','actor_absorbing_done','reference_timeout','row_id','family_id','episode_id','step_k','impulse22x6','meanFn22','coverage22'}

def _resolve(pin,registry=None):
    if registry is not None:return registry.resolve_pin(pin)
    path=Path(pin['path']).resolve(strict=True);b=path.read_bytes()
    if len(b)!=pin['bytes'] or hashlib.sha256(b).hexdigest()!=pin['sha256']:raise ValueError('Dataset array bytes differ from report pin')
    return path

@dataclass
class ReplayDataset:
    report_path:Path
    report:dict
    role:str
    raw_arrays:dict
    canonical:dict
    def wm_indices(self):return np.flatnonzero(self.canonical['WM_eligible'])
    def actor_indices(self,horizon=2):
        if type(horizon) is not int or horizon<1:raise ValueError('Positive imagination horizon')
        d=self.canonical;return np.flatnonzero(d['Actor_start_eligible']&d['WM_eligible']&(d['step_k']+horizon<=100))
    def selection_receipt(self):
        d=self.canonical;return dict(role=self.role,total_rows=len(d['row_id']),WM_included_row_ids=d['row_id'][d['WM_eligible']].tolist(),WM_excluded_row_ids=d['row_id'][~d['WM_eligible']].tolist(),partial_row_ids=d['row_id'][~d['complete_H']].tolist(),raw_rows_retained=True)

def read_partition(report_path,*,role='train',for_training=False,registry=None):
    if for_training and role!='train':raise ValueError('Only explicitly declared TRAIN partition may fit')
    path=Path(report_path).resolve(strict=True);r=json.loads(path.read_text(encoding='utf8'));schema=r['schema']
    if schema=='phone_tripod_replay_training_dataset:1':
        if not r.get('engineering_qualified') or not r.get('checks') or not all(v is True for v in r['checks'].values()):raise ValueError('Qualified canonical replay report required')
        if role not in ('train','monitor'):raise ValueError('Declared tripod train/monitor only')
        array_pin=r['partition_arrays'][role]
    elif schema=='phone_tripod_online_common_replay_dataset:1':
        if role!='train' or r.get('data_training_ready') is not True:raise ValueError('Online common pool is TRAIN only and must be ready')
        array_pin=r['common_pool']
    else:raise ValueError('Unsupported replay schema; heldout action-response is never a training replay')
    with np.load(_resolve(array_pin,registry),allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
    fieldmap=r['field_map']
    if not REQUIRED<=fieldmap.keys():raise ValueError('Complete phone replay field map required')
    d={key:raw[field] for key,field in fieldmap.items()};n=len(d['row_id'])
    if any(len(v)!=n for v in d.values()):raise ValueError('All canonical fields retain the same rows')
    for key in ['complete_H','WM_eligible','Actor_start_eligible','actor_absorbing_done','reference_timeout']:
        if d[key].dtype!=np.bool_:raise ValueError('Typed explicit mask '+key)
    wm=d['WM_eligible'];actor=d['Actor_start_eligible'];complete=d['complete_H']
    if np.any(wm&~complete) or np.any(actor&~wm):raise ValueError('Unknown/partial rows cannot be upgraded')
    for key,width in [('x58',58),('u20',20),('y58',58),('memory83',83),('next_memory83',83),('obs222',222),('next_obs222',222),('action20',20)]:
        if d[key].shape!=(n,width) or not np.isfinite(d[key][wm]).all():raise ValueError('Eligible shape/finite '+key)
    if d['impulse22x6'].shape!=(n,22,6) or d['meanFn22'].shape!=(n,22) or d['coverage22'].shape!=(n,22):raise ValueError('Canonical22 target shapes')
    if any(not np.isfinite(d[key][wm]).all() for key in ['impulse22x6','meanFn22','coverage22']):raise ValueError('Known physical target finite')
    if not np.isnan(d['y58'][~complete]).all() or not np.all(d['h'][wm]*d['n'][wm]==.02) or not np.all(d['executed_n'][wm]==d['n'][wm]):raise ValueError('Exact20ms known targets and NaN partial-H targets')
    if schema=='phone_tripod_replay_training_dataset:1' and set(d['family_id'])!=set(r['family_split'][role]):raise ValueError('Actual families must match frozen partition')
    return ReplayDataset(path,r,role,raw,d)

def from_registry(registry,name,*,role='train',for_training=False):
    registered_role=registry.entries[name].role.lower()
    if for_training and ('heldout' in registered_role or 'parity' in registered_role or registered_role.startswith(('test','monitor','validation'))):raise ValueError('Artifact registry role prohibits fitting: '+registered_role)
    return read_partition(registry.resolve(name),role=role,for_training=for_training,registry=registry)

def prepare_WM_batch(dataset,geometry,physical,*,device='cpu',chunk_rows=4,codec=None):
    """Explicit eligible selection, physical targets and current-only cache."""
    d=dataset.canonical;ii=dataset.wm_indices()
    if not len(ii):raise ValueError('No complete known WM rows; never substitute partials')
    rows=dict(x=d['x58'][ii],u=d['u20'][ii],y=d['y58'][ii],h=d['h'][ii],n=d['n'][ii],J=d['impulse22x6'][ii],Fn=d['meanFn22'][ii],cov=d['coverage22'][ii])
    result,scope=prepare_physical_rows(rows,geometry,physical,device=device,chunk_rows=chunk_rows,codec=codec)
    return result,dict(selection=dataset.selection_receipt(),geometry_cache=scope)

def read_original_train(registry,report_name='old_original_TRAIN',labels_name='old_pair_labels'):
    """Original684 complete TRAIN rows, explicitly joined by saved row IDs.

    This deliberately opens only report.outputs.train and TRAIN720 pair labels.
    Original validation/test pins may be in the JSON but are never NPZ-opened.
    """
    r=json.loads(registry.resolve(report_name).read_text(encoding='utf8'));l=json.loads(registry.resolve(labels_name).read_text(encoding='utf8'))
    if r['schema']!='phone_episode_acquisition_result:1' or not r['partition_qualified']:raise ValueError('Original family-partitioned acquisition required')
    if l['schema']!='original_TRAIN_pair22_saved_contact_export_result:1' or not l['checks'] or not all(v is True for v in l['checks'].values()):raise ValueError('Qualified TRAIN pair22 labels required')
    with np.load(_resolve(r['outputs']['train'],registry),allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
    with np.load(_resolve(l['outputs']['labels'],registry),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    index={int(v):i for i,v in enumerate(a['sample_id_i64'])};keep=o['training_eligible_bool']
    if keep.dtype!=np.bool_ or len(keep)!=720 or int(keep.sum())!=684:raise ValueError('Frozen original TRAIN684 +36partial identity')
    rows=np.flatnonzero(keep);li=np.array([index[int(v)] for v in o['row_id_i64'][rows]])
    if not a['physical_H_labels_known'][li].all():raise ValueError('Known original physical H labels')
    result=dict(x=o['x58_F64'][keep],u=o['u20_F64'][keep],y=o['y58_F64'][keep],h=o['native_dt_s_F64'][keep],n=o['n_substeps_i64'][keep],J=a['impulse_source_BODY_moving_COM22x6'][li],Fn=a['mean_positive_Fn22_N'][li],cov=a['Fn_gt_0p1_coverage22'][li])
    return result,dict(included_row_ids=o['row_id_i64'][keep].tolist(),excluded_partial_row_ids=o['row_id_i64'][~keep].tolist(),raw_rows=720,complete_rows=684,pair_ids=a['pair_other_geom_ids'].tolist(),VAL_TEST_NPZ_reads=0)

def prepare_physical_rows(rows,geometry,physical,*,device='cpu',chunk_rows=4,codec=None):
    import torch
    from . import world_model as WM
    if codec is None:from supermimic_research.runtime import structured as codec
    g=physical['g'].detach().cpu().numpy() if isinstance(physical['g'],torch.Tensor) else np.asarray(physical['g']);T=rows['h']*rows['n']
    if not all(np.isfinite(rows[k]).all() for k in ['x','u','y','J','Fn','cov']):raise ValueError('Explicit selected complete physical rows must be finite')
    heads=codec.encode_observed_numpy(rows['x'],rows['y'],rows['J'],T,g)
    head=np.c_[heads['joint_delta40'],heads['COM_ballistic_residual3_m'],heads['world_rotation_delta3_rad'],rows['J'].reshape(-1,132)]
    F,scope=WM.precompute_features(rows,geometry,chunk_rows=chunk_rows)
    result=dict(x=torch.tensor(rows['x'],dtype=torch.float64,device=device),y=torch.tensor(rows['y'],dtype=torch.float64,device=device),T=torch.tensor(T,dtype=torch.float64,device=device),head=torch.tensor(head,dtype=torch.float64,device=device),Fn=torch.tensor(rows['Fn'],dtype=torch.float32,device=device),cov=torch.tensor(rows['cov'],dtype=torch.float32,device=device),feature_F64=torch.tensor(F,dtype=torch.float64,device=device))
    return result,scope
