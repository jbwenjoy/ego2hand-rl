"""Default SOURCE/JSON/hash route only; never imports task/Torch/NN/native."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def check(v):assert pin(v['path'])==v,'Changed source/operand pin'
def preflight(path):
    p=json.loads(Path(path).read_text());assert p['schema']=='phone_geometry_actor_integration_CPU_contract:1'
    for k in ['CPU_run_authorized','Actor_training_authorized','NN_forward_authorized','native_execution_authorized']:assert p[k] is False
    check(p['task_bundle']);b=json.loads(Path(p['task_bundle']['path']).read_text());assert b['schema']=='phone_fixed_wrist_ego61_tripod_task:1'
    for v in p['sources'].values():check(v)
    for v in [b['initial']['source_preload_report'],b['initial']['source_probe'],b['initial']['source_candidate'],b['reference_report'],b['reference_arrays'],b['basis']]:check(v)
    assert b['task_designated_geom_ids']==[15,28,41] and b['model_training_original_designated_ids']==[15,28]
    assert len(b['model_canonical_pair_geom_ids'])==22 and b['actor_obs_dim']==b['critic_dim']==222 and b['memory_dim']==83
    assert b['initial']['holding_qualified'] is False and b['initial']['source_full_integration300_receipt_available'] is False
    assert b['support']['observed_contact_known'] is False and b['support']['force_truth_known'] is False
    assert set(v['hand_geom_id'] for v in b['contact_template'])=={15,28,41}
    assert b['duration_s']==2 and b['control_dt_s']==.02 and b['control_intervals']==100 and b['native_substeps']==[64,128]
    assert p['preconditions']['manual_two_second_holding_success_required'] is False and p['binding']['old_DexSuperMimicRunner_drop_in_compatible'] is False
    for k in ['task','reader_replay']:ast.parse(Path(p['sources'][k]['path']).read_text())
    assert not any(k in sys.modules for k in ['torch','mujoco','warp'])
    return dict(schema='phone_geometry_actor_integration_source_metadata:1',status='SOURCE_METADATA_PASS_NOT_RUNTIME_OR_ACTOR_QUALIFICATION',protocol=pin(path),
        task_bundle=p['task_bundle'],modes=p['modes'],actor_obs_dim=222,memory_dim=83,current_geometry_CHORD_proxy_implemented=True,
        actual_contact_known=False,full_Actor_training_runner_implemented=False,runtime_bindings_pending=p['required_runtime_bindings'],
        metadata_NPZ_values_native_NN_FK_calls=0)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,default=H/'protocol_CPUFalse_v1.json');a=ap.parse_args();print(json.dumps(preflight(a.protocol)))
