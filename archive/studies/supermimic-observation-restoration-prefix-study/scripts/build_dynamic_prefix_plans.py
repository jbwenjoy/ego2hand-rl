"""Freeze two actual-bundle plans; CPU metadata only, no worker launch."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--driver',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,action='append',default=[])
    ap.add_argument('--authorize',action='store_true')
    ap.add_argument('--version',required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(ROOT/'scripts'))
    import track_prefix_warp as w
    import mujoco
    export=ROOT/'outputs/restored_prefix_control_export_v1'
    manifest=read(export/'reference_manifest.json')
    receipt=read(export/'receipt.json')
    if not all(v is True for v in receipt['gates'].values()):raise ValueError('Unqualified CPU bundle')
    if manifest['source_pins']!=receipt['source_pins']:raise ValueError('Different export identities')
    protocol=read(manifest['protocol']['path']);inputs=protocol['inputs']
    aliases={name:pin(export/(kind+'_manifest.json')) for name,kind in [('reference_manifest','reference'),('init_manifest','init'),('control_manifest','control')]}
    aliases.update({k:manifest[k] for k in ('protocol','constrained_protocol','input_parent_protocol')})
    aliases.update(export_receipt=pin(export/'receipt.json'),model=inputs['model'],context=inputs['actual_context'],robot_kinematics=inputs['kinematics'])
    pins={**aliases,'worker':pin(ROOT/'scripts/track_prefix_warp.py'),'supervisor':pin(args.driver),'runtime_IO':pin(ROOT/'vendor/runtime_io.py'),'measurement':pin(ROOT/'vendor/measured_contacts.py'),'reviewed_lifecycle':pin('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py'),'plan_builder':pin(__file__)}
    pins.update({'evidence__'+str(i):pin(path) for i,path in enumerate(args.evidence)})
    # No receipt self-reference: artifacts pin their sources; plans pin receipts.
    for key,value in manifest['source_pins'].items():pins['export_source__'+key]=value
    for value in pins.values():
        if pin(value['path'])['sha256']!=value['sha256']:raise ValueError('Pinned source drift')
    model=mujoco.MjModel.from_binary_path(inputs['model']['path'])
    ids=[model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)]
    policy={'schema':'matched_dynamic_prefix_campaign:1','conditions_in_order':list(w.CONDITIONS),'attempts_per_condition':1,'automatic_retry':False,'same_candidate_and_initialization':True,'same_14_control_write_events':True,'second_condition_requires_first_engineering_identity_full_horizon_capacity_OK':True,'first_quality_failure_does_not_change_second_plan':True,'no_repeat_or_history_selection':True,'no_learning':True,'scope':'Sequential known-history diagnostic N1 per condition; moving reference16/0.3s only. No method benefit/full3s/ego claim.'}
    tag=args.version
    for short,condition in zip(('hold','track'),w.CONDITIONS):
        plan={'schema':'restoration_dynamic_prefix_plan:1','retarget_variant_id':w.VARIANT,**aliases,'source_pins':pins,'control_condition':condition,'steps':60,'dt_s':.005,'duration_s':.3,'nworld':1,'nconmax':192,'njmax':4096,'hard_process_seconds':45,'soft_worker_seconds':40,'automatic_retry':False,'gates':w.FIXED_GATES,'joint_lower_rad':model.jnt_range[ids,0].tolist(),'joint_upper_rad':model.jnt_range[ids,1].tolist(),'output':str(ROOT/'outputs'/('dynamic_matched_'+short+'_'+tag)),'GPU_run_authorized':args.authorize,'training_authorized':False,'campaign_policy':policy}
        path=ROOT/'configs'/('dynamic-prefix-'+short+'-'+tag+'.json')
        with path.open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({'condition':condition,'plan':pin(path),'GPU_run_authorized':args.authorize}))
if __name__=='__main__':main()
