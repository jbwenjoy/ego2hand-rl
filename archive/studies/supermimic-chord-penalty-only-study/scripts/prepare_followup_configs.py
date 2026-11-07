"""CPU-only preparation of P_ONLY bindings/configurations; never trains."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('D:/workspace/supermimic-chord-study')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    assert not any((ROOT/'outputs').glob('penalty_only_paired_v2_2000iter_seed*')),'Do not rebind after training starts'
    baseline=ROOT/'source_baselines'
    files={'entry':ROOT/'scripts/train_controlled_pen.py','penalty_wrapper':ROOT/'src/penalty_study/penalty_only_objective.py',
           'followup_helper':ROOT/'src/penalty_study/followup.py','original_trainer':baseline/'train_controlled_pen.py',
           'entry_diff':baseline/'trainer_unified.diff','allowed_changes':baseline/'allowed_changes.json',
           'auditor':ROOT/'scripts/audit_training_checkpoint.py','queue':ROOT/'scripts/run_matched_training_queue.py',
           'launcher':ROOT/'scripts/start_matched_training_queue.ps1'}
    pins={name:{'path':str(path),'sha256':sha(path)} for name,path in files.items()}
    assert pins['original_trainer']['sha256']=='823711c1a7c5876942506986741f20171b3c09a292bf3656bda17c1b5a3fd65a'
    templates={str(seed):{'path':str(MAIN/f'outputs/training_configs/replication_seeds123_v1/a2_paired_v2_2000iter_seed{seed}.json'),
                         'sha256':sha(MAIN/f'outputs/training_configs/replication_seeds123_v1/a2_paired_v2_2000iter_seed{seed}.json')} for seed in (1,2,3)}
    native=MAIN/'outputs/a2_paired_v2_clip007_2000iter_seed1/source/inventory.json'
    vendor={path.relative_to(ROOT/'src').as_posix():sha(path) for path in sorted((ROOT/'src/chord_study').glob('*.py'))}
    assert len(vendor)==12
    smoke=read(MAIN/'outputs/training_configs/a2_bounded_10iter_smoke_seed0.json')
    inherited=ROOT/'configs/paired-contact-pilot-v2.json'
    protocol={'schema':'supermimic_chord_penalty_only_followup:1','method':'P_ONLY','seeds':[1,2,3],
              'status':'CPU_preparation_no_GPU_authorized','n3_outcomes_known_before_followup':True,
              'scientific_scope':'Exploratory after A1L/A2 n3, not retroactive preregistration; existing arms and gates remain unchanged',
              'inherited_paired_protocol_sha256':sha(inherited),'source_bindings':pins,'vendored_chord_study_sha256':vendor,
              'native_inventory':{'path':str(native),'sha256':sha(native)},'A2_templates':templates,
              'engineering_smoke_training':smoke['training'],
              'objective':{'support_coefficient':0.,'penetration_coefficient':.1,'caller_dt_s':.02,'formula':'-.1*penetration_prior'},
              'penalty_sharing':'Direct exact original A2 terms[geometry_intent_penetration_prior], no duplicated penalty or cancellation',
              'native_world_model_collect_control_and_thresholds':'unchanged frozen8237 entry body; only prior factory/label/provenance differs',
              'primary_budget':{'iterations':2000,'num_envs':128,'wm_updates':4000,'aux_failure_updates':4000,'actor_updates':1800,'critic_updates':7200,'real_transitions':648192},
              'secondary_OS_wall_seconds':1200,'same_physical_acceptance_and_512contact_diagnostic':True,
              'ordering':'P_ONLY seed1,seed2,seed3 on one GPU, fresh runs; no quality stopping/retry',
              'shared_physical_GPU_lock':'D:/workspace/supermimic-chord-study/.locks/single_gpu_training.lock',
              'historical_controls':'Prior A1L/A2 seeds1,2,3 are frozen; later new-arm batch is not a contemporaneously randomized3arm experiment',
              'identification':'P_ONLY-A1L: penalty increment; A2-P_ONLY: support increment conditional on penalty; no full2x2 interaction without support-only'}
    protocol_path=ROOT/'configs/penalty-only-followup-v1.json';write(protocol_path,protocol)
    configs=[]
    for seed in (1,2,3):
        cfg=read(templates[str(seed)]['path']);cfg.update(method='P_ONLY',label=f'Post-n3 exploratory P_ONLY seed{seed}: exact A2 penalty, support0, native matched2000',
          output_dir=f'outputs/penalty_only_paired_v2_2000iter_seed{seed}',followup_protocol_file='configs/penalty-only-followup-v1.json',followup_protocol_sha256=sha(protocol_path))
        path=ROOT/f'configs/training/penalty_only_2000iter_seed{seed}.json';write(path,cfg);configs.append((seed,path,cfg))
    engineering=dict(configs[0][2]);engineering.update(label='P_ONLY10iter engineering smoke, not task comparison',output_dir='outputs/penalty_only_10iter_smoke_seed1',training=smoke['training'],run_role='engineering_smoke',max_attempt_seconds=600)
    write(ROOT/'configs/training/penalty_only_10iter_smoke_seed1.json',engineering)
    original_queue=read(MAIN/'configs/replication-seeds123-queue-v1.json')
    manifest=dict(original_queue)
    manifest.update(queue_id='penalty_only_seeds123_v1',status='CPU_prepared_not_started',method='P_ONLY',seeds=[1,2,3],
                    queue_output=str(ROOT/'outputs/penalty_only_seeds123_queue_v1'),trainer=str(files['entry']),trainer_sha256=sha(files['entry']),
                    audit_script=str(files['auditor']),audit_script_sha256=sha(files['auditor']),
                    queue_runner=str(files['queue']),queue_runner_sha256=sha(files['queue']),launcher=str(files['launcher']),launcher_sha256=sha(files['launcher']),
                    paired_protocol_file=str(inherited),paired_protocol_sha256=sha(inherited),followup_protocol_file=str(protocol_path),followup_protocol_sha256=sha(protocol_path),
                    source_inventory=str(native),source_inventory_sha256=sha(native))
    # Remove inherited claims/order intended specifically for six original-arm runs.
    manifest.pop('seed0_is_pilot_excluded_from_primary_replication_summary',None)
    manifest['ordering_rule']={'seed_order':[1,2,3],'method':'P_ONLY','single_GPU_serial':True,'scope':'Historical A1L/A2 controls remain immutable; no randomized3-arm batch claim'}
    manifest['templates']={'P_ONLY':{'path':str(configs[0][1]),'sha256':sha(configs[0][1])}}
    manifest['jobs']=[{'position':index,'seed':seed,'method':'P_ONLY','config':str(path),'config_sha256':sha(path),
                       'output':str(ROOT/cfg['output_dir']),'observed_seed0_seconds_for_ETA':None} for index,(seed,path,cfg) in enumerate(configs,1)]
    manifest['estimated_total_seconds_from_pilot_costs']=3*1346.715981766698
    manifest['ETA_scope']='Conservative provisional A2 mean wall cost as upper planning reference; measure P_ONLY smoke before launch, not assumed equal compute'
    manifest['shared_physical_GPU_lock']=protocol['shared_physical_GPU_lock']
    write(ROOT/'configs/penalty-only-queue-v1.json',manifest)
    sys.path.insert(0,str(ROOT/'src'));from penalty_study.followup import validate
    for _,_,cfg in configs:validate(cfg,ROOT)
    validate(engineering,ROOT)
    print(json.dumps({'status':'prepared','vendor_exact_count':len(vendor),'entry_sha256':pins['entry']['sha256'],
                      'wrapper_sha256':pins['penalty_wrapper']['sha256'],'followup_protocol_sha256':sha(protocol_path),
                      'queue_manifest_sha256':sha(ROOT/'configs/penalty-only-queue-v1.json'),'no_torch_import':'torch' not in sys.modules},indent=2))


if __name__=='__main__':main()
