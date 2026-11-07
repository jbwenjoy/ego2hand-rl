"""Root metadata-only preregistration of matched public native15 training."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).absolute()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def checks(r):
    assert r['checks'] and all(v is True for v in r['checks'].values())
    assert r['passed']==r['total']==len(r['checks'])


def main():
    old=read(ROOT/'protocols/full151105_boundary_impulse_matched_training_CPU_run_v1.json')
    inputs=dict(old['inputs'])
    cached=ROOT/'outputs/full151105_boundary_cached_self15_saved_CPU_v1'
    inputs.update(cached_sensor_labels=pin(cached/'joined152_cached_sensor_labels.json'),
                  cached_sensor_report=pin(cached/'report.json'),
                  cached_sensor_saved7=pin(ROOT/'outputs/full151105_boundary_cached_self15_saved_independent_v1/report.json'),
                  target_identities=pin(Path(inputs['target_report']['path']).parent/'group152_identities.json'))
    assert len(inputs)==16
    for key in ('target_saved8','initial_saved_health29','initial_parity_saved_review','cached_sensor_saved7'):checks(read(inputs[key]['path']))
    recipe=pin(ROOT/'protocols/full151105_boundary_public_self15_matched_training_recipe_v1.json')
    assert recipe['sha256']=='9e0765fd562c3e24bdb09d52f78219204af4228bb6e8b1d8ffd67104ce094bce'
    contract=pin(ROOT/'docs/full151_boundary_public_self15_matched_training_contract_v1.md')
    assert contract['sha256']=='5a12bb913b21a45f6cfc3bd6df39a8bbe4074e0588ac47f1791f9561ec3f4ed7'
    ready_path=ROOT/'outputs/full151105_boundary_public_self15_training_preparation_CPU_v1/READY.json'
    health_path=ROOT/'outputs/full151105_boundary_public_self15_checkpoint_auditor_preparation_CPU_v1/READY.json'
    ready=read(ready_path);health=read(health_path)
    checks(read(ready['cpu_report']['path']));checks(read(health['cpu_report']['path']))
    review_path=ROOT/'outputs/full151105_boundary_public_self15_training_source_independent_v1/report.json'
    review=read(review_path);checks(review)
    assert review['schema']=='full151105_boundary_public_self15_training_source_independent:1' and review['source_review_qualified'] is True
    sources=dict(old['source_pins'])
    for tag,document in [('public15_trainer',ready),('public15_checkpoint_reader',health)]:
        for k,v in document['source_pins'].items():sources[tag+'_'+k]=v
        sources[tag+'_proof']=document['cpu_report']
    sources.update(public15_trainer_READY=pin(ready_path),public15_reader_READY=pin(health_path),
                   public15_source_contract=contract,public15_source_review=pin(review_path),
                   public15_proposal=pin(ROOT/'docs/full151_boundary_public_self15_supervision_recipe_proposal_v1.md'),
                   cached_label_source_contract=pin(ROOT/'docs/full151_boundary_final_self15_evaluation_source_contract_v1.md'),
                   cached_label_output_manifest=pin(cached/'output_pins.json'),protocol_builder=pin(Path(__file__)))
    for v in [*inputs.values(),*sources.values(),recipe,contract]:assert pin(v['path'])['sha256']==v['sha256']
    output=ROOT/'outputs/full151105_boundary_public_self15_matched_training_CPU_v1';assert not output.exists()
    scientific={
        'schema':'full151105_boundary_public_self15_matched_training_preregistration:1',
        'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'FROZEN_BEFORE_ACTUAL_PUBLIC_NATIVE15_TRAINING','recipe':recipe,'source_contract':contract,
        'input_pins':inputs,'source_pins':sources,'source_review':pin(review_path),
        'hypothesis':'Restore the original native15 public supervision in BOTH arms before comparing consistency. Prior no-self15 fixed1000 negative results remain. This is an exploratory correction, not outcome-tuned consistency weight search or blind confirmation.',
        'public_loss':'For each row FP32 s=mean52((D52/s52)^2), c=sum15BCE; native_pair=mean152(s+.1*c)FP32. Public=FP32(native_pair+modeBCE)+impulseF64. Only method adds1*consF64; control0. Lstate remains mean152(s), self15sum/native_pair have separate names. No15mean,dt,windowfactor,classweights,detach or modegate.',
        'same_initial':'Both restart unmodified ea051 original initial, independently copy zero14 head/context. No continuation from prior different final1000s. Same fixed normalizers, actual physical context, order and Adam from empty state.',
        'budget':{'iterations_each_arm':1000,'own_initial_torch_load':1,'native_WM_constructor':1,'raw_deepcopies':2,'native_FW':2000,'fullbatch_rows':152,'predicted_rows':304000,'backward_steps':2000,'optimizer_steps':2000,'extra_validation_FW':0,'Actor_old119f_MJB_Env_physics_GPU':0},
        'checkpoints':[100,1000],'parameter_tensors':16,
        'snapshots':'PRE-I raw named gradients before/after clip1 and nativeD/squared52/pointrows/self15logits/target/BCEelements/selfrows/nativepairrows/8scalars; AFTER-I parameters/Adam and exact saved logprefix. All detachedCPU snapshots, no extraFW.',
        'initial_parity':'First actual training forward of each arm must match e427 native58/self15 bits and positivezero14. New point-row FP32 reduction may differ from oldflatmean byULP; do not compare scalar parity across definitions.',
        'label_scope':'Label-only joined152 JSON, exact original path/k/branch/snapshot/group/substep/order and raw25->15 mapping. Cachedlastsolver~15ms labels vs POST20 state, allrowsW1 including k149->150, no u150/donor/fresh. Labels never enter forward.',
        'first_checkpoint_pause':'After BOTH new100 checkpoints saved, NEW independent savedhealth positive and ETA saved, pause Goal per explicit user instruction. Background worker explicitly continues to fixed1000 independently. Do not substitute historical checkpoint health.',
        'ETA':'First100 paired elapsed costs with first3 trimmed; remaining900 pairs plus measuredsave and explicit20s finalization planning allowance/range0.85..1.3. Heuristic, not CI or speedup.',
        'no_resume_retry_or_model_selection':True,
        'limits':'Single cached robot lineage, 152 dependent training rows; not task/Actor/CHORD, forcecapacity/physicalgradient/wholehidden, multiple trajectory/ego/hardware qualification. Original reference/task failures remain; finalfit diagnostics require separate evaluation after resume.',
    }
    prereg_path=ROOT/'protocols/full151105_boundary_public_self15_matched_training_CPU_prereg_v1.json';write(prereg_path,scientific)
    protocol={'schema':'full151105_boundary_public_self15_matched_training_CPU_protocol:1',
        'role':'matched_native_self15_public_aux_vs_cross_target_consistency','CPU_run_authorized':True,'GPU_run_authorized':False,
        'automatic_retry':False,'background_continue_to_final':True,'preregistration':pin(prereg_path),'recipe':recipe,
        'source_contract':contract,'source_review':pin(review_path),'inputs':inputs,'source_pins':sources,
        'output_prefix':'full151105_boundary_public_self15_matched_training_CPU_','output':str(output),
        'first_checkpoint_user_pause_condition':scientific['first_checkpoint_pause']}
    pp=ROOT/'protocols/full151105_boundary_public_self15_matched_training_CPU_run_v1.json';write(pp,protocol)
    print(json.dumps({'protocol':pin(pp),'preregistration':pin(prereg_path),'output':str(output),'actual_arrays_weights_models_FW_AD':0}))


if __name__=='__main__':main()
