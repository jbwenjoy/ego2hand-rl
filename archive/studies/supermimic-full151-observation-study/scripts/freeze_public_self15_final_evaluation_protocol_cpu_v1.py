"""Root metadata-only freeze of a fixed public-native15 AFTER1000 evaluation."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).absolute()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def checks(v):
    assert v['checks'] and all(x is True for x in v['checks'].values())
    assert v['passed']==v['total']==len(v['checks'])


def main():
    tp=ROOT/'protocols/full151105_boundary_public_self15_matched_training_CPU_run_v1.json';train=read(tp);old=train['inputs'];run=Path(train['output'])
    assert read(run/'result.json')['status']=='COMPLETED_FIXED1000_MATCHED_PUBLIC_SELF15_COMPONENT_TRAINING'
    hp=ROOT/'outputs/full151105_boundary_public_self15_final1000_saved_independent_v1/report.json';health=read(hp);checks(health)
    assert health['iteration']==1000 and health['final_iteration_scope'] is True and health['own_CPU_checkpoint_loads']==3
    rp=ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_preparation_CPU_v1/READY.json';ready=read(rp);checks(read(ready['cpu_report']['path']))
    srp=ROOT/'outputs/full151105_public_self15_statistics_independent_preparation_v1/READY.json';sr=read(srp);checks(read(sr['cpu_report']['path']))
    reviewpath=ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_source_independent_v1/report.json';review=read(reviewpath);checks(review)
    assert review['schema']=='full151105_boundary_public_self15_final_evaluation_source_independent:1' and review['source_review_qualified'] is True
    contract=pin(ROOT/'docs/full151_boundary_public_self15_final_evaluation_contract_v1.md');assert contract['sha256']=='94268782d3f1cb282e123145cd85245d8f51ee8eabddc9e872c279e27e4d9046'
    statistics=pin(ROOT/'scripts/describe_public_self15_predictions_saved_cpu_v1.py');assert statistics['sha256']=='fb4465bcf12ae98db966420ce24b25961673f523353119ffa4ec9a91843cc2c8'
    inputs={k:old[k] for k in ('target_report','target_arrays','target_identities','target_saved8','target_source_inventory','initial_validation','initial_parity_report','cached_sensor_labels','cached_sensor_report','cached_sensor_saved7')}
    inputs.update(original_initial_component=old['initial_component'],initial_saved_parity10=old['initial_parity_saved_review'],initial_saved_baseline=old['initial_parity_baseline'],
                  training_protocol=pin(tp),training_result=pin(run/'result.json'),final_pair_manifest=pin(run/'paired_checkpoint_1000.json'),final_saved_health=pin(hp))
    assert len(inputs)==17
    sources=dict(train['source_pins'])
    for prefix,document in [('new_final_evaluator',ready),('new_statistics',sr)]:
        sources[prefix+'_proof']=document['cpu_report']
        for k,v in document['source_pins'].items():sources[prefix+'_'+k]=v
    sources.update(new_evaluation_READY=pin(rp),new_statistics_READY=pin(srp),new_scientific_contract=contract,new_statistics=statistics,
                   new_evaluation_source_review=pin(reviewpath),protocol_builder=pin(Path(__file__)))
    for value in [*inputs.values(),*sources.values()]:assert pin(value['path'])['sha256']==value['sha256']
    output=ROOT/'outputs/full151105_boundary_public_self15_final_evaluation_CPU_v1';assert not output.exists()
    prereg={'schema':'full151105_boundary_public_self15_final_evaluation_preregistration:1','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'FROZEN_BEFORE_ACTUAL_PUBLIC15_AFTER1000_PREDICTIONS','contract':contract,'input_pins':inputs,'source_pins':sources,
        'roles':['unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency'],
        'budget':{'own_initial_and_new_final_torch_loads':3,'native_WM_constructor':1,'raw_WM_deepcopies':2,'native_B152_FW':3,'aux_head_FW':2,'predicted_rows':456,'AD_optimizer_Actor_119f_MJB_Env_physics_GPU':0},
        'baseline':'Original ea051 native58/self15 actually evaluated and bitmatches e427 saved baseline; initialzero14 algebraic template, no thirdheadFW.',
        'loss':'Use frozen796c public_self15_losses on same outputs: state=rowmean52mean152, self=sum15BCEmean152, native_pair=mean152(pointrow+.1*selfrow)FP32, +modeFP32 then+impF64, cons0/control1/method. Capture8 independentnames and rawrowops. Never compare different consistency-weighted totals as a winner.',
        'metrics':'All152/13 overlapping cohorts,39cohort and456row tables; state52 six physical blocks+rawR9 separately, mode2/z/J/K/source totals/observeddefectmatch; cached15 BCE/Brier/TPFPFNTN/positive recall and perfeature/emptydenomnull, intensity/Penproxy, initialdrift. CapturedFP32 vs savedF64rescore distinguished.',
        'data_scope':'Training152 oldphase0/49/99/149 robotlineage. Cachedself15~15ms vsPOST20 state; no donor/fresh. Existingfreshphase24/74/124/148 data and6stable labels are different and cannot be joined as same predictions.',
        'comparison_scope':'Exploratory known-training-data evaluation, not blind or independent trajectories. Preserve prior no-self15 negative results; Lpublic changed and FP32 rowreduction clarified, so old/newstage public totals are different objectives. Do not tune lambda/checkpoint from this run.',
        'failure':'Anyrequiredsource/finite/chart/solve failure preserves rawpartial and stops wholecohort. No filtering, projection, repair, automaticretry or extraFW.',
        'eligibility':'Finalnewfamilyhealth5,current tested evaluator/statistics and independentSOURCE review. These only permit this fixed3FW job, not newtraining/Actor/physics.',
        'limits':'No task/CHORD/Actor, physicalgradient/forcecapacity/wholehidden, real geometry/human/ego/hardware or multipletrajectory benefit qualification.',
        'next':'Independent saved-only arithmetic review before interpretation; afterward preregister response/generalization diagnostics with explicit old/fresh source scopes.'}
    pp=ROOT/'protocols/full151105_boundary_public_self15_final_evaluation_CPU_prereg_v1.json';write(pp,prereg)
    p={'schema':'full151105_boundary_public_self15_final_evaluation_CPU_protocol:1','role':'fixed_initial_and_two_public_self15_final1000_component_evaluation',
       'CPU_run_authorized':True,'GPU_run_authorized':False,'training_authorized':False,'automatic_retry':False,'preregistration':pin(pp),
       'inputs':inputs,'source_pins':sources,'output_prefix':'full151105_boundary_public_self15_final_evaluation_CPU_','output':str(output),'no_scientific_success_gate':True}
    path=ROOT/'protocols/full151105_boundary_public_self15_final_evaluation_CPU_run_v1.json';write(path,p)
    print(json.dumps({'protocol':pin(path),'preregistration':pin(pp),'output':str(output),'actual_model_or_array_reads':0}))


if __name__=='__main__':main()
