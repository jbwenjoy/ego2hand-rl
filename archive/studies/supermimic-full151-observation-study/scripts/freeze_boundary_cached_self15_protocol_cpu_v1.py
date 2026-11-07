"""Root metadata-only authorization of a saved-cache sensor-label evaluation."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).absolute()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    route_path = ROOT/'outputs/full151105_cached_self15_input_routes_source_only_v1/input_routes.json'
    routes = read(route_path);i = routes['inputs'];assert len(i)==13
    ready_path=Path(routes['cpu_ready']['path']);ready=read(ready_path);proof=read(ready['cpu_report']['path'])
    assert proof['passed']==proof['total']==5 and all(v is True for v in proof['checks'].values())
    assert ready['source_pins']==proof['source_pins']==routes['source_pins']
    for v in [*i.values(),*ready['source_pins'].values(),routes['source_contract'],routes['cpu_ready'],ready['cpu_report']]:
        assert pin(v['path'])['sha256']==v['sha256']
    assert routes['source_contract']['sha256']=='a535112324eeddb9113ebf47b2d68e379c3ddf5f191604eea498e756a30254f5'
    n=read(i['native_eval_report']['path'])
    assert n['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_AFTER1000_EVAL_NOT_CONFIRMATION' and n['counts']['native_FW_returned']==3 and n['counts']['predicted_rows']==456
    assert all(v is True for v in n['checks'].values())
    sources=dict(ready['source_pins']);sources.update(READY=pin(ready_path),cpu_proof=ready['cpu_report'],input_route_source=pin(route_path),Root_protocol_builder=pin(Path(__file__)))
    out=ROOT/'outputs/full151105_boundary_cached_self15_saved_CPU_v1';assert not out.exists()
    prereg={'schema':'full151105_cached_self15_saved_preregistration:1','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'FROZEN_BEFORE_ACTUAL_CACHED_LABEL_SCORING','source_contract':routes['source_contract'],'input_pins':i,'source_pins':sources,
        'root_source_review':'Read metadata bindings, exact152 row/snapshot/path/phase/branch/group/substeps, raw25->15 sensor mapping, stable F64 logistic scores, strict>0 zero-negative and all13 cohort definitions; no known blocking issue. Owner single Tiny fixture5/5 delegated, not rerun.',
        'budget':{'saved_forward_NPZ_reads':3,'label_rows':152,'channels':15,'roles':3,'new_weights_models_NN_FW_AD_MJB_Env_physics_GPU':0},
        'phase_scope':'Cached lastsolver_preintegration~kdt+.015 native finger sensor targets versus POST20 state; not post20 geometric clearance or HO/human labels. Keep all original152 including terminal k149->150, no u150.',
        'metrics':'Stable F64 BCE/Brier, per-channel and total strict logit>0 confusion and exactzero ties; soft intensity and original Pen proxy -0.02*sum(sigmoid) differences versus same cached binary labels. All3 roles/13 overlapping cohorts and456 role rows; no filtering, threshold tuning, accuracy success gate or composite winner.',
        'precision':'F64 saved-logit postprocessing, not exact original FP32 loss/reward replay. rewarddtF64=.02 is Pen50Hz dt, distinct from actual4*FP32 physics h.',
        'observed_before_this_stage':'Native evaluation and initial-relative self15 drift already observed. This stage is exploratory source-dependent cached-label accuracy, not blinded or independent trajectory evaluation.',
        'original_native_evaluation_contract_unchanged':True,'no_training_or_model_selection':True,
        'further_review':'Separate saved-only arithmetic crosscheck; even label accuracy does not qualify post20 collision geometry, full reward, physicalgradient, forcecapacity, task/Actor/CHORD or ego benefit.'}
    pp=ROOT/'protocols/full151105_boundary_cached_self15_saved_CPU_prereg_v1.json';write(pp,prereg)
    p={'schema':'full151105_cached_self15_label_evaluation_CPU_protocol:1','role':'all152_cached_native15_sensor_labels_saved_only',
       'CPU_run_authorized':True,'GPU_run_authorized':False,'training_authorized':False,'automatic_retry':False,'accuracy_gate':None,
       'preregistration':pin(pp),'source_contract':routes['source_contract'],'inputs':i,'source_pins':sources,
       'output_prefix':'full151105_boundary_cached_self15_saved_CPU_','output':str(out)}
    ppath=ROOT/'protocols/full151105_boundary_cached_self15_saved_CPU_run_v1.json';write(ppath,p)
    print(json.dumps({'preregistration':pin(pp),'protocol':pin(ppath),'output':str(out),'actual_array_or_label_scoring':0}))


if __name__=='__main__':main()
