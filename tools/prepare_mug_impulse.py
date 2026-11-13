"""Before fitting: pin actual residual labels, verify oracle and frozen views."""
from pathlib import Path
import argparse,json,os,sys,tomllib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.impulse_codec import observed_defect_numpy,close_velocities
from supermimic_research.learning.impulse_response import ImpulseView,HEAD_SCHEMA
from supermimic_research.learning.response_training import restore_response
from supermimic_research.learning.response_rollout import load_rollout_cases,rollout_metrics
from supermimic_research.learning.paired_response import intervention_metrics


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main(args):
    config_path=args.config.resolve();config=tomllib.loads(config_path.read_text());out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified preparation output')
    paths={k:(config_path.parent/config[k]).resolve() for k in ['warm_health','frozen_paired_health','intervention_review','old_packed','old_probes']};docs={k:json.loads(p.read_text()) for k,p in paths.items()}
    for doc in docs.values():
        for pin in doc['sources'] if 'sources' in doc else [doc['source']]:verify_file_pin(pin)
    warm,paired,review,old=[docs[k] for k in ['warm_health','frozen_paired_health','intervention_review','old_packed']]
    if (warm['passed'],warm['total'],paired['passed'],paired['total'],review['passed'],review['total'])!=(12,12,40,40,8,8):raise ValueError('Reviewed inputs required')
    new={r:dict(np.load(verify_file_pin(p))) for r,p in review['outputs'].items()};old_data={r:dict(np.load(verify_file_pin(p))) for r,p in old['outputs'].items()};cases,physics,rollout_pins=load_rollout_cases(review)
    raw={'TRAIN':{k:np.concatenate([old_data['TRAIN'][k],new['TRAIN'][k]]) for k in ['x','y']},'CHECK':new['CHECK']}
    mass,I,g=[physics[k] for k in ['mass','inertia_BODY','gravity']];m=torch.tensor(mass,dtype=torch.float64);it=torch.tensor(I);inv=torch.linalg.inv(it);gt=torch.tensor(g)
    targets={};oracle={}
    for role,data in raw.items():
        d=observed_defect_numpy(data['x'],data['y'],mass,I,g);heads=data['y'].copy();heads[:,46:52]=d;targets[role]=heads
        x=torch.tensor(data['x']);h=torch.tensor(heads)
        exact=close_velocities(x,h,m,it,inv,gt,torch.tensor(d)).numpy();zero=close_velocities(x,h,m,it,inv,gt).numpy()
        quantized=close_velocities(x.float(),h.float(),m,it,inv,gt,torch.tensor(d)).numpy()
        oracle[role]=dict(rows=len(x),observed_defect_RMS6=np.sqrt(np.mean(d*d,axis=0)).tolist(),observed_defect_abs_max6=abs(d).max(0).tolist(),
            true_heads_with_native_defect_velocity_max_error=float(abs(exact[:,46:52]-data['y'][:,46:52]).max()),
            true_heads_zero_defect_velocity_max_error=float(abs(zero[:,46:52]-data['y'][:,46:52]).max()),
            F32_inputs_heads_with_native_defect_velocity_max_error=float(abs(quantized[:,46:52]-data['y'][:,46:52]).max()))
    warm_saved=torch.load(verify_file_pin(warm['checkpoint']),map_location='cpu',weights_only=False);scales=warm_saved['scales'];head_mean=scales['y_mean'].copy();head_scale=scales['y_scale'].copy();d=targets['TRAIN'][:,46:52]
    floors=np.r_[np.full(3,config['linear_defect_scale_floor_Ns']),np.full(3,config['angular_defect_scale_floor_Nms'])];head_mean[46:52]=d.mean(0);head_scale[46:52]=np.maximum(d.std(0),floors)
    sources=[file_pin(ROOT/p) for p in ['tools/prepare_mug_impulse.py','src/supermimic_research/mechanics/impulse_codec.py','src/supermimic_research/learning/impulse_response.py',
        'src/supermimic_research/learning/paired_response.py','src/supermimic_research/learning/response_training.py','src/supermimic_research/learning/response_rollout.py',
        'src/supermimic_research/mechanics/response_decode.py','src/supermimic_research/mechanics/transition_records.py']]
    out.mkdir();write(out/'intent.json',dict(schema='mug_impulse_head_preparation:1',config=file_pin(config_path),sources=sources,inputs={k:file_pin(p) for k,p in paths.items()},
        head_schema=HEAD_SCHEMA,scales_fit_role='TRAIN_ONLY',native_contact_labels='Preserved separately measured mean source-BODY moving-COM wrenches; residuals never replace them.',
        supervised_defect='Endpoint WORLD linear Ns/angular Nms residual, label only, unavailable as inference input.',neural_updates=0,native_steps=0))
    for role,h in targets.items():np.savez_compressed(out/(role+'_heads.npz'),heads=h)
    np.savez_compressed(out/'head_scales.npz',head_mean=head_mean,head_scale=head_scale)
    frozen=torch.load(verify_file_pin(paired['checkpoint']),map_location='cpu',weights_only=False);device=torch.device(config['device']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    views={}
    for mode in ['ABS','CF']:
        base,_,_=restore_response(frozen['arms'][mode],device);view=ImpulseView(base,physics).to(device)
        metrics,arrays=intervention_metrics(view,new['CHECK'],device);np.savez_compressed(out/(mode+'_zero_defect_view_CHECK.npz'),**arrays)
        rollout,arrays=rollout_metrics(view,cases,physics,device);np.savez_compressed(out/(mode+'_zero_defect_view_rollout.npz'),**arrays);views[mode]=dict(response=metrics,recorded_control=rollout)
    checks=dict(exact_oracle_roundtrip=all(v['true_heads_with_native_defect_velocity_max_error']<1e-11 for v in oracle.values()),
        TRAIN_counts=len(targets['TRAIN'])==3258,CHECK_counts=len(targets['CHECK'])==1920,nonzero_native_defects=bool(np.any(targets['TRAIN'][:,46:52]!=0)),
        finite_positive_head_scales=bool(np.isfinite(np.r_[head_mean,head_scale]).all() and np.all(head_scale>0)),preserved_force_labels=all(np.array_equal(targets[r][:,78:],raw[r]['y'][:,78:]) for r in raw))
    for pin in sources:verify_file_pin(pin)
    result=dict(schema='mug_impulse_head_preparation:1',sources=sources,config=file_pin(config_path),inputs={k:file_pin(p) for k,p in paths.items()},rollout_inputs=rollout_pins,
        physics={k:np.asarray(physics[k]).tolist() for k in ['mass','inertia_BODY','gravity']},head_schema=HEAD_SCHEMA,head_scales=file_pin(out/'head_scales.npz'),
        targets={r:file_pin(out/(r+'_heads.npz')) for r in targets},checks={k:bool(v) for k,v in checks.items()},passed=sum(checks.values()),total=len(checks),oracle=oracle,
        frozen_zero_defect_views=views,neural_updates=0,native_steps=0,scope='Label/decoder oracle and fixed-model reinterpretation only. Small residual by construction is not contact/response accuracy or physical control qualification. Same task, no RGB, policy or holding claim.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),passed=result['passed'],total=result['total'],oracle=oracle,
        frozen_small_object_response={mode:views[mode]['response']['by_amplitude']['0.04']['object12']['AD'] for mode in views})))
    if not all(checks.values()):raise ValueError('Bad impulse preparation; no fitting')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
