import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.pen_response import PenResponseWM
from supermimic_research.learning.pen_branch_evaluation import common_one_step
from supermimic_research.learning.pen_diagnostics import rollout_diagnostics
from supermimic_research.learning.pen_response_evaluation import response_diagnostics
from supermimic_research.learning.pen_stage import write


def main(a):
    fit,reference=[json.loads(p.read_text()) for p in [a.fit,a.reference]]
    if (fit['schema']!='pen_response_loss_ablation_result:1' or fit['update']!=2000 or fit['training_running']
            or reference['schema']!='pen_branch_old_new_evaluation:1' or fit['dataset_pins']!=reference['dataset_pins']
            or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')):
        raise ValueError('Terminal ablation and unchanged predeclared development evaluation')
    for pin in fit['checkpoint_health'].values():
        health=json.loads(verify_file_pin(pin).read_text())
        if not all(health['checks'].values()):raise ValueError('Exact-health and frozen-baseline gates')
    ck=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False)
    for pin in ck['sources']+reference['sources']:verify_file_pin(pin)
    prep=json.loads(verify_file_pin(fit['prepared']).read_text());collection=json.loads(verify_file_pin(prep['inputs']['collection']).read_text())
    directions=np.load(verify_file_pin(collection['directions']))['CHECK']
    original=torch.load(verify_file_pin(reference['old_checkpoint']),map_location='cpu',weights_only=False)
    scale=np.asarray(original['arms']['DIRECT']['scales']['y_scale'])
    data={r:dict(np.load(verify_file_pin(p))) for r,p in ck['dataset_pins'].items()}
    pairs=reference['physical_pair_classifications']
    device=torch.device('cuda:0');torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    a.output.mkdir();models={};outputs={};checks={}
    for name,saved in ck['arms'].items():
        model=PenResponseWM(saved['scales'],saved['head_scales'],saved['physics'],width=saved['width'],coupled=saved['coupled']).to(device)
        model.load_state_dict(saved['model'],strict=True);model.eval();one_step={};arrays={}
        for role in ['TRAIN','CHECK','SEQUENCE_CHECK']:
            one_step[role],prediction=common_one_step(model,data[role],scale,device);arrays[role+'_prediction']=prediction
        rollout,rollout_arrays=rollout_diagnostics(model,data['SEQUENCE_CHECK'],device);arrays.update(rollout_arrays)
        physical=response_diagnostics(model,data['CHECK'],pairs,directions,scale,device)
        path=a.output/(name+'.npz');np.savez_compressed(path,**arrays);outputs[name]=file_pin(path)
        models[name]=dict(one_step=one_step,recorded_control_rollout=rollout,physical_response=physical)
        if name.startswith('POINT_'):
            old_prediction=np.load(verify_file_pin(reference['outputs']['V2_'+saved['mode']]))
            checks[name+'_same_CHECK_prediction_as_frozen_V2']=np.array_equal(arrays['CHECK_prediction'],old_prediction['CHECK_prediction'])
        del model
    result=dict(schema='pen_response_loss_ablation_evaluation:1',fit=file_pin(a.fit),reference=file_pin(a.reference),checkpoint=fit['checkpoint'],
        sources=[file_pin(__file__)]+[file_pin(ROOT/p) for p in ['src/supermimic_research/learning/pen_response_evaluation.py',
            'src/supermimic_research/learning/pen_branch_evaluation.py','src/supermimic_research/learning/pen_native_response.py','src/supermimic_research/learning/pen_diagnostics.py']],
        dataset_pins=fit['dataset_pins'],common_scale='Original frozen V1 TRAIN output scales for all models',
        physical_pair_classifications=pairs,eligible_CHECK_pairs=sum(p['eligible'] for p in pairs),total_CHECK_pairs=len(pairs),models=models,outputs=outputs,
        checks={k:bool(v) for k,v in checks.items()},passed=sum(checks.values()),total=len(checks),optimizer_updates=0,native_integrations=0,
        scope='Same64 development action-direction CHECK,80 sequence CHECK and11 eligible physical contrasts. Correlated fixed clip/initial state;1 execution per sign. Contrast labels are finite amplitudes, not exact infinitesimal derivatives. No independent final holdout, policy/Actor/task success or ego-generalization qualification.')
    write(a.output/'report.json',result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),passed=result['passed'],total=result['total'],models={n:dict(
        CHECK=d['one_step']['CHECK']['common_v1_scale_MSE'],SEQUENCE=d['one_step']['SEQUENCE_CHECK']['common_v1_scale_MSE'],
        physical=d['physical_response']['summary']['ALL']) for n,d in models.items()})))
    if not all(checks.values()):raise ValueError('Retain failed baseline evaluation; no outcome claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
