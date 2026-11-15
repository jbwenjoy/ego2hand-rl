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
from supermimic_research.learning.pen_diagnostics import rollout_diagnostics
from supermimic_research.learning.pen_branch_evaluation import classify_pair,common_one_step,physical_pair_derivative
from supermimic_research.learning.pen_stage import write


def main(a):
    fit=json.loads(a.fit.read_text());old_health=json.loads(a.old_health.read_text())
    if fit['schema']!='pen_branch_paired_fit_result:1' or fit['training_running'] or fit['update']!=2000 or not all(old_health['checks'].values()):
        raise ValueError('Healthy terminal model pair results')
    for pin in fit['checkpoint_health'].values():
        health=json.loads(verify_file_pin(pin).read_text())
        if not all(health['checks'].values()):raise ValueError('Both checkpoint health gates required')
    new=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False)
    old=torch.load(verify_file_pin(old_health['checkpoint']),map_location='cpu',weights_only=False)
    for pin in new['sources']+old['sources']+old['parent_sources']:verify_file_pin(pin)
    prep=json.loads(verify_file_pin(fit['prepared']).read_text());collection=json.loads(verify_file_pin(prep['inputs']['collection']).read_text())
    proposal=json.loads(verify_file_pin(collection['proposal']).read_text());selected=np.load(verify_file_pin(proposal['artifacts']['selected']))
    qref=np.load(verify_file_pin(proposal['clip']))['reference_qpos'];directions=np.load(verify_file_pin(collection['directions']))['CHECK']
    data={r:dict(np.load(verify_file_pin(p))) for r,p in new['dataset_pins'].items()}
    common_scale=np.asarray(old['arms']['DIRECT']['scales']['y_scale'])
    if not np.array_equal(common_scale,old['arms']['IMPULSE']['scales']['y_scale']):raise ValueError('Common fixed v1 output scale')
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified evaluation')
    a.output.mkdir();device=torch.device('cuda:0');torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    check=data['CHECK'];pairs=[]
    for plus in [r for r in prep['rows'] if r['role']=='CHECK' and r['sign']==1 and r['reached']]:
        minus=next(r for r in prep['rows'] if r['role']=='CHECK' and r['sign']==-1 and r['anchor']==plus['anchor'] and r['direction']==plus['direction'] and r['dt']==plus['dt'])
        if not minus['reached']:continue
        ip=np.flatnonzero(check['case_id']==plus['case_id']).item();im=np.flatnonzero(check['case_id']==minus['case_id']).item()
        classification=classify_pair(check['x'][ip],check['x'][im],check['valid'][ip] and check['valid'][im],
            qref[plus['anchor']+1]+selected['delta20'],directions[plus['direction']],plus['epsilon'])
        pairs.append(dict(plus_index=ip,minus_index=im,anchor=plus['anchor'],dt=plus['dt'],direction=plus['direction'],epsilon=plus['epsilon'],**classification))
    results={};outputs={}
    for version,checkpoint in [('V1',old),('V2',new)]:
        for mode,saved in checkpoint['arms'].items():
            key=version+'_'+mode;m=PenResponseWM(saved['scales'],saved['head_scales'],saved['physics'],width=saved['width'],coupled=saved['coupled']).to(device)
            m.load_state_dict(saved['model'],strict=True);m.eval();one_step={};arrays={}
            for role in ['TRAIN','CHECK','SEQUENCE_CHECK']:
                one_step[role],prediction=common_one_step(m,data[role],common_scale,device);arrays[role+'_prediction']=prediction
            rollouts,rollout_arrays=rollout_diagnostics(m,data['SEQUENCE_CHECK'],device);arrays.update(rollout_arrays)
            physical=[]
            for pair in pairs:
                if not pair['eligible']:continue
                ip,im=pair['plus_index'],pair['minus_index']
                score=physical_pair_derivative(m,check['x'][ip],check['x'][im],check['y'][ip],check['y'][im],
                    directions[pair['direction']],pair['epsilon'],common_scale,device)
                physical.append(dict(anchor=pair['anchor'],dt=pair['dt'],direction=pair['direction'],epsilon=pair['epsilon'],**score))
            aggregate={}
            for anchor in ['ALL',0,2,4,6]:
                chosen=[r for r in physical if anchor=='ALL' or r['anchor']==anchor]
                row=dict(pairs=len(chosen))
                for block in ['object12','wrench6']:
                    if chosen:row[block]=dict(mean_relative_error=float(np.mean([r[block]['relative_L2_error'] for r in chosen])),
                        mean_cosine=float(np.mean([r[block]['cosine'] for r in chosen])),max_relative_error=max(r[block]['relative_L2_error'] for r in chosen))
                aggregate[str(anchor)]=row
            path=a.output/(key+'.npz');np.savez_compressed(path,**arrays);outputs[key]=file_pin(path)
            results[key]=dict(one_step=one_step,recorded_control_rollout=rollouts,physical_same_source_CHECK_pairs=physical,physical_summary=aggregate)
            del m
    result=dict(schema='pen_branch_old_new_evaluation:1',fit=file_pin(a.fit),old_health=file_pin(a.old_health),new_checkpoint=fit['checkpoint'],old_checkpoint=old_health['checkpoint'],
        sources=[file_pin(__file__)]+[file_pin(ROOT/p) for p in ['src/supermimic_research/learning/pen_branch_evaluation.py','src/supermimic_research/learning/pen_diagnostics.py','src/supermimic_research/learning/pen_native_response.py']],
        prepared=fit['prepared'],dataset_pins=new['dataset_pins'],common_error_scale='Frozen original v1 TRAIN y_scale for ALL models and comparisons',outputs=outputs,models=results,
        physical_pair_classifications=pairs,eligible_CHECK_pairs=sum(p['eligible'] for p in pairs),total_CHECK_pairs=len(pairs),
        optimizer_updates=0,native_integration_steps=0,
        scope='Held-out fresh action-direction CHECK at four known source-time anchors and original80 sequence CHECK. Source-mismatched or clipped pairs excluded ONLY from action derivative metrics, never dropped from one-step evaluation or original datasets. Native contrasts have1 execution per sign, no fresh repeat uncertainty bound. All correlated same clip/initial pose; not unseen-state/ego generalization, full derivative qualification or Actor success.')
    write(a.output/'report.json',result)
    summary={m:dict(CHECK=v['one_step']['CHECK']['common_v1_scale_MSE'],SEQUENCE_CHECK=v['one_step']['SEQUENCE_CHECK']['common_v1_scale_MSE'],physical=v['physical_summary']) for m,v in results.items()}
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),eligible_CHECK_pairs=result['eligible_CHECK_pairs'],total_CHECK_pairs=len(pairs),models=summary)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--old-health',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
