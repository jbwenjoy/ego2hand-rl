"""Fresh diagnostic native amplitude/spread and frozen-network AD comparison."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.pen_response_targets import response_xy
from supermimic_research.learning.pen_response import PenResponseWM
from supermimic_research.learning.pen_native_response import compare_native_response,vector_comparison
from supermimic_research.learning.pen_stage import write


def main(a):
    review=json.loads(a.review.read_text());r=json.loads(verify_file_pin(review['run']).read_text())
    if not review['physical_response_interpretation_allowed'] or not all(review['gates_passed'].values()):raise ValueError('ALL original same-source/native fidelity gates required')
    supplement=json.loads(a.models.read_text())
    if verify_file_pin(supplement['run'])!=verify_file_pin(review['run']) or supplement['passed']!=supplement['total']:raise ValueError('Matching fixed-model supplement')
    fit=json.loads(a.fit.read_text())
    if fit['schema']!='pen_response_loss_ablation_result:1' or fit['update']!=2000 or fit['training_running']:raise ValueError('Frozen terminal ablation')
    for pin in fit['checkpoint_health'].values():
        if not all(json.loads(verify_file_pin(pin).read_text())['checks'].values()):raise ValueError('Healthy frozen networks')
    ck=torch.load(verify_file_pin(fit['checkpoint']),map_location='cpu',weights_only=False)
    for pin in ck['sources']+r['sources']+[review['source'],review['comparison_source']]:verify_file_pin(pin)
    response=json.loads(verify_file_pin(r['responses']).read_text());efc=json.loads(verify_file_pin(review['response_EFC']).read_text())
    direction=np.load(verify_file_pin(r['direction']))['direction20'];captures={(c['dt'],c['anchor']):c for c in r['source_captures']}
    sources={key:dict(np.load(verify_file_pin(c['arrays']))) for key,c in captures.items()};model_tables={c['dt']:dict(np.load(verify_file_pin(c['arrays']))) for c in supplement['receipts']}
    models={};physics={}
    for dt in [.005,.0025]:
        c=next(c for c in response['cases'] if c['dt']==dt);m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));models[dt]=m
        b=c['model_layout']['object_body'];p=model_tables[dt];q=p['body_iquat'][0,b];R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        physics[dt]=dict(mass=float(p['body_mass'][0,b]),inertia_BODY=(R@np.diag(p['body_inertia'][0,b])@R.T).tolist(),gravity=p['opt.gravity'][0].tolist())
    for saved in ck['arms'].values():
        if any(saved['physics']!=p for p in physics.values()):raise ValueError('Frozen neural physical prior differs')
    xs=[];ys=[];metadata=[]
    for c in response['cases']:
        m=models[c['dt']];jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids,dofs=m.jnt_qposadr[jids],m.jnt_dofadr[jids];aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids]);body=c['model_layout']['object_body']
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt']);w=np.load(verify_file_pin(e['wrenches']))['world_COM_wrench']
        packet=dict(np.load(verify_file_pin(c['trace'])))
        x,y=response_xy(packet,w,m,sources[(c['dt'],c['anchor'])],qids,dofs,aids,body,c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr'],c['dt'])
        xs.append(x);ys.append(y);metadata.append({k:c[k] for k in ['name','dt','anchor','profile','epsilon','sign','repeat','clamp_max_rad']})
    x,y=np.array(xs),np.array(ys)
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Fresh diagnostic analysis')
    a.output.mkdir();np.savez_compressed(a.output/'diagnostic_xy.npz',x=x,y=y)
    scale=np.array(ck['arms']['POINT_DIRECT']['scales']['y_scale'])
    # Use ONE declared scale for every network/native condition: frozen V2 TRAIN.
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');networks={}
    for name,saved in ck['arms'].items():
        model=PenResponseWM(saved['scales'],saved['head_scales'],saved['physics'],width=saved['width'],coupled=saved['coupled']).to(device)
        model.load_state_dict(saved['model'],strict=True);model.eval();networks[name]=model
    comparisons=[];native={};amplitudes=[]
    for dt in [.005,.0025]:
        for anchor in [0,2,4,6]:
            ids=[i for i,c in enumerate(metadata) if c['dt']==dt and c['anchor']==anchor];base=[i for i in ids if metadata[i]['sign']==0]
            if not all(np.array_equal(x[i,:81],x[base[0],:81]) and x[i,-1]==dt for i in ids):raise ValueError('Lost common source chart')
            center=torch.tensor(x[base[0]:base[0]+1],dtype=torch.float32,device=device);tangent=torch.zeros_like(center);tangent[:,81:101]=torch.tensor(direction,dtype=torch.float32,device=device)
            gradients={}
            for name,model in networks.items():
                _,g=torch.autograd.functional.jvp(model.prediction,center,tangent,create_graph=False);gradients[name]=g[0].detach().cpu().numpy()
            for epsilon in [.002,.005]:
                plus=[i for i in ids if metadata[i]['epsilon']==epsilon and metadata[i]['sign']==1]
                minus=[i for i in ids if metadata[i]['epsilon']==epsilon and metadata[i]['sign']==-1]
                unclipped=all(metadata[i]['clamp_max_rad']<=2e-7 for i in plus+minus)
                if not unclipped:continue
                for name,ad in gradients.items():
                    row=dict(model=name,dt=dt,anchor=anchor,epsilon=epsilon)
                    for block,sl in [('object12',slice(40,52)),('wrench6',slice(78,84))]:
                        metric,fd=compare_native_response(y[plus,sl],y[minus,sl],ad[sl],scale[sl],epsilon)
                        # Absolute norm retained so zero native responses cannot vanish
                        # from pooled error by multiplying a floored relative ratio.
                        metric['error_L2']=float(np.linalg.norm(ad[sl]/scale[sl]-fd))
                        row[block]=metric;native[(dt,anchor,epsilon,block)]=fd
                    comparisons.append(row)
            row=dict(dt=dt,anchor=anchor)
            for block in ['object12','wrench6']:
                if all((dt,anchor,eps,block) in native for eps in [.002,.005]):row[block]=vector_comparison(native[(dt,anchor,.002,block)],native[(dt,anchor,.005,block)])
            amplitudes.append(row)
    summaries={}
    for name in networks:
        subsets={}
        for anchor in ['ALL',0,2,4,6]:
            chosen=[row for row in comparisons if row['model']==name and (anchor=='ALL' or row['anchor']==anchor)]
            blocks={}
            for block in ['object12','wrench6']:
                target_energy=sum(row[block]['target_L2']**2 for row in chosen);error_energy=sum(row[block]['error_L2']**2 for row in chosen)
                blocks[block]=dict(conditions=len(chosen),pooled_relative_error=float(np.sqrt(error_energy/target_energy)) if target_energy>0 else None,
                    absolute_error_L2=float(np.sqrt(error_energy)),native_response_L2=float(np.sqrt(target_energy)),
                    contrasts_exceed_observed_spread=sum(row[block]['contrast_exceeds_observed_spread'] for row in chosen),
                    zero_native_energy=target_energy==0)
            subsets[str(anchor)]=blocks
        summaries[name]=subsets
    result=dict(schema='pen_same_source_native_AD_diagnostics:1',review=file_pin(a.review),models_supplement=file_pin(a.models),fit=file_pin(a.fit),checkpoint=fit['checkpoint'],
        sources=[file_pin(__file__)]+[file_pin(ROOT/p) for p in ['src/supermimic_research/mechanics/pen_response_targets.py','src/supermimic_research/mechanics/controlled_pen_records.py',
            'src/supermimic_research/mechanics/transition_records.py','src/supermimic_research/learning/pen_native_response.py']],
        xy=file_pin(a.output/'diagnostic_xy.npz'),metadata=metadata,comparisons=comparisons,native_amplitude_comparison=amplitudes,model_summaries=summaries,
        common_scale='Unchanged V2 TRAIN y_scale; all four networks share it',neural_updates=0,native_integrations=0,data_role='DIAGNOSTIC_NEVER_FIT',
        scope='New seed7801 direction at8 known actual sources,2 finite amplitudes and3 same-source execution repeats after original native fidelity gates. Observed spread is not a confidence bound or proof of infinitesimal differentiability. No model selection, independent-task/ego generalization, Actor or task success.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),summaries=summaries,native_amplitude_comparison=amplitudes)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['review','models','fit','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
