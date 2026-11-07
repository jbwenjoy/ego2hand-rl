"""Saved metrics-only WM heatmaps; no model imports or additional inference."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
MODELS=('identity','constant_velocity','A1L','P_ONLY','A2')
TRAJECTORIES=('A0','A1L','P_ONLY','A2')
PANELS=(
    ('q_RMSE_rad','q_RMSE_mrad','Joint position RMSE','mrad',1000.),
    ('qd_RMSE_rad_s','qd_RMSE_rad_s','Joint velocity RMSE','rad/s',1.),
    ('position_m_RMS','position_RMS_mm','Object position RMS','mm',1000.),
    ('rotation_rad_RMS','rotation_RMS_deg','Object rotation RMS','deg',180./math.pi),
    ('linear_velocity_m_s_RMS','linear_velocity_RMS_m_s','Linear velocity RMS','m/s',1.),
    ('angular_velocity_rad_s_RMS','angular_velocity_RMS_rad_s','Angular velocity RMS','rad/s',1.),
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))


def pin(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'outputs/WM_onpolicy_CPU_diagnostic_v1')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if any(name in sys.modules for name in ('torch','warp','mujoco_warp')):raise RuntimeError('Saved-only plotting must not import model/device frameworks')
    source=a.input.resolve(strict=True);metrics=read(source/'metrics.json');diagnostic=read(source/'report.json')
    if diagnostic.get('status')!='COMPLETED_CPU_DIAGNOSTIC_NOT_PHYSICAL_CERTIFICATE' or diagnostic.get('physics_steps_added')!=0 or diagnostic.get('GPU_used') is not False or diagnostic.get('RNG_unchanged') is not True or diagnostic.get('one_step_prediction_rows')!=180:raise ValueError('Require the completed saved CPU diagnostic')
    if len(metrics)!=20:raise ValueError('Expected exactly20 saved model/trajectory metrics')
    lookup={}
    for row in metrics:
        key=(row['model'],row['trajectory'])
        if key in lookup or key[0] not in MODELS or key[1] not in TRAJECTORIES or type(row['rows']) is not int or row['rows']!=15:raise ValueError('Wrong, duplicate or incomplete15-transition cell')
        for field,_,_,_,_ in PANELS:
            value=row[field]
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Invalid saved RMS metric '+field)
        lookup[key]=row
    if set(lookup)!={(model,path) for model in MODELS for path in TRAJECTORIES}:raise ValueError('Model/path coverage is incomplete')
    source_pins={name:pin(source/name) for name in ('metrics.json','report.json','runplan.json','protocol.json','codec_receipts.json','predictions_truth_baselines_gradients.npz','loader_A1L.json','loader_P_ONLY.json','loader_A2.json')}
    for key in ('entry','protocol','runplan'):
        value=diagnostic[key]
        if pin(value['path'])['sha256']!=value['sha256']:raise ValueError('Pinned diagnostic source changed: '+key)
        source_pins['diagnostic_'+key]=value
    data={key:np.asarray([[lookup[(model,path)][field]*scale for path in TRAJECTORIES] for model in MODELS],float) for field,key,_,_,scale in PANELS}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15.2,9.2),layout='constrained')
    scales={}
    for ax,(_,key,title,unit,_) in zip(axes.flat,PANELS):
        matrix=data[key];maximum=float(matrix.max());upper=maximum if maximum>0 else 1.
        image=ax.imshow(matrix,cmap='viridis_r',vmin=0.,vmax=upper,aspect='auto',interpolation='nearest')
        for i in range(5):
            for j in range(4):
                rgba=image.cmap(image.norm(matrix[i,j]));luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(j,i,format(matrix[i,j],'.3g'),ha='center',va='center',fontsize=10,color='black' if luminance>.48 else 'white')
        ax.set_xticks(range(4),TRAJECTORIES,fontsize=10);ax.set_yticks(range(5),MODELS,fontsize=9)
        ax.set(title=title+' ('+unit+')',xlabel='Saved trajectory',ylabel='Predictor / baseline')
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True);ax.grid(which='minor',color='white',lw=.8,alpha=.5);ax.tick_params(which='minor',bottom=False,left=False)
        bar=fig.colorbar(image,ax=ax,shrink=.88,pad=.025);bar.set_label(unit)
        scales[key]={'unit':unit,'data_min':float(matrix.min()),'data_max':maximum,'color_scale_min':0.,'color_scale_max':upper,'independent_panel_scale':True}
    fig.suptitle('Saved one-step prediction errors | one controlled source, seed0\n15 dependent transitions per cell; lower is better; independent units and color scale per panel',fontsize=13)
    fig.supxlabel('FP32 CPU diagnostic only · self-collision15 truth UNKNOWN · no physical-Jacobian, contact-accuracy or generalization qualification',fontsize=10)
    fig.savefig(a.output/'WM_onpolicy_prediction_heatmaps.png',dpi=180);fig.savefig(a.output/'WM_onpolicy_prediction_heatmaps.svg');plt.close(fig)
    table=[]
    for i,model in enumerate(MODELS):
        for j,path in enumerate(TRAJECTORIES):table.append({'model':model,'trajectory':path,'dependent_transitions':15,**{key:float(data[key][i,j]) for _,key,_,_,_ in PANELS}})
    with (a.output/'metrics_table.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    np.savez_compressed(a.output/'display_matrices.npz',**data)
    report={'schema':'saved_WM_onpolicy_CPU_heatmap_summary:1','status':'SAVED_METRICS_ONLY_DESCRIPTIVE','source_pins':source_pins,'plotter':pin(__file__),'model_order':list(MODELS),'trajectory_order':list(TRAJECTORIES),'panels':scales,'metric_cells':20,'dependent_transitions_per_cell':15,'underlying_saved_trajectories':4,'underlying_dependent_transitions':60,'learned_WM_prediction_rows':180,'source_lineage_N':1,'independent_ego_N':0,'training_seed_per_method':0,'selfcollision15_truth':'UNKNOWN_not_saved_found_sensor','data_table_rows':table,'model_forward_calls_added':0,'physics_steps_added':0,'GPU_used':False,'Torch_imported':'torch' in sys.modules,'Warp_imported':'warp' in sys.modules,'scope':'Saved aggregate errors across15 temporally dependent transitions per cell, not independent samples or held-out generalization. Identity/CV are predictive baselines. No new model forward, physical Jacobian, contact/force accuracy or global method-benefit qualification.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'output':str(a.output),'metric_cells':20,'panels':6,'model_forward_calls_added':0},indent=2))


if __name__=='__main__':main()
