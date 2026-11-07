"""Describe all saved responses and secants; no model/physics/gradient calls."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run=a.run.resolve(strict=True)
    files=[run/'decomposition_arrays.npz',run/'all152_branch_raw.npz',run/'all64_pair_records.json',run/'report.json']
    inputs=[pin(p) for p in files]
    with np.load(files[0],allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    with np.load(files[1],allow_pickle=False) as z:clipped=z['legacy_clipped'].copy()
    pairs=json.loads(files[2].read_text(encoding='utf-8-sig'));report=json.loads(files[3].read_text(encoding='utf-8-sig'))
    assert report['counts']['all_epsilon_pairs']==len(pairs)==64 and report['counts']['perturbation_endpoints']==128
    assert report['fixed_visible_state_interpretation_permitted'] is True
    assert d['centered_response_error_F64'].shape==(8,16,58) and d['constant_baseline_error_F64'].shape==(8,58)
    defined=d['secant_defined_mask'];assert defined.shape==clipped.shape==(8,4,2)
    assert defined.all()  # This actual artifact has all64 defined; no pair selection.
    assert [(r['snapshot_id'],r['j'],r['epsilon_index']) for r in pairs]==[(s,j,e) for s in range(8) for j in range(4) for e in range(2)]
    for r in pairs:assert r['legacy_clipped']==bool(clipped[r['snapshot_id'],r['j'],r['epsilon_index']])
    a.output.mkdir(parents=True,exist_ok=False)
    blocks=[('q',0,20,1000,'mrad','rad / control rad'),('qd',20,40,1,'rad/s','(rad/s) / control rad'),
            ('p',40,43,1000,'mm','m / control rad'),('R',43,52,1,'dimensionless','1 / control rad'),
            ('v',52,55,1,'m/s','(m/s) / control rad'),('omega',55,58,1,'rad/s','(rad/s) / control rad')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    keys=['constant_baseline_error_F64','centered_physical_response_F64','centered_model_response_F64','centered_response_error_F64']
    labels=['Baseline bias\n8 centers','Physical response\n128 endpoints','WM response\n128 endpoints','Response error\n128 endpoints']
    colors=['#8b96a8','#2463a8','#cf7937','#a64956']
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    bar_values={}
    for ax,(name,lo,hi,scale,unit,_) in zip(axes.flat,blocks):
        values=[float(np.sqrt(np.mean(d[k][...,lo:hi]**2)))*scale for k in keys]
        assert all(np.isfinite(values)) and all(v>0 for v in values)
        bar_values[name]={'units':unit,'values':values}
        ax.bar(range(4),values,color=colors,width=.65)
        ax.set_yscale('log');ax.set_ylabel(f'{name}: component RMS ({unit}), log scale')
        ax.set_xticks(range(4),labels,fontsize=8)
        for i,v in enumerate(values):ax.text(i,v*1.2,f'{v:.4g}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(min(values)*.45,max(values)*4);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Saved visible-state diagnosis: baseline bias and finite action responses',fontsize=14,y=.99)
    fig.text(.02,.01,'All 8 centers / 128 endpoints retained. Responses depend on the selected 1 and 2 mrad interventions; no accuracy threshold or Jacobian claim.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(a.output/'baseline_bias_and_all128_responses.png',dpi=180);plt.close(fig)
    physical=d['physical_coordinate_secant_F64'].reshape(64,58);model=d['model_coordinate_secant_F64'].reshape(64,58)
    assert np.isfinite(physical).all() and np.isfinite(model).all()
    fig,axes=plt.subplots(2,3,figsize=(14,9));export=[];point_counts={}
    for ax,(name,lo,hi,_,_,unit) in zip(axes.flat,blocks):
        xx=physical[:,lo:hi];yy=model[:,lo:hi];clip=np.broadcast_to(clipped.reshape(64,1),xx.shape)
        for flag,color,label in [(False,'#2463a8','Unclipped pairs'),(True,'#cf7937','Clipped pairs')]:
            ax.scatter(xx[clip==flag],yy[clip==flag],s=10,alpha=.45,c=color,label=label)
        bound=float(max(abs(xx).max(),abs(yy).max()))*1.06
        if bound==0:bound=1.
        ax.plot([-bound,bound],[-bound,bound],linestyle='--',color='#777777',linewidth=.8)
        ax.set(xlim=(-bound,bound),ylim=(-bound,bound),xlabel=f'Observed secant ({unit})',ylabel=f'WM secant ({unit})',title=f'{name}: all 64 x {hi-lo} components')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.12);point_counts[name]=int(xx.size)
        for i,r in enumerate(pairs):
            for feature in range(lo,hi):export.append({'pair_id':i,'snapshot_id':r['snapshot_id'],'path':r['path'],'k':r['k'],'joint_index':r['joint_index'],
                'epsilon_index':r['epsilon_index'],'feature':feature,'block':name,'units':unit,'physical':float(physical[i,feature]),'model':float(model[i,feature]),'legacy_clipped':r['legacy_clipped']})
    handles,text=axes.flat[0].get_legend_handles_labels();fig.legend(handles,text,loc='lower center',ncol=2,bbox_to_anchor=(.5,.025))
    fig.suptitle('All 64 actual-coordinate finite secants; dashed line is equality',fontsize=14,y=.99)
    fig.text(.02,.008,'Dependent FP32 saved endpoints. Non-symmetric/clipped chords are retained; encoded zeros do not certify a zero physical derivative.',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.95));fig.savefig(a.output/'all64_secants_all3712_components.png',dpi=180);plt.close(fig)
    assert len(export)==64*58 and sum(point_counts.values())==3712
    with (a.output/'all3712_secant_components.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    assert inputs==[pin(p) for p in files]
    result={'schema':'full151105_saved_action_response_figures:1','inputs':inputs,'renderer':pin(__file__),'all_pairs':64,'all_endpoints':128,
            'bar_values':bar_values,'scatter_components_by_block':point_counts,'exported_components':3712,'new_model_or_physics_calls':0,
            'scope':'Full saved cohort; per-block units; baseline magnitude and finite response are distinct quantities, no stable-case selection or gradient qualification.'}
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(a.output),'pairs':64,'endpoints':128,'secant_components':3712}))


if __name__=='__main__':main()
