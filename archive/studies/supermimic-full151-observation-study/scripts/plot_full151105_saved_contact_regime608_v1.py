"""Render all 608 saved generated-contact annotations; no classification/query."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=args.run_dir.resolve(strict=True);out=args.output.resolve()
    report=json.loads((base/'report.json').read_text(encoding='utf8'))
    assert report['status']=='COMPLETED_ALL608_SAVED_GENERATED_CONTACT_REGIME_NOT_PHYSICAL_CERTIFICATE'
    samples=json.loads((base/'sample608.json').read_text(encoding='utf8'))
    assert len(samples)==608 and [r['sample_id'] for r in samples]==list(range(608))
    assert not out.exists()
    out.mkdir()
    matrices=[]
    specs=[('hand_object','generated_candidate_count','Generated hand-object candidate count','viridis',1),
           ('hand_object','secondary_positive_force_count','Hand-object candidates with qualified saved Fn > 1e-6 N','viridis',1),
           ('hand_object','observed_candidate_depth_m','Maximum observed hand-object candidate depth [mm]','magma',1000),
           ('object_nonhand','generated_candidate_count','Generated object-with-nonhand candidate count','viridis',1)]
    for cat,key,_,_,factor in specs:
        matrices.append(np.asarray([r['categories'][cat][key]*factor for r in samples]).reshape(8,76))
    absent=np.asarray([r['categories']['hand_object']['generated_candidate_absent'] for r in samples]).reshape(8,76)
    labels=[f"{samples[s*76]['path']} / k={samples[s*76]['k']}" for s in range(8)]
    for s in range(8):
        rows=samples[s*76:(s+1)*76]
        assert all(r['snapshot_id']==s for r in rows)
        assert [(r['branch_id']%19,r['substep']) for r in rows]==[(b,t) for b in range(19) for t in range(1,5)]
    fig,axes=plt.subplots(4,1,figsize=(18,13),layout='constrained')
    for panel,(ax,a,(_,key,title,cmap,_)) in enumerate(zip(axes,matrices,specs)):
        im=ax.imshow(a,aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=max(float(a.max()),1e-12))
        ax.set_title(title,loc='left',fontsize=13)
        ax.set_yticks(range(8),labels,fontsize=10)
        ax.set_xticks(np.arange(19)*4+1.5,[f'B{b+1}' if b<3 else str(b) for b in range(19)],fontsize=9)
        ax.set_xlim(-.5,75.5)
        for b in range(1,19):ax.axvline(b*4-.5,color='white',alpha=.35,lw=.5)
        cb=fig.colorbar(im,ax=ax,pad=.012,shrink=.82)
        if key.endswith('count'):cb.locator=MaxNLocator(integer=True);cb.update_ticks()
        if panel==2:
            yy,xx=np.where(absent)
            ax.scatter(xx,yy,marker='x',s=5,c='#89d8ff',linewidths=.4,label='No saved HO candidate; clearance UNKNOWN')
            ax.legend(loc='upper right',fontsize=8,framealpha=.95)
    axes[-1].set_xlabel('Original branch offset within each snapshot (B1-B3 baseline; 3-18 perturbations); each cell is substep 1, 2, 3 or 4')
    fig.suptitle('All 608 branch solver-PRE samples: 8 snapshots x 19 restored branches x 4 substeps',fontsize=16)
    fig.text(.5,-.025,'Generated candidates and source-decoded force evidence only. Empty candidates do not prove clearance.\n'
             'Each branch is a separate restored interval; columns are not one continuous trajectory. No new force/wrench decoding.',
             ha='center',fontsize=10)
    png=out/'all608_saved_contact_regime.png'
    fig.savefig(png,dpi=160,bbox_inches='tight');plt.close(fig)
    np.savez_compressed(out/'all608_plot_arrays.npz',generated_HO_count=matrices[0],positive_force_HO_count=matrices[1],
                        observed_HO_depth_mm=matrices[2],generated_object_nonhand_count=matrices[3],HO_candidate_absent=absent)
    rows=[]
    for s in range(8):
        group=samples[s*76:(s+1)*76]
        r={'snapshot_id':s,'path':group[0]['path'],'k':group[0]['k'],'sample_count':76}
        for cat in ('hand_object','hand_self','other','object_nonhand','hand_other'):
            cats=[x['categories'][cat] for x in group]
            r.update({cat+'_candidate_rows':sum(c['generated_candidate_count'] for c in cats),
                      cat+'_samples_with_candidates':sum(not c['generated_candidate_absent'] for c in cats),
                      cat+'_samples_with_secondary_positive_force':sum(c['secondary_positive_force_count']>0 for c in cats),
                      cat+'_max_observed_depth_mm':max(c['observed_candidate_depth_m'] for c in cats)*1000})
        rows.append(r)
    with (out/'all8_snapshot_descriptors.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'schema':'full151105_saved_contact608_figures:1','producer':pin(base/'report.json'),
            'samples':pin(base/'sample608.json'),'renderer':pin(__file__),'sample_count':608,'snapshot_count':8,
            'all8_snapshot_descriptors':rows,'outputs':{p.name:pin(p) for p in out.iterdir()},
            'new_classification_model_AD_physics_GPU':0,'global_clearance_qualified':False}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'figure':str(png),'sample_count':608,'snapshots':8}))


if __name__=='__main__':main()
