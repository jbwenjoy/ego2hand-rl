"""Saved-only description of the complete old152 target cohort."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def vrms(a):return float(np.sqrt(np.mean(np.sum(a*a,axis=-1))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--independent-review',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();base=a.run_dir.resolve(strict=True);review=read(a.independent_review)
    assert review['passed']==review['total']==8 and all(x is True for x in review['checks'].values())
    producer=read(base/'report.json');assert producer['enrollment_qualified'] is True
    ids=read(base/'group152_identities.json')
    with np.load(base/'all152_boundary_impulse_targets.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    assert len(ids)==152 and len(np.unique(data['source_substep_rows152x4']))==608
    modes=data['boundary_constraint_witness'];rv=data['r_v_saved_m_s_F64'];rw=data['r_w_saved_rad_s_F64']
    cohorts=[('ALL',np.ones(152,bool))]
    for path in ('A0','A2'):
        cohorts.append((path,np.asarray([r['path']==path for r in ids])))
    for path in ('A0','A2'):
        for k in (0,49,99,149):cohorts.append((f'{path}/k{k}',np.asarray([r['path']==path and r['k']==k for r in ids])))
    for mode in ('baseline','direction'):cohorts.append((mode,np.asarray([r['mode']==mode for r in ids])))
    summaries=[]
    for name,mask in cohorts:
        row={'cohort':name,'groups':int(mask.sum()),'HO_witness':int(modes[mask,0].sum()),'nonhand_witness':int(modes[mask,1].sum()),
             'both_witness':int(modes[mask].all(axis=1).sum()),'neither_witness':int((~modes[mask].any(axis=1)).sum()),
             'r_v_vector_RMS_m_s':vrms(rv[mask]),'r_w_vector_RMS_rad_s':vrms(rw[mask]),
             'r_v_norm_max_m_s':float(np.linalg.norm(rv[mask],axis=1).max()),'r_w_norm_max_rad_s':float(np.linalg.norm(rw[mask],axis=1).max())}
        for field,prefix in [('J_palm_F64','J_Ns'),('K_palm_F64','K_Nms')]:
            for c,label in enumerate(('HO','nonhand')):row[f'{prefix}_{label}_vector_RMS']=vrms(data[field][mask,c])
        summaries.append(row)
    out=a.output.resolve();assert not out.exists();out.mkdir()
    with (out/'all13_cohort_summary.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained');fig.get_layout_engine().set(rect=(0,.10,1,.90))
    for ax,(field,title,unit) in zip(axes[0],[('J_palm_F64','Source-separated linear impulse norms','mN s'),('K_palm_F64','Source-separated COM angular impulse norms','mN m s')]):
        for c,label,color in [(0,'Hand-object','#2675b8'),(1,'Nonhand / terrain','#d77826')]:
            ax.plot(range(152),np.linalg.norm(data[field][:,c],axis=-1)*1000,'o',ms=3,alpha=.7,c=color,label=label)
        ax.set(title=title,ylabel=f'Vector norm [{unit}]');ax.set_ylim(bottom=0);ax.legend(fontsize=9);ax.grid(alpha=.2)
    for ax,(value,title,unit,scale) in zip(axes[1],[(rv,'Observed linear velocity defect','mm/s',1000),(rw,'Observed angular velocity defect','rad/s',1)]):
        ax.plot(range(152),np.linalg.norm(value,axis=1)*scale,'o',ms=3,alpha=.7,c='#526b7b')
        ax.set(title=title,ylabel=f'Vector norm [{unit}]');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    labels=[f'{path}/k{k}' for path in ('A0','A2') for k in (0,49,99,149)]
    for ax in axes.flat:
        for boundary in range(19,152,19):ax.axvline(boundary-.5,c='gray',lw=.6,ls='--')
        ax.set_xticks([19*i+9 for i in range(8)],labels,rotation=30,ha='right',fontsize=8)
        ax.set_xlabel('All 152 original branch rows; 19 dependent rows per snapshot')
    fig.suptitle('Old152 / 608 substeps: boundary impulse supervision targets, not model performance',fontsize=15)
    fig.text(.5,.025,'All eight source snapshots retained. No donor/fresh enrollment, model training, label filtering or residual accuracy gate.\nWitness = recorded solver boundary constraint; not positive-force, self15, human contact or global clearance.',ha='center',fontsize=10)
    fig.savefig(out/'all152_impulse_and_observed_defect_targets.png',dpi=145,bbox_inches='tight');plt.close(fig)
    fixed={}
    for key in ('palm_rot_raw_FP32','palm_pos_raw_FP32','gravity_palm_F64','DeltaT_F64'):
        x=data[key];fixed[key]={'all152_bitexact_first_row':x.tobytes()==np.ascontiguousarray(np.broadcast_to(x[0],x.shape)).tobytes(),
                              'unique_numeric_rows':len(np.unique(x.reshape(152,-1),axis=0))}
    p=int(np.linalg.norm(rw,axis=1).argmax())
    d={'schema':'full151105_boundary_impulse_target_description:1','producer':pin(base/'report.json'),'independent8':pin(a.independent_review),'renderer':pin(__file__),
       'all152_groups_608_source_substeps':True,'summary':summaries,'fixed_context_descriptors':fixed,'r_w_peak_group':ids[p],
       'native_R0_det_range':[float(data['native_R0_det_F64'].min()),float(data['native_R0_det_F64'].max())],
       'native_R1_det_range':[float(data['native_R1_det_F64'].min()),float(data['native_R1_det_F64'].max())],
       'I_palm1_condition_range':[float(data['I_palm1_condition_number_F64'].min()),float(data['I_palm1_condition_number_F64'].max())],
       'solve_equation_residual_RMS':vrms(data['rawR_solve_residual_F64']),
       'scope':'Full-cohort descriptors, not a new accuracy/physical/learning-benefit gate. Native rawR and observed nonzero defects retained.',
       'new_Torch_NN_MJB_physics':0,'outputs':{p.name:pin(p) for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'ALL':summaries[0],'fixed_context':fixed,'output':str(out)}))


if __name__=='__main__':main()
