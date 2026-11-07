"""Saved-only complete momentum description; additional summaries are exploratory."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pin(p):
    p = Path(p).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--root-crosscheck', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    base = a.run_dir.resolve(strict=True)
    report = read(base/'report.json')
    cross = read(a.root_crosscheck)
    assert cross['rows'] == 1808 and cross['closure_eligible_rows'] == 1808
    with np.load(base/'all1808_momentum_arrays.npz', allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    rows = [json.loads(x) for x in (base/'all1808_rows.jsonl').read_text(encoding='utf8').splitlines()]
    contacts = [json.loads(x) for x in (base/'all_generated_contacts.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows) == 1808 and len(contacts) == report['all_generated_contact_rows']
    assert data['closure_interpretation_eligible'].all()
    residual = data['force_residual_N_F64']
    norm = np.linalg.norm(residual, axis=1)
    required_contact = data['delta_p_Ns_F64']/data['dt_used_s_F64'][:,None] - data['Fgravity_N_F64']
    # Explicitly post hoc descriptive omission; no threshold or causal experiment.
    HO_only_residual = required_contact-data['FHO_N_F64']
    out = a.output.resolve()
    assert not out.exists()
    out.mkdir()
    summaries = []
    for path in ('ALL','A0','A2'):
        for mode in ('ALL','donor','baseline','direction'):
            mask = np.asarray([(path == 'ALL' or r['path'] == path) and (mode == 'ALL' or r['mode'] == mode) for r in rows])
            x = norm[mask]
            omission = HO_only_residual[mask]
            summaries.append({'path':path,'mode':mode,'rows':int(mask.sum()),
                'force_residual_vector_RMS_N':float(np.sqrt(np.mean(x*x))),
                **{f'force_residual_norm_{name}_N':float(np.quantile(x,q)) for name,q in [('min',0),('median',.5),('p90',.9),('p99',.99),('max',1)]},
                'HO_only_omission_vector_RMS_N_posthoc':float(np.sqrt(np.mean(np.sum(omission*omission,axis=1)))),
                'residual_accuracy_gate':None})
    with (out/'all_path_mode_summary.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    np.savez_compressed(out/'all1808_descriptive_arrays.npz',force_residual_norm_N=norm,
                        required_contact_force_from_velocity_N=required_contact,HO_only_omission_residual_N_posthoc=HO_only_residual)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    ax=axes[0]
    for j,(name,color) in enumerate(zip(('x','y','z'),('#2675b8','#d77826','#549e77'))):
        ax.scatter(required_contact[:,j],data['Ftotal_N_F64'][:,j],s=9,alpha=.5,c=color,label=name)
    lim=float(max(np.abs(required_contact).max(),np.abs(data['Ftotal_N_F64']).max()))*1.05
    ax.plot([-lim,lim],[-lim,lim],c='gray',ls='--',lw=1)
    ax.set(xlabel='m*(v_post-v_pre)/dt - gravity [N]',ylabel='All object boundary contact force [N]',title='All 1808 substeps, all three world components')
    ax.set_aspect('equal',adjustable='box');ax.legend(title='World axis');ax.grid(alpha=.2)
    ax=axes[1];ax.plot(np.arange(1808),norm*1e6,'o',ms=2,alpha=.65,c='#4c667b')
    for boundary in (600,904,1504):ax.axvline(boundary,c='gray',ls='--',lw=.8)
    ax.set_yscale('symlog',linthresh=.1);ax.set_ylim(bottom=0)
    ax.set(xlabel='Original serialized row index (boundaries marked)',ylabel='Force residual vector norm [micro-N]',title='All rows; display symlog, linear below 0.1 micro-N')
    ax.grid(alpha=.2)
    fig.suptitle('Conditional discrete object linear momentum: all contacts + known gravity',fontsize=15)
    fig.text(.5,-.035,'One cached simulation lineage; donor/restored rows are dependent. No residual accuracy pass gate.\nRoot arithmetic crosscheck; metadata/omitted terms remain source-backed. Finite solver and implicit integrator terms were not saved.',ha='center',fontsize=10)
    fig.savefig(out/'all1808_linear_momentum_consistency.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for ax,path in zip(axes,('A0','A2')):
        idx=[i for i,r in enumerate(rows) if r['path']==path and r['mode']=='donor']
        assert len(idx)==600
        t=np.arange(600)*float(data['dt_used_s_F64'][0])
        for key,label,color in [('FHO_N_F64','Hand-object contact','#2675b8'),('Fother_N_F64','Other object boundary contacts','#d77826')]:
            ax.plot(t,data[key][idx,2],lw=1.1,c=color,label=label)
        ax.plot(t,required_contact[idx,2],lw=.9,c='#4d5863',ls='--',label='m*delta_v/dt - gravity')
        ax.set(xlabel='Canonical pre-tick time [s]',ylabel='World z force [N]',title=f'{path}: all 600 donor substeps, external sources separated')
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Contact force decomposition: other contacts cannot be assigned to the hand',fontsize=15)
    fig.text(.5,-.03,'Forces are solver-PRE, paired with the same substep velocity increment. Complete x/y/z arrays are also saved.\nThis is measured simulation contact force, not recovered human force or a six-dimensional support certificate.',ha='center',fontsize=10)
    fig.savefig(out/'both_donors_force_source_decomposition.png',dpi=150,bbox_inches='tight');plt.close(fig)
    counts={k:sum(c['category']==k for c in contacts) for k in ('HO','other_external','internal','not_object')}
    peak=int(np.argmax(norm))
    d={'schema':'full151105_object_momentum_descriptive_figures:1','producer':pin(base/'report.json'),'root_crosscheck':pin(a.root_crosscheck),'renderer':pin(__file__),
       'all_rows':1808,'contact_categories_all9473':counts,'summary':summaries,'peak_row':{k:rows[peak][k] for k in ('row_id','path','mode','branch_id','phase','substep')},
       'peak_force_residual_norm_N':float(norm[peak]),'gravity_force_norm_N':float(np.linalg.norm(data['Fgravity_N_F64'][0])),
       'additional_quantiles_and_HO_only_omission':'post hoc exploratory description, no selection, causal comparison or eligibility threshold',
       'new_MJB_backend_model_physics_GPU':0,'outputs':{p.name:pin(p) for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'ALL':summaries[0],'contacts':counts,'peak_row':d['peak_row'],'output':str(out)}))


if __name__=='__main__':
    main()
