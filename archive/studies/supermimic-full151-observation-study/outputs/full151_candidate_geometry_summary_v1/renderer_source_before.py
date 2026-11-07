"""Saved native geometry samples only; no geometry queries or physical calls."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main(source,output):
    if output.exists():raise FileExistsError('Fresh summary directory required')
    receipt_path=source/'receipt.json';r=json.loads(receipt_path.read_text('utf-8'))
    if r.get('schema')!='full151_sampled_geometry_diagnostic_receipt:1' or r.get('status')!='COMPLETED_SAMPLED_GEOMETRY_DIAGNOSTIC':raise ValueError('Incomplete/wrong diagnostic receipt')
    amap={str(Path(x['path']).resolve()):x['sha256'] for x in r['artifacts']}
    inputs={'receipt':pin(receipt_path)}
    def used(name,relative):
        p=(source/relative).resolve()
        if amap.get(str(p))!=sha(p):raise ValueError('Saved diagnostic artifact mismatch: '+str(p))
        inputs[name]=pin(p);return load(p)
    raw=used('raw','sampled_native_geometry.npz');saved=used('summary','generated_depth_summary.npz');cat=used('catalog','catalog.npz')
    d,mask,ties,count=raw['selected_distance_m'],raw['generated_mask'],raw['tie_count'],raw['contact_count']
    if d.shape!=(601,273) or mask.shape!=d.shape or mask.dtype!=np.bool_ or not np.all(raw['queried']):raise ValueError('Malformed/incomplete601 native samples')
    if not np.isnan(d[~mask]).all() or not np.isfinite(d[mask]).all():raise ValueError('Absent gap must remainNaN, generated distances finite')
    t=saved['time_s']
    if not np.array_equal(t,np.arange(601)*.005):raise ValueError('601 sample clock changed')
    summary={};category={};checks={'all601_queried':bool(np.all(raw['queried'])),'absent_gaps_NaN':bool(np.isnan(d[~mask]).all())}
    for name in ('hand_object','hand_self'):
        cols=cat['pair_kind']==name
        present=mask[:,cols].any(axis=1)
        # Zero here is the existing aggregate "no generated overlap observed";
        # it is never used as a clearance estimate or plotted at absent samples.
        depth=np.max(np.where(mask[:,cols],np.maximum(-np.nan_to_num(d[:,cols],nan=0),0),0),axis=1)
        if not np.array_equal(depth,saved[name+'_depth_m']):raise ValueError('Saved generated depth summary differs from actual selected distances')
        observed=np.where(present,depth,np.nan)
        tied_pairs=np.sum(ties[:,cols]>1,axis=1)
        category[name]={'present':present,'raw_aggregate_depth_m':depth,'observed_depth_m':observed,'generated_pairs':mask[:,cols].sum(axis=1),'native_contact_count':count[:,cols].sum(axis=1),'tied_pairs':tied_pairs}
        k=int(depth.argmax())
        summary[name]={'maximum_generated_depth_m':float(depth[k]),'worst_sample_index':k,'worst_sample_time_s':float(t[k]),'samples_over_original3mm':int(np.count_nonzero(depth>.003)),'samples_with_generated_contact':int(present.sum()),'samples_without_generated_contact_unknown_clearance':int((~present).sum()),'samples_with_tied_pairs':int(np.count_nonzero(tied_pairs))}
        checks[name+'_raw_depth_exact_saved']=True
    ho,ss=category['hand_object'],category['hand_self']
    anyties=(ties>1).any(axis=1)
    checks['HO_depth_max_matches_receipt']=summary['hand_object']['maximum_generated_depth_m']==r['summary']['maximum_generated_hand_object_depth_m']
    checks['self_depth_max_matches_receipt']=summary['hand_self']['maximum_generated_depth_m']==r['summary']['maximum_generated_hand_self_depth_m']
    checks['HO_over3mm_count_matches_receipt']=summary['hand_object']['samples_over_original3mm']==r['summary']['sample_counts_over_original3mm']['hand_object']
    checks['self_over3mm_count_matches_receipt']=summary['hand_self']['samples_over_original3mm']==r['summary']['sample_counts_over_original3mm']['hand_self']
    checks['HO_absent_count_matches_receipt']=summary['hand_object']['samples_without_generated_contact_unknown_clearance']==r['summary']['sample_counts_without_generated_hand_object']
    checks['ties_count_matches_receipt']=int(anyties.sum())==r['summary']['sample_counts_with_tied_pairs']
    if not all(checks.values()):raise ValueError('Saved diagnostic numerical inconsistency')
    output.mkdir(parents=True)
    (output/'renderer_source_before.py').write_bytes(Path(__file__).read_bytes())
    with (output/'sampled_geometry601.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['sample_index','time_s','HO_generated_present','HO_native_contacts','HO_generated_pairs','HO_observed_depth_m_BLANK_if_absent','HO_saved_generated_overlap_aggregate_m_NOT_clearance','HO_over3mm_observed','HO_tied_pairs','self_generated_present','self_native_contacts','self_generated_pairs','self_observed_depth_m_BLANK_if_absent','self_saved_generated_overlap_aggregate_m_NOT_clearance','self_over3mm_observed','self_tied_pairs','any_tied_pair','absent_pair_unknown_gap_count'])
        for k in range(601):
            row=[k,float(t[k])]
            for c in (ho,ss):
                row += [bool(c['present'][k]),int(c['native_contact_count'][k]),int(c['generated_pairs'][k]),float(c['observed_depth_m'][k]) if c['present'][k] else None,float(c['raw_aggregate_depth_m'][k]),bool(c['present'][k] and c['raw_aggregate_depth_m'][k]>.003),int(c['tied_pairs'][k])]
            row += [bool(anyties[k]),int(np.count_nonzero(~mask[k]))]
            w.writerow(row)
    with (output/'category_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['category']+list(summary['hand_object']));w.writeheader()
        for name,value in summary.items():w.writerow({'category':name,**value})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True,gridspec_kw={'height_ratios':[1.2,1.2,.8]})
    fig.suptitle('Full151 candidate: parent numerical FAIL + sampled geometry FAIL',fontsize=15,fontweight='bold',color='#a52828',y=.98)
    for ax,name,color,label in [(axs[0],'hand_object','#225d96','Hand–object'),(axs[1],'hand_self','#80613c','Hand–self')]:
        c=category[name];s=summary[name]
        ax.plot(t,c['observed_depth_m']*1000,color=color,lw=1.1,marker='.',ms=1.6,label='generated native overlap only; gaps where absent')
        ax.axhline(3,color='#b33232',ls='--',lw=1,label='original3 mm diagnostic criterion')
        k=s['worst_sample_index'];value=s['maximum_generated_depth_m']*1000
        ax.scatter([t[k]],[value],color='#b33232',s=25,zorder=4)
        ax.annotate(f'max {value:.4f} mm at{t[k]:.3f} s',(t[k],value),xytext=(15,-3),textcoords='offset points',ha='left',va='top',fontsize=9)
        ax.set(title=f"{label}: {s['samples_over_original3mm']}/601 samples exceed3 mm; {s['samples_with_generated_contact']}/601 have generated contacts",ylabel='generated depth (mm)',ylim=(-.3,max(3.8,value*1.24)))
        ax.legend(loc='upper right',fontsize=8,framealpha=.9)
        ax.grid(alpha=.18)
    ax=axs[2]
    ax.scatter(t[ho['present']],np.full(int(ho['present'].sum()),2),s=6,marker='|',color='#225d96',label='generated contact observed')
    ax.scatter(t[~ho['present']],np.full(int((~ho['present']).sum()),2),s=15,marker='x',color='#777777',label='HO absent: clearance UNKNOWN')
    ax.scatter(t[ss['present']],np.full(int(ss['present'].sum()),1),s=6,marker='|',color='#80613c')
    ax.scatter(t[anyties],np.zeros(int(anyties.sum())),s=8,marker='|',color='#bd7222',label='native selected-distance ties≥2')
    ax.set(yticks=[0,1,2],yticklabels=['tied pair≥2','self generated','HO generated'],ylim=(-.55,2.7),xlabel='prescribed sample time (s)')
    ax.text(.01,.95,f"HO absent: {int((~ho['present']).sum())}/601; tied-pair samples: {int(anyties.sum())}/601",transform=ax.transAxes,va='top',fontsize=9)
    ax.legend(loc='lower right',fontsize=8,framealpha=.9)
    ax.grid(axis='x',alpha=.18)
    fig.text(.05,.025,'Saved-only601 samples at5 ms; connecting lines do not bound overlap between samples. Absent pair gaps stayNaN; blank depth is UNKNOWN clearance.\nA zero generated-depth aggregate means no native overlap observed in that category, not positive clearance. Prescribed source64 q/objectpose, zero physics steps; no force/dynamics/contact-intent qualification.',fontsize=9,va='bottom')
    fig.tight_layout(rect=(.02,.09,.99,.95),h_pad=1.7)
    fig.savefig(output/'full151_sampled_geometry.png',dpi=180);fig.savefig(output/'full151_sampled_geometry.svg');plt.close(fig)
    if any(n in sys.modules for n in ('torch','warp','mujoco','scipy')):raise AssertionError('Saved-only renderer imported a model/physics numerical runtime')
    result={'schema':'full151_sampled_geometry_saved_summary:1','status':'SUMMARY_OF_FAILED_NUMERICAL_AND_SAMPLED_GEOMETRY_CANDIDATE','source':pin(__file__),'inputs':inputs,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'parent_candidate_status':r['parent_candidate_status'],'sampled_depth3mm_pass':False,'category_summary':summary,'samples_with_any_tied_pair':int(anyties.sum()),'source_query_calls_record_only':r['actual_calls'],'new_queries_or_physics_calls':0,'geometry_or_physics_qualified':False,'no_clearance_or_force_qualification':True,'visual_QA':'PENDING_ACTUAL_VIEW','artifacts':[pin(p) for p in sorted(output.iterdir()) if p.is_file()]}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({'PNG':str(output/'full151_sampled_geometry.png'),'summary':pin(output/'summary.json'),'counts':{k:v['samples_over_original3mm'] for k,v in summary.items()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.run:main(Path(args.input).resolve(),Path(args.output).resolve())
    else:print(json.dumps({'status':'NOT_RUN','source':pin(__file__),'scope':'Default does not read diagnostic outputs or render'}))
