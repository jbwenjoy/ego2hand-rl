"""Saved-only full151 SCP result: failures/trials retained, no new solver/query."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def csvfile(p,columns,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)


def main(source,out):
    if out.exists():raise FileExistsError('New summary directory required')
    receipt_path=source/'receipt.json';r=json.loads(receipt_path.read_text('utf-8'))
    if r.get('schema')!='full151_contact_coupled_CPU_receipt:1' or len(r.get('majors',[]))!=8:raise ValueError('Expected saved full8-major attempt')
    artifacts={str(Path(x['path']).resolve()):x['sha256'] for x in r['artifacts']};inputs={'receipt':pin(receipt_path)}
    def use(name,relative):
        p=(source/relative).resolve()
        if artifacts.get(str(p))!=sha(p):raise ValueError('Saved artifact mismatch: '+str(p))
        inputs[name]=pin(p);return p
    cat=arrays(use('catalog','catalog.npz'));kind=cat['pair_kind'];time=np.arange(601)*.005
    data={};summary={};checks={}
    for stage,key in [('base','base_summary'),('final','final_geometry')]:
        grid=arrays(use(stage+'_grid',stage+'_geometry/native_grid.npz'))
        mask=grid['generated_mask'];d=grid['selected_distance_m']
        if mask.shape!=(601,273) or d.shape!=mask.shape or not np.isnan(d[~mask]).all() or not np.isfinite(d[mask]).all():raise ValueError('Malformed native grid or missing-gap NaN changed')
        data[stage]={};summary[stage]={}
        for name in ('hand_object','hand_self'):
            cols=kind==name;present=mask[:,cols].any(1)
            depth=np.max(np.where(mask[:,cols],np.maximum(-np.nan_to_num(d[:,cols],nan=0),0),0),axis=1)
            ties=(grid['tie_count'][:,cols]>1).any(1);k=int(depth.argmax())
            data[stage][name]={'present':present,'depth':depth,'observed':np.where(present,depth,np.nan),'ties':ties}
            summary[stage][name]={'max_depth_m':float(depth[k]),'worst_sample':k,'worst_time_s':float(time[k]),'samples_over3mm':int(np.count_nonzero(depth>.003)),'generated_present':int(present.sum()),'absent_unknown':int((~present).sum()),'tied_samples':int(ties.sum())}
            checks[stage+'_'+name+'_max_raw_exact']=summary[stage][name]['max_depth_m']==r[key][name+'_max_m']
            checks[stage+'_'+name+'_over3mm_raw_exact']=summary[stage][name]['samples_over3mm']==r[key][name+'_samples_over3mm']
        checks[stage+'_absent_gap_NaN']=bool(np.isnan(d[~mask]).all())
    majors=r['majors'];accepted=[];phi_after=[float(majors[0]['old_actual_phi'])];records=[]
    for k,row in enumerate(majors,1):
        if row['major']!=k:raise ValueError('Major order changed')
        saved=json.loads(use(f'major{k:02d}_receipt',f'major{k:02d}/receipt.json').read_text('utf-8'))
        if saved!=row:raise ValueError('Saved individual and global major record differ')
        raw=arrays(use(f'major{k:02d}_merit',f'major{k:02d}/merit_raw.npz'))
        for nk,rk in [('old_phi','old_actual_phi'),('predicted_phi','predicted_phi'),('new_phi','new_actual_phi'),('eta_m','eta_m')]:
            if nk in raw and float(raw[nk])!=row[rk]:raise ValueError('Raw major merit mismatch')
        if row['accepted']:accepted.append(k)
        phi_after.append(float(row['new_actual_phi']) if row['accepted'] else phi_after[-1])
        records.append([k,row['accepted'],row['radius_rad'],row['next_radius_rad'],row['solver_success'],row['linear_audit']['raw_primal_pass'],row['eta_m'],row['direct_fit_cost'],row['old_actual_phi'],row['predicted_phi'],row['new_actual_phi'],phi_after[-1],row['predicted_gain'],row['actual_gain'],row['acceptance_ratio'],row['geometry_summary']['hand_object_max_m'],row['geometry_summary']['hand_self_max_m'],row['geometry_summary']['hand_object_samples_over3mm'],row['geometry_summary']['hand_self_samples_over3mm']])
    checks['accepted_major_set_matches_last_source']=accepted[-1]==r['last_accepted_major']
    checks['lastaccepted_phi_matches_final']=phi_after[-1]==r['final_actual_phi']
    checks['lastaccepted_eta_matches_final']=majors[accepted[-1]-1]['eta_m']==r['last_accepted_eta_m']
    checks['overall_failed_status_retained']=r['status']=='CPU_COUPLED_CANDIDATE_FAIL_RETAINED'
    if not all(checks.values()):raise ValueError('Saved data consistency check failed')
    out.mkdir(parents=True);(out/'renderer_source_before.py').write_bytes(Path(__file__).read_bytes())
    columns=['sample_index','time_s']
    for stage in ('base','final'):
        for name in ('hand_object','hand_self'):columns += [stage+'_'+name+'_generated_present',stage+'_'+name+'_observed_depth_m_BLANK_if_absent',stage+'_'+name+'_aggregate_depth_m_NOT_clearance',stage+'_'+name+'_tied_pair_present']
    rows=[]
    for k in range(601):
        row=[k,float(time[k])]
        for stage in ('base','final'):
            for name in ('hand_object','hand_self'):
                d=data[stage][name];row += [bool(d['present'][k]),float(d['observed'][k]) if d['present'][k] else None,float(d['depth'][k]),bool(d['ties'][k])]
        rows.append(row)
    csvfile(out/'base_final_geometry601.csv',columns,rows)
    csvfile(out/'majors8.csv',['major','accepted','trust_radius_before_rad','trust_radius_next_rad','solver_success','raw_linear_pass','eta_m','direct_original_fit_smooth_cost','old_actual_Phi','predicted_Phi','trial_actual_Phi','lastaccepted_Phi_after_major','predicted_drop','actual_drop','actual_over_predicted_drop','trial_HO_max_m','trial_self_max_m','trial_HO_over3mm_samples','trial_self_over3mm_samples'],records)
    csvfile(out/'base_final_category_summary.csv',['stage','category','max_depth_m','worst_sample','worst_time_s','over3mm_samples','generated_present_samples','absent_UNKNOWN_samples','tied_samples'],[[stage,name,*summary[stage][name].values()] for stage in ('base','final') for name in ('hand_object','hand_self')])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(16,10))
    fig.suptitle('8-major contact-coupled SCP — OVERALL FAIL / NO READY',fontsize=16,color='#a52727',fontweight='bold',y=.98)
    fig.text(.5,.944,'Historical parent numerical FAIL and sampled-geometry FAIL remain unchanged; final is the last accepted major8.',ha='center',fontsize=10)
    colors={'base':'#888888','final':'#225f96'}
    for ax,name,label in [(axes[0,0],'hand_object','Hand–object'),(axes[0,1],'hand_self','Hand–self')]:
        for stage in ('base','final'):
            ax.plot(time,1000*data[stage][name]['observed'],color=colors[stage],lw=1.1,ls='--' if stage=='base' else '-',label=stage+' generated overlap')
        ax.axhline(3,color='#b03333',ls=':',lw=1.2,label='original3 mm criterion')
        sb,sf=summary['base'][name],summary['final'][name]
        ax.set(title=label+': all601 prescribed samples',xlabel='reference time (s)',ylabel='generated native depth (mm)',ylim=(-.3,17.8),xlim=(0,3))
        ax.text(.98,.97,f"max {1000*sb['max_depth_m']:.3f} → {1000*sf['max_depth_m']:.3f} mm\n>3mm: {sb['samples_over3mm']} → {sf['samples_over3mm']} /601",ha='right',va='top',transform=ax.transAxes,fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
        ax.legend(loc='center right',fontsize=7)
    major=np.arange(1,9);accept=np.array([x['accepted'] for x in majors]);trialphi=np.array([x['new_actual_phi'] for x in majors]);predphi=np.array([x['predicted_phi'] for x in majors])
    ax=axes[0,2]
    ax.plot(np.arange(9),phi_after,'-',lw=1.6,color='#225f96',label='last-accepted actual Phi')
    ax.plot(major,predphi,':o',lw=.7,ms=2.5,color='#888888',label='trial predicted Phi')
    ax.scatter(major[accept],trialphi[accept],s=27,color='#225f96',marker='o',zorder=4,label='accepted trial')
    ax.scatter(major[~accept],trialphi[~accept],s=55,color='#b03333',marker='x',zorder=5,label='rejected trial (major6)')
    ax.annotate('6 rejected',(6,trialphi[5]),xytext=(-4,14),textcoords='offset points',ha='center',fontsize=8,color='#b03333')
    ax.set(title='True/predicted elastic merit; all8 trials retained',xlabel='major attempt (0=base)',ylabel='Phi = direct fit+4smooth +1e4 max violation')
    ax.set_xticks(np.arange(9));ax.legend(fontsize=7,loc='upper right')
    ax=axes[1,0];etas=np.array([x['eta_m'] for x in majors])*1000
    ax.plot(major,etas,'o-',color='#99642c',lw=1.2,ms=4,label='LP elastic eta (mm)')
    ax.axhline(1e-6,color='#b03333',ls=':',lw=1.2,label='final eta gate1e-6 mm (=1e-9 m)')
    ax.scatter([6],[etas[5]],s=50,marker='x',color='#b03333',zorder=5)
    ax.set(title='Elastic slack FAIL; trust radius reduction',xlabel='major attempt',ylabel='eta (mm)',ylim=(-.2,11),xticks=major)
    ax.text(.98,.98,f'final eta {1000*r["last_accepted_eta_m"]:.6f} mm\nrequired ≤0.000001 mm',transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax.legend(fontsize=7,loc='center right')
    inset=ax.inset_axes([.12,.16,.78,.25])
    inset.step(major,[x['radius_rad'] for x in majors],where='mid',color='#444444',lw=1)
    inset.plot(major,[x['radius_rad'] for x in majors],'s',color='#444444',ms=2)
    inset.set(ylim=(.06,.165),xlim=(.7,8.3),yticks=[.075,.15],xticks=[],ylabel='r (rad)')
    inset.tick_params(labelsize=7);inset.yaxis.label.set_size(7)
    inset.text(.98,.75,'major6 reject → next r/2',transform=inset.transAxes,ha='right',fontsize=7)
    ax=axes[1,1];x=np.array([0,1]);width=.31
    for stage,shift in [('base',-width/2),('final',width/2)]:
        vals=[1000*summary[stage][n]['max_depth_m'] for n in ('hand_object','hand_self')]
        bars=ax.bar(x+shift,vals,width,color=colors[stage],alpha=.8,label=stage)
        for bar,name in zip(bars,('hand_object','hand_self')):
            s=summary[stage][name]
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.4,f'{bar.get_height():.3f}\n{s["samples_over3mm"]}/601 >3mm',ha='center',va='bottom',fontsize=8)
    ax.axhline(3,color='#b03333',ls=':',lw=1)
    ax.set(title='Category tradeoff; no overall benefit claim',xticks=x,xticklabels=['hand–object','hand–self'],ylabel='maximum generated depth (mm)',ylim=(0,20.5));ax.legend(fontsize=7,loc='upper right')
    ax=axes[1,2];ax.axis('off')
    failed=[k for k,v in r['checks'].items() if not v]
    lines=[f"Final: {sum(r['checks'].values())}/{len(r['checks'])} gates; FAIL / no READY",'Failed:',*['  '+k for k in failed],'',f"8 normal LPs / raw certificates pass",f"7 accepted; major6 rejected",'r:0.15 rad →0.075 rad, no re-expansion','',f"true Phi {phi_after[0]:.6f} → {r['final_actual_phi']:.6f}",f"fit+smooth {r['base_fit_cost']:.6f} → {r['final_fit_cost']:.6f}",'Merit decrease is not overall task benefit.','',f"HO absent: {summary['base']['hand_object']['absent_unknown']} → {summary['final']['hand_object']['absent_unknown']} /601 (UNKNOWN gap)",f"{r['cost']['grid_audits']} grids /6010 native collisions",'0 physics / Env / WM / GPU / training','Original control/FP32 gates pass only.','No dynamics/force/clearance qualification.']
    ax.text(0,.99,'\n'.join(lines),va='top',fontsize=8.5,family='monospace',linespacing=1.32)
    for ax in axes.flat:
        if ax.axison:ax.grid(alpha=.16)
    fig.text(.04,.025,'Saved-only601 dependent grid points and8 trials from one legacy observation lineage[200,351). Base/final are prescribed geometric references, not physical states.\nAbsent native pair gaps remainNaN; depth curves break where a category has no generated contact. Zero aggregate overlap is not positive clearance.\nLocal all-witness linearizations/elastic eta are not global derivatives or contact forces. Original3 mm criterion fails; no new solve, FK, query, WM or physics in this renderer.',fontsize=9,va='bottom')
    fig.tight_layout(rect=(.02,.10,.99,.92),h_pad=2,w_pad=2)
    fig.savefig(out/'contact_coupled_scp_summary.png',dpi=180);fig.savefig(out/'contact_coupled_scp_summary.svg');plt.close(fig)
    if any(x in sys.modules for x in ('torch','mujoco','warp','scipy')):raise AssertionError('Saved-only renderer imported model/physics numerical runtime')
    result={'schema':'full151_coupled_SCP_saved_figure_summary:1','status':'SUMMARY_OF_OVERALL_FAILED_COUPLED_CANDIDATE','source':pin(__file__),'inputs':inputs,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'base_final_summary':summary,'accepted_majors':accepted,'rejected_majors':[x['major'] for x in majors if not x['accepted']],'Phi_lastaccepted_path':phi_after,'eta_final_m':r['last_accepted_eta_m'],'final_source_status':r['status'],'parent_failed_status_unchanged':True,'new_solver_FK_queries_physics_WM_GPU':0,'visual_QA':'PENDING_ACTUAL_VIEW','scope':'Merit/tradeoff description only, no overall method or physical benefit; unknown missing gaps remainNaN.','artifacts':[pin(p) for p in sorted(out.iterdir()) if p.is_file()]}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({'PNG':str(out/'contact_coupled_scp_summary.png'),'summary':pin(out/'summary.json'),'raw_checks':[result['passed'],result['total']]}))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--run',action='store_true');arg=a.parse_args()
    if arg.run:main(Path(arg.input).resolve(),Path(arg.output).resolve())
    else:print(json.dumps({'status':'NOT_RUN','source':pin(__file__),'scope':'Default does not load results or render'}))
