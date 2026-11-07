"""Saved-only author figures/template; not an independent numerical audit.

This script is prepared before outcomes and must not run until Root terminal.
No state conversion, objective, geometry, model, Torch, native or GPU call.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np


def require(v,m):
    if not v:raise ValueError(m)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def pin(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}


def write(out,name,value):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def rows_for(rows,path,kind):
    return [r for r in rows if r['path']==path and ((r['mode']=='donor') if kind=='donor' else (r['mode']!='donor'))]


def confusion(rows):
    return {'rows':len(rows),'native_gt3_proxy_positive':sum(r['native_HO_depth_gt3mm'] and r['proxy_P_positive'] for r in rows),
        'native_gt3_proxy_zero':sum(r['native_HO_depth_gt3mm'] and not r['proxy_P_positive'] for r in rows),
        'proxy_alert_without_observed_native_gt3':sum(not r['native_HO_depth_gt3mm'] and r['proxy_P_positive'] for r in rows),
        'neither_observed':sum(not r['native_HO_depth_gt3mm'] and not r['proxy_P_positive'] for r in rows)}


def coverage(rows,field):
    miss=[r for r in rows if r['primary_miss']]
    gap=lambda r:bool(r[field]['geom_set_gap_ids'])
    covered=lambda r:bool(r[field]['covered_proxy_undersensitivity_ids'])
    return {'native_positive_proxy_zero_rows':len(miss),'geom_gap_rows':sum(gap(r) for r in miss),
        'covered_undersensitivity_rows':sum(covered(r) for r in miss),'overlap_rows':sum(gap(r) and covered(r) for r in miss),
        'neither_flag_rows':sum(not gap(r) and not covered(r) for r in miss),'categories_overlap_not_unique_causes':True}


def savefig(fig,out,name,plt):
    fig.savefig(out/(name+'.png'),dpi=190,facecolor='white');fig.savefig(out/(name+'.svg'),facecolor='white');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    run=Path(args.run_dir).resolve(strict=True);out=Path(args.output).resolve()
    require(not out.exists(),'Fresh author-render directory required')
    out.mkdir(parents=True);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    names=('report.json','rows_all1808.json','statistics.json','converter_qualification.json','geometry_ID_mapping.json','output_pins.json')
    before={n:pin(run/n) for n in names};expected=read(run/'output_pins.json')
    for n,v in before.items():
        if n!='output_pins.json':require(v['sha256']==expected[n]['sha256'],'Saved pin mismatch: '+n)
    try:
        report=read(run/'report.json');rows=read(run/'rows_all1808.json');gates=read(run/'converter_qualification.json');geom=read(run/'geometry_ID_mapping.json')
        require(report['status']=='COMPLETED_SAME_SOLVER_PHASE_PROXY_OBSERVABILITY_NOT_PHYSICS_CERTIFICATE' and len(rows)==1808,'Complete actual dense run required')
        require(gates['all_converter_gates_passed'] is True and report['converted_rows']==2260 and report['scored_rows']==1808,'Prerequisite conversion scope differs')
        for path in ('A0','A2'):
            require(len(rows_for(rows,path,'donor'))==600 and len(rows_for(rows,path,'branches'))==304,'All600 donor/304 branch rows required perpath')
        total=confusion(rows);primary=coverage(rows,'primary_deepest_tie_coverage');secondary=coverage(rows,'secondary_all_violating_coverage')
        table=[]
        for path in ('A0','A2'):
            for mode in ('donor','baseline','direction'):
                rr=[r for r in rows if r['path']==path and r['mode']==mode]
                table.append({'path':path,'mode':mode,**confusion(rr)})
        data=[]
        for r in rows:
            data.append({k:r[k] for k in ('row_id','path','mode','k','branch_id','substep','canonical_pre_tick','native_generated_HO_count','native_generated_HO_depth_m','native_HO_depth_gt3mm','P','BP','minimum_sampled_sdf_m','proxy_sample_depth_m','proxy_P_positive','primary_miss','proxy_min_sdf_exactly_minus_FP32_allowance','native_depth_minus_nominal_3mm_m')})
        for name,values in [('all1808_descriptive.csv',data),('all_strata_confusion.csv',table)]:
            with (out/name).open('x',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
        class_defs=[('Both above threshold','#2563eb',lambda r:r['native_HO_depth_gt3mm'] and r['proxy_P_positive']),
          ('Native >3mm, proxy P=0','#b91c1c',lambda r:r['primary_miss']),
          ('Proxy alert, no observed native >3mm','#d97706',lambda r:r['proxy_P_positive'] and not r['native_HO_depth_gt3mm']),
          ('Neither observed','#9ca3af',lambda r:not r['proxy_P_positive'] and not r['native_HO_depth_gt3mm'])]
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for pi,path in enumerate(('A0','A2')):
            for mi,kind in enumerate(('donor','branches')):
                ax=axes[pi,mi];rr=rows_for(rows,path,kind)
                lim=max([3.5]+[max(r['native_generated_HO_depth_m'],r['proxy_sample_depth_m'])*1000 for r in rr])*1.16
                for label,color,fn in class_defs:
                    chosen=[r for r in rr if fn(r)]
                    ax.scatter([r['native_generated_HO_depth_m']*1000 for r in chosen],[r['proxy_sample_depth_m']*1000 for r in chosen],s=18,c=color,alpha=.55,linewidths=0,zorder=3)
                ax.plot([0,lim],[0,lim],color='.65',lw=.8);ax.axvline(3,color='#111827',ls='--',lw=.8);ax.axhline(3,color='#111827',ls='--',lw=.8)
                ax.set_xscale('symlog',linthresh=3);ax.set_yscale('symlog',linthresh=3);ax.set_xlim(-.15,lim);ax.set_ylim(-.15,lim);ax.grid(color='.9',lw=.6)
                ax.set_title(f'{path}: {kind}, all{len(rr)} saved substeps')
                ax.set_xlabel('Native generated HO depth at PRE pose (mm)');ax.set_ylabel('Sampled proxy depth at SAME pose (mm)')
                cf=confusion(rr)
                ax.text(.025,.97,'both+ {native_gt3_proxy_positive}; native+P0 {native_gt3_proxy_zero}\nproxy alert {proxy_alert_without_observed_native_gt3}; neither {neither_observed}'.format(**cf),transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
        fig.suptitle('Same solver phase: native candidates versus fixed sampled penalty',fontsize=14,y=.99)
        fig.legend([Line2D([],[],color=c,marker='o',ls='') for _,c,_ in class_defs],[n for n,_,_ in class_defs],ncol=2,loc='upper center',bbox_to_anchor=(.5,.955),frameon=False,fontsize=9)
        fig.text(.02,.014,'All1808 rows retained; coincident points overlap. Dashed lines: nominal3mm. Generated-contact absence is not global no-penetration evidence.',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.895));savefig(fig,out,'same_pose_all1808_depth_scatter',plt)
        # Donor time is continuous; branch horizontal positions are restore IDs,
        # never a fictitious continuous trajectory or an independent seed count.
        fig,axes=plt.subplots(2,2,figsize=(13,8.5))
        for pi,path in enumerate(('A0','A2')):
            for mi,kind in enumerate(('donor','branches')):
                ax=axes[pi,mi];rr=rows_for(rows,path,kind)
                xx=[r['canonical_pre_tick']*.005 if kind=='donor' else (r['branch_id']-pi*76)+(r['substep']-1)/4 for r in rr]
                ax.scatter(xx,[r['native_generated_HO_depth_m']*1000 for r in rr],c='#2563eb',s=10,alpha=.7,label='Native generated HO depth')
                ax.scatter(xx,[r['proxy_sample_depth_m']*1000 for r in rr],c='#d94801',s=11,marker='x',alpha=.75,label='Proxy sampled depth')
                ax.axhline(3,color='#111827',ls='--',lw=.8);ax.set_yscale('symlog',linthresh=3);ax.grid(axis='y',color='.9')
                all_depth=[max(r['native_generated_HO_depth_m'],r['proxy_sample_depth_m'])*1000 for r in rr]
                ax.set_ylim(-.2,max([3.5]+all_depth)*1.2)
                ax.set_title(f'{path}: {kind} ({len(rr)} dependent substeps)');ax.set_ylabel('Depth (mm), symlog')
                if kind=='donor':ax.set_xlabel('Donor canonical PRE time (s)')
                else:
                    ax.set_xlabel('Restored branch ID within path + substep offset')
                    for boundary in (19,38,57):ax.axvline(boundary-.125,c='.65',lw=.7)
                    ax.set_xticks([9,28,47,66],['k=0','k=49','k=99','k=149'])
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.956),ncol=2,frameon=False)
        fig.suptitle('Complete donor coverage and restored branches kept separate',y=.995,fontsize=14)
        fig.text(.02,.014,'No lines across restores. All600 donor and304 branch substeps perpath shown; repeated baselines/nearby steps are not independentN.',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.91));savefig(fig,out,'same_pose_donor_vs_branches_complete',plt)
        # Nonstacked because geom coverage evidence can overlap in one tied set.
        fig,ax=plt.subplots(figsize=(12,5.5));labels=[];gap=[];under=[];overlap=[];miss=[]
        for path in ('A0','A2'):
            for mode in ('donor','baseline','direction'):
                rr=[r for r in rows if r['path']==path and r['mode']==mode];cc=coverage(rr,'primary_deepest_tie_coverage')
                labels.append(path+'/'+mode);gap.append(cc['geom_gap_rows']);under.append(cc['covered_undersensitivity_rows']);overlap.append(cc['overlap_rows']);miss.append(cc['native_positive_proxy_zero_rows'])
        x=np.arange(6);width=.2
        for shift,values,color,label in [(-1.5,gap,'#b91c1c','Native-only geom in deepest tie set'),(-.5,under,'#2563eb','Covered deepest geom, proxy SDF ≥ −3mm'),(.5,overlap,'#7c3aed','Both flags in same row'),(1.5,miss,'#6b7280','All native-positive/proxy-zero rows')]:
            bars=ax.bar(x+shift*width,values,width,color=color,label=label)
            ax.bar_label(bars,fontsize=8,padding=2)
        ax.set_xticks(x,labels,rotation=12);ax.set_ylabel('Saved rows (dependent)');ax.grid(axis='y',color='.9',zorder=0);ax.set_axisbelow(True)
        ax.set_ylim(0,max([1]+gap+under+overlap+miss)*1.25)
        fig.suptitle('Primary deepest exact-tie coverage evidence: overlapping, not unique causes',fontsize=12,y=.99)
        fig.legend(*ax.get_legend_handles_labels(),ncol=2,loc='upper center',bbox_to_anchor=(.5,.93),frameon=False,fontsize=9)
        fig.text(.02,.015,'All tied deepest pairs retained. Secondary all-violating-pair evidence is reported separately; no firstargmax or exclusive causal diagnosis.',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.79));savefig(fig,out,'same_pose_deepest_tie_coverage',plt)
        md='# Same solver-phase penalty observability: author-side saved description\n\n'
        md+='This describes the completed saved consumer; it is not Contact’s independent numerical audit. No new conversion, objective, model, geometry query or forward is executed by this renderer.\n\n'
        md+=f"All1808 pre-state rows are retained:1200 donor,96 baseline and512 perturbation substeps. The converter used2260 rows (1808pre +452knownpost20);160 leading-anchor comparisons do not add conversions. Native generated-HO depth above3mm occurs in{total['native_gt3_proxy_positive']+total['native_gt3_proxy_zero']} rows; proxyP is positive in{total['native_gt3_proxy_positive']+total['proxy_alert_without_observed_native_gt3']} rows. Native-positive/proxy-zero:{total['native_gt3_proxy_zero']}; bothpositive:{total['native_gt3_proxy_positive']}; proxyalert without observednative>3mm:{total['proxy_alert_without_observed_native_gt3']}; neitherobserved:{total['neither_observed']}. These are repeated dependent rows, not independent trials.\n\n"
        md+='|Path|Stratum|Rows|Both+|Native+ / P0|Proxy alert|Neither observed|\n|---|---|---:|---:|---:|---:|---:|\n'
        for r in table:md+='|{path}|{mode}|{rows}|{native_gt3_proxy_positive}|{native_gt3_proxy_zero}|{proxy_alert_without_observed_native_gt3}|{neither_observed}|\n'.format(**r)
        md+='\n## Interpretation boundaries\n\n'
        md+='Contactdist and object_solver_pose are from the solver PRE state. ProxyP is evaluated at that same reconstructed PRE qpos/qvel; post20ms scores are not substituted. The actual nativeReader fixed palm frame is recorded per452 groups, while fixed-root source/topology is separate evidence; this is not1808 directmocap readings.\n\n'
        md+='Primary coverage uses every EXACT tied deepest pair; all>3mm violating pairs are a separately named secondary set. Gap and covered-undersensitivity flags can coexist. A covered-geom miss can combine finite sampling, object-exterior/mesh/narrowphase differences; it does not identify a unique sampling cause or implementationbug. A missing generatedHO candidate is not global proof of no penetration.\n\n'
        md+=f"Primary deepest-tie evidence among{primary['native_positive_proxy_zero_rows']} miss rows: geomgap{primary['geom_gap_rows']}, coveredundersensitivity{primary['covered_undersensitivity_rows']}, overlap{primary['overlap_rows']}. Secondary allviolating-pair evidence: geomgap{secondary['geom_gap_rows']}, coveredundersensitivity{secondary['covered_undersensitivity_rows']}, overlap{secondary['overlap_rows']}. Counts overlap and must not be summed as exclusive causes.\n\n"
        borders=sum(r['proxy_min_sdf_exactly_minus_FP32_allowance'] for r in rows)
        md+=f"The original nominal native gate remains>0.003m; actualP uses the unchanged FP32 scalar allowance{report['threshold_arithmetic']['proxy_FP32_scalar_allowance_m']:.17g}m. Exact proxy-boundary rows:{borders}. Continuous native_depth_minus_nominal_3mm_m and fullSDF remain saved; no threshold was tuned or relaxed.\n\n"
        md+='P is the original normalized squared sampled penetration; −0.1P has no additional dt inside the module. There is no WM/Actor, derivative, optimizer or new physics here. Local proxy/compiled qualifications and wholehiddenstate/physicalgradient remain unqualified. Nsource1/Nseed1/Nego0; fresh saved-action-source replay is not original archive closed-loop reproduction.\n\n'
        md+='Runtime RNG/buffer invariance flags are consumer source-backed assertions; this renderer only hashes its saved inputs. FK-local body indices, native body/geom IDs and mocap IDs are distinct and the saved geometry_ID_mapping.json is authoritative.\n\n'
        for n in ('same_pose_all1808_depth_scatter.png','same_pose_donor_vs_branches_complete.png','same_pose_deepest_tie_coverage.png'):md+=f'![{n}]({(out/n).as_posix()})\n\n'
        for n in ('all1808_descriptive.csv','all_strata_confusion.csv'):md+=f'- [{n}]({(out/n).as_posix()})\n'
        (out/'diagnostic.md').write_text(md,encoding='utf-8')
        require(all(pin(run/n)==v for n,v in before.items()),'Saved inputs changed')
        require(all(x not in sys.modules for x in ('torch','mujoco','warp')),'Renderer imported a backend')
        result={'schema':'full151105_same_phase_penalty_saved_author_description:1','status':'COMPLETED_AUTHOR_SAVED_DESCRIPTION_NOT_INDEPENDENT_AUDIT','source':pin(__file__),'input_pins':before,
           'confusion_all':total,'strata':table,'primary_deepest_tie_coverage':primary,'secondary_allviolating_coverage':secondary,
           'new_conversion_objective_model_geometry_WM_AD_Env_physics_GPU':0,'wholehiddenstate_qualified':False,'physical_gradient_qualified':False,
           'visual_QA':'PENDING_ACTUAL_VIEW','outputs':{p.name:pin(p) for p in out.iterdir() if p.is_file()}}
        write(out,'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'diagnostic':pin(out/'diagnostic.md')},indent=2))
    except BaseException as error:
        write(out,'failure.json',{'status':'FAILED_SAVED_AUTHOR_RENDER_PRESERVED','error':repr(error),'input_pins':before,'source':pin(__file__),'new_science_execution':False});raise


if __name__=='__main__':main()
