"""Static scientific figures and descriptive151frame intent diagnostics."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/contact_intent_v2'
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def main():
    report=json.loads((OUT/'report.json').read_text());f0=json.loads((OUT/'frame0_rankings.json').read_text());z=np.load(OUT/'surface_proximity_151.npz',allow_pickle=False);meta=report['surface_geometry'];r,L=meta['radius_m']*1000,meta['cylinder_half_length_bodyx_m']*1000
    variants=['source_rigid_mapped','morphology_normalized_target','robot_candidate_FK'];colors=['#b84a45','#2869aa','#df8d20','#8c61a5','#397b54']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.titlesize':11,'axes.labelsize':9})
    fig=plt.figure(figsize=(16.5,7.2));axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)]
    allpoints=np.concatenate([z[name+'__points_o'][0]*1000 for name in variants]);lo=np.minimum(allpoints.min(0),[-125,-9,-9])-10;hi=np.maximum(allpoints.max(0),[125,9,9])+10
    titles=['Source landmarks, rigid mapped\n(source relation preserved)','Robot-length normalized targets\n(different anatomical correspondence)','New geometric candidate FK\n(robot macro-landmarks only)']
    for ax,name,title in zip(axes,variants,titles):
        theta=np.linspace(0,2*np.pi,65);xx,tt=np.meshgrid([-L,L],theta);ax.plot_surface(xx,r*np.cos(tt),r*np.sin(tt),color='#58a3a6',alpha=.23,linewidth=.2,edgecolor='#458f93',shade=False)
        for b in meta['box_specs']:
            center=np.array(b['center_body_m'])*1000;half=np.array(b['halfsize_body_m'])*1000;end=b['end_sign']
            for axis in range(3):
                for sign in [-1,1]:
                    if axis==0 and sign==-end:
                        th,blend=np.meshgrid(np.linspace(0,2*np.pi,97),[0.,1.]);ct,st=np.cos(th),np.sin(th);outer=np.minimum(half[1]/np.maximum(np.abs(ct),1e-12),half[2]/np.maximum(np.abs(st),1e-12));radius=r+blend*(outer-r);ax.plot_surface(np.full_like(radius,center[0]+sign*half[0]),radius*ct,radius*st,color='#58a3a6',alpha=.23,shade=False)
                    else:
                        other=[a for a in range(3) if a!=axis];pts=[]
                        for v,w in [(-1,-1),(-1,1),(1,1),(1,-1)]:
                            p=center.copy();p[axis]+=sign*half[axis];p[other[0]]+=v*half[other[0]];p[other[1]]+=w*half[other[1]];pts.append(p)
                        ax.add_collection3d(Poly3DCollection([pts],facecolors='#58a3a6',edgecolors='#458f93',alpha=.22,linewidths=.35))
        points=z[name+'__points_o'][0]*1000;ax.scatter(*points[0],s=12,color='#555555')
        for finger,color in enumerate(colors):
            ids=np.arange(1+4*finger,5+4*finger);path=points[ids];ax.plot(*path.T,'-o',color=color,ms=3.5,lw=1.6,label=f'C{finger+1} '+['thumb*','index*','middle*','ring*','little*'][finger]);ax.plot(*np.vstack([points[0],path[0]]).T,color=color,lw=.5,alpha=.35,ls=':')
        for finger in [2,3]:
            p=next(p for p in f0['variants'][name]['nearest_per_finger_distal2'] if p['finger_chain']==finger);k=p['landmark_index'];point=np.array(p['point_o'])*1000;close=np.array(p['object_surface_closest_o'])*1000;n=np.array(p['outward_normal_o']);color=colors[finger-1];ax.plot(*np.vstack([point,close]).T,color=color,lw=1.1,ls='--');ax.scatter(*close,s=24,facecolors='white',edgecolors=color,linewidths=1.3);ax.quiver(*close,*n,length=14,color=color,lw=1.1,arrow_length_ratio=.25)
            offset=np.array([0,3 if finger==2 else -6,3]);ax.text(*(point+offset),f'{k}: {p["signed_distance_m"]*1000:+.2f} mm',color=color,fontsize=8)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='object x (mm)',ylabel='object y (mm)',zlabel='object z (mm)',title=title);ax.set_box_aspect(hi-lo);ax.view_init(elev=23,azim=-58);ax.tick_params(labelsize=7,pad=1)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False;axis._axinfo['grid']['color']=(.8,.8,.8,.35)
    axes[0].legend(loc='upper left',bbox_to_anchor=(-.03,.96),frameon=False,fontsize=8)
    fig.suptitle('Frame 0 — known pen exterior vs observed / normalized / robot points',fontsize=15,y=.97)
    fig.text(.5,.08,'All panels share the object frame and axis scale. Open markers: closest exterior points; arrows: object outward normals.\n* Anatomical names are ordinal assumptions. Points are joint/macro-landmarks, not finger skin or measured contacts. Confidence unknown.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.02,right=.94,bottom=.17,top=.80,wspace=.07);extra=[label for ax in axes for label in [ax.xaxis.label,ax.yaxis.label,ax.zaxis.label]];fig.savefig(OUT/'frame0_surface_proximity.png',dpi=180,bbox_inches='tight',bbox_extra_artists=extra);fig.savefig(OUT/'frame0_surface_proximity.svg',bbox_inches='tight',bbox_extra_artists=extra);plt.close(fig)
    fig,(a,b)=plt.subplots(1,2,figsize=(12.4,4.5),gridspec_kw={'width_ratios':[1.6,1]});labels=['source (rigid map)','normalized target','robot FK'];vc=['#2b6f94','#d28d32','#796393'];xx=np.arange(5)
    for j,(name,label,color) in enumerate(zip(variants,labels,vc)):
        vals=[next(p['signed_distance_m']*1000 for p in f0['variants'][name]['nearest_per_finger_distal2'] if p['finger_chain']==i) for i in range(1,6)];a.bar(xx+(j-1)*.24,vals,width=.23,label=label,color=color)
    a.set(xticks=xx,xticklabels=['C1','C2','C3','C4','C5'],ylabel='Nearest distal landmark → pen surface (mm)',title='Frame0: same ordinal distal points, different geometry');a.grid(axis='y',alpha=.2);a.set_axisbelow(True);a.legend(frameon=False,fontsize=8)
    angle=[]
    for name in variants:
        pair=next(x for x in f0['variants'][name]['opposition_pairs_sorted_by_proximity_then_opposition'] if x['finger_chains']==[2,3]);angle.append(pair['outward_normal_angle_deg'])
    b.bar(np.arange(3),angle,color=vc);b.set(xticks=np.arange(3),xticklabels=['source','normalized','robot FK'],ylim=(0,190),ylabel='Object outward-normal angle (deg)',title='C2–C3: relative surface sides');b.grid(axis='y',alpha=.2);b.set_axisbelow(True)
    for i,v in enumerate(angle):b.text(i,v+4,f'{v:.2f}°',ha='center')
    fig.text(.5,.015,'Surface-normal opposition is a hypothesis cue, not contact evidence or a force-closure test.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(OUT/'frame0_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'frame0_comparison.svg',bbox_inches='tight');plt.close(fig)
    temporal={};pairrecords={}
    for name in ['source_original']+variants:
        distances=z[name+'__unsigned_distance_m'];normals=z[name+'__outward_normal_o'];indices=np.stack([3+4*i+distances[:,3+4*i:5+4*i].argmin(1) for i in range(5)],axis=1);gap=np.take_along_axis(distances,indices,axis=1);nn=normals[np.arange(151)[:,None],indices];pairrecords[name]=[];pairsummary=[]
        for i,j in itertools.combinations(range(5),2):
            dot=np.clip(np.sum(nn[:,i]*nn[:,j],axis=1),-1,1);maxgap=np.maximum(gap[:,i],gap[:,j]);counts={str(mm):int(((maxgap<=mm/1000)&(dot<=-.5)).sum()) for mm in [1,3,5,10,20]};pairsummary.append({'finger_chains':[i+1,j+1],'descriptive_frame_counts_both_within_mm_and_normals_over120deg':counts,'maximum_normal_angle_deg':float(np.rad2deg(np.arccos(dot)).max()),'minimum_pair_max_distance_m':float(maxgap.min()),'not_contact_labels':True});pairrecords[name].append({'finger_chains':[i+1,j+1],'nearest_distal_landmark_indices':indices[:,[i,j]].tolist(),'maximum_surface_distance_m':maxgap.tolist(),'outward_normal_dot':dot.tolist()})
        temporal[name]={'finger_chain_summary':[{'finger_chain':i+1,'minimum_distal_distance_m':float(gap[:,i].min()),'median_distal_distance_m':float(np.median(gap[:,i])),'max_distal_distance_m':float(gap[:,i].max()),'frame0_signed_distal_distance_m':next(p['signed_distance_m'] for p in f0['variants'][name]['nearest_per_finger_distal2'] if p['finger_chain']==i+1)} for i in range(5)],'pair_diagnostics':pairsummary,'scope':'151time samples of one seencontrolledclip, not151independentN. Fixed distancebins and normalangle thresholds only describe geometry; no contact phase/label is inferred.'}
    (OUT/'opposition_metrics_151.json').write_text(json.dumps(pairrecords,indent=2)+'\n',encoding='utf-8');(OUT/'temporal_descriptive_summary.json').write_text(json.dumps(temporal,indent=2)+'\n',encoding='utf-8')
    artifacts={p.name:pin(p) for p in OUT.glob('*') if p.suffix in ['.png','.svg','.json','.npz'] and p.name!='figure_and_summary_manifest.json'};(OUT/'figure_and_summary_manifest.json').write_text(json.dumps({'code':pin(__file__),'artifacts':artifacts,'scope':'Static scientific figures and descriptive singleclip geometry, not contactlabels/physicalinference.'},indent=2)+'\n',encoding='utf-8');print(json.dumps({'figures':[str(OUT/f) for f in ['frame0_surface_proximity.png','frame0_surface_proximity.svg','frame0_comparison.png','frame0_comparison.svg']],'temporal_summary':str(OUT/'temporal_descriptive_summary.json')},indent=2))
if __name__=='__main__':main()
