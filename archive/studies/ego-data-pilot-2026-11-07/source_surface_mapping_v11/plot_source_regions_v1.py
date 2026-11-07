"""Phone cross sections and geometric surface normals; no dynamics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull

H=Path(__file__).resolve().parent; P=H.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
old=read(P/'contact_patch_v7/paired_surface_anchors_v1.json')
exact=read(H/'exact_source_original_hand_anchors_v1.json')
sens=read(H/'source1mm_opposed_side_original_hand_anchors_v1.json')
with np.load(P/'contact_patch_v7/geometry_arrays_v1.npz') as z:
    Rg=z['geom71_current_world_R']; pg=z['geom71_current_world_p']
    vg=z['geom71_vertices_geom_local_FP32'].astype(float); triangles=z['geom71_hull_triangles']
Rp=np.array(old['source_pose']['phone_body_R_world']); pp=np.array(old['source_pose']['phone_root_world_m'])
V=np.linalg.solve(Rp,(vg@Rg.T+pg-pp).T).T
edges=sorted({tuple(sorted((int(t[i]),int(t[(i+1)%3])))) for t in triangles for i in range(3)})
def section(z):
    out=[]
    for ia,ib in edges:
        a,b=V[ia],V[ib]
        if (a[2]-z)*(b[2]-z)<=0 and a[2]!=b[2]:out.append((a+(b-a)*(z-a[2])/(b[2]-a[2]))[:2])
    xy=np.array(out); ids=ConvexHull(xy).vertices
    return xy[np.r_[ids,ids[0]]]*1000

fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
colors=['#888888','#1565c0','#d95f02']; labels=['Prior CAD mapping','Exact ego proximity','1 mm sensitivity']
for panel,ax in enumerate(axes):
    z=float(np.mean([r['original_ego_material_target_phone_body_m'][2] for r in exact['anchors']])) if panel==0 else exact['anchors'][panel-1]['original_ego_material_target_phone_body_m'][2]
    xy=section(z);ax.fill(xy[:,0],xy[:,1],color='#f1f2f3');ax.plot(xy[:,0],xy[:,1],color='#333333',lw=1)
    for j,bundle in enumerate([old,exact,sens]):
        for i,row in enumerate(bundle['anchors']):
            if panel and i!=panel-1:continue
            s=row['surfaces']['phone']; point=np.array(s['projected_point_phone_body_m'])*1000;n=np.array(s['outward_normal_phone_body'])
            ax.plot(point[0],point[1],marker='o' if i==0 else 's',color=colors[j],ms=5,label=labels[j] if i==0 or panel==2 else None)
            length=2.1 if panel==0 else .65
            ax.annotate('',xy=point[:2]+length*n[:2],xytext=point[:2],arrowprops=dict(arrowstyle='->',color=colors[j],lw=1.6))
    if panel:
        q=np.array(exact['anchors'][panel-1]['original_ego_material_target_phone_body_m'])*1000
        ax.add_patch(Circle(q[:2],1,fill=False,ls='--',color='#d95f02',lw=.9))
        ax.set_xlim(q[0]-1.7,q[0]+1.7);ax.set_ylim(q[1]-1.7,q[1]+1.7)
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('Phone body x [mm]');ax.set_ylabel('Phone body y [mm]')
    ax.set_title(['Cross section near source contacts','Thumb: 36 um relocation','Index: 495 um relocation'][panel])
handles,names=axes[0].get_legend_handles_labels()
fig.legend(handles,names,loc='upper left',bbox_to_anchor=(.04,.86),fontsize=9)
fig.suptitle('Mesh outward normals at source proximity regions (arrows are not measured forces)',fontsize=11)
fig.savefig(H/'source_regions_v1.png',dpi=180,bbox_inches='tight')
fig.savefig(H/'source_regions_v1.pdf',bbox_inches='tight')
print(str(H/'source_regions_v1.png'))
