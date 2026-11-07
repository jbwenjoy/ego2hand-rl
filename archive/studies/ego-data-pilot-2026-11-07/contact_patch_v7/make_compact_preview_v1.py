"""Saved catalog formatting/preview only; no new mesh or physical query."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
H=Path(__file__).resolve().parent
pin=lambda p:dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
c=json.loads((H/'facet_catalog_v1.json').read_text());d=json.loads((H/'contact_localization_v1.json').read_text());out={}
with np.load(H/'geometry_arrays_v1.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
fig=plt.figure(figsize=(12,4.8))
for col,(gid,g) in enumerate(c['geometries'].items()):
    referenced={level:set() for level in c['thresholds']}
    for row in d['contacts']:
        if gid in row['localization']:
            for level,entries in row['localization'][gid]['groups'].items():referenced[level].update(e['group_id'] for e in entries)
    if gid=='71':
        for row in d['ego_frame42_material_candidates']:
            for level,entries in row['localization']['groups'].items():referenced[level].update(e['group_id'] for e in entries)
    out[gid]={k:g[k] for k in ['name','body','coordinate_system','compiled_FP32_vertex_count','convex_extreme_vertex_count','hull_triangle_count','local_AABB_min_m','local_AABB_max_m']}
    out[gid]['levels']={}
    for level,rows in g['levels'].items():
        top=sorted(rows,key=lambda x:-x['triangle_area_sum_m2'])[:8]
        keep={x['group_id']:x for x in top}
        keep.update({i:rows[i] for i in sorted(referenced[level])})
        out[gid]['levels'][level]={'full_group_count':len(rows),'retained_top8_or_localized_groups':[keep[i] for i in sorted(keep)]}
    V=a[f'geom{gid}_vertices_geom_local_FP32'].astype(float)*1000;T=a[f'geom{gid}_hull_triangles']
    ax=fig.add_subplot(1,3,col+1,projection='3d');ax.add_collection3d(Poly3DCollection(V[T],facecolor='#bac5cf',edgecolor='none',alpha=.20))
    for group in sorted(g['levels']['planar'],key=lambda x:-x['triangle_area_sum_m2'])[:3]:ax.add_collection3d(Poly3DCollection(V[T[group['triangle_ids']]],facecolor='#e5a943',edgecolor='none',alpha=.45))
    for row in d['contacts']:
        if gid in row['localization']:
            q=np.array(row['localization'][gid]['input_geom_local_m'])*1000;ax.scatter(*q,color='red',s=25);ax.text(*q,str(row['original_contact_index']),fontsize=8)
    if gid=='71':
        for row in d['ego_frame42_material_candidates']:
            q=np.array(row['localization']['input_geom_local_m'])*1000;ax.scatter(*q,color='blue',s=25);ax.text(*q,row['finger'],fontsize=8)
    lo=V.min(0);hi=V.max(0);center=(lo+hi)/2;r=max(hi-lo)/2
    ax.set(xlim=(center[0]-r,center[0]+r),ylim=(center[1]-r,center[1]+r),zlim=(center[2]-r,center[2]+r),title={'15':'Thumb collider','28':'Index collider','71':'Phone convex collider'}[gid]);ax.set_box_aspect((1,1,1));ax.tick_params(labelsize=7,pad=0)
    ax.set_xlabel('local x (mm)',fontsize=8,labelpad=1);ax.set_ylabel('local y (mm)',fontsize=8,labelpad=1);ax.set_zlabel('local z (mm)',fontsize=8,labelpad=1)
fig.suptitle('CAD geometry only: red native points / blue ego material candidates\nOrange = three largest planar groups; functional pad and load-bearing patch UNKNOWN',fontsize=10,y=.99)
fig.subplots_adjust(left=.015,right=.97,bottom=.12,top=.82,wspace=.02)
preview=H/'facet_localization_preview_v2.png';fig.savefig(preview,dpi=180,bbox_inches='tight',pad_inches=.15);plt.close(fig)
catalog=dict(schema='saved_contact_facet_compact_catalog:7',full_catalog=pin(H/'facet_catalog_v1.json'),geometry_arrays=pin(H/'geometry_arrays_v1.npz'),contact_localization=pin(H/'contact_localization_v1.json'),thresholds=c['thresholds'],geometries=out,scope='Top8 area groups plus every localized native/ego group, preserving IDs and vertex/triangle support. Remaining groups retained in full catalog/arrays. No functional pad or pressure-patch qualification.')
with (H/'catalog_compact_v1.json').open('x',encoding='utf8') as f:json.dump(catalog,f,indent=2,allow_nan=False)
with (H/'compact_preview_receipt_v1.json').open('x',encoding='utf8') as f:json.dump(dict(source=pin(Path(__file__)),compact_catalog=pin(H/'catalog_compact_v1.json'),preview=pin(preview),new_geometry_native_or_physics_calls=0),f,indent=2)
print(json.dumps({'compact_catalog':pin(H/'catalog_compact_v1.json'),'preview':pin(preview)}))
