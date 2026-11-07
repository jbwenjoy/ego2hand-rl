"""Scientific figure from the already-derived response NPZ; no raw replay."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
def pin(p):
    b=Path(p).read_bytes();return dict(path=str(Path(p).resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
with np.load(H/'all320_actualdu_responses.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
axes=a['axis'];dt=a['native_dt_s'];COM=a['COM_endpoint_response_m'];J=a['J22x6_response'];ids=a['canonical_pair_geom_ids']
fig,(top,heat)=plt.subplots(2,1,figsize=(13,8.6),gridspec_kw={'height_ratios':[1,1.7]})
colors=['#2b6f9f','#d4792c']
for c,h in zip(colors,sorted(set(dt),reverse=True)):
    value=[np.sqrt(np.mean(np.sum(COM[(axes==j)&(dt==h)]**2,axis=1)))*1000 for j in range(20)]
    top.plot(range(20),value,marker='o',ms=4,lw=1.7,color=c,label=f'Native dt = {h*1000:.5g} ms (160 sides)')
top.set(xlim=(-.5,19.5),xticks=range(20),ylabel='COM endpoint response\nvector-norm RMS (mm)',xlabel='Absolute actuator-control axis')
top.legend(frameon=False,loc='upper right');top.grid(axis='y',alpha=.23);top.spines[['top','right']].set_visible(False)
top.set_title('20 ms finite-action response to baseline: all 320 sides retained',loc='left',fontweight='bold')
matrix=np.array([[np.sqrt(np.mean(np.sum(J[axes==j,p,:3]**2,axis=1))) for p in range(22)] for j in range(20)])
im=heat.imshow(matrix,aspect='auto',interpolation='nearest',cmap='magma',vmin=0)
labels=[str(g)+('*' if int(g) in [15,28,41] else '') for g in ids]
heat.set(xticks=range(22),xticklabels=labels,yticks=range(20),ylabel='Absolute actuator-control axis',xlabel='Other geom ID in canonical phone pair (* designated pad)')
heat.set_title('Per-pair force-impulse response RMS; pooled over both dt / families / source times',loc='left',fontsize=11,pad=10)
cb=fig.colorbar(im,ax=heat,pad=.015,fraction=.025);cb.set_label('Force-impulse vector-norm RMS (N s)')
fig.suptitle('Native local coupling at 8 known MON integration states',x=.065,y=.985,ha='left',fontsize=15,fontweight='bold')
fig.text(.065,.018,'Actual signed du is approximately ±0.01 rad; no side filtering. 290 Actor-EMA reachable / 30 outside (descriptor only).\nFinite 20 ms effects, not an infinitesimal Jacobian, independent trajectories or policy benefit. Zero and very small values remain descriptive.',fontsize=9,color='#444444')
fig.subplots_adjust(left=.075,right=.95,top=.925,bottom=.12,hspace=.48)
dest=H/'native_finite_action_COM_and_pair_impulse_v1.png';fig.savefig(dest,dpi=180,bbox_inches='tight');plt.close(fig)
report=dict(schema='phone_tripod_saved_response_scientific_figure:1',source=pin(__file__),derived_input=pin(H/'all320_actualdu_responses.npz'),figure=pin(dest),derived_NPZ_reads=1,raw_branch_NPZ_reads=0,new_native_NN_calls=0,units=dict(COM='mm',force_impulse='N s'),moment_impulse_not_mixed_into_force_heatmap=True)
with (H/'plot_receipt_v1.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
print(json.dumps(report['figure']))
