"""Figures from frozen saved evaluations only; zero inference or native calls."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent.parent
CHECK=P/'phone_geometry_momentum_evaluation_v1/completed_v1'
FRESH=P/'phone_causal_fresh_evaluation_v1/completed_v1'
OUT=P/'outputs/geometry_WM5000_description_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
check=read(CHECK/'endpoint_statistics.json');fresh=read(FRESH/'endpoint_statistics.json');responses=read(CHECK/'side_response_statistics.json')
groups=[('Known-state new actions',check,'NEW_ACTION_SIDES'),('New controlled families',fresh,'ALL_COMPLETE_FRESH_ENDPOINTS')]
roles=['model5000','CV','ballistic'];labels=['Geometry/momentum WM5000','Constant velocity prior','Ballistic prior'];colors=['#3475a8','#d18b43','#9a9fa5']
values={}
for name,records,cohort in groups:
    values[name]={role:next(v['state'] for v in records if v['cohort']==cohort and v['role']==role) for role in roles}
side=next(v for v in responses if v['cohort']=='ALL_KNOWN_SIDES');ratios=np.array([e['RMS']/t['RMS'] for t,e in zip(side['native_response_blocks'],side['model_response_error_blocks'])])
with np.load(CHECK/'CHECK_all1440_side_response_arrays.npz',allow_pickle=False) as z:
    known=z['defined'];native=z['native_delta52_response'];model=z['model_delta52_response'];du=z['actual_du_F64'];mask=known&np.all(native[:,40:43]==0,axis=1)
    ghost=np.linalg.norm(model[mask,40:43]/du[mask,None],axis=1)*1000
OUT.mkdir();fig,axs=plt.subplots(2,2,figsize=(13,8.7),layout='constrained')
for ax,metric,title,ylabel in [(axs[0,0],'COM_vector_RMS_mm','20 ms COM endpoint error','Vector RMS [mm]'),(axs[0,1],'SO3_geodesic_RMS_deg','20 ms orientation endpoint error','Geodesic RMS [deg]')]:
    xx=np.arange(2);width=.24
    for i,(role,label,color) in enumerate(zip(roles,labels,colors)):
        yy=[values[name][role][metric] for name,_,_ in groups];bars=ax.bar(xx+(i-1)*width,yy,width,label=label,color=color)
        ax.bar_label(bars,fmt='%.3g',padding=3,fontsize=9)
    ax.set_xticks(xx,[name for name,_,_ in groups]);ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.18)
axs[0,0].legend(fontsize=8,loc='upper left')
ax=axs[1,0];bars=ax.bar(np.arange(6),ratios,color=['#3475a8']*6)
ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax.axhline(1,color='#555',linestyle='--',linewidth=1,label='Zero-action-response prior')
ax.set_xticks(np.arange(6),['q','q velocity','COM','World rotation','Pair force\nimpulse','Pair moment\nimpulse'],fontsize=9)
ax.set_ylim(0,max(1.4,float(ratios.max())*1.18));ax.set_ylabel('Response RMSE / native response RMS');ax.set_title('CHECK finite action response: lower is better');ax.legend(fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
ax=axs[1,1];ax.hist(ghost,bins=30,color='#3475a8',alpha=.85);ax.axvline(0,color='#555',linestyle='--',label='Observed native COM response = 0')
ax.axvline(np.median(ghost),color='#c17831',label=f'Model median = {np.median(ghost):.3f} mm/rad');ax.set_xlabel('Model finite COM response / actual action step [mm/rad]');ax.set_ylabel('Side records');ax.set_title(f'Remaining spurious response on {len(ghost)} observed-zero cases');ax.legend(fontsize=8)
fig.suptitle('Phone WM5000: one-step predictions and action-response limits',fontsize=15)
fig.supxlabel('Same nominal object/model and known source poses. Rows within families are correlated. No Actor, holding, CHORD or real-ego benefit established.',fontsize=9)
fig.savefig(OUT/'WM5000_predictions_and_response_limits.png',dpi=160);fig.savefig(OUT/'WM5000_predictions_and_response_limits.svg');plt.close(fig)
summary=dict(schema='geometry_WM5000_saved_scientific_description:1',inputs=dict(check_endpoints=pin(CHECK/'endpoint_statistics.json'),fresh_endpoints=pin(FRESH/'endpoint_statistics.json'),check_response=pin(CHECK/'side_response_statistics.json'),check_arrays=pin(CHECK/'CHECK_all1440_side_response_arrays.npz')),
    source=pin(__file__),state_metrics=values,response_error_relative_to_zero_response_prior=ratios.tolist(),native_exactzero_COM_side_count=int(mask.sum()),known_side_count=int(known.sum()),
    model_COM_secant_on_native_zero_mm_per_rad=dict(median=float(np.median(ghost)),p95=float(np.quantile(ghost,.95)),maximum=float(np.max(ghost))),
    NN_calls=0,native_calls=0,training_updates=0,plots=[pin(OUT/'WM5000_predictions_and_response_limits.png'),pin(OUT/'WM5000_predictions_and_response_limits.svg')],
    scope='Saved endpoint and empirical finite-response descriptions. Response ratio1 is zero response prior; model world-rotation response is worse here despite smaller endpoint errors. Native exactzero refers to saved endpoint response only, not an all-H structural-zero certificate. No selected checkpoint or independent ego/policy result.')
with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(dict(report=pin(OUT/'report.json'))))
