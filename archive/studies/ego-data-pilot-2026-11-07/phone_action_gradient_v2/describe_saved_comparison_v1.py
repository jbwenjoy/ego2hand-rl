"""Saved-only response plots/strata; no Torch, native or weight update."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
def write(p,j):
    with Path(p).open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False)
reportpath=H/'comparison_completed_v1/report.json';report=read(reportpath)
metrics=read(checked(report['side_metrics']))
with np.load(checked(report['array_output']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
native_report=read(H/'native_completed_v1/report.json');records=read(checked(native_report['records']))
record_by_id={r['branch_id']:r for r in records};base_signatures={}
def signature(record):
    with np.load(checked(record['output']),allow_pickle=False) as z:c=z['contact_rows24']
    return sorted((int(t),int(g1),int(g2),int(dim),int(addr>=0)) for t,g1,g2,dim,addr in c[:,:5])
for r in records:
    if r['kind']=='baseline':base_signatures[r['state_id']]=signature(r)
changes=[]
for bid,sid in zip(a['branch_id'],a['state_id']):changes.append(signature(record_by_id[int(bid)])!=base_signatures[int(sid)])
changes=np.array(changes,dtype=bool);assert changes.sum()==126 and len(changes)==476
ids=a['state_id'].astype(int);primary=a['joint']<8
nchange=np.linalg.norm(a['native_side_observable59'][:,40:43]-a['native_baseline59'][ids,40:43],axis=1)
pchange=np.linalg.norm(a['model_side_observable59'][:,40:43]-a['model_baseline59'][ids,40:43],axis=1)
nnorm=np.linalg.norm(a['native_secant_F64'][:,40:43],axis=1)
pnorm=np.linalg.norm(a['model_secant_F64_denominator'][:,40:43],axis=1)
out=H/'saved_description_v1';out.mkdir()
csvpath=out/'all_side_COM_and_signature.csv'
with csvpath.open('x',newline='',encoding='utf8') as f:
    w=csv.writer(f);w.writerow(['branch_id','state_id','joint','cohort','signature_changed','native_COM_change_um','model_COM_change_um','native_response_m_per_rad','model_response_m_per_rad'])
    for i in range(476):w.writerow([int(a['branch_id'][i]),int(ids[i]),int(a['joint'][i]),'primary8' if primary[i] else 'secondary12',bool(changes[i]),nchange[i]*1e6,pchange[i]*1e6,nnorm[i],pnorm[i]])
stats={}
for group,mask in [('primary8',primary),('secondary12',~primary),('all20',np.ones(476,dtype=bool))]:
    stats[group]=dict(sides=int(mask.sum()),native_COM_exact_zero=int(np.sum(nnorm[mask]==0)),
        native_COM_response_p50_p90_max_m_per_rad=np.quantile(nnorm[mask],[.5,.9,1]).tolist(),
        model_COM_response_p50_p90_max_m_per_rad=np.quantile(pnorm[mask],[.5,.9,1]).tolist(),
        model_COM_actual_change_p50_p90_max_mm=np.quantile(pchange[mask]*1000,[.5,.9,1]).tolist(),
        signature_changed=int(np.sum(changes[mask])))
fig,axes=plt.subplots(1,2,figsize=(10.5,4.9),layout='constrained')
limit=max(float(nchange.max()),float(pchange.max()))*1e6*1.15
for ax,(name,mask) in zip(axes,[('Original 8 actuators',primary),('Additional 12 actuators',~primary)]):
    for flag,color,label in [(False,'#1565c0','Same contact signature'),(True,'#d95f02','Changed contact signature')]:
        chosen=mask&(changes==flag)
        ax.scatter(nchange[chosen]*1e6,pchange[chosen]*1e6,s=17,alpha=.6,color=color,label=label,edgecolors='none')
    ax.plot([0,limit],[0,limit],ls='--',color='#444444',lw=1)
    ax.set_xscale('symlog',linthresh=.1);ax.set_yscale('symlog',linthresh=.1)
    ax.set_xlim(-.03,limit);ax.set_ylim(-.03,limit);ax.grid(alpha=.25)
    ax.set_xlabel('Native COM change from base action [um]');ax.set_ylabel('WM COM change from base action [um]')
    ax.set_title(name+f" ({int(mask.sum())} one-sided branches)")
    ax.text(.04,.95,f"Exact zero native response: {int(np.sum(nnorm[mask]==0))}",transform=ax.transAxes,va='top',fontsize=9)
axes[0].legend(fontsize=8,loc='lower right')
fig.suptitle('Fixed WM5000 vs native 20ms action responses\nMagnitudes only; vector directions are reported separately',fontsize=11)
png=out/'COM_response_magnitudes_v1.png';fig.savefig(png,dpi=180,bbox_inches='tight')
j=dict(schema='phone_full20_saved_response_description:1',source=pin(__file__),comparison=pin(reportpath),native_report=pin(H/'native_completed_v1/report.json'),
    signatures='Sorted PRE tick/geom1/geom2/dim/(efc_address>=0), including multiplicity. No force threshold, location, normal or smoothness certificate.',
    stats=stats,signature_changes=126,signature_same=350,csv=pin(csvpath),figure=pin(png),
    new_Torch_NN_native_or_training_calls=0,scope='Description of all predeclared matched contexts; magnitude differences do not establish analytic gradients or policy gains. Zero values remain zero on symlog axes.')
write(out/'report.json',j)
print(json.dumps(stats))
