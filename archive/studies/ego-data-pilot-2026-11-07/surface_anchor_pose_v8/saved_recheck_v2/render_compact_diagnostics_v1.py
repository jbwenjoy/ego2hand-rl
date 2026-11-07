"""Format saved assessment scalars only; no data/model/query reprocessing."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
pin=lambda p:dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
r=json.loads((H/'report.json').read_text());rows=r['results'];x=np.arange(len(rows));labels=[f"s{v['seed']}r{v['rank']}" for v in rows]
fig,ax=plt.subplots(1,3,figsize=(11,3.4),constrained_layout=True)
ax[0].plot(x,[v['conditional8']['scaled_loss'] for v in rows],'o-',label='conditional 8-ray')
ax[0].plot(x,[v['native4']['scaled_loss'] for v in rows],'s-',label='native 4-ray')
ax[0].set(title='Different points and normals',ylabel='scaled 6D residual')
for k,finger in [(0,'thumb'),(1,'index')]:
    ax[1].plot(x,[v['normal_opposition_error_deg'][k] for v in rows],'o-',label=finger)
    ax[2].plot(x,[v['failed_geometry_constraints'][k]['violation_m']*1e6 for v in rows],'o-',label=finger)
ax[1].axhline(1,color='black',ls=':',label='1 degree gate')
ax[1].set(title='Fixed surface-normal error',ylabel='degrees')
ax[2].axhline(0,color='black',ls=':')
ax[2].set(title='Pad compression beyond 100 micrometres',ylabel='excess (micrometre)')
for a in ax:
    a.set_xticks(x,labels,rotation=25);a.grid(alpha=.25);a.legend(fontsize=8);a.title.set_fontsize(10)
fig.suptitle('All six candidates failed; no cone-only attribution or holding qualification',fontsize=11)
out=H/'six_candidate_diagnostics_compact.png';fig.savefig(out,dpi=180,bbox_inches='tight');plt.close(fig)
receipt=dict(source=pin(Path(__file__)),saved_assessment=pin(H/'report.json'),plot=pin(out),saved_numbers_unchanged=True,new_native_or_LP_calls=0)
with (H/'compact_plot_receipt_v1.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
