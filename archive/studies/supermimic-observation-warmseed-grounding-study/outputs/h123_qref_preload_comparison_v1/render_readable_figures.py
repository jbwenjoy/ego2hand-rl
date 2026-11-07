"""Artifact-only re-render of frozen arrays after actual image QA; no analysis/physics."""
from pathlib import Path
import numpy as np,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('D:/workspace/supermimic-observation-warmseed-grounding-study')
OUT=Path(__file__).resolve().parent
JOINT_TO_PLOT=3
def plot(data,targetdelta):
 colors={'QREF':'#31688e','PRELOAD':'#b63679'};plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
 fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True);j=JOINT_TO_PLOT
 for label,d in data.items():
  c=colors[label];z=d['raw'];st=z['raw_time_s']*1000;t=d['t']*1000
  axes[0,0].plot(st,1000*d['position'],color=c,label=label);axes[0,1].plot(st,d['angle'],color=c,label=label);axes[0,2].plot(st,1000*d['excess'].max(axis=1),color=c,label=label)
  axes[1,0].plot(t,d['force'][:,2],color=c,label=label);axes[1,1].plot(t,1000*np.linalg.norm(d['moment'],axis=1),color=c,label=label);axes[1,2].step(t,d['activefingers'].sum(axis=1),where='post',color=c,label=label)
  axes[2,0].plot(st,1000*(z['joint_rad'][:,j]-z['joint_lower_rad'][j]),color=c,label=label);axes[2,1].plot(st,z['joint_velocity_rad_s'][:,j],color=c,label=label);axes[2,2].plot(t,d['actual'][:,j],color=c,label=label+' measured');axes[2,2].plot(t,d['pred'][:,j],color=c,ls='--',lw=1,label=label+' affine(pre)')
 for ax,title,y in zip(axes.flat,['Object position error','Directed shaft error','Maximum actual joint excess','Measured force on object','Measured total COM moment','Positive-force fingers','Thumb joint4 clearance from lower limit','Thumb joint4 actual velocity','Thumb joint4 actuator force'],['mm','deg','mrad','Fz [N]','norm [mNm]','fingers','clearance [mrad]','qd [rad/s]','Nm']):ax.set(title=title,ylabel=y,xlabel='time [ms]',xlim=(0,300));ax.grid(alpha=.2)
 for ax,value in [(axes[0,0],20),(axes[0,1],20),(axes[0,2],1)]:ax.axhline(value,color='0.5',ls=':',lw=1)
 axes[1,0].axhline(data['QREF']['weight'],color='0.5',ls=':',label='weight');axes[2,0].axhline(0,color='0.5',ls=':',label='hard lower');axes[1,2].set(ylim=(0,5),yticks=range(6))
 axes[0,0].legend();axes[1,0].legend();axes[2,2].legend(fontsize=7);fig.suptitle('H123 canonical same-q exploratory rest | QREF earlier, PRELOAD later | N=1 each\nStates: 0-300ms; force/PD: solver prestate 0-295ms. Gates unchanged; no method-effect inference.',fontsize=12)
 fig.savefig(OUT/'sameq_rest_comparison_readable.png');fig.savefig(OUT/'sameq_rest_comparison_readable.svg');plt.close(fig)
 fig,ax=plt.subplots(2,3,figsize=(13,9),constrained_layout=True);qrange=max(np.abs(d['raw']['joint_rad']-d['raw']['joint_rad'][0]).max() for d in data.values());vrange=max(np.abs(d['raw']['joint_velocity_rad_s']).max() for d in data.values())
 for col,(label,d) in enumerate(data.items()):
  q=d['raw']['joint_rad']-d['raw']['joint_rad'][0];v=d['raw']['joint_velocity_rad_s'];im=ax[0,col].imshow(q.T,aspect='auto',origin='lower',extent=(0,300,.5,20.5),cmap='RdBu_r',vmin=-qrange,vmax=qrange);im2=ax[1,col].imshow(v.T,aspect='auto',origin='lower',extent=(0,300,.5,20.5),cmap='RdBu_r',vmin=-vrange,vmax=vrange);ax[0,col].set_title(label+' q minus initial [rad]');ax[1,col].set_title(label+' joint velocity [rad/s]');fig.colorbar(im,ax=ax[0,col],shrink=.75);fig.colorbar(im2,ax=ax[1,col],shrink=.75)
 ax[0,2].bar(np.arange(1,21),targetdelta*1000,color='#b63679');ax[0,2].set(title='Actual fixed u: PRELOAD minus QREF',xlabel='joint index (4 per finger)',ylabel='mrad');ax[0,2].axhline(0,color='0.4',lw=.5)
 for label,d in data.items():ax[1,2].plot(d['t']*1000,np.abs(d['actual']-d['pred']).max(axis=1),color=colors[label],label=label+' correct prestate');ax[1,2].plot(d['t']*1000,np.abs(d['actual']-d['wrong_post']).max(axis=1),color=colors[label],ls=':',label=label+' WRONG poststate')
 ax[1,2].set(title='PD prediction timing diagnostic',xlabel='solver prestate [ms]',ylabel='maximum force error [Nm]',yscale='log');ax[1,2].legend(fontsize=7)
 for a in [ax[0,0],ax[0,1],ax[1,0],ax[1,1]]:a.set(xlabel='state time [ms]',ylabel='joint index (4 per finger)');a.set_yticks([1,4,8,12,16,20]);[a.axhline(v,color='0.6',lw=.4,alpha=.6) for v in [4.5,8.5,12.5,16.5]]
 fig.suptitle('All20 joint trajectories and control inputs | shared color ranges | descriptive N1 contrast',fontsize=12);fig.savefig(OUT/'all_joint_control_comparison_readable.png');fig.savefig(OUT/'all_joint_control_comparison_readable.svg');plt.close(fig)

report=json.loads((OUT/'report.json').read_text());before=hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest()
with np.load(OUT/'comparison_arrays.npz') as z:arrays={k:z[k].copy() for k in z.files}
data={}
for label,dir in [('QREF','h123_rest_baseline_v1'),('PRELOAD','h123_rest_baseline_preload_v1')]:
 with np.load(ROOT/'outputs'/dir/'raw_states.npz') as z:raw={k:z[k].copy() for k in z.files}
 data[label]={'raw':raw,'weight':float(np.linalg.norm(report['LP_witness']['static_object_contact_wrench_world'][:3]))}
 for key in ['t','force','moment','activefingers','actual','pred','wrong_post','position','angle','excess']:data[label][key]=arrays[label+'__'+key]
delta=data['PRELOAD']['raw']['fixed_target_FP32'].astype(float)-data['QREF']['raw']['fixed_target_FP32'].astype(float)
plot(data,delta)
assert hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest()==before
print('Artifact-only rendering complete; report and data unchanged.')
