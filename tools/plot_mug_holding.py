"""Scientific plot of failed native holds; no new simulation or fitting."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run,out=args.run.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((run/'report.json').read_text());colors=plt.get_cmap('viridis')(np.linspace(.05,.9,5))
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for i,case in enumerate(report['cases']):
        verify_file_pin(case['trace']);z=np.load(case['trace']['path']);dt=float(z['native_dt_s']);t=(np.arange(len(z['post_COM']))+1)*dt
        style='-' if dt==.0003125 else '--';color=colors[i//2];label=f"{case['overclosure_m']*1000:g} mm" if i%2==0 else None
        drift=np.linalg.norm(z['post_COM']-z['initial_COM'],axis=1)*1000
        angle=2*np.rad2deg(np.arccos(np.clip(np.abs(z['post_quat']@z['initial_quat']),0,1)))
        axes[0].plot(t,drift,style,color=color,label=label,lw=1.1);axes[1].plot(t,angle,style,color=color,lw=1.1)
        if i==9:
            R=Rotation.from_quat(z['pre_qpos'][:,23:27][:,[1,2,3,0]]).as_matrix()
            force=np.einsum('nij,nj->ni',R,z['pre_wrench_BODY_COM'][:,:3])
            axes[2].plot(np.arange(len(force))*dt,force[:,2],color=color,label='contact force world z, 2 mm / fine dt',lw=1)
    axes[0].axhline(5,color='#dc2626',ls=':',lw=1);axes[0].set(xlabel='Time (s)',ylabel='COM displacement (mm)',title='No case reached required 2 seconds')
    axes[1].axhline(45,color='#dc2626',ls=':',lw=1);axes[1].set(xlabel='Time (s)',ylabel='Rotation from reset (deg)',title='All10 cases exceeded45 degrees')
    axes[2].axhline(.25*9.81,color='#dc2626',ls=':',lw=1,label='weight 2.4525 N');axes[2].set(xlabel='Time (s)',ylabel='Force (N)',title='Transient load is not sustained balance')
    axes[0].legend(title='Geometric command overclosure',fontsize=8,title_fontsize=8);axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Same static-qualified pose, no object aid | solid dt=0.3125 ms, dashed dt=0.15625 ms',fontsize=11)
    out.mkdir(parents=True);fig.savefig(out/'mug_noaid_holding.png',dpi=160);plt.close(fig)
    result=dict(source=file_pin(__file__),input=file_pin(run/'report.json'),figure=file_pin(out/'mug_noaid_holding.png'),integration_steps=0,neural_updates=0)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
