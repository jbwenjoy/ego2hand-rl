"""Saved-prefix physical diagnostics only; no native/NN/model invocation."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)

def main():
    report_path=H/'projection_completed_run1/report.json';r=read(report_path);out=H/'saved_description_v1';out.mkdir(exist_ok=False)
    routes=['no_preload_ctrl_equal_q','normal_1N_each_pad','normal_3N_each_pad']
    ids=[]
    for c in r['cases']:
        if c['candidate_id'] not in ids:ids.append(c['candidate_id'])
    assert len(ids)==2 and len(r['cases'])==12
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    ff,faxes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    rows=[];colors=['tab:blue','tab:orange','tab:green'];lines={.0003125:'-',.00015625:'--'}
    for c in r['cases']:
        i=ids.index(c['candidate_id']);j=routes.index(c['route']);dt=c['dt_s'];style=lines[dt]
        with np.load(checked(c['trace']),allow_pickle=False) as z:
            a={k:z[k].copy() for k in ['post_time','pre_time_from_integer_tick','post_COM','initial_COM_world3','post_angle',
                'pre_pad_normal_N','pre_phone_nonpad_normal_N','pre_actuator_force_margin20x2','pre_joint_actuator_force_margin20x2']}
        time=a['post_time'];drift=np.linalg.norm(a['post_COM']-a['initial_COM_world3'],axis=1)*1000
        axes[i,j].plot(time*1000,drift,style,label=f'h={dt*1e6:g} us')
        axes[i,j].plot(time[-1]*1000,drift[-1],'x',color=axes[i,j].lines[-1].get_color())
        for pad in range(3):faxes[i,j].plot(a['pre_time_from_integer_tick']*1000,a['pre_pad_normal_N'][:,pad],style,color=colors[pad],label=f'pad {pad+1}' if dt==.0003125 else None,alpha=.9)
        faxes[i,j].plot(a['pre_time_from_integer_tick']*1000,a['pre_phone_nonpad_normal_N'],style,color='black',label='other hand total' if dt==.0003125 else None,alpha=.8)
        triple=np.all(a['pre_pad_normal_N']>.1,axis=1);idx=np.flatnonzero(triple)
        rows.append(dict(candidate_id=c['candidate_id'],route=c['route'],native_dt_s=dt,actual_steps=c['finite_steps_saved'],
            terminal_prefix_s=float(time[-1]),termination=c['termination'],complete2s=c['gates']['duration_complete'],
            final_COM_drift_mm=float(drift[-1]),final_angle_deg=float(a['post_angle'][-1]*180/np.pi),
            executed_prefix_triple_coverage=float(triple.mean()),first_PRE_triple_Fn_over_point1_s=None if not len(idx) else float(a['pre_time_from_integer_tick'][idx[0]]),
            initial_RESET_FORWARD_Fn3_N=c['initial_native']['pad_normal_N3'],
            executed_prefix_max_Fn3_N=a['pre_pad_normal_N'].max(0).tolist(),other_hand_sum_Fn_max_N=float(a['pre_phone_nonpad_normal_N'].max()),
            actual_actuator_force_margin_min_Nm=float(a['pre_actuator_force_margin20x2'].min()),
            actual_joint_actuator_force_margin_min_Nm=float(a['pre_joint_actuator_force_margin20x2'].min()),
            tail_metrics_available=c['metrics']['tail_triple_pad_force_coverage'] is not None,
            new3pad_pass=c['nominal_holding_dynamics_pass'],old2pad_pass=c['old_two_pad_pass']))
    for i in range(2):
        for j in range(3):
            for ax in [axes[i,j],faxes[i,j]]:ax.set_title(('exact-source' if i==0 else '1mm algorithmic branch')+' / '+['ctrl=q','normal target 1N / pad','normal target 3N / pad'][j]);ax.grid(alpha=.25);ax.set_xlabel('executed prefix time [ms]')
            axes[i,j].axhline(5,color='gray',linestyle=':',label='5mm final gate' if i==j==0 else None)
            axes[i,j].axhline(50,color='red',linestyle=':',label='50mm abort' if i==j==0 else None)
            axes[i,j].set_ylabel('COM drift [mm]');axes[i,j].legend(fontsize=8)
            faxes[i,j].axhline(.1,color='gray',linestyle=':');faxes[i,j].set_yscale('symlog',linthresh=.1);faxes[i,j].set_ylim(bottom=0);faxes[i,j].set_ylabel('positive Fn sum [N]');faxes[i,j].legend(fontsize=7)
    fig.suptitle('All 12 native runs aborted; none completed the 2s holding test\nShown times are exit prefixes, not stable-holding durations',fontsize=13)
    ff.suptitle('Native PRE forces: pad1 / pad2 / pad3 and other-hand total\nSolid h=312.5us, dashed h=156.25us; desired normal targets are not measured loads',fontsize=13)
    fig.savefig(out/'COM_abort_prefixes.png',dpi=150);ff.savefig(out/'native_PRE_forces.png',dpi=150);plt.close(fig);plt.close(ff)
    write(out/'case_diagnostics.json',rows)
    with (out/'case_diagnostics.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'report.json',dict(schema='functional_tripod_native_saved_prefix_description:1',source=pin(__file__),producer_report=pin(report_path),
        rows=12,source_trace_reads=12,native_model_NN_AD_training_calls=0,
        all_cases_aborted=all(c['termination']=='ABORT_LOSS_OF_HOLD' for c in rows),
        all_complete2s=False,tail_metrics_available_count=sum(c['tail_metrics_available'] for c in rows),
        minimum_exit_prefix_s=min(c['terminal_prefix_s'] for c in rows),maximum_exit_prefix_s=max(c['terminal_prefix_s'] for c in rows),
        outputs={p.name:pin(p) for p in out.iterdir() if p.is_file()},
        scope='Author saved descriptive plots and prefix statistics, not new dynamics/independent geometry proof. Prefix contact coverage is never tail1.5..2s holding coverage; 1mm branch is algorithmic, not measured uncertainty.'))
    print(json.dumps(dict(report=pin(out/'report.json'))))

if __name__=='__main__':main()
