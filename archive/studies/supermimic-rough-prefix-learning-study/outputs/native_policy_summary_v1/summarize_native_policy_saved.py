"""Plot saved complete native rollouts only; no environment or new physics."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin,loadnp
from rough_prefix.metadata import write
from rough_prefix.bundle_v2 import load_bundle,rotation


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b=load_bundle(ROOT/'outputs/input_bundle_v1/bundle.json');ref=b['arrays']['reference']
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.4),layout='constrained')
    entries=[('A0','native_a0_v2','#69717b'),('A1L','eval_smoke_a1l_v2','#2865a7'),
             ('P_ONLY','eval_smoke_p_only_v2','#d98222'),('A2','eval_smoke_a2_v2','#23896c')]
    rows=[];sources={}
    for method,directory,color in entries:
        run=ROOT/'outputs'/directory;v=read(run/'evaluation.json');raw=loadnp(run/'raw_states.npz')
        if not v['complete'] or len(raw['raw_time_s'])!=61:raise ValueError('Require complete saved61states')
        if method!='A0':
            proof=read(ROOT/'outputs'/('policy_saved_audit_'+method.lower()+'_v2')/'report.json')
            if not proof['engineering_qualified']:raise ValueError('Unqualified saved engineering')
        time=raw['raw_time_s'];pos=np.linalg.norm(raw['object_pose_w'][:,:3]-ref['object_COM_world_m'],axis=1)*1000
        rot=rotation(raw['object_pose_w'][:,3:]);angle=np.rad2deg(np.arccos(np.clip(
            np.sum(rot[:,:,0]*ref['object_rotation_w'][:,:,0],axis=1),-1,1)))
        solver=read(run/'solver_samples.json');events=read(run/'control_events.json')
        applied=np.asarray([e['actualu'] for e in events]).reshape(15,20)
        rates=np.abs(np.diff(np.vstack((raw['ctrl_rad'][0],applied)),axis=0)).max(axis=1)/.02
        depth=np.asarray([x['penetration_m'] for x in solver])*1000
        style={'color':color,'label':method,'lw':1.6,'alpha':.9}
        axes[0,0].plot(time,angle,**style);axes[0,1].plot(time,pos,**style)
        axes[1,0].plot([s['solver_t_s'] for s in solver],depth,**style)
        axes[1,1].plot(np.arange(15)*.02,rates,marker='.',markersize=3,**style)
        row={'method':method,'source':pin(run/'evaluation.json'),'maximum_position16_mm':v['max_object_position_error_16_m']*1000,
             'maximum_shaft16_deg':v['max_shaft_angle_16_deg'],'maximum_penetration_mm':v['max_penetration_m']*1000,
             'maximum_target_rate_including_startup_rad_s':float(rates.max()),
             'failed_quality_gates':[k for k,value in v['gates'].items() if not value],
             'quality_pass':v.get('native_policy_pass',v.get('native_A0_pass'))}
        rows.append(row)
        for name in ('raw_states.npz','solver_samples.json','control_events.json','evaluation.json'):
            sources[directory+'/'+name]=pin(run/name)
    for ax,title,ylabel,threshold in [
        (axes[0,0],'Directed shaft error','deg',20),(axes[0,1],'Object position error','mm',20),
        (axes[1,0],'Solved contact penetration (pre-integration)','mm',3),
        (axes[1,1],'Applied target updates, including startup','rad/s',15)]:
        ax.axhline(threshold,color='#b74444',ls='--',lw=1,label='Declared limit')
        ax.set(title=title,xlabel='Time (s)',ylabel=ylabel,xlim=(0,.3));ax.grid(alpha=.18);ax.set_ylim(bottom=0)
    axes[0,0].legend(loc='upper left',fontsize=8,ncol=3)
    fig.suptitle('Controlled prefix16 | native A0 + three seed0 / 10-iteration policies\n'
                 'One saved rollout each; no replication or method-benefit claim',fontsize=12)
    fig.savefig(a.output/'native_prefix_policy_diagnostics.png',dpi=170);plt.close(fig)
    write(a.output/'report.json',{'schema':'rough_native_prefix_saved_summary:1','status':'SAVED_ONLY_DESCRIPTIVE',
          'source_lineage_N':1,'independent_ego_N':0,'physical_reference_qualified':False,
          'rows':rows,'source_pins':sources,'plotter':pin(__file__),'physics_steps_added':0,
          'rate_convention':'Startup ctrl0->first appliedu0 divided by20ms; subsequent14 differences. Not instantaneous derivative.',
          'scope':'Native cadence A0 comparison only; historical directWarp T0 remains separate. Three single-seed10iter policies are engineering checkpoints, not benefit replication.'})
    print({'status':'SAVED_ONLY_DESCRIPTIVE','output':str(a.output),'rows':rows})

if __name__=='__main__':main()
