"""Saved v2/v3 push histories; adjacent-dt descriptions, no physics or selection."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('saved_orientation_math_v1',BASE/'warp_diagnose_push_orientation_v1.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def main():
    out=BASE/'push_orientation_diagnostic_v2.json'
    if out.exists():raise FileExistsError(out)
    v2_path=BASE/'cpu_screen_v2/result.json'
    v3_path=BASE.parent/'simulation_timestep_v3/cpu_screen_v3/result.json'
    p2,p3=json.loads(v2_path.read_text()),json.loads(v3_path.read_text())
    assert p2['schema']=='phone_convex_contact_CPU_screening:2'
    assert p3['schema']=='phone_convex_timestep_CPU_screening:3'
    rows2=[r for r in p2['cases'] if r['test']=='force_push_release' and r['timeconst_s']==.0025]
    rows3=[r for r in p3['cases'] if r['test']=='force_push_release']
    assert sorted(r['dt_s'] for r in rows3)==[.00015625,.0003125,.000625]
    rows=[next(r for r in rows2 if r['dt_s']==.00125)]+sorted(rows3,key=lambda r:r['dt_s'],reverse=True)
    traces=[];desc=[];saved={};summaries=[]
    for i,r in enumerate(rows):
        with np.load(M.checked(r['trajectory']),allow_pickle=False) as z:a={k:z[k] for k in z.files}
        assert np.array_equal(a['tick'],np.arange(round(3/r['dt_s'])+1))
        d=M.describe(a,r['dt_s']);traces.append(a);desc.append(d)
        summaries.append({'source_stage':'v2' if i==0 else 'v3','case':r['case'],'dt_s':r['dt_s'],
                          'final_heading_relative_initial_deg':float(d['heading_deg'][-1]),
                          'max_thin_axis_tilt_deg':float(d['tilt_deg'].max()),
                          'final_thin_axis_tilt_deg':float(d['tilt_deg'][-1]),
                          'contact_count_histogram':d['contact_count_histogram'],
                          'generated_contact_count_change_events':d['contact_count_changes']})
        for key,value in {'tick':a['tick'],'time_s':a['tick']*r['dt_s'],'raw_clock':a['time_s'],
                          'heading_deg':d['heading_deg'],'thin_axis_tilt_deg':d['tilt_deg'],
                          'contact_count':a['contact_count'],'COM_world_m':a['com_world_m'],
                          'penetration_m':a['penetration_m']}.items():saved[f'dt{i}__{key}']=value
    pairs=[]
    for i in range(3):
        ratio=round(rows[i]['dt_s']/rows[i+1]['dt_s']);assert ratio==2
        angle,axis,rv,mask=M.rotation_difference(desc[i]['q'],desc[i+1]['q'][::2])
        shared=traces[i]['tick']*rows[i]['dt_s']
        pairs.append({'source_pair':'v2_history' if i==0 else 'v3_screening_adjacent',
                      'coarse_dt_s':rows[i]['dt_s'],'fine_dt_s':rows[i+1]['dt_s'],
                      'final_relative_angle_deg':float(angle[-1]),'final_relative_world_axis':axis[-1].tolist(),
                      'max_relative_angle_deg':float(angle.max()),'max_time_s':float(shared[np.argmax(angle)]),
                      'first_above_existing_2deg_reference_time_s':float(shared[np.flatnonzero(angle>2)[0]]) if np.any(angle>2) else None,
                      'shared_contact_count_differences':int(np.count_nonzero(traces[i]['contact_count']!=traces[i+1]['contact_count'][::2]))})
        saved.update({f'pair{i}__time_s':shared,f'pair{i}__angle_deg':angle,f'pair{i}__axis_world':axis,
                      f'pair{i}__rotvec_world_deg':rv,f'pair{i}__axis_defined':mask})
    original_fine=next(r for r in rows2 if r['dt_s']==.000625)
    with np.load(M.checked(original_fine['trajectory']),allow_pickle=False) as z:
        repeated={k:bool(np.array_equal(z[k].view(np.uint8),traces[1][k].view(np.uint8))) for k in ['qpos','qvel','contact_count']}
    curves=BASE/'push_orientation_diagnostic_v2_arrays.npz';np.savez_compressed(curves,**saved)
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,ax=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True);axes=ax.ravel()
    colors=['#7f7f7f','#d95f02','#1f77b4','#009e73']
    for i,(r,a,d) in enumerate(zip(rows,traces,desc)):
        t=a['tick']*r['dt_s'];label=f"{'v2' if i==0 else 'v3'} {r['dt_s']*1000:g} ms"
        style='--' if i==0 else '-'
        for axis,y in [(axes[0],d['heading_deg']),(axes[1],d['tilt_deg']),
                       (axes[3],a['contact_count']),(axes[4],(a['com_world_m'][:,0]-a['com_world_m'][0,0])*1000),
                       (axes[5],a['penetration_m']*1000)]:
            axis.plot(t,y,color=colors[i],ls=style,lw=.9,alpha=.85,label=label)
    for i,pair in enumerate(pairs):axes[2].plot(saved[f'pair{i}__time_s'],saved[f'pair{i}__angle_deg'],
        label=f"{pair['coarse_dt_s']*1000:g} vs {pair['fine_dt_s']*1000:g} ms",color=colors[i],ls='--' if i==0 else '-')
    axes[2].axhline(2,color='black',lw=.7,ls=':',label='Existing 2° final gate')
    labels=[('Long-axis projected heading','Change from initial (deg)'),('Thin-axis tilt from world +Z','Tilt (deg)'),
            ('Adjacent-step orientation differences','Relative angle (deg)'),('All generated phone–plane candidates','Count'),
            ('World COM travel','Δ COM X (mm)'),('Observed-candidate penetration','Depth (mm)')]
    for axis,(title,ylabel) in zip(axes,labels):
        axis.set(title=title,ylabel=ylabel,xlabel='Integer-tick time (s)',xlim=(0,3))
        axis.axvspan(1,1.2,color='grey',alpha=.15);axis.grid(alpha=.2);axis.legend(fontsize=6.7)
    fig.suptitle('Saved push histories · τ=2.5 ms, m=0.2 kg, μ=0.6 · original v2 plus all three v3 step sizes\nPost-hoc description only; selection remains with the independent full-gate assessment',fontsize=12)
    png=BASE/'push_orientation_diagnostic_v2.png';svg=BASE/'push_orientation_diagnostic_v2.svg'
    fig.savefig(png,dpi=180);fig.savefig(svg);plt.close(fig)
    report={'schema':'phone_convex_saved_push_orientation_diagnostic:2','status':'SAVED_ONLY_DESCRIPTION_NOT_SELECTION',
            'source':M.pin(__file__),'math_source':M.pin(BASE/'warp_diagnose_push_orientation_v1.py'),
            'v2_result':M.pin(v2_path),'v3_result':M.pin(v3_path),'inputs':[r['trajectory'] for r in rows],
            'all_trace_summaries':summaries,'all_adjacent_pair_summaries':pairs,
            'v2_0p625ms_vs_fresh_v3_0p625ms_raw_bit_identity':repeated,
            'outputs':{'full_curves':M.pin(curves),'figure_PNG':M.pin(png),'figure_SVG':M.pin(svg)},
            'limits':['All four curves and both new adjacent pairs are preserved; finest dt is not ground truth.',
                      'No acceptance decision is made here; Contact independently applies all nine cases and all original gates.',
                      'Final yaw-like differences coexist with transient tilt/contact-topology switching; no unique causal inference.',
                      'No model, FK, collision query, new physics, training or GPU. v2 null selection remains unchanged.']}
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'report':M.pin(out),'pairs':pairs,'repeated_capture':repeated}))


if __name__=='__main__':main()
