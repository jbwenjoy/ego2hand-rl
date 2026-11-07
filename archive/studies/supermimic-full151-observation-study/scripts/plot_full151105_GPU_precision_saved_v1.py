"""Saved-only full105 GPU-mode gradient figure. No Torch/model/backend import."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

MODES = ('FP32_STORAGE_TF32_ALLOWED', 'BF16_NET_FP32_INTEGRATE_TF32_ALLOWED')
MODULE_SHA = 'b292630b290dd2d05f2987c3f0156bb2d301968680e8282ae086f66fb498964f'
PHASES = (0,49,99,149)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path): return {'path':str(Path(path).resolve(strict=True)),'sha256':sha(path)}
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(value,message):
    if not value: raise ValueError(message)
def checked(value):
    path=Path(value['path']).resolve(strict=True)
    require(sha(path)==value['sha256'],'Input pin drift: '+str(path));return path
def same_pin(a,b): return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    root=args.input.resolve(strict=True);source=pin(__file__)
    report=read(root/'report.json');pins=read(root/'output_pins.json');config=read(checked(report['config']))
    require(report['schema']=='full151105_GPU_precision_diagnostic_report:1' and report['status']=='COMPLETED_GPU_PRECISION_DIAGNOSTIC'
            and report['modes']==list(MODES) and all(v is True for v in report['invariants'].values()),'Complete invariant-preserving precision result required')
    require(report['WM_forward_calls']==report['u_autograd_calls']==2 and report['predicted_rows']==16
            and report['physical_comparison_rows_per_mode']==64 and report['dependent_direction_groups']==32,'Actual finite mode counts differ')
    for value in pins.values():checked(value)
    require(same_pin(pins['report.json'],pin(root/'report.json')),'GPU report alias differs')
    cpu_pins=read(checked(config['CPU_output_pins']))
    for name in ('report.json','WM_predictions_gradients.npz','comparisons.json'):checked(cpu_pins[name])
    require(same_pin(cpu_pins['report.json'],config['CPU_report']) and same_pin(cpu_pins['WM_predictions_gradients.npz'],config['CPU_predictions']),'Same actual CPU authority required')
    cpu_report=read(checked(config['CPU_report']));require(cpu_report['status']=='COMPLETED_CPU_DIAGNOSTIC' and all(cpu_report['invariants'].values()),'CPU reference invariants incomplete')
    code=[v for v in config['source_pins'].values() if v['sha256']==MODULE_SHA]
    require(bool(code),'Actual immutable precision source not in input closure')
    for value in code:checked(value)
    inputs={'GPU_output_pins':pin(root/'output_pins.json'),'GPU_report':pins['report.json'],'GPU_arrays':pins['precision_arrays.npz'],
            'CPU_output_pins':config['CPU_output_pins'],'CPU_report':config['CPU_report'],'CPU_arrays':config['CPU_predictions'],
            'CPU_comparisons':cpu_pins['comparisons.json'],'GPU_config':report['config'],
            **{'GPU_comparisons_'+str(i):pins[mode+'_comparisons.json'] for i,mode in enumerate(MODES)}}
    if not args.run:
        print(json.dumps({'status':'PASS_SAVED_INPUT_PINS_ONLY','source':source,'inputs':inputs,'rendered':False,'Torch_imported':'torch' in sys.modules},indent=2));return
    out=args.output.resolve();require(not out.exists(),'Fresh figure output required; no overwrite/retry')
    out.mkdir(parents=True);(out/'executed_plot_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'source_before.json',{'source':source,'precision_module':code,'inputs':inputs})
    with np.load(checked(config['CPU_predictions']),allow_pickle=False) as z:cpu=z['gradient_u'].copy()
    with np.load(checked(pins['precision_arrays.npz']),allow_pickle=False) as z:
        fp32=z[MODES[0]+'__gradient_u'].copy();bf16=z[MODES[1]+'__gradient_u'].copy()
    require(all(v.dtype==np.float32 and v.shape==(8,20) and np.isfinite(v).all() for v in (cpu,fp32,bf16)),'All three saved FP32 gradient8x20 arrays required')
    differences=(fp32.astype(np.float64)-cpu.astype(np.float64),bf16.astype(np.float64)-fp32.astype(np.float64))
    amplitude=max(float(abs(v).max()) for v in differences)
    # Allzero data keep their actual zero CSV values and an explicitly arbitrary
    # display range, never a claimed nonzero measurement.
    display_amplitude=amplitude if amplitude>0 else 1e-12
    compare={ 'CPU_FP32':read(checked(cpu_pins['comparisons.json'])),
              MODES[0]:read(checked(pins[MODES[0]+'_comparisons.json'])),
              MODES[1]:read(checked(pins[MODES[1]+'_comparisons.json'])) }
    require(all(len(rows)==64 for rows in compare.values()),'All64 rows per mode/reference required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig=plt.figure(figsize=(15.8,11.3))
    grid=fig.add_gridspec(3,2,width_ratios=[1,.027],height_ratios=[1,1,.66],left=.09,right=.945,top=.895,bottom=.185,hspace=.39,wspace=.035)
    norm=TwoSlopeNorm(vmin=-display_amplitude,vcenter=0,vmax=display_amplitude)
    labels=['A%d k%d'%(0 if s<4 else 2,PHASES[s%4]) for s in range(8)]
    titles=['GPU non-AMP FP32 storage (TF32 allowed) minus CPU FP32',
            'GPU native-net BF16 minus GPU non-AMP (same held flags, TF32 allowed)']
    for index,(values,title) in enumerate(zip(differences,titles)):
        ax=fig.add_subplot(grid[index,0]);im=ax.imshow(values,cmap='RdBu_r',norm=norm,aspect='auto',interpolation='nearest')
        ax.set_title(title,fontsize=11,pad=10)
        ax.set_yticks(range(8),labels,fontsize=9)
        ax.set_xticks(range(20),['F%d\nJ%d'%(j//4+1,j%4+1) for j in range(20)],fontsize=8)
        ax.set_xticks(np.arange(-.5,20,1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.4,alpha=.5);ax.tick_params(which='minor',bottom=False,left=False)
        for s in range(8):
            for j in range(20):
                value=values[s,j];text='0' if value==0 else '%.1e'%value
                ax.text(j,s,text,ha='center',va='center',fontsize=5.5,color='white' if abs(value)>.58*display_amplitude else '#171717')
    cax=fig.add_subplot(grid[:2,1]);fig.colorbar(im,cax=cax,label='Signed gradient difference [controlled score/rad]')
    table_ax=fig.add_subplot(grid[2,0]);table_ax.axis('off')
    table_rows=[];summaries=[]
    names=['CPU FP32','GPU non-AMP / TF32 allowed','GPU native BF16 / TF32 allowed']
    authorities=[cpu_report['agreement'],report['models'][MODES[0]]['agreement'],report['models'][MODES[1]]['agreement']]
    for index,(name,agreement) in enumerate(zip(names,authorities)):
        pair=[agreement[str(eps)] for eps in (.002,.001)]
        fractions=['%d/%d'%(v['sign_correct'],v['eligible']) for v in pair]
        rho=['NA' if v['pearson'] is None else '%.3f'%v['pearson'] for v in pair]
        delta=np.zeros((8,20),np.float64) if index==0 else (fp32 if index==1 else bf16).astype(float)-cpu.astype(float)
        maximum=float(abs(delta).max());rms=float(np.sqrt(np.mean(delta*delta)))
        table_rows.append([name,fractions[0],fractions[1],rho[0]+' / '+rho[1],'%.3e'%maximum,'%.3e'%rms])
        summaries.append(dict(mode=name,eps2_sign_correct=pair[0]['sign_correct'],eps2_eligible=pair[0]['eligible'],eps1_sign_correct=pair[1]['sign_correct'],eps1_eligible=pair[1]['eligible'],
                              eps2_Pearson=pair[0]['pearson'],eps1_Pearson=pair[1]['pearson'],maximum_abs_gradient_minusCPU=maximum,RMS_gradient_minusCPU=rms))
    table=table_ax.table(cellText=table_rows,colLabels=['Saved mode','sign / stable\neps2 mrad','sign / stable\neps1 mrad','Pearson\neps2 / eps1','max |grad-CPU|\nscore/rad','RMS grad-CPU\nscore/rad'],
                        colWidths=[.32,.12,.12,.15,.145,.145],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(8.5);table.scale(1,1.8)
    for (r,c),cell in table.get_celld().items():
        if r==0:cell.set_facecolor('#e8edf2');cell.set_text_props(weight='bold')
        cell.set_edgecolor('#b5bec8');cell.set_linewidth(.6)
    fig.suptitle('Frozen105 A2 WM: saved numerical-mode comparison',fontsize=15,y=.975)
    fig.text(.5,.937,'All 160 entries in each heatmap; one shared symmetric full-range color scale; all values in score/rad',ha='center',fontsize=10)
    caption=('One source, seed0; 8 dependent fresh action-source donor states (A0/A2), not historical closed-loop rollout reproduction.\n'
             'The same 9 stable groups at two eps are not 18 independent samples; all 64 comparison rows/mode remain in CSV, including clipped/flat/terminal rows.\n'
             'One GPU pass/mode, no variance estimate or unique attribution to precision. CPU-vs-GPU differences also mix backend/hardware/TF32.\n'
             'Controlled post20ms score only, not original15ms reward or HO force gradient; hidden graph scratch UNKNOWN. No physical-gradient qualification.')
    fig.text(.5,.055,caption,ha='center',va='bottom',fontsize=8.4,linespacing=1.5)
    fig.savefig(out/'GPU_precision_gradient_differences.png',dpi=190);fig.savefig(out/'GPU_precision_gradient_differences.svg');plt.close(fig)
    with (out/'all160_gradient_entries.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['snapshot_id','donor_action_source','k','joint_index','finger','joint','CPU_FP32','GPU_nonAMP_TF32_allowed','GPU_BF16_TF32_allowed','GPU_nonAMP_minusCPU','GPU_BF16_minusGPU_nonAMP','units'])
        for s in range(8):
            for j in range(20):w.writerow([s,'A0' if s<4 else 'A2',PHASES[s%4],j,j//4+1,j%4+1,float(cpu[s,j]),float(fp32[s,j]),float(bf16[s,j]),float(differences[0][s,j]),float(differences[1][s,j]),'controlled score/rad'])
    fields=['mode','snapshot_id','path','k','joint_index','epsilon_index','epsilon_rad','physical_slope','WM_AD_effective','WM_AD_nominal','physical_label','stable_nonflat','clipped','center_offset_maxabs','signal_floor']
    with (out/'all192_physical_comparisons.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for mode,rows in compare.items():w.writerows(dict(row,mode=mode) for row in rows)
    write(out/'summary.json',{'schema':'full151105_saved_precision_summary:1','mode_summaries':summaries,
          'GPU_BF16_minus_GPU_nonAMP':{'maximum_absolute':float(abs(differences[1]).max()),'RMS':float(np.sqrt(np.mean(differences[1]**2)))},
          'rows_per_heatmap':8,'columns_per_heatmap':20,'difference_values_preserved':320,'physical_comparison_rows_retained':192,
          'heatmap_symmetric_limit':display_amplitude,'actual_maximum_absolute_difference':amplitude,'zero_display_range_artificial':amplitude==0,
          'scope':caption,'numeric_raw_audit':'SEPARATE_CONTACT_REVIEW_NOT_DUPLICATED'})
    require(pin(__file__)==source,'Plot source changed during render')
    for value in inputs.values():checked(value)
    for value in code:checked(value)
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Saved renderer crossed framework boundary')
    write(out/'receipt.json',{'schema':'full151105_GPU_precision_saved_figure:1','status':'RENDERED_AWAITING_ACTUAL_VIEW_QA',
          'source':source,'precision_module':code,'inputs':inputs,'outliers_dropped':0,'heatmap_entries_per_panel':160,
          'comparisons_GPU_rows':128,'comparisons_CPU_rows':64,'shared_color_limits':[-display_amplitude,display_amplitude],
          'source_unchanged':True,'input_pins_unchanged':True,'Torch_imported':False,'Warp_imported':False,'WM_forwards':0,'physics_steps':0,'GPU_used':False,
          'output_pins':{v.name:pin(v) for v in out.iterdir() if v.is_file()},'scope':caption})
    print(json.dumps({'status':'RENDERED_AWAITING_ACTUAL_VIEW_QA','PNG':pin(out/'GPU_precision_gradient_differences.png'),'receipt':pin(out/'receipt.json')},indent=2))


if __name__=='__main__':main()
