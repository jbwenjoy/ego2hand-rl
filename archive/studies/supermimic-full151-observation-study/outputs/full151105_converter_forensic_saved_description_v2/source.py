"""Author-side saved arithmetic figures; never an independent audit or new gate.

Prepared before forensic outcomes. Do not execute until Root terminal notice.
JSON/NPZ/Matplotlib only: no converter, Torch, MJB, objective, model or physics.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np

BLOCKS={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'position':(40,43,'m'),
        'rotation':(43,52,'matrix entry'),'linear_velocity':(52,55,'m/s'),'angular_velocity':(55,58,'rad/s, reader/wrist frame')}


def require(v,m):
    if not v:raise ValueError(m)


def pin(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def write(out,name,value):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def savefig(fig,out,name,plt):
    fig.savefig(out/(name+'.png'),dpi=190,facecolor='white');fig.savefig(out/(name+'.svg'),facecolor='white');plt.close(fig)


def boundaries(ax,labels):
    for i in range(1,len(labels)):
        if labels[i]['path']!=labels[i-1]['path']:ax.axvline(i-.5,color='black',lw=.8)


def state_heatmap(data,labels,out,name,title,plt):
    from matplotlib.colors import TwoSlopeNorm,ListedColormap
    n=len(labels);require(data['actual'].shape==(n,58),'All state58 rows required')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,(block,(start,end,unit)) in zip(axes.ravel(),BLOCKS.items()):
        fail=~data['old_flags'][:,start:end]
        if block in ('q','qd'):
            im=ax.imshow(fail.T.astype(int),aspect='auto',origin='lower',cmap=ListedColormap(['#f3f4f6','#c026d3']),vmin=0,vmax=1,interpolation='nearest')
            cb=fig.colorbar(im,ax=ax,pad=.015,ticks=[0,1]);cb.ax.set_yticklabels(['bits equal','different'])
            detail='Exact FP32 bit comparison, including signed zero'
        else:
            ratio=abs(data['difference_F64'][:,start:end])/data['old_bound_F64'][:,start:end]
            color=np.log10(np.maximum(ratio,1e-6));hi=max(.5,float(color.max()))
            im=ax.imshow(color.T,aspect='auto',origin='lower',cmap='coolwarm',norm=TwoSlopeNorm(vmin=-6,vcenter=0,vmax=hi),interpolation='nearest')
            cb=fig.colorbar(im,ax=ax,pad=.015);cb.set_label('log10(abs(error) / OLD bound)',fontsize=8)
            detail='White: ratio1; rings mark OLD failed elements'
        failed_feature,failed_row=np.where(fail.T)
        if len(failed_row):ax.scatter(failed_row,failed_feature,s=24,facecolors='none',edgecolors='#111827',linewidth=.8,zorder=4)
        boundaries(ax,labels);ax.set_title(f'{block}: {int(fail.sum())} failed elements /{n} rows',fontsize=10)
        ax.set_xlabel(f'All{n} saved rows, original order');ax.set_ylabel('Joint / component index')
        ax.text(.01,-.24,detail,transform=ax.transAxes,fontsize=7.5)
    fig.suptitle(title,fontsize=14,y=.995)
    fig.text(.015,.012,'All rows/features shown, not only failures. Color floor1e−6 is display-only. F64 subtraction of saved FP32 values; no replacement gate or dense-scoring eligibility.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96));savefig(fig,out,name,plt)


def pose_heatmap(position,rotation,out,plt):
    from matplotlib.colors import TwoSlopeNorm
    fig,axes=plt.subplots(1,2,figsize=(13,4.7))
    for ax,d,title in zip(axes,(position,rotation),('Solver PRE position (3 coordinates)','Solver PRE orientation (9 matrix entries)')):
        require(d['actual'].shape[0]==1808,'All1808 solver pose rows required')
        ratio=abs(d['difference_F64'])/d['old_bound_F64'];c=np.log10(np.maximum(ratio,1e-6));hi=max(.5,float(c.max()))
        im=ax.imshow(c.T,aspect='auto',origin='lower',cmap='coolwarm',norm=TwoSlopeNorm(vmin=-6,vcenter=0,vmax=hi),interpolation='nearest')
        fig.colorbar(im,ax=ax,pad=.02).set_label('log10(abs(error) / OLD bound)',fontsize=8)
        ax.set_title(f'{title}\n{int((~d["old_flags"]).sum())} failed elements');ax.set_xlabel('All1808 saved solver rows');ax.set_ylabel('Coordinate / matrix component')
    fig.suptitle('Solver pose versus raw PRE freejoint pose: full saved comparison',fontsize=13,y=.99)
    fig.text(.02,.015,'Old output-scaled8eps32 rule is displayed unchanged. This is arithmetic/phase diagnosis, not a new geometry, restore or physical-gradient certificate.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.91));savefig(fig,out,'all1808_solver_pose_old_rule',plt)


def conditioning_figure(condition,out,plt):
    ratio=condition['cancellation_ratios_DIAGNOSTIC_ONLY'];cpu=condition['converted_minus_F64_shadow'];gpu=condition['endpoint_GPU_reference_minus_F64_shadow']
    require(ratio.shape==cpu.shape==(2260,58) and gpu.shape==(452,58),'All2260/452 conditioning rows required')
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    for ax,block in zip(axes.ravel(),('position','rotation','linear_velocity','angular_velocity')):
        start,end,unit=BLOCKS[block];scale=1000 if block=='position' else 1;label='mm' if scale==1000 else 'rad/s' if block=='angular_velocity' else unit
        xx=ratio[:,start:end].max(-1);yy=abs(cpu[:,start:end]).max(-1)*scale;gg=abs(gpu[:,start:end]).max(-1)*scale
        ax.scatter(xx[:1808],yy[:1808],c='#9ca3af',s=9,alpha=.35,linewidths=0,label='CPU shadow difference:1808pre')
        ax.scatter(xx[1808:],yy[1808:],c='#2563eb',s=15,alpha=.6,linewidths=0,label='CPU shadow difference:452post20')
        ax.scatter(xx[1808:],gg,c='#d94801',s=16,marker='x',alpha=.6,label='GPU reference difference:452post20')
        ax.set_xscale('symlog',linthresh=1);ax.set_yscale('symlog',linthresh=1e-9)
        ax.set_title(block+(' (reader/wrist)' if block=='angular_velocity' else '')+' | all rows retained');ax.set_xlabel('Max component operand/output ratio (descriptive only)',fontsize=8)
        ax.set_ylabel(f'Max |FP32 − secondary F64| ({label})',fontsize=9);ax.grid(color='.9',lw=.6)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,.95),frameon=False,fontsize=8)
    fig.suptitle('All2260 arithmetic conditioning descriptors; F64 is not replacement truth',fontsize=13,y=.99)
    fig.text(.02,.014,'Per-row component maxima only for display; full arrays/CSV remain saved. Cancellation correlation is not a unique cause, error theorem, accepted tolerance or TF32 attribution.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.9));savefig(fig,out,'all2260_conditioning_secondary',plt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    run=Path(args.run_dir).resolve(strict=True);out=Path(args.output).resolve();require(not out.exists(),'Fresh author forensic description required')
    out.mkdir(parents=True);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    names=('report.json','summary.json','endpoint_identities.json','anchor_identities.json','raw_conversion_all2260.npz','endpoint452_old_rule.npz','leading160_old_rule.npz',
           'solver_position1808_old_rule.npz','solver_rotation1808_old_rule.npz','conditioning_secondary_all2260.npz','output_pins.json')
    pins={name:pin(run/name) for name in names};recorded=read(run/'output_pins.json')
    for name,value in pins.items():
        if name!='output_pins.json':require(value['sha256']==recorded[name]['sha256'],'Saved forensic file changed: '+name)
    try:
        report=read(run/'report.json');summary=read(run/'summary.json')
        require(report['status']=='COMPLETED_ARITHMETIC_FORENSIC_NOT_DENSE_ELIGIBILITY' and report['dense_scoring_eligibility'] is False and summary['original_failed_gate_unchanged']=='FAIL','Diagnostic scope changed')
        endpoints=load(run/'endpoint452_old_rule.npz');leading=load(run/'leading160_old_rule.npz');position=load(run/'solver_position1808_old_rule.npz');rotation=load(run/'solver_rotation1808_old_rule.npz');condition=load(run/'conditioning_secondary_all2260.npz')
        endpoint_labels=read(run/'endpoint_identities.json');anchor_labels=read(run/'anchor_identities.json')
        # Export every saved component, including passing elements and signedzero.
        with (out/'all_old_component_diagnostics.csv').open('x',newline='',encoding='utf-8') as stream:
            w=csv.writer(stream);w.writerow(['dataset','row','feature','actual_FP32','reference_FP32','difference_F64','old_bound','old_flag','abs_error_over_old_bound'])
            for name,data in [('endpoint452',endpoints),('anchor160',leading),('solverposition1808',position),('solverrotation1808',rotation)]:
                for i in range(data['actual'].shape[0]):
                    for j in range(data['actual'].shape[1]):w.writerow([name,i,j,float(data['actual'][i,j]),float(data['reference'][i,j]),float(data['difference_F64'][i,j]),float(data['old_bound_F64'][i,j]),bool(data['old_flags'][i,j]),float(abs(data['difference_F64'][i,j])/data['old_bound_F64'][i,j])])
        with (out/'all2260_conditioning_by_block.csv').open('x',newline='',encoding='utf-8') as stream:
            w=csv.writer(stream);w.writerow(['row','stratum','block','unit','max_operand_abs_sum','max_cancellation_ratio','max_CPU_abs_diff_from_F64_shadow','max_GPUref_abs_diff_from_F64_shadow'])
            for i in range(2260):
                for block,(start,end,unit) in BLOCKS.items():
                    w.writerow([i,'pre' if i<1808 else 'post20',block,unit,float(condition['absolute_operand_sums_F64'][i,start:end].max()),float(condition['cancellation_ratios_DIAGNOSTIC_ONLY'][i,start:end].max()),
                      float(abs(condition['converted_minus_F64_shadow'][i,start:end]).max()),None if i<1808 else float(abs(condition['endpoint_GPU_reference_minus_F64_shadow'][i-1808,start:end]).max())])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
        state_heatmap(endpoints,endpoint_labels,out,'all452_endpoint_old_rule','All452 post20 endpoints: ORIGINAL rule replay diagnostics',plt)
        state_heatmap(leading,anchor_labels,out,'all160_anchor_old_rule','All160 leading anchors: ORIGINAL rule replay diagnostics',plt)
        pose_heatmap(position,rotation,out,plt);conditioning_figure(condition,out,plt)
        md='# Converter arithmetic forensics: author-side saved description\n\n'
        md+='This is saved-output rendering, not Contact’s independent mathematics audit. No conversion/model/query/scoring is repeated. The original dense attempt FAILED its output-scaled8eps32 endpoint gate before factory/scoring; that FAIL remains unchanged. This new arithmetic-only condition cannot grant dense-scoring eligibility.\n\n'
        md+='|Set|Block|Unit|Failed elements|Failed rows|Maximum absolute difference|\n|---|---|---|---:|---:|---:|\n'
        for dataset,key in [('452 endpoints','endpoints'),('160 anchors','leadinganchors')]:
            for block,v in summary[key].items():md+=f"|{dataset}|{block}|{BLOCKS[block][2]}|{v['failed_elements']}|{v['failed_rows']}|{v['maximum_absolute_difference']:.9g}|\n"
        md+='\nAll452 endpoint and160 anchor feature rows are shown, not only failures. q/qd use exactFP32 bits, including signedzero; their numeric difference being0 does not override a bit mismatch. Other components retain the ORIGINAL bound8eps32·max(1,|actual|,|reference|). Black path dividers are source grouping, not a continuous time axis across restores.\n\n'
        md+='All1808 solverPRE pose comparisons and all2260 conditioning descriptors are retained. State58 omega is reader/wrist-frame angular velocity: raw freejoint BODYomega → objectrotation toWORLD → inversepalmrotation. It is neither untouched bodyomega nor generic Euler-angle velocity. Position, rotation entries, linear velocity and angular velocity use different units and have no combined mixed-unit score.\n\n'
        md+='The F64 shadow is SECONDARY arithmetic on the same saved inputs, not replacement physical state or reference truth. Operand absolute sums and cancellation ratios are descriptors, not a proved numerical bound or newly accepted tolerance. Different F64 summation orders can matter nearzero. A correlation with cancellation does not uniquely prove roundoff as the cause, does not eliminate wrongframe/mapping/phase, and does not uniquely implicate TF32. Both CPU and recorded GPU differences from the shadow are shown.\n\n'
        md+='The forensic producer reports only passiveMJB1/nativeState2260 and diagnostics, with factory/score/WM/Actor/AD/optimizer/Env/physics0. RNG/in-memory invariance flags remain source-backed runtime evidence. Existing sources/config/failure/data are immutable; originalFAILED gate is not rescued by a successful diagnostic serialization or any newer replay flags.\n\n'
        for n in ('all452_endpoint_old_rule.png','all160_anchor_old_rule.png','all1808_solver_pose_old_rule.png','all2260_conditioning_secondary.png'):md+=f'![{n}]({(out/n).as_posix()})\n\n'
        for n in ('all_old_component_diagnostics.csv','all2260_conditioning_by_block.csv'):md+=f'- [{n}]({(out/n).as_posix()})\n'
        (out/'diagnostic.md').write_text(md,encoding='utf-8')
        require(all(pin(run/n)==v for n,v in pins.items()),'Forensic saved input changed')
        require(all(m not in sys.modules for m in ('torch','mujoco','warp')),'Saved renderer imported backend')
        result={'schema':'full151105_converter_forensic_saved_author_description:1','status':'COMPLETED_AUTHOR_DESCRIPTION_NOT_INDEPENDENT_AUDIT','source':pin(__file__),'input_pins':pins,
          'endpoint_rows':452,'anchor_rows':160,'solver_pose_rows':1808,'conditioning_rows':2260,'original_failed_gate_unchanged':'FAIL','dense_scoring_eligibility':False,
          'new_converter_MJB_objective_model_WM_Actor_geometry_AD_Env_physics_GPU':0,'new_gate_or_error_theorem_defined':False,'visual_QA':'PENDING_ACTUAL_VIEW',
          'outputs':{p.name:pin(p) for p in out.iterdir() if p.is_file()}}
        write(out,'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'diagnostic':pin(out/'diagnostic.md')},indent=2))
    except BaseException as error:
        write(out,'failure.json',{'status':'FAILED_AUTHOR_FORENSIC_RENDER_PRESERVED','error':repr(error),'input_pins':pins,'source':pin(__file__),'new_science_execution':False});raise


if __name__=='__main__':main()
