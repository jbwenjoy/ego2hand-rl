"""Saved-only full151 LP candidate figures; no fitting, model or physics calls."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def write_csv(p,columns,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)


def summarize(source,output):
    if output.exists():raise FileExistsError('Summary output must be new; never overwrite a figure')
    receipt_path=source/'receipt.json'
    receipt=json.loads(receipt_path.read_text('utf-8'))
    if receipt.get('schema')!='full151_input_cpu_receipt:1':raise ValueError('Wrong receipt schema')
    if receipt.get('cost',{}).get('LP_calls')!=20:raise ValueError('This renderer requires the saved complete20 LP attempt')
    if len(receipt.get('joint_results',[]))!=20:raise ValueError('Missing saved joint attempt')
    artifacts={str(Path(p['path']).resolve()):p['sha256'] for p in receipt['artifacts']}
    inputs={'receipt':pin(receipt_path)}
    def used(name,relative):
        p=source/relative
        if str(p.resolve()) not in artifacts or sha(p)!=artifacts[str(p.resolve())]:
            raise ValueError('Saved artifact hash absent/mismatched: '+str(p))
        inputs[name]=pin(p)
        return p
    def optional(name,relative):
        return load(used(name,relative)) if (source/relative).exists() else None
    fit=optional('fit_solutions','fit_solutions.npz')
    ref=optional('reference','reference_bspline151.npz')
    control=optional('control','planned_control150.npz')
    pd=optional('PD_parameters','actual_PD_parameters.npz')
    init=optional('init','init_requested.npz')
    groups=[];joints=[]
    for j,row in enumerate(receipt['joint_results']):
        if row['joint']!=j:raise ValueError('Saved joint order changed')
        result=load(used(f'joint{j:02d}_result',f'joint_LPs/joint{j:02d}_result.npz'))
        contract=json.loads(used(f'joint{j:02d}_contract',f'joint_LPs/joint{j:02d}_contract.json').read_text('utf-8'))
        residual=result['raw_Ax_minus_b']
        if residual.size:
            if residual.shape!=(2402,) or not np.isfinite(residual).all():raise ValueError('Bad saved residual shape/value')
            maximum=float(residual.max())
            if maximum!=row['maximum_raw_Ax_minus_b']:raise ValueError('Saved maximum and actual residual array differ')
            for group in contract['row_groups']:
                value=float(residual[group['start']:group['stop']].max())
                if value!=row['row_group_max_raw_Ax_minus_b'][group['name']]:raise ValueError('Saved row-group mismatch')
                groups.append([j,group['name'],group['units'],value,1e-12,value<=1e-12])
        else:maximum=None
        joints.append([j,row['success'],row['status'],row['iterations'],row['wall_s'],row.get('objective'),maximum,row['arithmetic_1e12_gate']])
    if ref is not None:
        if fit is None or control is None or pd is None or init is None:raise ValueError('Incomplete exported-array set')
        shapes=[(ref['q_goal_rad'],(151,20)),(ref['planned_q601_rad'],(601,20)),(ref['analytic_qd151_rad_s'],(151,20)),(fit['fit_target151_rad'],(151,20)),(control['u_postC2_150_FP32_rad'],(150,20)),(control['postC2_real_affine_rawforce_Bernstein_Nm'],(150,4,20)),(control['reference_affine601_FP32_Nm'],(601,20)),(control['reference_difference601_FP32_Nm'],(601,20))]
        if any(a.shape!=s or not np.isfinite(a).all() for a,s in shapes):raise ValueError('Malformed/nonfinite exported arrays')
        if control['u_postC2_150_FP32_rad'].dtype!=np.float32:raise ValueError('Planned targets lost FP32 identity')
        t=ref['time_s'];u=control['u_postC2_150_FP32_rad'];delta=ref['q_goal_rad']-fit['fit_target151_rad']
        rate=np.diff(u.astype(float),axis=0)/.02
        if not np.array_equal(rate,control['planned_rate149_rad_s']):raise ValueError('Saved rates do not match finalU32')
        lower=control['force_bounds_Nm'][:,0];upper=control['force_bounds_Nm'][:,1]
        force=control['postC2_real_affine_rawforce_Bernstein_Nm']
        bern_margin=np.minimum(force-lower,upper-force).min(axis=(1,2))
        aff=control['reference_affine601_FP32_Nm'];diff=control['reference_difference601_FP32_Nm']
        affmargin=np.minimum(aff-lower,upper-aff).min(axis=1)
        diffmargin=np.minimum(diff-lower,upper-diff).min(axis=1)
        innerlo=pd['target_lower'].astype(np.float32)+np.float32(.020)
        innerhi=pd['target_upper'].astype(np.float32)-np.float32(.020)
        domainmargin=np.minimum(u-innerlo,innerhi-u).min(axis=1)
        initchecks={
          'source64_P0':bool(np.array_equal(ref['control_points_rad'][0],fit['P0'])),
          'source64_P1':bool(np.array_equal(ref['control_points_rad'][1],fit['P1'])),
          'requested_q0_FP32':bool(np.array_equal(ref['q_goal_rad'][0].astype(np.float32),init['q_init_requested_FP32_rad'])),
          'requested_analytic_qd0_FP32':bool(np.array_equal(ref['analytic_qd151_rad_s'][0].astype(np.float32),init['qd_init_requested_FP32_rad_s'])),
          'fixed_u0_FP32':bool(np.array_equal(u[0].view(np.uint32),init['u_planned0_FP32_rad'].view(np.uint32))),
          'startup_ctrl_to_first_target_jump_zero':bool(not np.any(u[0].astype(float)-init['u_planned0_FP32_rad'].astype(float))),
        }
    else:
        initchecks={}
    output.mkdir(parents=True)
    (output/'renderer_source_before.py').write_bytes(Path(__file__).read_bytes())
    write_csv(output/'joint_LP_numerical_status.csv',['joint_index0','solver_success','solver_status','iterations','wall_s','objective','max_raw_Ax_minus_b','raw_gate_1e12'],joints)
    write_csv(output/'raw_constraint_group_residuals.csv',['joint_index0','row_group','row_units','max_raw_Ax_minus_b','protocol_numerical_threshold','passed'],groups)
    failed=[r[0] for r in joints if not r[-1]]
    numeric={}
    if ref is not None:
        write_csv(output/'fit_time151.csv',['time_s','max_abs_error_rad','RMSE_rad']+[f'joint{j:02d}_signed_error_rad' for j in range(20)],[[t[k],float(np.abs(delta[k]).max()),float(np.sqrt(np.mean(delta[k]**2)))]+delta[k].tolist() for k in range(151)])
        write_csv(output/'fit_per_joint.csv',['joint_index0','MAE_rad','RMSE_rad','max_abs_rad','max_after_initial_rad','initial_signed_error_rad'],[[j,float(np.abs(delta[:,j]).mean()),float(np.sqrt(np.mean(delta[:,j]**2))),float(np.abs(delta[:,j]).max()),float(np.abs(delta[1:,j]).max()),float(delta[0,j])] for j in range(20)])
        write_csv(output/'planned_targets150.csv',['interval_index0','left_time_s','original_C2_min_domain_margin_rad','min_Bernstein_rawforce_cap_margin_Nm']+[f'joint{j:02d}_planned_u_FP32_rad' for j in range(20)],[[k,t[k],float(domainmargin[k]),float(bern_margin[k])]+u[k].astype(float).tolist() for k in range(150)])
        write_csv(output/'planned_updates149.csv',['new_interval_index','update_time_s','max_abs_planned_rate_rad_s']+[f'joint{j:02d}_signed_rate_rad_s' for j in range(20)],[[k+1,t[k+1],float(np.abs(rate[k]).max())]+rate[k].tolist() for k in range(149)])
        write_csv(output/'reference_PD_margin601.csv',['time_s','affine_grouping_min_cap_margin_Nm','difference_grouping_min_cap_margin_Nm'],np.c_[ref['planned_time601_s'],affmargin,diffmargin])
        numeric={'fit_MAE_rad':float(np.abs(delta).mean()),'fit_max_abs_rad':float(np.abs(delta).max()),'fit_max_after_initial_rad':float(np.abs(delta[1:]).max()),'max_planned_rate_rad_s':float(np.abs(rate).max()),'min_original_C2_domain_margin_rad':float(domainmargin.min()),'min_postC2_real_affine_Bernstein_cap_margin_Nm':float(bern_margin.min()),'min_FP32_affine601_cap_margin_Nm':float(affmargin.min()),'min_FP32_difference601_cap_margin_Nm':float(diffmargin.min()),'C2_changed_components150':int(np.count_nonzero(control['u_pre150_FP32_rad']!=u))}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axs=plt.subplots(2,3,figsize=(15,9))
    failed_status=receipt['status']!='CPU_INPUT_GATES_PASS_PHYSICS_PENDING'
    fig.suptitle('Full151 CPU candidate — '+('NUMERICAL GATE FAIL / NOT QUALIFIED' if failed_status else 'CPU reference/control gates pass; physics pending'),fontsize=15,fontweight='bold',color='#a72525' if failed_status else '#222222',y=.985)
    if ref is not None:
        ax=axs[0,0]
        ax.plot(t,1000*np.abs(delta).max(axis=1),label='max over20 joints',color='#9e3f3f')
        ax.plot(t,1000*np.sqrt(np.mean(delta**2,axis=1)),label='RMSE over20 joints',color='#205c93')
        ax.scatter([0],[1000*np.abs(delta[0]).max()],color='#9e3f3f',s=23,zorder=4)
        ax.set(title='151-knot fit against derived539d geometry',xlabel='reference time (s)',ylabel='q fit error (mrad)');ax.legend(fontsize=8)
        ax.text(.98,.96,f'MAE {1000*np.abs(delta).mean():.3f} mrad\ninitial mismatch {1000*np.abs(delta[0]).max():.3f} mrad',transform=ax.transAxes,ha='right',va='top',fontsize=8)
        ax=axs[0,1]
        ax.step(np.r_[t[:150],3.],np.vstack((u,u[-1:])),where='post',linewidth=.8,alpha=.7)
        ax.set(title='150 post-C2 planned targets (20 joints)',xlabel='left interval time / ZOH (s)',ylabel='absolute target (rad)')
        ax.text(.02,.97,f'Original C2 domain: PASS; minimum margin {domainmargin.min():.3e} rad\nPlanning domain tightened inward by2e-6 rad',transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
        ax=axs[0,2]
        ax.scatter(np.repeat(t[1:150],20),np.abs(rate).ravel(),s=2,color='#9ba5ae',alpha=.45)
        ax.plot(t[1:150],np.abs(rate).max(axis=1),'o',ms=2.8,color='#225c92',label='maximum over20 joints')
        ax.axhline(15,color='#b43f37',lw=1,label='original gate15 rad/s')
        ax.axhline(14.9999,color='#222',ls=':',lw=.8,label='LP planning14.9999 rad/s')
        ax.set(title='149 planned updates at20..2980 ms',xlabel='update time (s)',ylabel='absolute planned target rate (rad/s)',ylim=(-.3,16.6))
        ax.text(.02,.96,f'max final-U32 rate {np.abs(rate).max():.9f} rad/s\nstartup ctrl→first target:0 (same u0)',transform=ax.transAxes,va='top',fontsize=8)
        ax.legend(loc='lower right',fontsize=7)
        ax=axs[1,0]
        ax.step(np.r_[t[:150],3.],np.r_[bern_margin,bern_margin[-1]],where='post',color='#205c93',label='600 Bernstein coefficient min / interval')
        ax.plot(ref['planned_time601_s'],affmargin,color='#b17629',lw=.8,alpha=.8,label='601 FP32 affine reference samples')
        ax.plot(ref['planned_time601_s'],diffmargin,color='#568847',lw=.8,alpha=.65,label='601 FP32 difference reference samples')
        ax.axhline(0,color='#b43f37',lw=1)
        ax.set(title='Reference raw-PD force cap slack; not measured force',xlabel='reference time / interval (s)',ylabel='min slack to both actual caps (Nm)')
        ax.legend(fontsize=7)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    else:
        for ax in (axs[0,0],axs[0,1],axs[0,2],axs[1,0]):
            ax.axis('off');ax.text(.5,.5,'Export arrays NOT GENERATED\nNo values imputed',ha='center',va='center',transform=ax.transAxes)
    ax=axs[1,1]
    values=[np.nan if r[6] is None else r[6] for r in joints]
    colors=['#ae3030' if j in failed else '#326b86' for j in range(20)]
    ax.scatter(np.arange(20),values,c=colors,s=25,zorder=3)
    ax.axhline(1e-12,color='#ae3030',ls='--',lw=1,label='frozen numerical gate1e-12')
    ax.set_yscale('symlog',linthresh=1e-13,linscale=.5)
    ax.set(title=f'Raw LP residuals: {len(failed)}/20 joints fail',xlabel='joint index (zero based)',ylabel='max saved A x − b (mixed row units)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j in failed:ax.annotate(str(j),(j,values[j]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8,color='#ae3030')
    ax.legend(fontsize=7,loc='upper left')
    ax=axs[1,2];ax.axis('off')
    init_ok=bool(initchecks) and all(initchecks.values())
    lines=[f"Receipt: {sum(bool(v) for v in receipt['checks'].values())}/{len(receipt['checks'])} gates",f"20/20 solver success; raw gate failures: {failed}",'',f'P0/P1, q0/analytic qd0/u0 retained: {init_ok}', 'Initial q0 was not spliced to derived frame1.','No C2 projection changes: '+str(numeric.get('C2_changed_components150')==0),'','Failed raw maxima (not rounded to zero):']
    for j in failed:lines.append(f'  joint{j:02d}   {values[j]:.4e}')
    target4=next((r[3] for r in groups if r[0]==4 and r[1]=='target_upper'),None)
    if target4 is not None:lines+=['',f'j4 tightened target row: {target4:.4e}', 'Original post-C2 domain still passes.']
    lines+=['','Geometry / contact / physics NOT VALIDATED.','Reference PD slack is not actual force.']
    ax.text(0,.99,'\n'.join(lines),va='top',fontsize=9,family='monospace',linespacing=1.38)
    for ax in axs.flat:
        if ax.axison:ax.grid(alpha=.17)
    fig.text(.04,.035,'Saved-only diagnostic; one legacy observation lineage[200,351), one derived geometric target; no robot joint labels or independent repetitions.\nFixed H123 initialization; continuous goal q / planned u differ from actual state and actuator force. Tracking-only sufficient constraints; no hold arm, new physics or optimization in this renderer.',fontsize=9,va='bottom')
    fig.tight_layout(rect=(.02,.09,.99,.95),w_pad=2,h_pad=2)
    fig.savefig(output/'full151_candidate_summary.png',dpi=180)
    fig.savefig(output/'full151_candidate_summary.svg')
    plt.close(fig)
    if any(n in sys.modules for n in ('torch','warp','mujoco','scipy')):raise AssertionError('Renderer imported forbidden numerical/physics runtime')
    summary={'schema':'full151_CPU_candidate_saved_summary:1','status':'SAVED_ONLY_SUMMARY_NUMERICAL_GATES_FAIL' if failed_status else 'SAVED_ONLY_CPU_SUMMARY_PHYSICS_PENDING',
      'source':pin(Path(__file__)),'inputs':inputs,'producer_status':receipt['status'],'producer_gates':receipt['checks'],'initial_arrays_checks':initchecks,
      'failed_raw_gate_joints0':failed,'numerical_summary':numeric,'reported_joint_results':receipt['joint_results'],
      'scope':'No LP/FK/collision/physics/WM/GPU; the original numerical gate failure is retained. Reference force predictions are not measured. No geometry/dynamics/training qualification.',
      'figure_QA':'PENDING_ACTUAL_VIEW','artifacts':[pin(p) for p in sorted(output.iterdir()) if p.is_file()]}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({'status':summary['status'],'PNG':str(output/'full151_candidate_summary.png'),'summary':pin(output/'summary.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:print(json.dumps({'status':'NOT_RUN','source':pin(__file__),'scope':'Default does not load candidate data or render'}))
    else:summarize(Path(args.input).resolve(),Path(args.output).resolve())
