"""Independent H8 donor indexing, recurrence traces and raw58-unit arithmetic.

NumPy only. Fixed 2 paths x 143 overlapping windows x 8 transitions. This does
not evaluate a model or assert independent samples, physical success or H8
policy/Actor/memory equivalence. Restored branches have no H8 observations.
"""
import numpy as np

PATHS=('A0','A2');HORIZON=8
BLOCKS={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),
        'R_raw9':(43,52,'dimensionless'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}


def require(v,message):
    if not v:raise ValueError(message)


def f32(a,shape,label):
    a=np.asarray(a);require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite FP32 '+label);return a


def bits(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def donor_window_mapping(identities):
    """Resolve original row addresses by keys, never by first300 row slicing."""
    require(type(identities) is list and len(identities)==452,'All452 original identity rows')
    donor={};mode_counts={k:0 for k in ('donor','baseline','direction')}
    for index,row in enumerate(identities):
        require(type(row) is dict and type(row.get('group_id')) is int and row['group_id']==index,'Exact original row ID')
        path,mode,k=row.get('path'),row.get('mode'),row.get('k')
        require(path in PATHS and mode in mode_counts and type(k) is int and 0<=k<150,'Original path/mode/control phase')
        require(type(row.get('source_frame')) is int and row['source_frame']==200+k and type(row.get('terminal_observation')) is int and row['terminal_observation']==k+1 and row.get('no_u150') is True,'Native source/terminal phase')
        mode_counts[mode]+=1
        if mode=='donor':
            require(row.get('branch_id') is None and (path,k) not in donor,'No restored or duplicate donor substitution')
            donor[(path,k)]=index
    require(mode_counts=={'donor':300,'baseline':24,'direction':128},'Exact original452 population before donor-only restriction')
    require(set(donor)=={(p,k) for p in PATHS for k in range(150)},'Every path/control phase0..149 required')
    donor_rows=np.asarray([[donor[(p,k)] for k in range(150)] for p in PATHS],dtype=np.int64)
    starts=np.tile(np.arange(143,dtype=np.int64),2);path_index=np.repeat(np.arange(2,dtype=np.int64),143)
    u_phase=starts[:,None]+np.arange(8,dtype=np.int64)[None]
    rows=donor_rows[path_index[:,None],u_phase]
    return {'donor_row_indices2x150':donor_rows,'window_path_index286':path_index,'window_start_k286':starts,
      'window_initial_row_indices286':rows[:,0].copy(),'window_transition_row_indices286x8':rows,
      'actual_control_phase286x8':u_phase,'target_observation_phase286x8':u_phase+1,
      'link_row_pairs298x2':np.concatenate([np.column_stack((r[:-1],r[1:])) for r in donor_rows],axis=0)}


def source_byte_guards(pre,target,u,all_substep_u,mapping):
    pre=f32(pre,(452,58),'original PRE452');target=f32(target,(452,58),'native POST452')
    u=f32(u,(452,20),'actual absoluteu452');held=f32(all_substep_u,(452,4,20),'four substep controls452')
    rows=mapping['donor_row_indices2x150'].reshape(-1);links=mapping['link_row_pairs298x2']
    require(bits(target[links[:,0]],pre[links[:,1]]),'All298 same-path donor full58 links must match bytes')
    require(bits(held[rows],np.repeat(u[rows,None,:],4,axis=1)),'All300 donor intervals use identical actual control bytes across4 substeps')
    return {'native_full58_link_count':298,'held_actual_control_intervals':300,'full58_links_bitexact':True,'four_substep_controls_bitexact':True}


def expected_recursive_state_inputs(initial,predicted):
    initial=f32(initial,(286,58),'window initial PRE');predicted=f32(predicted,(286,8,58),'all H8 predictions')
    return np.concatenate((initial[:,None,:],predicted[:,:-1,:]),axis=1)


def expected_dynamics_inputs(state_inputs,xi46):
    x=f32(state_inputs,(286,8,58),'actual recurrent state inputs');xi=f32(xi46,(46,),'fixed physical xi46')
    return np.concatenate((x,np.broadcast_to(xi,(286,8,46)),np.zeros((286,8,174),dtype=np.float32)),axis=-1)


def trace_byte_guards(initial,predicted,state_inputs,dynamics_state_inputs,xi46,dynamics_inputs=None):
    """Saved traces must come from actual call arguments, not posthoc synthesis."""
    wanted=expected_recursive_state_inputs(initial,predicted)
    state=f32(state_inputs,(286,8,58),'captured forward state arguments')
    dynstate=f32(dynamics_state_inputs,(286,8,58),'captured dynamics-state slices')
    require(bits(state,wanted) and bits(dynstate,wanted),'Each h must consume previous prediction in x AND dynamics; no truth reset or stale dynamics')
    expected=expected_dynamics_inputs(wanted,xi46)
    if dynamics_inputs is not None:require(bits(f32(dynamics_inputs,(286,8,278),'captured dynamics278'),expected),'Exact fixed xi46/zero174/state recurrence')
    return {'initial_native_PRE_only':True,'previous_predicted_state_each_h':True,'dynamics_state_refreshed_each_h':True,
      'captured_full_dynamics_bytes_checked':dynamics_inputs is not None,'state_conversion_or_model_execution':0}


def raw58_error_statistics(predicted,target,window_path_index):
    p=f32(predicted,(286,8,58),'H8 predicted states');t=f32(target,(286,8,58),'H8 native truth')
    paths=np.asarray(window_path_index)
    require(paths.dtype==np.int64 and paths.shape==(286,) and np.array_equal(paths,np.repeat(np.arange(2,dtype=np.int64),143)),'Full2x143 path population/order')
    error=p.astype(np.float64)-t.astype(np.float64)
    def describe(v,windows,steps):
        a=v.reshape(-1,58);per={'signed_mean':a.mean(0).tolist(),'mean_absolute':abs(a).mean(0).tolist(),
          'RMSE':np.sqrt((a*a).mean(0)).tolist(),'maximum_absolute':abs(a).max(0).tolist()}
        blocks={}
        for name,(lo,hi,unit) in BLOCKS.items():
            b=a[:,lo:hi];sq=b*b;r={'units':unit,'start':lo,'stop_exclusive':hi,'component_count':hi-lo,
              'component_RMSE':float(np.sqrt(sq.mean())),'mean_absolute':float(abs(b).mean()),'maximum_absolute':float(abs(b).max())}
            if name in ('p','v','omega'):r['vector_RMS']=float(np.sqrt(sq.sum(-1).mean()))
            if name=='R_raw9':r['Frobenius_RMS']=float(np.sqrt(sq.sum(-1).mean()))
            blocks[name]=r
        return {'overlapping_windows':windows,'steps_per_window':steps,'dependent_state_comparisons':len(a),'per_feature':per,'blocks':blocks}
    stats={'all_windows_all_steps':describe(error,286,8),
      'per_horizon':{str(h+1):describe(error[:,h],286,1) for h in range(8)},
      'by_path_all_steps':{path:describe(error[paths==pi],143,8) for pi,path in enumerate(PATHS)},
      'by_path_per_horizon':{path:{str(h+1):describe(error[paths==pi,h],143,1) for h in range(8)} for pi,path in enumerate(PATHS)}}
    return {'signed_error286x8x58_F64':error,'absolute_error286x8x58_F64':abs(error),'statistics':stats,
      'scope':'Raw q unwrapped and R9 element differences; no angle/projection/mixed-unit scalar, accuracy gate, independent-N or CI.'}
