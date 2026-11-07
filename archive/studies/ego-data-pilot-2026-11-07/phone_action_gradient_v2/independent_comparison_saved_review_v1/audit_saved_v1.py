"""One saved-only scalar-arithmetic review; no producer import or NN/native calls."""
from pathlib import Path
import collections, hashlib, json, math, sys
import numpy as np

P = Path(__file__).resolve().parent.parent
O = Path(__file__).resolve().parent
BLOCKS = {'q20': (0,20), 'qd20': (20,40), 'COM3': (40,43),
          'world_rotation3': (43,46), 'COM_velocity3': (46,49),
          'world_omega3': (49,52), 'coverage2': (52,54),
          'log_min_force2': (54,56), 'log_endpoint_force2': (56,58),
          'log_nonpad_force1': (58,59)}
METRICS = ['prediction_L2','native_or_reference_L2','error_L2','RMSE','relative_L2','cosine']
ATOL, RTOL = 2e-12, 2e-12  # Independent scalar/f-sum reduction rounding only.
checks, errors, deviations = {}, [], collections.defaultdict(lambda: [0.0, 0.0, 0])

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p = Path(p).resolve(strict=True); b = p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    if pin(r['path']) != r: raise ValueError('Pin mismatch: '+r['path'])
    return Path(r['path'])
def arrays(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def bit_equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def check(name, value, detail=None):
    checks[name] = bool(value)
    if not value: errors.append(dict(check=name,detail=detail))
def near(x,y,tag):
    if x is None or y is None:
        if x is not None or y is not None: errors.append(dict(location=tag,expected=x,actual=y))
        return
    x,y=float(x),float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        errors.append(dict(location=tag,error='nonfinite')); return
    ae=abs(x-y); re=ae/max(abs(x),abs(y),1e-300)
    d=deviations[tag.split('/')[0]];d[0]=max(d[0],ae);d[1]=max(d[1],re);d[2]+=1
    if ae>ATOL+RTOL*max(abs(x),abs(y)):
        errors.append(dict(location=tag,expected=x,actual=y,absolute_error=ae))
def norm(v): return math.sqrt(math.fsum(float(t)*float(t) for t in v))
def metric(pred,ref):
    p,a=list(map(float,pred)),list(map(float,ref))
    if not p or not all(map(math.isfinite,p+a)): return None
    pn,an=norm(p),norm(a);e=[x-y for x,y in zip(p,a)];en=norm(e)
    return dict(features=len(p),prediction_L2=pn,native_or_reference_L2=an,
                error_L2=en,RMSE=math.sqrt(math.fsum(t*t for t in e)/len(e)),
                relative_L2=None if an==0 else en/an,
                cosine=None if an==0 or pn==0 else math.fsum(x*y for x,y in zip(p,a))/(pn*an))
def quantile(v,q):
    v=sorted(v);x=(len(v)-1)*q;i=int(math.floor(x));f=x-i
    return v[i] if i==len(v)-1 else (1-f)*v[i]+f*v[i+1]
def agg(v):
    v=[float(t) for t in v if t is not None and math.isfinite(t)]
    return dict(defined_count=len(v),median=None if not v else quantile(v,.5),
                mean=None if not v else math.fsum(v)/len(v),p90=None if not v else quantile(v,.9))

def main():
    report_path=P/'comparison_completed_v1/report.json'
    r=read(report_path);protocol=read(checked(r['protocol']))
    for q in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:checked(q)
    a=arrays(checked(r['array_output']));rows=read(checked(r['side_metrics']))
    selection=read(protocol['inputs']['selection']['path']);op=arrays(protocol['inputs']['operands']['path'])
    native_report=read(protocol['inputs']['native_report']['path']);records=read(checked(native_report['records']))
    terminal_path=P/'process_comparator_run1/terminal.json';terminal=read(terminal_path)
    for key in ['source','runner','protocol','stdout','stderr']:checked(terminal[key])
    counts=dict(checkpoint_loads=1,model_constructors=1,NN_forward=500,
                action_AD_jacobians=12,NN_parameter_updates=0,native_steps=0,TEST_arrays_read=0)
    check('current_input_source_output_and_terminal_pins',
          terminal['returncode']==0 and terminal['stderr']['bytes']==0 and
          terminal['protocol']==r['protocol'] and terminal['runner']==protocol['source_pins']['runner'])
    check('counts_and_fixed5000_weight_identity_SOURCE_RUNTIME',r['counts']==counts==protocol['counts_budget'] and
          protocol['checkpoint_iteration']==5000 and protocol['parameter_updates_authorized'] is False and
          protocol['inputs']['checkpoint']['sha256']=='f2e51432377bcd68b235b0f5dbeb6953a3a411f586e008e0ee4a08111a3b1f1f')
    expected_shapes={k:(476,59) for k in ['native_secant_F64','model_secant_F64_denominator',
        'model_secant_FP32_denominator','model_AD_column','model_side_observable59','native_side_observable59']}
    expected_shapes.update({k:(476,) for k in ['du64','du32','state_id','joint','branch_id']})
    expected_shapes.update(model_baseline59=(12,59),native_baseline59=(12,59),
        model_action_AD12x59x20=(12,59,20),TRAIN_std59=(59,),TRAIN_nonconstant59=(59,))
    check('all_saved_array_shapes_dtypes_finite',set(a)==set(expected_shapes) and all(a[k].shape==v for k,v in expected_shapes.items()) and
          all(np.isfinite(v).all() for v in a.values()) and all(a[k].dtype==np.int64 for k in ['state_id','joint','branch_id']) and
          a['TRAIN_nonconstant59'].dtype==np.bool_ and all(v.dtype==np.float64 for k,v in a.items() if k not in ['state_id','joint','branch_id','TRAIN_nonconstant59']))
    check('TRAIN_nonzero_mask_and_frozen_normalizer_operands',
          bit_equal(a['TRAIN_std59'],op['TRAIN_observable59_population_std_F64']) and
          bit_equal(a['TRAIN_nonconstant59'],op['TRAIN_observable59_nonconstant_bool']) and
          np.array_equal(a['TRAIN_nonconstant59'],a['TRAIN_std59']>0) and np.all(a['TRAIN_std59']>=0))
    check('model_capture_is_exactly_FP32_representable',all(bit_equal(a[k],a[k].astype(np.float32).astype(np.float64))
          for k in ['model_side_observable59','model_baseline59','model_action_AD12x59x20']))
    states={s['state_id']:s for s in selection['states']};byid={v['branch_id']:v for v in records}
    bases={v['state_id']:v for v in records if v['kind']=='baseline'}
    expected_order=[v['branch_id'] for sid in range(12) for v in records if v['state_id']==sid and v['kind']=='finite_action_side']
    check('all12states_476sides_and_original_order_no_dropped_available_sides',
          list(states)==list(range(12)) and len(bases)==12 and len(rows)==476 and len(records)==488 and
          [v['branch_id'] for v in rows]==expected_order==a['branch_id'].tolist() and len(set(expected_order))==476 and
          sum(p['available'] for s in states.values() for p in s['perturbations'])==476 and
          sum(not p['available'] for s in states.values() for p in s['perturbations'])==4)
    native_bases=[];native_pin_count=0
    for sid in range(12):
        b=bases[sid];v=arrays(checked(b['output']));native_pin_count+=1;native_bases.append(v['observable59'])
        if not bit_equal(v['observable59'],a['native_baseline59'][sid]) or not bit_equal(v['observable59'],op['baseline_observable59_F64'][sid]):
            errors.append(dict(location='native_baseline',state_id=sid))
    e_ns=[];e_ms64=[];e_ms32=[];e_ad=[];independent_rows=[];metadata_ok=True
    for i,row in enumerate(rows):
        sid=int(a['state_id'][i]);j=int(a['joint'][i]);bid=int(a['branch_id'][i]);state=states[sid];rec=byid[bid];plan=rec['perturbation']
        native=arrays(checked(rec['output']));native_pin_count+=1
        u0=np.asarray(state['baseline_ctrl20_F64'],dtype=np.float64);u=np.asarray(plan['actual_ctrl20_F64'],dtype=np.float64)
        du=float(u[j])-float(u0[j]);du32=float(np.float32(u[j]))-float(np.float32(u0[j]))
        met=(rec['state_id']==sid and plan['direction_joint']==j and plan in state['perturbations'] and plan['available'] is True and
             rec['usable_observable59'] is True and rec['complete_H20ms'] is True and rec['finite'] is True and
             rec['source_receipt']['sha256']==state['source_before_action_receipt']['sha256'] and
             bit_equal(np.delete(u,j),np.delete(u0,j)) and bit_equal(u,native['actual_control20']) and
             du==plan['actual_signed_du_F64_rad']==row['du_F64']==a['du64'][i] and
             du32==plan['actual_signed_du_FP32_rad']==row['du_FP32']==a['du32'][i] and du!=0 and du32!=0 and
             row['joint']==j and row['state_id']==sid and row['family_id']==state['family_id'] and
             row['dt_s']==state['native_dt_s'] and row['interval']==state['interval_index'] and
             row['source_branch']==state['source_branch'] and row['source_native_output']==rec['output'] and
             row['side']==plan['side'] and row['cohort']==('primary_original8' if j<8 else 'secondary_added12') and
             row['FP32_step_ratio']==du32/du and bit_equal(native['observable59'],a['native_side_observable59'][i]))
        metadata_ok=metadata_ok and met
        ns=np.asarray([(float(v)-float(b))/du for v,b in zip(native['observable59'],native_bases[sid])])
        pred=a['model_side_observable59'][i];mb=a['model_baseline59'][sid]
        ms64=np.asarray([(float(v)-float(b))/du for v,b in zip(pred,mb)])
        ms32=np.asarray([(float(v)-float(b))/du32 for v,b in zip(pred,mb)])
        ad=a['model_action_AD12x59x20'][sid,:,j]
        e_ns.append(ns);e_ms64.append(ms64);e_ms32.append(ms32);e_ad.append(ad)
        blocks={}
        for name,(lo,hi) in BLOCKS.items():
            ids=[k for k in range(lo,hi) if a['TRAIN_nonconstant59'][k]];std=a['TRAIN_std59']
            relations=dict(native_vs_model_FD=metric(ms64[lo:hi],ns[lo:hi]),
                           model_AD_vs_own_FD=metric(ad[lo:hi],ms32[lo:hi]),
                           normalized_native_vs_model_FD=metric([ms64[k]/std[k] for k in ids],[ns[k]/std[k] for k in ids]),
                           normalized_model_AD_vs_own_FD=metric([ad[k]/std[k] for k in ids],[ms32[k]/std[k] for k in ids]))
            blocks[name]=relations
            for relation,value in relations.items():
                saved=row['blocks'][name][relation]
                if value is None or saved is None:
                    if value!=saved:errors.append(dict(location=f'side/{i}/{name}/{relation}',error='undefined mismatch'))
                else:
                    if value['features']!=saved['features']:errors.append(dict(location='features',row=i))
                    for m in METRICS:near(value[m],saved[m],f'side/{i}/{name}/{relation}/{m}')
        independent_rows.append(dict(cohort=row['cohort'],blocks=blocks))
    check('all_native_output_pins_and_control_context_join',metadata_ok and native_pin_count==488)
    for k,v in [('native_secant_F64',e_ns),('model_secant_F64_denominator',e_ms64),
                ('model_secant_FP32_denominator',e_ms32),('model_AD_column',e_ad)]:
        check('exact_'+k,bit_equal(a[k],np.asarray(v)))
    side_errors=sum(1 for e in errors if str(e.get('location','')).startswith(('side/','features')))
    check('all476_side_block_metrics_scalar_fsum_and_undefined_masks',side_errors==0)
    summaries={}
    summary_error_before=len(errors)
    for cohort in ['primary_original8','secondary_added12','all20']:
        subset=independent_rows if cohort=='all20' else [v for v in independent_rows if v['cohort']==cohort]
        summary=dict(sides=len(subset),blocks={})
        if r['summaries'][cohort]['sides']!=len(subset):errors.append(dict(location='summary_count',cohort=cohort))
        for block in BLOCKS:
            summary['blocks'][block]={}
            for relation in ['native_vs_model_FD','model_AD_vs_own_FD','normalized_native_vs_model_FD','normalized_model_AD_vs_own_FD']:
                values=[v['blocks'][block][relation] for v in subset if v['blocks'][block][relation] is not None]
                dest={}
                for m in METRICS:
                    stat=agg([v[m] for v in values]);dest[m]=stat;saved=r['summaries'][cohort]['blocks'][block][relation][m]
                    if stat['defined_count']!=saved['defined_count']:errors.append(dict(location='summary_defined_count',cohort=cohort,block=block,metric=m))
                    for q in ['mean','median','p90']:near(stat[q],saved[q],f'summary/{cohort}/{block}/{relation}/{m}/{q}')
                summary['blocks'][block][relation]=dest
        summaries[cohort]=summary
    check('all3cohorts_summaries_and_defined_counts_scalar_fsum',len(errors)==summary_error_before)
    pidx=np.nonzero(a['joint']<8)[0];sidx=np.nonzero(a['joint']>=8)[0]
    pzero=[i for i in pidx if all(float(v)==0 for v in a['native_secant_F64'][i,40:43])]
    szero=[i for i in sidx if all(float(v)==0 for v in a['native_secant_F64'][i,40:43])]
    key_findings=dict(primary_original8_sides=len(pidx),secondary_added12_sides=len(sidx),
        primary_COM_exact_zero=len(pzero),primary_COM_nonzero=len(pidx)-len(pzero),
        primary_COM_nonzero_cosine_median=summaries['primary_original8']['blocks']['COM3']['native_vs_model_FD']['cosine']['median'],
        primary_COM_nonzero_relative_L2_median=summaries['primary_original8']['blocks']['COM3']['native_vs_model_FD']['relative_L2']['median'],
        secondary_COM_exact_zero=len(szero),secondary_COM_nonzero=len(sidx)-len(szero),
        secondary_all_model_COM_L2_median_m_per_rad=quantile([norm(a['model_secant_F64_denominator'][i,40:43]) for i in sidx],.5),
        secondary_native_zero_model_COM_L2_median_m_per_rad=quantile([norm(a['model_secant_F64_denominator'][i,40:43]) for i in szero],.5),
        secondary_COM_relative_defined_count=summaries['secondary_added12']['blocks']['COM3']['native_vs_model_FD']['relative_L2']['defined_count'],
        secondary_COM_cosine_defined_count=summaries['secondary_added12']['blocks']['COM3']['native_vs_model_FD']['cosine']['defined_count'])
    check('188_288_COM_masks_and_nulls_retained',len(pidx)==188 and len(sidx)==288 and len(pzero)==78 and len(szero)==233 and
          all(rows[i]['blocks']['COM3']['native_vs_model_FD']['relative_L2'] is None and rows[i]['blocks']['COM3']['native_vs_model_FD']['cosine'] is None for i in pzero+szero))
    check('scope_not_analytic_physical_gradient_and_order_deviation_preserved',r['accuracy_or_physics_Jacobian_qualified'] is False and
          protocol['native_strict_protocol_order_conformance'] is False and protocol['native_context_data_qualified'] is True and
          bool(r['native_protocol_order_deviation']))
    passed=not errors and all(checks.values())
    output=dict(schema='phone_full20_WM_native_comparison_saved_independent:1',status='PASS_SAVED_FINITE_RESPONSE_ARITHMETIC_AND_MASKS' if passed else 'FAILED_SAVED_COMPARISON_REVIEW',
        engineering_qualified=passed,passed=sum(checks.values()),total=len(checks),checks=checks,
        producer_report=pin(report_path),protocol=r['protocol'],source_auditor=pin(__file__),
        inputs=dict(array_output=r['array_output'],side_metrics=r['side_metrics'],operands=protocol['inputs']['operands'],
                    selection=protocol['inputs']['selection'],native_report=protocol['inputs']['native_report'],
                    native_records=native_report['records'],native_saved_review=protocol['inputs']['native_saved_review'],
                    checkpoint=protocol['inputs']['checkpoint'],terminal=pin(terminal_path)),
        native_outputs_checked=488,actual_producer_counts=r['counts'],reviewer_counts=dict(NN_loads=0,NN_constructors=0,NN_forward=0,AD=0,native_calls=0,parameter_updates=0),
        key_findings=key_findings,reduction_comparison=dict(absolute_tolerance=ATOL,relative_tolerance=RTOL,
            reason='Independent scalar fsum versus NumPy BLAS/reduction rounding, not a scientific accuracy gate',
            deviations={k:dict(max_absolute=v[0],max_relative=v[1],comparisons=v[2]) for k,v in deviations.items()}),
        errors=errors,scope=['Independent arithmetic uses saved model baseline/side outputs and saved AD matrix; it does not independently validate the NN forward or autodiff values.',
            'AD column indexing and model self-FD agreement are internal model diagnostics, not independent physical-gradient truth.',
            'Native response authority is the original saved 488 branches and their restore review; no new native call or quaternion/log recomputation.',
            'TRAINstd values are byte-bound to frozen operands; this review does not recompute TRAIN data normalizers.',
            'Only per-unit raw blocks and separately TRAIN-scaled nonzero dimensions; no mixed-unit raw global norm.',
            '476 directions share 12 states, VAL families and lineages; not independent episodes or an infinitesimal physical Jacobian.',
            'Counts/CPU/parameter-freeze phase rely on source, runtime assertions and terminal record; fixed checkpoint bytes were hashed without loading it.',
            'Per-state baseline-then-sides order deviation remains explicit: data qualified, strict prose-order conformance false.'])
    with (O/'report.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(O/'report.json'),passed=output['passed'],total=output['total'],status=output['status'],findings=key_findings,errors=errors[:5])))
    return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
