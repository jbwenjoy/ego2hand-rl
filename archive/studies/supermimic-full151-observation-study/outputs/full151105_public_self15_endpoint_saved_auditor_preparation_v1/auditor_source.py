"""Independent saved endpoint arithmetic. Metadata default; --run is explicit.

No producer module/analyze import, model, checkpoint, Torch, AD or physics.
Arrays use scalar endpoint subtraction; reductions use math.fsum. The fixed
comparison tolerances cover reduction roundoff, not scientific accuracy.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,zipfile
from pathlib import Path

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
SCHEMA='full151105_public_self15_endpoint_response_saved_independent:1'
PROTOCOL_SHA='0d53136b87ec5689d8be1cc1678914e8896debb7c776d60ff7fd2bea108f3529'
PRODUCER_SHA='d6178807bf2fa7e22f26dcd528c2cd5a524c3c2dabb170a034cb57e6ae35df72'
ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
PHASES=(0,49,99,149);COORDINATES=(0,4,8,12);EPS=(.002,.001)
BLOCKS=(('q',0,20,'rad'),('qd',20,40,'rad/s'),('p',40,43,'m'),('R_raw9',43,52,'dimensionless'),('v',52,55,'m/s'),('omega',55,58,'rad/s'))
INPUTS={'parent_protocol','parent_report','parent_saved_review','prepared_inputs','group152_identities','initial_forward152','control_forward152','method_forward152','old_branch_index'}
ATOL=1e-12;RTOL=2e-12
PREP=ROOT/'outputs/full151105_public_self15_endpoint_saved_auditor_preparation_v1'
CHECKER=ROOT/'scripts/check_public_self15_endpoint_saved_auditor_cpu_v1.py'


def need(x,msg):
    if not x:raise ValueError(msg)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def pin(path):
    p=Path(path);need(p.is_absolute(),'Absolute pins and absolute protocol argument required')
    need(p.suffix.lower()!='.pt','This auditor never reads checkpoint bytes')
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(v):
    need(type(v) is dict and set(v)=={'path','sha256'} and pin(v['path'])['sha256']==v['sha256'],'Current absolute source/input pin')
    return Path(v['path'])
def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))
def same_recorded_identity(a,b):
    # The positive parent saved review owns checkpoint-byte qualification.
    # Join its recorded semantic pins by SHA+samefile without opening any.pt.
    return (type(a) is dict and type(b) is dict and set(a)==set(b)=={'path','sha256'} and a['sha256']==b['sha256']
            and Path(a['path']).is_absolute() and Path(b['path']).is_absolute() and Path(a['path']).samefile(Path(b['path'])))
def allchecks(r):need(type(r['checks']) is dict and bool(r['checks']) and all(v is True for v in r['checks'].values()) and r['passed']==r['total']==len(r['checks']),'Positive saved/source proof')


def close_json(actual,expected,label,maxima):
    if type(expected) is dict:
        need(type(actual) is dict and set(actual)==set(expected),'JSON keys '+label)
        for k in expected:close_json(actual[k],expected[k],label+'/'+k,maxima)
    elif type(expected) is list:
        need(type(actual) is list and len(actual)==len(expected),'JSON length '+label)
        for j,(a,b) in enumerate(zip(actual,expected)):close_json(a,b,label+'/'+str(j),maxima)
    elif type(expected) is float:
        need(type(actual) in (float,int) and math.isfinite(actual),'Finite numeric '+label)
        error=abs(float(actual)-expected);maxima['JSON_reduction_maxabs']=max(maxima.get('JSON_reduction_maxabs',0.),error)
        need(error<=ATOL+RTOL*abs(expected),'Reduction arithmetic '+label)
    else:need(type(actual) is type(expected) and actual==expected,'Exact metadata '+label)


def source_mapping(ids,metadata,groups,substeps,np):
    need(type(ids) is list and type(metadata) is list and len(ids)==len(metadata)==152,'Original152 metadata')
    need(groups.dtype==substeps.dtype==np.int64 and groups.shape==(152,) and substeps.shape==(152,4),'Group array ABI')
    directions={};ids_by_j={j:set() for j in range(4)}
    for s in range(8):
        path=('A0','A2')[s//4];phase=PHASES[s%4]
        schedule=[(None,None,0,i) for i in range(3)]
        schedule += [(j,e,sign,None) for e in range(2) for j in range(4) for sign in (-1,1)]
        for offset,(j,e,sign,repeat) in enumerate(schedule):
            bid=19*s+offset;gid=226*(s//4)+150+19*(s%4)+offset;base=j is None
            a=ids[bid];b=metadata[bid]
            expected={'row_id':bid,'snapshot_id':s,'path':path,'mode':'baseline' if base else 'direction','k':phase,'branch_id':bid,'group_id':gid,'physical_rows':[4*gid+h for h in range(4)],'terminal_observation':phase+1}
            for k,v in expected.items():need(type(a.get(k)) is type(v) and a[k]==v,'Identity '+k)
            fields={'row_id':bid,'snapshot_id':s,'path':path,'k':phase,'type':expected['mode'],'group_id':gid,'j':j,'epsilon_index':e,'epsilon_rad':None if base else EPS[e],'sign':sign,'baseline_repeat':repeat,'joint_index':None if base else COORDINATES[j]}
            for k,v in fields.items():need(type(b.get(k)) is type(v) and b[k]==v,'Response metadata '+k)
            need(int(groups[bid])==gid and substeps[bid].tolist()==expected['physical_rows'],'Prepared source identity')
            if base:need(b['native_actuator_id'] is None and b['native_joint_id'] is None,'No selected baseline actuator')
            else:
                need(type(b['native_actuator_id']) is int and b['native_actuator_id']>=0 and type(b['native_joint_id']) is int and b['native_joint_id']>=0,'Native IDs')
                directions[(s,j,e,sign)]=bid;ids_by_j[j].add((b['native_actuator_id'],b['native_joint_id']))
    need(all(len(values)==1 for values in ids_by_j.values()),'One native ID pair per selected coordinate')
    pairs=[]
    for s in range(8):
        for j,q in enumerate(COORDINATES):
            for e,epsilon in enumerate(EPS):
                minus=directions[(s,j,e,-1)];plus=directions[(s,j,e,1)];native=metadata[minus]
                pairs.append({'pair_id':len(pairs),'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'j':j,'joint_index':q,'epsilon_index':e,'epsilon_rad':epsilon,
                    'minus_branch_id':minus,'plus_branch_id':plus,'minus_group_id':int(groups[minus]),'plus_group_id':int(groups[plus]),'native_actuator_id':native['native_actuator_id'],'native_joint_id':native['native_joint_id']})
    return pairs


def statistics(matrix,indices,per_rad=False):
    if not indices:return {'reported_rows':0,'perfeature':None,'blocks':None}
    rows=[[float(v) for v in matrix[i]] for i in indices];n=len(rows)
    def describe(values):return (math.sqrt(math.fsum(v*v for v in values)/len(values)),math.fsum(abs(v) for v in values)/len(values),max(abs(v) for v in values))
    features=[describe([row[j] for row in rows]) for j in range(58)]
    blocks={}
    for name,lo,hi,unit in BLOCKS:
        rms,mae,maxabs=describe([v for row in rows for v in row[lo:hi]])
        norm=math.sqrt(math.fsum(math.fsum(v*v for v in row[lo:hi]) for row in rows)/n)
        blocks[name]={'unit':unit+'/rad' if per_rad else unit,'component_RMS':rms,'vector_or_raw9_Frobenius_RMS':norm,'MAE':mae,'maxabs':maxabs}
    return {'reported_rows':n,'perfeature':{'RMS':[v[0] for v in features],'MAE':[v[1] for v in features],'maxabs':[v[2] for v in features]},'blocks':blocks}


def independent(x,u,target,predictions,ids,meta,groups,substeps,np):
    for a,shape in [(x,(152,58)),(u,(152,20)),(target,(152,58))]+[(predictions[r],(152,58)) for r in ROLES]:
        need(a.dtype==np.float32 and a.shape==shape and bool(np.isfinite(a).all()),'Finite full original FP32 arrays')
    rows=source_mapping(ids,meta,groups,substeps,np);du=[];dx=[];physical=[];den=[];nominal=[];defined=[];diagnostics=[]
    for row in rows:
        mi,pl,q=row['minus_branch_id'],row['plus_branch_id'],row['joint_index']
        control=[float(u[pl,c])-float(u[mi,c]) for c in range(20)]
        state=[float(x[pl,c])-float(x[mi,c]) for c in range(58)]
        truth=[float(target[pl,c])-float(target[mi,c]) for c in range(58)]
        d=control[q];offaxis=list(control);offaxis[q]=0.
        same_numeric=all(float(a)==float(b) for a,b in zip(x[mi],x[pl]));same_bits=x[mi].tobytes()==x[pl].tobytes()
        row.update(actual_coordinate_denominator_rad=d,actual_defined=d!=0,actual_delta_u20_rad_F64=control,offaxis_delta_u20_rad_F64=offaxis,actual_delta_x58_F64=state,
                   same_visible_x58_numeric=same_numeric,same_visible_x58_bits=same_bits,clip_scope='Original branch JSON has no clip flag; none inferred or invented')
        du.append(control);dx.append(state);physical.append(truth);den.append(d);nominal.append(2*row['epsilon_rad']);defined.append(d!=0)
        diagnostics.append({'pair_id':row['pair_id'],'selected_abs_delta_rad':abs(d),'delta_u20_L2_rad':math.sqrt(math.fsum(a*a for a in control)),
                            'offaxis_L2_rad':math.sqrt(math.fsum(a*a for a in offaxis)),'same_visible_x58_numeric':same_numeric,'same_visible_x58_bits':same_bits})
    def secant(values,divisor):return np.asarray([[v/divisor[i] if divisor[i]!=0 else math.nan for v in row] for i,row in enumerate(values)],dtype=np.float64)
    arrays={'minus_branch_id':np.asarray([r['minus_branch_id'] for r in rows],np.int64),'plus_branch_id':np.asarray([r['plus_branch_id'] for r in rows],np.int64),
        'actual_delta_u20_rad_F64':np.asarray(du,np.float64),'actual_delta_x58_F64':np.asarray(dx,np.float64),'physical_response58_F64':np.asarray(physical,np.float64),
        'actual_coordinate_denominator_rad_F64':np.asarray(den,np.float64),'actual_defined_mask64':np.asarray(defined,bool),'nominal_2epsilon_rad_F64':np.asarray(nominal,np.float64),
        'physical_actual_secant58_F64':secant(physical,den),'physical_nominal_secant58_F64':secant(physical,nominal)}
    cohorts={'ALL':list(range(64)),'A0':[i for i,r in enumerate(rows) if r['path']=='A0'],'A2':[i for i,r in enumerate(rows) if r['path']=='A2']}
    for s in range(8):cohorts['snapshot_'+str(s)]=[i for i,r in enumerate(rows) if r['snapshot_id']==s]
    for e,eps in enumerate(EPS):cohorts['epsilon_'+str(eps)]=[i for i,r in enumerate(rows) if r['epsilon_index']==e]
    stats={}
    for role in ROLES:
        pred=predictions[role]
        response=[[float(pred[r['plus_branch_id'],c])-float(pred[r['minus_branch_id'],c]) for c in range(58)] for r in rows]
        error=[[v-t for v,t in zip(row,truth)] for row,truth in zip(response,physical)]
        actual=secant(response,den);actual_error=secant(error,den);nominal_response=secant(response,nominal);nominal_error=secant(error,nominal)
        metrics={'response':response,'physical_response':physical,'response_error':error,'actual_secant':actual,'actual_secant_error':actual_error,'nominal_secant':nominal_response,'nominal_secant_error':nominal_error}
        for key,value in {'response58_F64':response,'response_error58_F64':error,'actual_secant58_F64':actual,'actual_secant_error58_F64':actual_error,'nominal_secant58_F64':nominal_response,'nominal_secant_error58_F64':nominal_error}.items():arrays[role+'__'+key]=np.asarray(value,np.float64)
        stats[role]={}
        for name,indices in cohorts.items():
            valid=[j for j in indices if defined[j]]
            stats[role][name]={'reported_pairs':len(indices),'actual_defined_pairs':len(valid),'actual_undefined_pairs':len(indices)-len(valid),
                **{key:statistics(value,valid if key.startswith('actual_') else indices,per_rad='secant' in key) for key,value in metrics.items()}}
    baseline=[]
    for s in range(8):
        selected=list(range(19*s,19*s+3));descriptors={}
        for name,a in {'pre':x,'physical':target,**predictions}.items():
            ranges=[];variances=[];maxdev=[]
            for c in range(58):
                values=[float(a[b,c]) for b in selected];mean=math.fsum(values)/3;centered=[v-mean for v in values]
                ranges.append(max(values)-min(values));variances.append(math.fsum(v*v for v in centered)/3);maxdev.append(max(abs(v) for v in centered))
            descriptors[name]={'range58':ranges,'centered_population_variance58':variances,'maxabs_centered58':maxdev}
        baseline.append({'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'baseline_branch_ids':selected,'descriptors':descriptors,
                         'baseline_u_three_bits_equal':all(u[b].tobytes()==u[selected[0]].tobytes() for b in selected)})
    return arrays,rows,stats,baseline,diagnostics


def metadata(protocol_path,run_dir):
    protocol_path=Path(protocol_path);run_dir=Path(run_dir)
    need(protocol_path.is_absolute() and run_dir.is_absolute(),'Do not repeat relative-protocol pin failure')
    pp=pin(protocol_path);need(pp['sha256']==PROTOCOL_SHA,'Root fixed successful v2 protocol only')
    p=read(protocol_path);i=p['inputs'];r=read(run_dir/'report.json');allchecks(read(checked(i['parent_saved_review'])))
    need(p['schema']=='full151105_public_self15_endpoint_response_CPU_protocol:1' and p['role']=='old152_finite_endpoint_response_fitting_not_uJac_or_holdout' and set(i)==INPUTS,'Exact successful protocol/input ABI')
    need(r['schema']=='full151105_public_self15_endpoint_response_saved_analysis:1' and r['status']=='COMPLETED_ALL3_OLD64_FINITE_ENDPOINT_RESPONSE_FITTING_NOT_JACOBIAN' and same(r['protocol'],pp),'Only terminal v2 report')
    need(set(r['input_pins'])==set(i) and all(same(v,r['input_pins'][k]) for k,v in i.items()),'One exact input cohort')
    for v in [*i.values(),*p['source_pins'].values()]:checked(v)
    source=ROOT/'src/full151_gradient/public_self15_endpoint_response105_v1.py';need(pin(source)['sha256']==PRODUCER_SHA,'Frozen unchanged d617 science')
    parent=read(checked(i['parent_report']));parent_protocol=read(checked(i['parent_protocol']));review=read(checked(i['parent_saved_review']));allchecks(parent);allchecks(review)
    need(parent['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1' and review['schema']=='full151105_boundary_public_self15_final_prediction_saved_independent:1' and review['engineering_qualified'] is True and same(review['producer_report'],i['parent_report']) and same(review['protocol'],i['parent_protocol']),'New public15 predictions/positive saved12')
    need(same(parent['protocol'],i['parent_protocol']) and set(parent_protocol['inputs'])==set(parent['input_pins']) and all(same_recorded_identity(v,parent['input_pins'][k]) for k,v in parent_protocol['inputs'].items()),'Parent recorded protocol source map; checkpoint bytes delegated, never read here')
    target=read(checked(parent['input_pins']['target_report']))
    need(same(i['old_branch_index'],target['input_pins']['response_metadata152']),'Exact old response metadata alias')
    for alias,name in {'prepared_inputs':'prepared_inputs_and_context.npz','group152_identities':'group152_identities.json',**dict(zip(('initial_forward152','control_forward152','method_forward152'),(s+'_forward152.npz' for s in ROLES)))}.items():need(same(i[alias],parent['outputs'][name]),'Parent immutable input route')
    origin=parent['input_pins']['target_identities'];need(i['group152_identities']['sha256']==origin['sha256'] and checked(i['group152_identities']).read_bytes()==checked(origin).read_bytes(),'Copied identity bytes, not false samefile')
    return p,r


def audit(protocol_path,run_dir,output):
    need(not any(k in sys.modules for k in ('torch','mujoco','warp')),'No model/native framework')
    p,r=metadata(protocol_path,run_dir);run=Path(run_dir);out=Path(output)
    need(out.is_absolute() and not out.exists(),'Fresh absolute audit output');out.mkdir()
    proof=read(PREP/'report.json');allchecks(proof);need(proof['actual_array_reads']==0 and same(proof['source_pins']['auditor'],pin(Path(__file__))) and same(proof['source_pins']['checker'],pin(CHECKER)),'Executed same-source single closedform proof')
    checks={};maxima={};counts={'NPZ_reads':0,'Torch_pt_models_FW_AD_MJB_Env_physics_GPU':0};i=p['inputs']
    try:
        import numpy as np
        outputs=read(run/'output_pins.json')
        for name,v in outputs.items():need(checked(v).parent.samefile(run) and checked(v).name==name,'Output pins route')
        need(same(outputs['report.json'],pin(run/'report.json')) and read(run/'protocol.json')==p and read(run/'source_inventory.json')==p['source_pins'],'Protocol/report/source inventory identity')
        with zipfile.ZipFile(run/'execution_sources.zip') as z:
            expected={name+'/'+Path(v['path']).name:v for name,v in p['source_pins'].items()};need(set(z.namelist())==set(expected),'Exact source archive entries')
            for name,v in expected.items():need(hashlib.sha256(z.read(name)).hexdigest()==v['sha256'],'Archive source bytes')
        checks['v2_absolute_protocol_parent_inputs_outputs_and_source_archive']=True
        with np.load(checked(i['prepared_inputs']),allow_pickle=False) as f:data={k:f[k].copy() for k in ('x58_FP32','u20_FP32','target58_FP32','source_group_id152','source_substep_rows152x4')};counts['NPZ_reads']+=1
        predictions={}
        for role,alias in zip(ROLES,('initial_forward152','control_forward152','method_forward152')):
            with np.load(checked(i[alias]),allow_pickle=False) as f:predictions[role]=f['prediction58_FP32'].copy();counts['NPZ_reads']+=1
        arrays,rows,stats,baseline,diagnostics=independent(data['x58_FP32'],data['u20_FP32'],data['target58_FP32'],predictions,read(checked(i['group152_identities'])),read(checked(i['old_branch_index'])),data['source_group_id152'],data['source_substep_rows152x4'],np)
        checks['all152_oldphase_nativeID_and_all64_ordered_endpoint_joins']=True
        with np.load(run/'consumed_old152_inputs_predictions.npz',allow_pickle=False) as f:
            wanted={**data,**{role+'__prediction58_FP32':v for role,v in predictions.items()}};need(set(f.files)==set(wanted),'Complete copied input arrays')
            for k,v in wanted.items():need(f[k].dtype==v.dtype and f[k].shape==v.shape and f[k].tobytes()==v.tobytes(),'Consumed source bytes '+k)
            counts['NPZ_reads']+=1
        checks['all_consumed_FP32_inputs_predictions_and_int64_identity_arrays_byte_exact']=True
        with np.load(run/'all64_endpoint_response_arrays.npz',allow_pickle=False) as f:
            need(set(f.files)==set(arrays),'All endpoint array keys')
            for k,v in arrays.items():need(f[k].dtype==v.dtype and f[k].shape==v.shape and np.array_equal(f[k],v,equal_nan=True),'Scalar difference/secant/mask '+k)
            counts['NPZ_reads']+=1
        checks['scalar_all64_response_error_control_state_deltas_two_secants_and_masks_exact']=True
        close_json(read(run/'all64_pair_records.json'),rows,'pair_records',maxima)
        checks['all64_pair_metadata_offaxis_dx_and_numeric_bit_drift']=True
        close_json(read(run/'all13_pair_cohort_statistics.json'),stats,'all39_stats',maxima)
        checks['all3_by13_full_unit_statistics_independent_fsum_reductions']=True
        close_json(read(run/'baseline_repeat_descriptors8.json'),baseline,'all8_baselines',maxima)
        checks['all8_triple_baseline_ranges_population_variance_and_control_bits']=True
        with (run/'all64_pairs.csv').open(newline='',encoding='utf-8') as f:
            csvrows=list(csv.DictReader(f));fields=list(csvrows[0])
        wanted_fields=['pair_id','snapshot_id','path','k','j','joint_index','epsilon_index','epsilon_rad','minus_branch_id','plus_branch_id','actual_coordinate_denominator_rad','actual_defined','same_visible_x58_numeric','same_visible_x58_bits']
        need(fields==wanted_fields and len(csvrows)==64,'All CSV64 rows/fields')
        for saved,row in zip(csvrows,rows):need(all(saved[k]==str(row[k]) for k in fields),'CSV mechanical values')
        checks['all64_CSV_rows_columns_and_raw_denominators']=True
        expected_counts={'NPZ_file_reads':4,'model_loads_FW_AD_MJB_physics_GPU':0,'branches':152,'baselines':24,'snapshots':8,'pairs':64,'pair_cohorts':13,'actual_defined_pairs':sum(row['actual_defined'] for row in rows),'actual_undefined_pairs':sum(not row['actual_defined'] for row in rows)}
        need(r['counts']==expected_counts and r['whole_hidden_state_qualified'] is False and r['physical_gradient_qualified'] is False and r['Actor_or_task_benefit_qualified'] is False,'Exact counts and qualification ceiling')
        close_json(r['summary'],stats,'report_summary',maxima)
        checks['terminal_report_counts_all_rows_and_no_physical_or_task_upgrade']=True
        failed=ROOT/'outputs/full151105_public_self15_endpoint_response_CPU_v1';failure=read(failed/'failure.json')
        need(failure['counts']=={'NPZ_file_reads':4,'model_loads_FW_AD_MJB_physics_GPU':0} and not (failed/'report.json').exists(),'Preserve original incomplete v1 and its four reads')
        checks['relative_protocol_v1_failure_preserved_counts4_not_formal_completion']=True
        need(counts=={'NPZ_reads':6,'Torch_pt_models_FW_AD_MJB_Env_physics_GPU':0} and not any(k in sys.modules for k in ('torch','mujoco','warp')),'Narrow independent saved budget')
        np.savez_compressed(out/'independent_all64_endpoint_arrays.npz',**arrays)
        write(out/'independent_all39_statistics.json',stats);write(out/'independent_baseline8_descriptors.json',baseline);write(out/'independent_control_norm_descriptors64.json',diagnostics)
        report={'schema':SCHEMA,'status':'PASS_SAVED_ALL3_ALL64_MATCHED_ENDPOINT_RESPONSE_NOT_JACOBIAN','engineering_qualified':True,'checks':checks,'passed':len(checks),'total':len(checks),
            'protocol':pin(protocol_path),'producer_report':pin(run/'report.json'),'run_dir':str(run),'source_auditor':pin(Path(__file__)),'CPU_proof':pin(PREP/'report.json'),'input_pins':i,
            'counts':counts,'producer_counts':r['counts'],'comparison':{'raw_arrays':'Exact numeric equality including identical NaN masks; input arrays byte-exact','fsum_reduction_rtol':RTOL,'fsum_reduction_atol':ATOL,**maxima},
            'preserved_failed_v1':pin(failed/'failure.json'),'source_pins':{k:pin(v) for k,v in {'auditor':Path(__file__),'checker':CHECKER}.items()},
            'scope':'Known old152/64 matched training endpoints only. Selected du_j is not full-control norm; offaxis and actual PRE-state drift retained. Defined-only secant statistics declare their denominator while raw undefined pairs remain. Old0/49/99/149 does not inherit fresh24/74/124/148 or old9/fresh6 stable labels. R9 is raw dimensionless, no Log/projection, scientific accuracy gate, winner or CI.',
            'extra_control_norm_descriptors':'All64 L2 du20/offaxis and selected absolute delta are audit descriptors, not derivative qualification.',
            'physical_gradient_qualified':False,'pure_u_Jacobian_qualified':False,'heldout_or_generalization_qualified':False,'whole_hidden_state_qualified':False,'Actor_or_task_benefit_qualified':False}
        write(out/'report.json',report);write(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()})
        print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(out/'report.json')}));return report
    except BaseException as error:
        write(out/'failure.json',{'error':repr(error),'checks_completed':checks,'counts':counts,'automatic_retry':False,'source_auditor':pin(Path(__file__))});raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path);ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.protocol is None:print(json.dumps({'schema':SCHEMA,'default':'SOURCE_METADATA_ONLY','NPZ_Torch_models_FW_AD_physics':0}));return
    if a.run:
        need(a.run_dir is not None and a.output is not None,'--run requires absolute protocol/run-dir and fresh output');audit(a.protocol,a.run_dir,a.output)
    else:
        need(a.run_dir is not None,'--protocol requires absolute --run-dir');metadata(a.protocol,a.run_dir);print(json.dumps({'status':'PASS_ABSOLUTE_V2_METADATA_ONLY','actual_array_reads':0}))


if __name__=='__main__':main()
