"""Saved old152 endpoint-response fitting; stdlib/NumPy, never a model call.

Default is JSON/hash metadata only. Actual saved arrays require a separate Root
protocol and --run. Raw R9 stays a dimensionless matrix, without Log/projection.
"""
from __future__ import annotations
import csv, hashlib, json, sys, time, zipfile
from pathlib import Path

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
SCHEMA='full151105_public_self15_endpoint_response_CPU_protocol:1'
ROLE='old152_finite_endpoint_response_fitting_not_uJac_or_holdout'
PREFIX='full151105_public_self15_endpoint_response_CPU_'
PREP=ROOT/'outputs/full151105_public_self15_endpoint_response_preparation_CPU_v1'
ROLES=('unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency')
PHASES=(0,49,99,149); JOINTS=(0,4,8,12); EPS=(.002,.001)
INPUTS={'parent_protocol','parent_report','parent_saved_review','prepared_inputs','group152_identities',
        'initial_forward152','control_forward152','method_forward152','old_branch_index'}
OWN={'consumer':ROOT/'src/full151_gradient/public_self15_endpoint_response105_v1.py',
     'CLI':ROOT/'scripts/describe_full151_public_self15_endpoint_response_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_public_self15_endpoint_response_cpu_v1.py'}
BLOCKS={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),
        'R_raw9':(43,52,'dimensionless'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}

def require(x,message):
    if not x:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def pin(path):
    path=Path(path);return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def checked(value):
    require(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str
            and type(value['sha256']) is str and len(value['sha256'])==64,'Strict file pin')
    path=Path(value['path']);require(path.is_absolute() and path.is_file(),'Absolute existing pinned file')
    require(pin(path)['sha256']==value['sha256'],'Current SHA mismatch: '+str(path));return path

def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))

def allchecks(r):
    checks=r.get('checks');require(type(checks) is dict and checks and all(type(v) is bool and v for v in checks.values()),'All explicit parent checks')
    require(type(r.get('passed')) is int and type(r.get('total')) is int and r['passed']==r['total']==len(checks),'Actual check count')

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def declaration():
    return {'schema':SCHEMA,'role':ROLE,'input_aliases':sorted(INPUTS),'roles':list(ROLES),
            'old_phases':list(PHASES),'counts':{'branches':152,'baselines':24,'pairs':64,'pair_cohorts':13},
            'default_array_reads':0,'models_FW_AD_MJB_physics':0,'output_prefix':PREFIX}

def validate_protocol(p,authorized=False):
    require(type(p) is dict and p.get('schema')==SCHEMA and p.get('role')==ROLE,'New finite endpoint fitting namespace')
    require(type(p.get('CPU_run_authorized')) is bool and p.get('cpu_threads')==1
            and type(p['cpu_threads']) is int and p.get('output_prefix')==PREFIX,'Explicit Root CPU flag/threads1/prefix')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS,'Exact9 saved input aliases')
    require(type(p.get('source_pins')) is dict and p['source_pins'],'Current source closure')
    if authorized:require(p['CPU_run_authorized'],'Root actual saved-array authorization required')
    return p

def metadata_preflight(protocol_path,output=None):
    p=validate_protocol(read(protocol_path));i=p['inputs']
    for value in [*i.values(),*p['source_pins'].values()]:checked(value)
    for path in OWN.values():require(any(same(pin(path),v) for v in p['source_pins'].values()),'Own source missing from closure')
    parent=read(checked(i['parent_report']));pp=read(checked(i['parent_protocol']));review=read(checked(i['parent_saved_review']))
    require(parent['schema']=='full151105_boundary_public_self15_final_evaluation_CPU_analysis:1'
        and parent['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_PUBLIC_SELF15_AFTER1000_EVAL_NOT_CONFIRMATION','Actual new public15 parent report')
    allchecks(parent);allchecks(review)
    require(review['schema']=='full151105_boundary_public_self15_final_prediction_saved_independent:1'
        and review['status']=='PASS_SAVED_ALL3_ALL152_PUBLIC_SELF15_FIXED_FINAL_PREDICTIONS'
        and review['engineering_qualified'] is True,'Positive current-family saved review')
    require(same(parent['protocol'],i['parent_protocol']) and same(review['protocol'],i['parent_protocol'])
        and same(review['producer_report'],i['parent_report']) and checked(i['parent_report']).parent.samefile(Path(review['run_dir'])),'Same parent run/protocol/review')
    require(pp['inputs'].keys()==parent['input_pins'].keys() and all(same(v,parent['input_pins'][k]) for k,v in pp['inputs'].items()),'Parent exact semantic input map')
    routes={'prepared_inputs':'prepared_inputs_and_context.npz','group152_identities':'group152_identities.json',
        **dict(zip(('initial_forward152','control_forward152','method_forward152'),(r+'_forward152.npz' for r in ROLES)))}
    for alias,name in routes.items():require(same(i[alias],parent['outputs'][name]) and checked(i[alias]).samefile(checked(i['parent_report']).parent/name),'Actual parent immutable output route '+alias)
    target=read(checked(parent['input_pins']['target_report']))
    require(same(i['old_branch_index'],target['input_pins']['response_metadata152']),'Original old152 branch metadata authority')
    # The parent deliberately byte-copies the original identity JSON to its own
    # output directory. These are different physical files, not samefile aliases.
    origin=parent['input_pins']['target_identities']
    require(i['group152_identities']['sha256']==origin['sha256'] and checked(i['group152_identities']).read_bytes()==checked(origin).read_bytes(),'Exact original identity JSON byte-copy authority')
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent.samefile(ROOT/'outputs') and out.name.startswith(PREFIX) and not out.exists(),'Fresh output only')
    require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'Default/consumer must not import model/native frameworks')
    return {'protocol':p,'protocol_pin':pin(protocol_path),'parent':parent,'output':out,'actual_array_reads':0}

def ready_for_run():
    ready=read(PREP/'READY.json');report=read(checked(ready['cpu_report']));allchecks(report)
    require(ready['schema']=='full151105_public_self15_endpoint_response_CPU_READY:1'
        and report['schema']=='full151105_public_self15_endpoint_response_CPU_contract:1'
        and report['status']=='PASS_SINGLE_SYNTHETIC_FINITE_ENDPOINT_RESPONSE_NO_NN','Executed narrow synthetic qualification')
    require(ready['cpu_checks']=={'passed':report['passed'],'total':report['total']} and report['actual_saved_array_reads']==0
        and report['model_FW_AD_physics_calls']==0,'Focused proof scope/counts')
    for key,path in OWN.items():require(same(ready['source_pins'][key],pin(path)) and same(report['source_pins'][key],pin(path)),'Current own source/proof binding')
    return pin(PREP/'READY.json'),ready['cpu_report']

def f32(a,shape,name,np):
    require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'Finite exact FP32 '+name)

def pair_mapping(ids,meta,u,groups,substeps,np):
    require(type(ids) is list and type(meta) is list and len(ids)==len(meta)==152,'All old152 original records')
    require(groups.dtype==substeps.dtype==np.int64 and groups.shape==(152,) and substeps.shape==(152,4),'Exact int64 group/substeps')
    pairs=[]
    for bid,(a,b) in enumerate(zip(ids,meta)):
        s,off=divmod(bid,19);pi,si=divmod(s,4);base=off<3;j=None if base else ((off-3)%8)//2;ei=None if base else (off-3)//8
        group=226*pi+150+19*si+off;path=('A0','A2')[pi];k=PHASES[si];mode='baseline' if base else 'direction'
        expected={'row_id':bid,'snapshot_id':s,'path':path,'mode':mode,'k':k,'branch_id':bid,'group_id':group,'physical_rows':list(range(4*group,4*group+4)),'terminal_observation':k+1}
        require(all(type(a.get(key)) is type(v) and a[key]==v for key,v in expected.items()),'Old target/source identity/order')
        wanted={'row_id':bid,'snapshot_id':s,'path':path,'k':k,'type':mode,'group_id':group,'j':j,'epsilon_index':ei,
            'epsilon_rad':None if base else EPS[ei],'sign':0 if base else (-1,1)[(off-3)%2],
            'baseline_repeat':off if base else None,'joint_index':None if base else JOINTS[j]}
        require(all(type(b.get(key)) is type(v) and b[key]==v for key,v in wanted.items()),'Original EPS-major metadata, not fresh labels')
        require(groups[bid]==group and np.array_equal(substeps[bid],np.arange(4*group,4*group+4)),'Actual prepared array identity join')
        if not base:require(type(b.get('native_actuator_id')) is int and b['native_actuator_id']>=0 and type(b.get('native_joint_id')) is int and b['native_joint_id']>=0,'Original direction native IDs')
        else:require(b.get('native_actuator_id') is None and b.get('native_joint_id') is None,'Baseline has no selected actuator')
    for j in range(4):require(len({(r['native_actuator_id'],r['native_joint_id']) for r in meta if r['j']==j})==1,'Same declared native actuator direction across snapshots/eps')
    for s in range(8):
        for j,q in enumerate(JOINTS):
            for ei,eps in enumerate(EPS):
                mi=s*19+3+8*ei+2*j;pl=mi+1
                require(meta[mi]['native_actuator_id']==meta[pl]['native_actuator_id'] and meta[mi]['native_joint_id']==meta[pl]['native_joint_id'],'Pair native direction identity')
                pairs.append({'pair_id':len(pairs),'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'j':j,'joint_index':q,'epsilon_index':ei,'epsilon_rad':eps,
                    'minus_branch_id':mi,'plus_branch_id':pl,'minus_group_id':int(groups[mi]),'plus_group_id':int(groups[pl]),
                    'native_actuator_id':meta[mi]['native_actuator_id'],'native_joint_id':meta[mi]['native_joint_id']})
    return pairs

def describe(a,np,per_control_rad=False):
    if len(a)==0:return {'reported_rows':0,'perfeature':None,'blocks':None}
    require(a.ndim==2 and a.shape[1]==58 and np.isfinite(a).all(),'Finite selected descriptive58 matrix')
    return {'reported_rows':len(a),'perfeature':{'RMS':np.sqrt(np.mean(a*a,axis=0)).tolist(),'MAE':np.mean(abs(a),axis=0).tolist(),'maxabs':np.max(abs(a),axis=0).tolist()},
        'blocks':{name:{'unit':unit+'/rad' if per_control_rad else unit,'component_RMS':float(np.sqrt(np.mean(a[:,lo:hi]**2))),
            'vector_or_raw9_Frobenius_RMS':float(np.sqrt(np.mean(np.sum(a[:,lo:hi]**2,axis=-1)))),
            'MAE':float(np.mean(abs(a[:,lo:hi]))),'maxabs':float(np.max(abs(a[:,lo:hi])))} for name,(lo,hi,unit) in BLOCKS.items()}}

def analyze(x,u,target,predictions,ids,meta,groups,substeps,np):
    for name,a,shape in [('x',x,(152,58)),('u',u,(152,20)),('target',target,(152,58))]:f32(a,shape,name,np)
    require(set(predictions)==set(ROLES),'All three fixed roles, no model selection')
    for role,a in predictions.items():f32(a,(152,58),role,np)
    rows=pair_mapping(ids,meta,u,groups,substeps,np);mi=np.asarray([r['minus_branch_id'] for r in rows]);pl=mi+1
    du=u[pl].astype(np.float64)-u[mi].astype(np.float64);dx=x[pl].astype(np.float64)-x[mi].astype(np.float64)
    q=np.asarray([r['joint_index'] for r in rows]);den=du[np.arange(64),q];defined=den!=0;nominal=2*np.asarray([r['epsilon_rad'] for r in rows],np.float64)
    physical=target[pl].astype(np.float64)-target[mi].astype(np.float64)
    def divide(a,d,mask):
        result=np.full(a.shape,np.nan,np.float64);np.divide(a,d[:,None],out=result,where=mask[:,None]);return result
    arrays={'minus_branch_id':mi.astype(np.int64),'plus_branch_id':pl.astype(np.int64),'actual_delta_u20_rad_F64':du,'actual_delta_x58_F64':dx,
        'physical_response58_F64':physical,'actual_coordinate_denominator_rad_F64':den,'actual_defined_mask64':defined,'nominal_2epsilon_rad_F64':nominal,
        'physical_actual_secant58_F64':divide(physical,den,defined),'physical_nominal_secant58_F64':physical/nominal[:,None]}
    offaxis=du.copy();offaxis[np.arange(64),q]=0
    for n,r in enumerate(rows):r.update(actual_coordinate_denominator_rad=float(den[n]),actual_defined=bool(defined[n]),
        actual_delta_u20_rad_F64=du[n].tolist(),offaxis_delta_u20_rad_F64=offaxis[n].tolist(),actual_delta_x58_F64=dx[n].tolist(),
        same_visible_x58_numeric=bool(np.array_equal(x[mi[n]],x[pl[n]])),same_visible_x58_bits=bool(np.array_equal(x[mi[n]].view(np.uint32),x[pl[n]].view(np.uint32))),
        clip_scope='Original branch JSON has no clip flag; none inferred or invented')
    cohorts={'ALL':np.ones(64,bool),'A0':np.asarray([r['path']=='A0' for r in rows]),'A2':np.asarray([r['path']=='A2' for r in rows])}
    cohorts.update({'snapshot_'+str(s):np.asarray([r['snapshot_id']==s for r in rows]) for s in range(8)})
    cohorts.update({'epsilon_'+str(eps):np.asarray([r['epsilon_index']==ei for r in rows]) for ei,eps in enumerate(EPS)})
    stats={};baseline=[]
    for role,pred in predictions.items():
        response=pred[pl].astype(np.float64)-pred[mi].astype(np.float64);error=response-physical
        actual=divide(response,den,defined);actual_error=divide(error,den,defined);nominal_response=response/nominal[:,None];nominal_error=error/nominal[:,None]
        for key,value in {'response58_F64':response,'response_error58_F64':error,'actual_secant58_F64':actual,'actual_secant_error58_F64':actual_error,
            'nominal_secant58_F64':nominal_response,'nominal_secant_error58_F64':nominal_error}.items():arrays[role+'__'+key]=value
        stats[role]={name:{'reported_pairs':int(mask.sum()),'actual_defined_pairs':int((mask&defined).sum()),'actual_undefined_pairs':int((mask&~defined).sum()),
            **{key:describe(value[mask&defined] if key.startswith('actual_') else value[mask],np,per_control_rad='secant' in key) for key,value in
                [('response',response),('physical_response',physical),('response_error',error),('actual_secant',actual),('actual_secant_error',actual_error),
                 ('nominal_secant',nominal_response),('nominal_secant_error',nominal_error)]}} for name,mask in cohorts.items()}
    for s in range(8):
        sl=slice(19*s,19*s+3);item={'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'baseline_branch_ids':list(range(19*s,19*s+3)),'descriptors':{}}
        for name,value in {'pre':x,'physical':target,**predictions}.items():
            a=value[sl].astype(np.float64);centered=a-a.mean(0)
            item['descriptors'][name]={'range58':np.ptp(a,axis=0).tolist(),'centered_population_variance58':np.mean(centered**2,axis=0).tolist(),'maxabs_centered58':np.max(abs(centered),axis=0).tolist()}
        item['baseline_u_three_bits_equal']=bool(np.array_equal(u[sl].view(np.uint32),np.broadcast_to(u[19*s],(3,20)).copy().view(np.uint32)));baseline.append(item)
    return arrays,rows,stats,baseline

def run_analysis(protocol_path,output):
    start=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);ready,proof=ready_for_run();out=m['output'];require(out is not None,'Output required');out.mkdir()
    write(out/'protocol.json',p);write(out/'source_inventory.json',p['source_pins']);phase='saved_inputs';counts={'NPZ_file_reads':0,'model_loads_FW_AD_MJB_physics_GPU':0}
    try:
        import numpy as np
        i=p['inputs']
        with np.load(checked(i['prepared_inputs']),allow_pickle=False) as z:
            data={k:z[k].copy() for k in ('x58_FP32','u20_FP32','target58_FP32','source_group_id152','source_substep_rows152x4')};counts['NPZ_file_reads']+=1
        predictions={}
        for role,alias in zip(ROLES,('initial_forward152','control_forward152','method_forward152')):
            with np.load(checked(i[alias]),allow_pickle=False) as z:predictions[role]=z['prediction58_FP32'].copy();counts['NPZ_file_reads']+=1
        arrays,rows,stats,baseline=analyze(data['x58_FP32'],data['u20_FP32'],data['target58_FP32'],predictions,read(checked(i['group152_identities'])),read(checked(i['old_branch_index'])),data['source_group_id152'],data['source_substep_rows152x4'],np)
        np.savez_compressed(out/'consumed_old152_inputs_predictions.npz',**data,**{role+'__prediction58_FP32':a for role,a in predictions.items()})
        np.savez_compressed(out/'all64_endpoint_response_arrays.npz',**arrays);write(out/'all64_pair_records.json',rows);write(out/'all13_pair_cohort_statistics.json',stats);write(out/'baseline_repeat_descriptors8.json',baseline)
        with (out/'all64_pairs.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['pair_id','snapshot_id','path','k','j','joint_index','epsilon_index','epsilon_rad','minus_branch_id','plus_branch_id','actual_coordinate_denominator_rad','actual_defined','same_visible_x58_numeric','same_visible_x58_bits']);w.writeheader();w.writerows({key:r[key] for key in w.fieldnames} for r in rows)
        for value in [*i.values(),*p['source_pins'].values()]:checked(value)
        require(same(pin(protocol_path),m['protocol_pin']) and counts['NPZ_file_reads']==4,'Fixed input identity/read budget')
        require(not any(k in sys.modules for k in ('torch','mujoco','warp')),'No model/native framework')
        with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name,value in p['source_pins'].items():z.write(checked(value),name+'/'+checked(value).name)
        report={'schema':'full151105_public_self15_endpoint_response_saved_analysis:1','status':'COMPLETED_ALL3_OLD64_FINITE_ENDPOINT_RESPONSE_FITTING_NOT_JACOBIAN',
            'protocol':m['protocol_pin'],'input_pins':i,'cpu_ready':ready,'cpu_report':proof,'counts':{**counts,'branches':152,'baselines':24,'snapshots':8,'pairs':64,'pair_cohorts':13,
                'actual_defined_pairs':int(arrays['actual_defined_mask64'].sum()),'actual_undefined_pairs':int((~arrays['actual_defined_mask64']).sum())},
            'summary':stats,'scope':'Old training entries; actual matched endpoints and controls retained. No fixed-hidden-state, pure u Jacobian, accuracy threshold, winner, noise bound, fresh6 stable or heldout claim. R9 raw dimensionless; all undefined rows retained.',
            'whole_hidden_state_qualified':False,'physical_gradient_qualified':False,'Actor_or_task_benefit_qualified':False,'elapsed_seconds':time.perf_counter()-start}
        write(out/'report.json',report);write(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as error:
        write(out/'failure.json',{'phase':phase,'error':repr(error),'counts':counts,'automatic_retry':False});raise
