"""Thin loaded-contact adapter; frozen dynamicv3 loop/gates are not rewritten.

Default JSON/hash metadata only. Four measured controls, both original dt,
no material-anchor inverse-PD or preload FW is called by this adapter.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,shutil,struct,time
H=Path(__file__).resolve().parent;P=H.parent
FROZEN=P/'functional_tripod_dynamics_v1/run_functional_tripod_dynamics_v3.py'
FROZEN_SHA='d433a51df307bf45b40e70bc2c74776654eb2ece65bd2de67ae8163c201fd960'
SCHEMA='functional_tripod_loaded_contact_dynamic_protocol:1'
RESULT_SCHEMA='functional_tripod_loaded_contact_dynamic_result:1'
ROUTES=['ctrl_equal_q','actual_contact_normal_1N','actual_contact_normal_3N','GN_best_final_fresh']
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    if pin(r['path'])!=r:raise ValueError('Changed pin '+r['path'])
    return Path(r['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def raw(value):
    return b''.join(raw(x) for x in value) if isinstance(value,list) else struct.pack('<d',float(value))
def physics():
    if pin(FROZEN)['sha256']!=FROZEN_SHA:raise ValueError('Frozen physics authority changed')
    s=importlib.util.spec_from_file_location('loaded_tripod_frozen_dynamicv3',FROZEN);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def metadata_preflight(path):
    p=read(path);f=physics()
    if p['schema']!=SCHEMA or type(p['CPU_run_authorized']) is not bool or p['integration_training_GPU_other_than_declared_CPU_steps_authorized'] is not False:raise ValueError('Explicit new CPU boundary')
    if (p['dt_s']!=f.DTS or p['duration_s']!=f.DURATION or p['gates']!=f.GATES or p['old_two_pad_gates']!=f.OLD_GATES or p['dt_pair_gates']!=f.PAIR_GATES or p['routes']!=ROUTES):raise ValueError('Original frozen gates/dt and four routes')
    if p['budget']!=dict(candidates=1,control_routes=4,dt_cases=2,maximum_cases=8,maximum_integration_steps=76800,preload_preparation_forward=0,maximum_initial_forward=8):raise ValueError('Exact finite budget')
    if set(p['inputs'])!={'model_mjb','model_binding','preload_report','preload_protocol','preload_saved_review'}:raise ValueError('Exact five inputs')
    for v in [*p['inputs'].values(),*p['source_pins'].values()]:checked(v)
    if p['source_pins']['runner']!=pin(__file__) or p['source_pins']['frozen_dynamicv3']!=pin(FROZEN):raise ValueError('Own/frozen source')
    out=Path(p['output'])
    if not out.is_absolute() or out.parent.resolve()!=H.resolve() or out.exists():raise ValueError('Fresh own output')
    binding=read(p['inputs']['model_binding']['path']);report=read(p['inputs']['preload_report']['path']);pre=read(p['inputs']['preload_protocol']['path']);review=read(p['inputs']['preload_saved_review']['path'])
    if binding['model']!=p['inputs']['model_mjb'] or pre['inputs']['model_mjb']!=binding['model'] or pre['inputs']['model_binding']!=p['inputs']['model_binding'] or report['protocol']!=p['inputs']['preload_protocol']:raise ValueError('Original model/source cohort')
    if report['schema']!='functional_tripod_native_preload_result:1' or report['status']!='COMPLETED_BOUNDED_ACTUAL_NATIVE_PRELOAD_NOT_HOLDING' or len(report['candidates'])!=1:raise ValueError('Completed own native-preload family')
    if (review['schema']!='functional_tripod_native_preload_saved_independent:1' or review['status']!='PASS_SAVED_ACTUAL_CONTACT_LOAD_ARITHMETIC_NOT_HOLDING'
        or review['passed']!=review['total'] or review['total']<=0 or len(review['checks'])!=review['total'] or not all(v is True for v in review['checks'].values()) or review['producer_report']!=p['inputs']['preload_report']):raise ValueError('Positive saved arithmetic review')
    c=report['candidates'][0];candidate=read(checked(c['candidate']));checked(c['state_npz'])
    if candidate['geometry_and_compression_target_pass'] is not True or candidate['geometry']['passed'] is not True or candidate['geometry']['unknown']!=[]:raise ValueError('Compressed full geometry')
    if [v['route'] for v in c['routes']]!=ROUTES[:3] or not all(v['available'] is True for v in c['routes']):raise ValueError('All3 original route controls retained')
    if c['native_loaded_entry'] is not True or not c['fresh_repeat_bit_equal'] or not all(v is True for v in c['fresh_repeat_bit_equal'].values()) or len(c['final_repeat_probes'])!=2:raise ValueError('Actual GN final measured loading/repeat')
    for rp in c['final_repeat_probes']:checked(rp)
    refs=[v['probe'] for v in c['routes']]+[c['final_repeat_probes'][0]];variants=[]
    for route,ref in zip(ROUTES,refs):
        probe=read(checked(ref));q=probe['raw']['qpos27'];u=probe['raw']['ctrl20'];pd=probe['PD']
        if (probe['schema']!='functional_tripod_native_preload_probe:1' or probe['candidate_id']!=c['candidate_id'] or raw(q)!=raw(candidate['qpos27_F64']) or any(probe['raw']['qvel26'])
            or not f.finite_vector(u,20) or not f.finite_vector(q,27)):raise ValueError('Exact measured probe state/control')
        if any(x<lo or x>hi for x,lo,hi in zip(u,pd['control_box_lower20'],pd['control_box_upper20'])):raise ValueError('No silent command repair')
        if not probe['measurements']['original_no_unwanted_penetration_control_bounds']:raise ValueError('Measured original unwanted-force/penetration/control bounds')
        variants.append(dict(name=route,candidate_id=c['candidate_id'],route=route,qpos27=q,ctrl20=u,
            ctrl_margin20x2=pd['ctrl_margin20x2'],candidate=c['candidate'],state_npz=c['state_npz'],source_native_probe=ref,
            source_native_initial_pad_normal_N3=probe['pad_normal_N3'],source_native_pad_wrench_world6=probe['pad_only_wrench_world6'],
            source_native_target_wrench_world6=probe['target_wrench_world6'],source_native_wrench_error_world6=probe['pad_wrench_error_world6'],
            source_native_PD_bounds_and_actual_margins=pd,source_native_loaded_entry=probe['measurements']['native_loaded_entry'],
            rule='Use raw measured u directly; no material-A or further inverse-PD preparation. Baseline loading flag does not filter any of four declared dynamic controls.'))
    return f,p,out,binding,variants
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    f,p,out,binding,variants=metadata_preflight(a.protocol)
    if not a.run:
        print(json.dumps(dict(status='METADATA_VALID_NO_BACKEND_MODEL_OR_STEPS',routes=[v['route'] for v in variants],cases=8,maximum_steps=76800,new_preload_forward=0)));return
    if p['CPU_run_authorized'] is not True:raise ValueError('Root explicit actual authorization required')
    out.mkdir();write(out/'protocol.json',p);shutil.copyfile(__file__,out/'execution_source.py');start=time.perf_counter();phase='backend'
    counts=dict(MJB_reads=0,MjData=0,preload_preparation_forward=0,preload_point_jacobians=0,initial_forward=0,integration_steps=0,contact_force_reads=0,neural_training_GPU=0);cases=[]
    try:
        import mujoco
        import numpy as np
        # One shared pinned raw state, checked before model construction.
        with np.load(checked(variants[0]['state_npz']),allow_pickle=False) as z:
            if z['qpos27_F64'].dtype!=np.float64 or z['qpos27_F64'].tobytes()!=np.asarray(variants[0]['qpos27'],dtype=np.float64).tobytes() or np.any(z['qvel26_F64']):raise ValueError('Actual compressed NPZ state bytes')
        m=mujoco.MjModel.from_binary_path(str(checked(p['inputs']['model_mjb'])));counts['MJB_reads']+=1
        phone,pg,table,pads=f.assert_model(m,mujoco,np,binding);before=f.model_digest_except_dt(m,np)
        endpoints={v['name']:[] for v in variants}
        for v in variants:
            for dt in f.DTS:
                phase=v['name']+'__dt'+str(round(dt*1e9))+'ns';d=out/phase;d.mkdir()
                # The original function owns exactly its initialFW and tick loop.
                summary,endpoint=f.evaluate_case(m,v,dt,d,phone,pg,table,pads,mujoco,np,counts)
                cases.append(summary);endpoints[v['name']].append(endpoint)
                print(json.dumps(dict(case=phase,termination=summary['termination'],steps=summary['finite_steps_saved'],tripod_pass=summary['nominal_holding_dynamics_pass'],old_two_pad_pass=summary['old_two_pad_pass'])),flush=True)
        pairs=f.dt_pair_summary(variants,cases,endpoints,np);after=f.model_digest_except_dt(m,np)
        if after!=before or counts['initial_forward']!=8 or counts['integration_steps']>76800:raise ValueError('Frozen model/budget')
        write(out/'report.json',dict(schema=RESULT_SCHEMA,status='COMPLETED_EIGHT_MEASURED_CONTROL_DYNAMIC_DIAGNOSTICS',protocol=pin(a.protocol),counts=counts,source_physics=pin(FROZEN),
            variants=variants,cases=cases,dt_pair_consistency=pairs,model_audit=dict(before=before,after=after,unchanged_except_dt=True),elapsed_seconds=time.perf_counter()-start,
            all_tripod_cases_and_dt_pairs_pass=bool(all(v['nominal_holding_dynamics_pass'] for v in cases) and all(v['dt_pair_consistency_pass'] for v in pairs)),
            all_old_two_pad_cases_and_dt_pairs_pass=bool(all(v['old_two_pad_pass'] for v in cases) and all(v['dt_pair_consistency_pass'] for v in pairs)),
            scope='Frozen2s physics loop/gates directly reused. Four measured u at one compressed pose, both dt retained. No new preload FW/material-anchor command, NN or object aid; measured initial loading is not a holding certificate.'))
    except BaseException as err:
        write(out/'failure.json',dict(phase=phase,error=repr(err),counts=counts,completed_cases=cases,automatic_retry=False));raise
if __name__=='__main__':main()
