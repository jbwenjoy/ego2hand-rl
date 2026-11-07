"""One384step baseline restore fixture, no retry/forward repair."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

R=Path(__file__).resolve().parent

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def checked(p):
    if pin(p['path'])!=p:raise ValueError('Changed pinned fixture input/source')
    return Path(p['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def save(p,d):
    with p.open('x',encoding='utf8') as f:json.dump(clean(d),f,indent=2,allow_nan=False);f.write('\n')
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    p=read(args.protocol)
    if p['schema']!='native_saved_action_restore_fixture_protocol:1':raise ValueError('Fixture namespace')
    for x in [*p['inputs'].values(),*p['source_pins'].values()]:checked(x)
    if p['source_pins']['fixture']!=pin(__file__):raise ValueError('Exact fixture source')
    selection=read(checked(p['inputs']['selection']));design=read(checked(p['inputs']['design']))
    if selection['operands']!=p['inputs']['operands'] or design['selection']!=p['inputs']['selection']:raise ValueError('Original v1 state/operand selection unchanged')
    selected=[s for s in selection['states'] if s['state_id'] in p['state_ids']]
    if len(selected)!=4 or {s['source_sample_id'] for s in selected}!={122,132,602,612} or sum(s['substeps'] for s in selected)!=384:raise ValueError('Only4 predeclared interval2 states')
    if any(s['split']!='validation' or s['interval_index']!=2 for s in selected):raise ValueError('VAL metadata selection')
    for s in selected:checked(s['source_trace'])
    if not args.execute:print(json.dumps({'status':'SOURCE_METADATA_ONLY_NO_NATIVE','protocol':pin(args.protocol),'states':[s['state_id'] for s in selected]}));return
    if p['CPU_execute_authorized'] is not True:raise ValueError('Explicit bounded native fixture boundary')
    out=Path(p['output']);out.mkdir();save(out/'protocol.json',p)
    spec=importlib.util.spec_from_file_location('frozen_native_restore_probe',checked(p['source_pins']['probe']));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    keys=['source_integration_F64','source_qpos27_F64','source_qvel26_F64','source_state58_F64','baseline_ctrl20_F64','baseline_observable59_F64','original_target_integration_F64','original_target_qpos27_F64','original_target_qvel26_F64','initial_qpos27_F64','initial_state58_F64']
    with np.load(checked(p['inputs']['operands']),allow_pickle=False) as z:a={k:z[k].copy() for k in keys}
    results=[];probes=[];current_state=None
    try:
        for dt in [.0003125,.00015625]:
            expected=design['model_parameter_audits'][str(dt)]
            probe=m.NativeSavedActionProbe(p['inputs']['model']['path'],dt_s=dt,expected_effective_model_digest=expected['model_digest']);probes.append(probe)
            for state in selected:
                if state['native_dt_s']!=dt:continue
                current_state=state;sid=state['state_id'];folder=out/f'state_{sid:02d}';folder.mkdir()
                with np.load(checked(state['source_trace']),allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
                value=probe.run(state,a,a['baseline_ctrl20_F64'][sid])
                # Preserve actual data before any comparison can reject the fixture.
                np.savez_compressed(folder/'actual_trace.npz',**value['trace'],source_before_action=value['source_receipt']['integration_state'],source_after_action=value['source_after_action_receipt']['integration_state'],target_integration=value['target_receipt']['integration_state'],observable59=value['observable59'])
                save(folder/'actual_result.json',{k:v for k,v in value.items() if k!='trace'})
                checks=m.baseline_checks(value,state,a,trace=old)
                checks.update(independent_source300_bytes=bits(value['source_receipt']['integration_state'],old['source_integration_state']),
                    independent_after_action300_bytes=bits(value['source_after_action_receipt']['integration_state'],old['source_after_action_integration_state']),
                    independent_target300_bytes=bits(value['target_receipt']['integration_state'],old['target_integration_state']),
                    independent_target_SHA=hashlib.sha256(value['target_receipt']['integration_state'].tobytes()).hexdigest()==state['original_target_receipt']['sha256'],
                    independent_all_POST_qpos_bytes=bits(value['trace']['post_qpos27'],old['post_qpos27']),
                    independent_all_POST_qvel_bytes=bits(value['trace']['post_qvel26'],old['post_qvel26']),
                    actual_original_control_held=bits(value['actual_control20'],old['actual_control20']) and hashlib.sha256(value['actual_control20'].tobytes()).hexdigest()==state['baseline_ctrl_raw_F64_sha256'],
                    global_ticks_and_native_target_time=value['source_global_tick']==state['source_before_action_receipt']['global_tick'] and value['target_global_tick']==state['original_target_receipt']['global_tick'] and value['target_receipt']['native_time_s']==state['original_target_receipt']['native_time_s'])
                record={'state_id':sid,'family_id':state['family_id'],'source_sample_id':state['source_sample_id'],'source_branch':state['source_branch'],'dt_s':dt,'interval_index':2,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'steps':value['returned_substeps'],'source_trace':state['source_trace'],'actual_trace':pin(folder/'actual_trace.npz'),'actual_result':pin(folder/'actual_result.json'),'termination':value['termination'],'forward_counters':value['forward_counters']}
                save(folder/'checks.json',record);results.append(record)
                if not all(checks.values()):raise ValueError('Original action replay mismatch: state'+str(sid)+' '+str([k for k,v in checks.items() if not v]))
            audit=probe.audit();save(out/f'model_dt_{round(dt*1e9)}.json',audit)
        counts={'MJB_reads':sum(q.reader.counts['MJB_reads'] for q in probes),'MjData':sum(q.reader.counts['MjData'] for q in probes),'state_sets':sum(q.counts['state_sets'] for q in probes),'branches_attempted':sum(q.counts['branches_attempted'] for q in probes),'branches_returned':sum(q.counts['branches_returned'] for q in probes),'integration_steps':sum(q.reader.counts['integration_steps'] for q in probes),'initial_forward':sum(q.reader.counts['initial_forward'] for q in probes),'observer_forward':sum(q.reader.counts['observer_forward'] for q in probes),'restore_forward':sum(q.counts['restore_forward'] for q in probes),'contact_force_reads':sum(q.reader.counts['contact_force_reads'] for q in probes),'NN_training_GPU':0}
        checks={'all4_full_target_and_trace_baselines_pass':len(results)==4 and all(x['passed']==x['total'] for x in results),'exact384_native_steps_2models_4fresh_states':counts['integration_steps']==384 and counts['MJB_reads']==2 and counts['state_sets']==counts['branches_returned']==4,'zero_initial_observer_restore_forward':counts['initial_forward']==counts['observer_forward']==counts['restore_forward']==0,'both_same_shared_physics_and_audits':len({q.reader.shared_physics_digest for q in probes})==1 and all(q.reader.initial_digest==q.reader.model_digest() for q in probes)}
        save(out/'report.json',{'schema':'native_saved_action_restore_actual_fixture:1','status':'PASS_FOUR_STATE_384STEP_ORIGINAL_ACTION_REPLAY' if all(checks.values()) else 'FAILED_PRESERVED_NO_RETRY','protocol':pin(args.protocol),'probe':p['source_pins']['probe'],'source':pin(__file__),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'states':results,'counts':counts,'full_MjData_restoration_or_general_Markov_qualified':False,'scope':'Direct full-integration restoration empirically reproduces these four original20ms actions including nonzero-history interval2. No outside forward, reset, perturbation, NN, LP or training; no claim of generic all-MjData restoration or physical gradients.'})
        print(json.dumps({'report':pin(out/'report.json'),'checks':checks,'counts':counts}))
    except BaseException as error:
        for i,probe in enumerate(probes):
            if probe.reader.d is not None:
                np.savez_compressed(out/f'failure_live_state_{i}.npz',qpos27=probe.reader.d.qpos.copy(),qvel26=probe.reader.d.qvel.copy(),ctrl20=probe.reader.d.ctrl.copy(),qacc26=probe.reader.d.qacc.copy())
        save(out/'failure.json',{'error':repr(error),'state_id':None if current_state is None else current_state['state_id'],'completed_state_records':results,'reader_counts':[q.reader.counts for q in probes],'probe_counts':[q.counts for q in probes],'automatic_retry':False});raise

if __name__=='__main__':main()
