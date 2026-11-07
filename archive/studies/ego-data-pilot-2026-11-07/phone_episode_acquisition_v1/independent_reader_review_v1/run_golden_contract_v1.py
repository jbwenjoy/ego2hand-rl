"""One192step native golden contract. No automatic retry or source mutation."""
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
    if pin(p['path'])!=p:raise ValueError('Changed contract source/input '+p['path'])
    return Path(p['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n')
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def shadow(q,v,ipos):
    # Quaternion scalar/vector formula, independent of producer's explicit matrix entries.
    w=float(q[23]);a=q[24:27];x,y,z=a
    S=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    M=(w*w-float(a@a))*np.eye(3)+2*np.outer(a,a)+2*w*S
    offset=M@ipos;omega=M@v[23:26]
    vel=v[20:23]+np.array([omega[1]*offset[2]-omega[2]*offset[1],omega[2]*offset[0]-omega[0]*offset[2],omega[0]*offset[1]-omega[1]*offset[0]])
    return np.r_[q[:20],v[:20],q[20:23]+offset,M.ravel(),vel,omega]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    protocol=read(args.protocol)
    if protocol['schema']!='native_phone_reader_golden_contract_protocol:1':raise ValueError('Contract namespace')
    for p in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:checked(p)
    if protocol['source_pins']['checker']!=pin(__file__):raise ValueError('Exact checker')
    if protocol['expected_steps_by_dt']!={'0.0003125':64,'0.00015625':128}:raise ValueError('Only prescribed192step contract')
    if not args.execute:print(json.dumps({'status':'METADATA_SOURCE_ONLY_NO_NATIVE_MODEL','protocol':pin(args.protocol)}));return
    if protocol['CPU_contract_execute_authorized'] is not True:raise ValueError('Explicit192step boundary')
    out=Path(protocol['output']);out.mkdir()
    save(out/'protocol.json',protocol)
    reader_path=checked(protocol['source_pins']['reader']);spec=importlib.util.spec_from_file_location('golden_frozen_phone_reader',reader_path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    old=read(checked(protocol['inputs']['old_constant_report']));results=[];readers=[];total=0
    try:
        for c in old['cases']:
            if c['name']!='matched_constant_u_seed0_rank0':raise ValueError('Exact oldbase family only')
            dt=c['dt_s'];n=protocol['expected_steps_by_dt'][str(dt)];assert current_trace(c['trace'],protocol['inputs'][f'trace_{n}'])
            with np.load(checked(c['trace']),allow_pickle=False) as z:oldtrace={k:z[k].copy() for k in z.files}
            reader=module.NativePhoneReader(protocol['inputs']['model']['path'],expected_model_sha256=protocol['inputs']['model']['sha256'],dt_s=dt);readers.append(reader)
            reset=reader.reset(oldtrace['initial_qpos27'],oldtrace['ctrl20'])
            case=out/f'dt_{n}';case.mkdir();save(case/'reset.json',reset)
            # Calls source operation exactly once; do not recover/retry if native fails.
            result=reader.step_control(oldtrace['ctrl20']);total+=result['returned_substeps'];trace=result['trace']
            np.savez_compressed(case/'trace.npz',**trace,source_state58=result['source_state58'],target_state58=result['target_state58'],actual_control20=result['actual_control20'],physical_world_delta52=result['physical_world_delta52'],initial_qpos27=reset['qpos27'],initial_qvel26=reset['qvel26'],body_ipos_body=reader.ipos)
            save(case/'result.json',{k:v for k,v in result.items() if k!='trace'})
            reference=np.array([shadow(q,v,reader.ipos) for q,v in zip(trace['post_qpos27'],trace['post_qvel26'])])
            errors={name:float(np.max(abs(trace['post_state58'][:,s]-reference[:,s]))) for name,s in [('COM',slice(40,43)),('Rbody9',slice(43,52)),('vCOMworld',slice(52,55)),('omegaworld',slice(55,58))]}
            mappings={'pre_pad_normal_N2':'pre_pad_normal_N','pre_phone_nonpad_normal_N':'pre_phone_nonpad_normal_N','pre_table_normal_N':'pre_table_normal_N','pre_penetration_m':'pre_penetration_m','pre_self_penetration_m':'pre_self_penetration_m','pre_energy2':'pre_native_energy','pre_actuator_force20':'pre_actuator_force'}
            contact_ref=oldtrace['contact_rows'][oldtrace['contact_rows'][:,0]<n]
            checks={'complete_finite_exact_one_H20ms':result['complete_H20ms'] is True and result['finite'] is True and result['returned_substeps']==n and result['termination']=='COMPLETE_INTERVAL',
              'all_POST_qpos27_bits_match_old_firstH':bits(trace['post_qpos27'],oldtrace['post_qpos'][:n]),
              'all_POST_qvel26_bits_match_old_firstH':bits(trace['post_qvel26'],oldtrace['post_qvel'][:n]),
              'all_PRE_force_energy_actuator_arrays_bits_match_old_firstH':all(bits(trace[k],oldtrace[v][:n]) for k,v in mappings.items()),
              'all_contact24_bits_match_old_firstH':bits(trace['contact_rows24'],contact_ref),
              'POST_COM_matches_old_native_COM_F64_domain':float(np.max(abs(trace['post_state58'][:,40:43]-oldtrace['post_COM'][:n])))<=protocol['F64_observer_abs_bound'],
              'COM_R_worldvel_worldomega_independent_shadow':all(v<=protocol['F64_observer_abs_bound'] for v in errors.values()),
              'nonzero_dynamic_worldomega_exercised':bool(np.max(abs(trace['post_state58'][:,55:58]))>0),
              'PRE_POST_integer_ticks_and_native_clock':np.array_equal(trace['pre_tick'],np.arange(n)) and np.array_equal(trace['post_tick'],np.arange(1,n+1)) and bits(trace['post_native_time_s'],oldtrace['post_time'][:n]) and bits(trace['pre_native_time_s'],np.r_[0.,oldtrace['post_time'][:n-1]]),
              'reset_forward_contact_phase_is_separate':reset['contact_phase'].startswith('RESET_FORWARD') and reset['initial_contact_rows24'].shape[1]==24,
              'same_u_source_before_and_after_action_receipts_equal':bits(result['source_receipt']['integration_state'],result['source_after_action_receipt']['integration_state']) and reset['receipt']['sha256']==result['source_receipt']['sha256'],
              'receipt_vectors_hashes_and_ticks':all(hashlib.sha256(x['integration_state'].tobytes()).hexdigest()==x['sha256'] for x in [result['source_receipt'],result['source_after_action_receipt'],result['target_receipt']]) and result['source_global_tick']==0 and result['target_global_tick']==n,
              'one_initial_forward_no_observer_forward_or_NN':reader.counts['initial_forward']==1 and reader.counts['observer_forward']==reader.counts['NN_training_GPU']==0 and reader.counts['integration_steps']==n,
              'initial_and_step_contact_force_counts_separate':reader.counts['initial_contact_force_reads']==len(reset['initial_contact_rows24']) and reader.counts['step_contact_force_reads']==len(trace['contact_rows24']) and reader.counts['contact_force_reads']==reader.counts['initial_contact_force_reads']+reader.counts['step_contact_force_reads']}
            audit=reader.audit();save(case/'model_audit.json',audit)
            results.append({'dt_s':dt,'steps':n,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'observer_max_abs_errors':errors,'source_old_trace':c['trace'],'reset':pin(case/'reset.json'),'result':pin(case/'result.json'),'trace':pin(case/'trace.npz'),'audit':pin(case/'model_audit.json'),'model_audit':audit})
        shared=[r['model_audit']['shared_physics_digest_without_dt'] for r in results]
        finalchecks={'bothdt_all_case_checks':all(all(r['checks'].values()) for r in results),'exactly192_steps_no_retry':total==192,'same_shared_physics_digest_excluding_dt':len(set(shared))==1,'both_model_audits_pass':all(r['model_audit']['passed'] is True for r in results)}
        report={'schema':'native_phone_reader_actual_golden_contract:1','status':'PASS_192_NATIVE_STEP_GOLDEN_READER_CONTRACT' if all(finalchecks.values()) else 'FAILED_PRESERVED_NO_RETRY','protocol':pin(args.protocol),'source':pin(__file__),'reader':pin(reader_path),'cases':results,'checks':finalchecks,'passed':sum(finalchecks.values()),'total':len(finalchecks),'counts':{'MJB_reads':sum(r.counts['MJB_reads'] for r in readers),'initial_forward':sum(r.counts['initial_forward'] for r in readers),'integration_steps':total,'observer_forward':sum(r.counts['observer_forward'] for r in readers),'contact_force_reads':sum(r.counts['contact_force_reads'] for r in readers),'initial_contact_force_reads':sum(r.counts['initial_contact_force_reads'] for r in readers),'step_contact_force_reads':sum(r.counts['step_contact_force_reads'] for r in readers),'NN_training_GPU':0},'scope':'Two golden20ms oldconstant prefixes only; matching pipeline/phase/dynamics and nonzero worldstate arithmetic, not holding, Markov, future episode coverage or learning qualification. F64 observer shadow bound is numeric codec-domain evidence, not physical accuracy.'}
        save(out/'report.json',report);print(json.dumps({'report':pin(out/'report.json'),'status':report['status'],'checks':finalchecks,'counts':report['counts']}))
    except BaseException as error:
        save(out/'failure.json',{'error':repr(error),'total_returned_steps':total,'reader_counts':[r.counts for r in readers],'automatic_retry':False});raise

def current_trace(a,b):return a==b and pin(a['path'])==a

if __name__=='__main__':main()
