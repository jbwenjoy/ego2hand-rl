"""Two fresh Reader reset+one-H executions of the actual loaded tripod bridge."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,time,traceback
H=Path(__file__).resolve().parent;P=H.parent;T=P/'phone_geometry_actor_integration_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):assert pin(v['path'])==v;return Path(v['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def mod(name,path):
    import sys
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def prepare():
    b=read(T/'task_bundle_v2.json');ip=read(T/'protocol_CPUFalse_v2.json')
    p=dict(schema='phone_loaded_tripod_native_reader_bridge_CPU_protocol:1',CPU_run_authorized=False,NN_training_GPU_authorized=False,source=pin(__file__),
        inputs=dict(bundle=pin(T/'task_bundle_v2.json'),integration_protocol=pin(T/'protocol_CPUFalse_v2.json'),catalog=pin(P/'phone_directional_gap_features_v1/geometry_catalog_v1.json'),model=b['model']),
        sources={k:ip['sources'][k] for k in ['task','reader_replay','native_reader','directional']},
        dt_s=[.0003125,.00015625],action='exact_zero_FP32_normalized20_GN_F64_ctrl_unchanged',
        budget=dict(MJB=2,initial_forward=2,maximum_native_steps=192,NN_constructors_FW_optimizer=0,GPU=0),output=str(H/'completed_run1'),
        scope='Two physical reset/control/replay interface executions, one20ms perdt. No training, no operator interventions, no2s holding or task gain. A2 score is an engineered geometry proxy read, not observed force reward.')
    write(H/'protocol_CPUFalse_v1.json',p);print(json.dumps(dict(protocol=pin(H/'protocol_CPUFalse_v1.json'))))
def preflight(path):
    p=read(path);assert p['schema']=='phone_loaded_tripod_native_reader_bridge_CPU_protocol:1' and p['source']==pin(__file__) and p['NN_training_GPU_authorized'] is False
    for v in [*p['inputs'].values(),*p['sources'].values()]:checked(v)
    assert p['dt_s']==[.0003125,.00015625] and p['budget']['maximum_native_steps']==192 and not Path(p['output']).exists();return p
def run(path):
    p=preflight(path);assert p['CPU_run_authorized'] is True
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import torch
    torch.set_num_threads(4);O=Path(p['output']);O.mkdir();started=time.perf_counter();phase='imports'
    try:
        taskmod=mod('native_bridge_task',checked(p['sources']['task']));bridge=mod('native_bridge_adapter',checked(p['sources']['reader_replay']))
        reader=mod('native_bridge_original_reader',checked(p['sources']['native_reader']));gap=mod('native_bridge_directional',checked(p['sources']['directional']))
        b=read(checked(p['inputs']['bundle']));summaries=[];audits=[];records=[]
        def sink(packet):
            i=len(records);arrays={}
            def convert(v):
                if isinstance(v,np.ndarray):
                    key='a'+str(len(arrays));arrays[key]=v
                    return dict(array=key,dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
                if isinstance(v,dict):return {k:convert(x) for k,x in v.items()}
                if isinstance(v,(tuple,list)):return [convert(x) for x in v]
                if isinstance(v,(bool,np.bool_)):return bool(v)
                if isinstance(v,np.integer):return int(v)
                if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
                return v
            meta=convert(packet);arrpath=O/f'raw_{i:02d}.npz';np.savez_compressed(arrpath,**arrays);write(O/f'raw_{i:02d}.json',dict(payload=meta,arrays=pin(arrpath)))
            records.append(pin(O/f'raw_{i:02d}.json'))
        for h in p['dt_s']:
            phase='reset_'+str(h);r=reader.NativePhoneReader(checked(p['inputs']['model']),dt_s=h);geometry=gap.DirectionalGapProvider(checked(p['inputs']['catalog']))
            task=taskmod.PhoneTask(checked(p['inputs']['bundle']),mode='A1_tracking');adapter=bridge.PhoneReaderTaskBridge(r,task,geometry,sink)
            x,obs=adapter.reset();initialcontact=np.load(O/f'raw_{len(records)-1:02d}.npz',allow_pickle=False)
            # Read initial contacts from the persisted named array descriptor.
            resetmeta=read(O/f'raw_{len(records)-1:02d}.json')['payload']['reset'];key=resetmeta['initial_contact_rows24']['array'];cr=initialcontact[key];initialcontact.close()
            Fn={str(g):0. for g in [15,28,41]};other=table=0.
            for row in cr:
                g1,g2=map(int,row[1:3])
                if 71 not in [g1,g2]:continue
                g=g2 if g1==71 else g1;f=max(0.,float(row[18]))
                if str(g) in Fn:Fn[str(g)]+=f
                elif g==0:table+=f
                else:other+=f
            initialx=x.numpy().copy();initialu=r.d.ctrl.copy();before=r.receipt()
            phase='one_H_'+str(h);result=adapter.step(torch.zeros(20,dtype=torch.float32));interval=result['raw_interval'];replay=result['replay_transition']
            a2=taskmod.PhoneTask(checked(p['inputs']['bundle']),mode='A2_engineered_material_proxy',geometry=geometry);score=a2.rewards(result['state58'],task.initial_memory(x,native_h_s=r.h,n_substeps=r.n),torch.zeros(20,dtype=torch.float32))
            initialerr=float(np.max(abs(initialx-np.asarray(b['initial']['native_PRE_state58_F64']))));actbits=np.asarray(interval['actual_control20']).tobytes()==np.array(b['initial']['ctrl20_F64']).tobytes()
            checks=dict(named_reset_state=initialerr<=1e-12,loaded_three_pad=all(v>.1 for v in Fn.values()),unwanted_initial_load=other<=.1 and table<=1e-8,
                exact_held_GN_control=actbits and initialu.tobytes()==np.asarray(interval['actual_control20']).tobytes(),
                complete_20ms=interval['complete_H20ms'] and interval['returned_substeps']==r.n,valid_finite_replay=bool(replay['valid'][0]) and bool(torch.isfinite(replay['x1']).all()),
                raw300_receipts=all(interval[k]['signature']==16383 and interval[k]['size']==300 for k in ['source_receipt','source_after_action_receipt','target_receipt']),
                memory_phase=int(adapter.memory['step'])==1 and int(replay['target_frame'][0])==1,
                observation_shape=obs.shape==(222,) and replay['obs'].shape==(1,222),task_reward_known=bool(result['reward']['objective_known']),
                engineered_proxy_known_not_actual=bool(score['objective_known']) and not bool(score['actual_contact_reward_known']),no_warnings=not np.any(interval['warning_numbers']))
            checks={k:bool(v) for k,v in checks.items()};audits.append(r.audit())
            summaries.append(dict(dt_s=h,n_substeps=r.n,checks=checks,initial_pad_positive_Fn_N=Fn,initial_other_positive_Fn_N=other,initial_table_positive_Fn_N=table,
                initial_state_maxabs=initialerr,post_state58=result['state58'].numpy().tolist(),drift_COM_m=float(np.linalg.norm(result['state58'].numpy()[40:43]-initialx[40:43])),
                triple_PRE_coverage=bridge.regroup_interval_contacts(interval)['triple_simultaneous_coverage'],A1_reward=float(result['reward']['total']),A2_engineered_proxy_reward=float(score['total']),
                native_termination=interval['termination'],reader_counts=dict(r.counts)))
        counts={k:sum(a['counts'][k] for a in audits) for k in audits[0]['counts']}
        assert counts['integration_steps']==192 and counts['initial_forward']==2 and counts['observer_forward']==0
        checks={f'dt{i}_{k}':v for i,s in enumerate(summaries) for k,v in s['checks'].items()};checks['CPU_no_GPU_context']=not torch.cuda.is_initialized()
        report=dict(schema='phone_loaded_tripod_native_reader_bridge_result:1',status='PASS_TWO_ACTUAL_ONE_H_INTERFACES' if all(checks.values()) else 'FAIL_PRESERVED_NO_RETRY',protocol=pin(path),source=pin(__file__),checks=checks,passed=sum(checks.values()),total=len(checks),
            summaries=summaries,native_counts=counts,model_audits=audits,raw_records=records,NN_constructor_FW_optimizer=0,elapsed_s=time.perf_counter()-started,scope=p['scope'])
        write(O/'report.json',report);print(json.dumps(dict(passed=report['passed'],total=report['total'],report=pin(O/'report.json'))));assert all(checks.values())
    except BaseException as e:
        write(O/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--protocol',type=Path);a.add_argument('--run',action='store_true');arg=a.parse_args()
    if arg.prepare:prepare()
    elif arg.run:run(arg.protocol)
    else:preflight(arg.protocol);print('METADATA_ONLY_NO_NATIVE_NN')
