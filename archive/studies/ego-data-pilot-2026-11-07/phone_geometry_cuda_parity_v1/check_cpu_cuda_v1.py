"""Bounded same-weight live geometry/decoder/input-AD device check, zero updates."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,time,traceback
H=Path(__file__).resolve().parent; P=H.parent; W=P/'phone_geometry_momentum_wm_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare():
    tp=read(W/'protocol_CPUTrue_resume1.json');health=W/'cpu_resume1/health_5000.json';hr=read(health)
    assert hr['passed']==hr['total']==15 and hr['iteration']==5000
    p=dict(schema='phone_geometry_CPU_CUDA_same_weight_input_AD_protocol:1',CPU_run_authorized=False,GPU_run_authorized=False,training_authorized=False,
        output=str(H/'completed_run1'),source=pin(__file__),training_protocol=pin(W/'protocol_CPUTrue_resume1.json'),health=pin(health),checkpoint=hr['checkpoint'],
        sources=tp['sources'],geometry_catalog=tp['inputs']['geometry_catalog'],current_only_prepared_FIT=pin(W/'cpu_resume1/prepared_FIT_targets_and_features.npz'),
        fixed_original_TRAIN_row_indices=[0,200,400,683],device='cuda:0',CPU_threads=4,
        budget=dict(own_CPU_checkpoint_loads=1,NN_constructors=2,NN_FW=2,input_AD_calls=2,NN_backward_for_parameter_training=0,optimizer_steps=0,native=0),
        thresholds=dict(current_feature_F64_maxabs=1e-10,raw222_rtol=1e-4,raw222_atol=1e-5,
            decoded_maxabs_by_block=[1e-5,1e-4,1e-6,1e-5,1e-4,1e-3],q_u_AD_relL2=5e-3,q_u_AD_absmax=1e-4),
        scope='Four fixed originalTRAIN states only, same final5000 weights and FIT normalization. Live F64 geometry, F32 NN, F64 structured decode; q/u-input AD device parity. No training, timing benchmark, native-gradient or task/ego claim. Thresholds engineering-only, no geometric/physics safety qualification.')
    write(H/'protocol_CPUFalse_v1.json',p);print(json.dumps(dict(prepared=True,protocol=pin(H/'protocol_CPUFalse_v1.json'))))
def preflight(path):
    p=read(path);assert p['schema']=='phone_geometry_CPU_CUDA_same_weight_input_AD_protocol:1' and p['training_authorized'] is False and p['source']==pin(__file__)
    for v in [p['training_protocol'],p['health'],p['checkpoint'],p['geometry_catalog'],p['current_only_prepared_FIT'],*p['sources'].values()]:checked(v)
    assert p['sources']['trainer']['sha256']=='224a0c9f47e196c626e862aa52d46402a454a6d16f0b497d3720ae74e25ca539'
    assert p['fixed_original_TRAIN_row_indices']==[0,200,400,683] and p['CPU_threads']==4
    assert p['thresholds']==dict(current_feature_F64_maxabs=1e-10,raw222_rtol=1e-4,raw222_atol=1e-5,decoded_maxabs_by_block=[1e-5,1e-4,1e-6,1e-5,1e-4,1e-3],q_u_AD_relL2=5e-3,q_u_AD_absmax=1e-4)
    assert not Path(p['output']).exists();return p
def run(path):
    p=preflight(path);assert p['CPU_run_authorized'] is True and p['GPU_run_authorized'] is True
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(p['output']);out.mkdir();started=time.perf_counter();phase='cuda_preflight'
    try:
        assert torch.cuda.is_available();assert '5090' in torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats(0)
        train=mod('parity_frozen_trainer',checked(p['sources']['trainer']))
        codec=mod('parity_frozen_codec',checked(p['sources']['structured_codec']))
        gap=mod('parity_frozen_directional',checked(p['sources']['directional']))
        geometry=gap.DirectionalGapProvider(checked(p['geometry_catalog']))
        with np.load(checked(p['current_only_prepared_FIT']),allow_pickle=False) as z: current=z['features366_F64'][p['fixed_original_TRAIN_row_indices'],:80].copy()
        ck=torch.load(checked(p['checkpoint']),map_location='cpu',weights_only=False)
        assert ck['iteration']==5000 and ck['sources']==p['sources']
        results={};memory={}
        for device in ['cpu',p['device']]:
            phase=device; model=train.make_model(torch).to(device);model.load_state_dict(ck['model_state'],strict=True);model.eval();model.requires_grad_(False)
            norm={k:v.to(device) for k,v in ck['normalizers'].items()};phys={k:v.to(device) for k,v in ck['physical_context'].items()}
            a=torch.tensor(current,dtype=torch.float64,device=device);q=a[:,:20].clone().requires_grad_();u=a[:,58:78].clone().requires_grad_()
            x=torch.cat([q,a[:,20:58]],dim=-1);h=a[:,78];n=a[:,79].to(torch.int64)
            features=train.current_features(x,u,h,n,geometry,torch)
            raw=model(((features-norm['input_mean'])/norm['input_std']).float())
            heads=raw[:,:178].double()*norm['head_std']+norm['head_mean']
            y=codec.decode_physical_heads(x,train.heads_from_physical(heads),h*n,phys['mass'],phys['I'],phys['g'])
            # Dimensionless fixed probe functional, not a policy or task reward.
            functional=y[:,:20].mean()+10*y[:,40:43].mean()+y[:,52:55].mean()+.1*y[:,55:58].mean()+.01*torch.log1p(torch.nn.functional.softplus(raw[:,178:200])).mean()
            dq,du=torch.autograd.grad(functional,(q,u),create_graph=False)
            assert all(torch.isfinite(v).all().item() for v in [features,raw,heads,y,dq,du])
            results[device]={k:v.detach().cpu().numpy() for k,v in dict(features=features,raw=raw,head=heads,decoded=y,dq=dq,du=du).items()}
            if device!='cpu':
                torch.cuda.synchronize();memory=dict(max_allocated_bytes=torch.cuda.max_memory_allocated(0),max_reserved_bytes=torch.cuda.max_memory_reserved(0),allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0))
            assert all(v.grad is None for v in model.parameters())
        c,g=results['cpu'],results[p['device']];t=p['thresholds'];blocks=[slice(0,20),slice(20,40),slice(40,43),slice(43,52),slice(52,55),slice(55,58)]
        maxima=[float(np.max(abs(c['decoded'][:,s]-g['decoded'][:,s]))) for s in blocks]
        gradmetrics={}
        for k in ['dq','du']:
            diff=g[k]-c[k];den=float(np.linalg.norm(c[k]));gradmetrics[k]=dict(maxabs=float(np.max(abs(diff))),reference_L2=den,relL2=float(np.linalg.norm(diff)/max(den,np.finfo(float).tiny)))
        checks=dict(features_F64=float(np.max(abs(c['features']-g['features'])))<=t['current_feature_F64_maxabs'],
            raw222_FP32=np.allclose(c['raw'],g['raw'],rtol=t['raw222_rtol'],atol=t['raw222_atol']),decoded_physical_blocks=all(v<=limit for v,limit in zip(maxima,t['decoded_maxabs_by_block'])),
            q_AD=gradmetrics['dq']['relL2']<=t['q_u_AD_relL2'] or gradmetrics['dq']['maxabs']<=t['q_u_AD_absmax'],
            u_AD=gradmetrics['du']['relL2']<=t['q_u_AD_relL2'] or gradmetrics['du']['maxabs']<=t['q_u_AD_absmax'],
            zero_parameter_gradients=True,same_weights_and_FIT_norms=True)
        checks={k:bool(v) for k,v in checks.items()}
        evidence=out/'device_values.npz';np.savez_compressed(evidence,**{device.replace(':','_')+'_'+k:v for device,values in results.items() for k,v in values.items()})
        report=dict(schema='phone_geometry_CPU_CUDA_saved_same_weight_input_AD_result:1',status='PASS_SCOPED_DEVICE_PARITY' if all(checks.values()) else 'FAIL_PRESERVED_NO_RETRY',checks=checks,passed=sum(checks.values()),total=len(checks),
            protocol=pin(path),source=pin(__file__),checkpoint=p['checkpoint'],evidence=pin(evidence),device=torch.cuda.get_device_name(0),torch_version=torch.__version__,CUDA_version=torch.version.cuda,
            feature_maxabs=float(np.max(abs(c['features']-g['features']))),raw222_maxabs=float(np.max(abs(c['raw']-g['raw']))),decoded_maxabs_q_qd_COM_R_v_w=maxima,AD=gradmetrics,GPU_torch_memory=memory,
            operation_counts=p['budget'],elapsed_s=time.perf_counter()-started,scope=p['scope'])
        write(out/'report.json',report);print(json.dumps(dict(passed=report['passed'],total=report['total'],report=pin(out/'report.json'))));assert all(checks.values())
    except BaseException as e:
        write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),no_automatic_retry=True));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--protocol',type=Path);ap.add_argument('--run',action='store_true');arg=ap.parse_args()
    if arg.prepare:prepare()
    elif arg.run:run(arg.protocol)
    else:preflight(arg.protocol);print('METADATA_ONLY_NO_TORCH_NPZ_NN_CUDA')
