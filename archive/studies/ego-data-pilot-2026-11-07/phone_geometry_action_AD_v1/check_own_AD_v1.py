"""Fixed same-model action AD vs central FD; not a native physics derivative."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,time
H=Path(__file__).resolve().parent;P=H.parent;W=P/'phone_geometry_momentum_wm_v1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(v):assert pin(v['path'])==v;return Path(v['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def mod(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def prepare():
    tp=read(W/'protocol_CPUTrue_resume1.json')
    p=dict(schema='phone_fixed_current_state_own_model_action_AD_protocol:1',CPU_run_authorized=False,GPU_run_authorized=False,training_authorized=False,
        output=str(H/'completed_run1'),source=pin(__file__),sources=tp['sources'],checkpoint=read(W/'cpu_resume1/health_5000.json')['checkpoint'],
        current_FIT_feature_archive=pin(W/'cpu_resume1/prepared_FIT_targets_and_features.npz'),physical_design=pin(P/'phone_episode_acquisition_v1/design_contract_v1.json'),
        fixed_original_TRAIN_row_indices=[0,200,400,683],epsilon_rad=[.001,.0005],engineering_relativeL2_threshold=.05,
        budget=dict(own_CPU_checkpoint_loads=1,NN_constructors=1,NN_FW=7,input_AD_Jacobians=4,optimizer_steps=0,native=0),
        scope='Action derivative of same fixed5000 model at4 originalTRAIN current states; cached current-only geometry is action-independent. SO3world-left tangent AD/central geodesic FD; no state/FK-gradient, native physics, Actor, or ego claim.')
    write(H/'protocol_CPUFalse_v1.json',p);print(json.dumps(dict(protocol=pin(H/'protocol_CPUFalse_v1.json'))))
def preflight(path):
    p=read(path);assert p['schema']=='phone_fixed_current_state_own_model_action_AD_protocol:1' and p['source']==pin(__file__) and p['GPU_run_authorized'] is False and p['training_authorized'] is False
    for v in [*p['sources'].values(),p['checkpoint'],p['current_FIT_feature_archive'],p['physical_design']]:checked(v)
    assert p['fixed_original_TRAIN_row_indices']==[0,200,400,683] and p['epsilon_rad']==[.001,.0005] and p['engineering_relativeL2_threshold']==.05
    assert not Path(p['output']).exists();return p
def run(path):
    p=preflight(path);assert p['CPU_run_authorized'] is True
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import torch
    torch.set_num_threads(4);out=Path(p['output']);out.mkdir();started=time.perf_counter()
    train=mod('own_AD_trainer',checked(p['sources']['trainer']));codec=mod('own_AD_codec',checked(p['sources']['structured_codec']))
    with np.load(checked(p['current_FIT_feature_archive']),allow_pickle=False) as z:F=z['features366_F64'][p['fixed_original_TRAIN_row_indices']].copy()
    ck=torch.load(checked(p['checkpoint']),map_location='cpu',weights_only=False);assert ck['iteration']==5000
    model=train.make_model(torch);model.load_state_dict(ck['model_state'],strict=True);model.requires_grad_(False);model.eval();norm=ck['normalizers'];phys=ck['physical_context']
    limits=np.array(read(checked(p['physical_design']))['actual_limits']['actuator_ctrlrange20x2_F64'])
    NN_FW=0
    def forward(features):
        nonlocal NN_FW
        normalized=((features-norm['input_mean'])/norm['input_std']).float();raw=model(normalized);NN_FW+=1
        head=raw[:,:178].double()*norm['head_std']+norm['head_mean']
        y=codec.decode_physical_heads(features[:,:58],train.heads_from_physical(head),features[:,78]*features[:,79],phys['mass'],phys['I'],phys['g'])
        return y,normalized
    ft=torch.tensor(F,dtype=torch.float64)
    with torch.no_grad():baseline,_=forward(ft)
    AD=[]
    for row in range(4):
        baseR=baseline[row,43:52].reshape(3,3).detach()
        def fun(u):
            features=torch.cat([ft[row,:58],u,ft[row,78:]],dim=-1).unsqueeze(0);y,_=forward(features);a=y[0];R=a[43:52].reshape(3,3)@baseR.T
            tangent=.5*torch.stack([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
            return torch.cat([a[:40],a[40:43],tangent,a[52:58]])
        AD.append(torch.autograd.functional.jacobian(fun,ft[row,58:78].clone().requires_grad_(),vectorize=True).detach().numpy())
    AD=np.array(AD);FD=np.full((2,4,52,20),np.nan);known=np.zeros((2,4,20),bool);du64=np.full((2,4,20),np.nan);du_encoded=np.full_like(du64,np.nan)
    for ei,eps in enumerate(p['epsilon_rad']):
        cases=[];ids=[]
        for row in range(4):
            for j in range(20):
                u=F[row,58+j]
                if u-eps<limits[j,0] or u+eps>limits[j,1]:continue
                for sign in [-1,1]:
                    a=F[row].copy();a[58+j]+=sign*eps;cases.append(a);ids.append((row,j,sign))
        with torch.no_grad():yp,nf=forward(torch.tensor(np.array(cases),dtype=torch.float64))
        yp=yp.numpy();nf=nf.numpy();index={k:i for i,k in enumerate(ids)}
        for row in range(4):
            for j in range(20):
                if (row,j,-1) not in index:continue
                lo,hi=index[row,j,-1],index[row,j,1];a,b=yp[lo],yp[hi];den=cases[hi][58+j]-cases[lo][58+j]
                encoded=(float(nf[hi,58+j])-float(nf[lo,58+j]))*float(norm['input_std'][58+j])
                if encoded==0:continue
                v=np.r_[b[:40]-a[:40],b[40:43]-a[40:43],codec.scalar.log_world(b[43:52].reshape(3,3)@a[43:52].reshape(3,3).T),b[52:58]-a[52:58]]/den
                FD[ei,row,:,j]=v;known[ei,row,j]=True;du64[ei,row,j]=den;du_encoded[ei,row,j]=encoded
    blocks=dict(q=slice(0,20),qd=slice(20,40),COM=slice(40,43),world_dR=slice(43,46),vCOM=slice(46,49),omega_world=slice(49,52));metrics=[]
    for ei,eps in enumerate(p['epsilon_rad']):
        entry=dict(epsilon_rad=eps,available_axes=int(known[ei].sum()),unavailable_axes=int((~known[ei]).sum()),blocks={})
        for key,s in blocks.items():
            aa=np.moveaxis(AD[:,s,:],-1,1)[known[ei]];ff=np.moveaxis(FD[ei,:,s,:],-1,1)[known[ei]]
            err=ff-aa;reference=float(np.linalg.norm(aa));entry['blocks'][key]=dict(reference_AD_L2=reference,error_L2=float(np.linalg.norm(err)),relativeL2=float(np.linalg.norm(err)/max(reference,1e-12)),error_RMS=float(np.sqrt(np.mean(err**2))),maxabs=float(np.max(abs(err))))
        metrics.append(entry)
    checks=dict(all_AD_finite=bool(np.isfinite(AD).all()),all_defined_FD_finite=bool(np.isfinite(np.moveaxis(FD,-1,-2)[known]).all()),
        preserved_unavailable_NaNs=bool(np.isnan(np.moveaxis(FD,-1,-2)[~known]).all()),
        finest_scale_engineering_agreement=all(v['relativeL2']<=p['engineering_relativeL2_threshold'] for v in metrics[-1]['blocks'].values()),
        exact_bounded_NN_FW=NN_FW==7,zero_parameter_gradients=all(v.grad is None for v in model.parameters()),CPU_only=not torch.cuda.is_initialized())
    ep=out/'same_model_AD_FD.npz';np.savez_compressed(ep,AD_physical52_per_action20=AD,FD_physical52_per_action20=FD,FD_known=known,actual_central_du_F64_rad=du64,normalized_F32_encoded_physical_du_rad=du_encoded,features_current366_F64=F)
    report=dict(schema='phone_same_model_action_AD_central_FD_result:1',status='PASS_SCOPED_MODEL_SELF_DIFFERENTIATION' if all(checks.values()) else 'DISCREPANCY_PRESERVED_NO_RETRY',checks=checks,passed=sum(checks.values()),total=len(checks),
        protocol=pin(path),checkpoint=p['checkpoint'],source=pin(__file__),evidence=pin(ep),metrics=metrics,operation_counts=p['budget'],actual_NN_FW=NN_FW,elapsed_s=time.perf_counter()-started,scope=p['scope'])
    write(out/'report.json',report);print(json.dumps(dict(passed=report['passed'],total=report['total'],report=pin(out/'report.json'))))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--protocol',type=Path);a.add_argument('--run',action='store_true');args=a.parse_args()
    if args.prepare:prepare()
    elif args.run:run(args.protocol)
    else:preflight(args.protocol);print('METADATA_ONLY_NO_NN_NATIVE')
