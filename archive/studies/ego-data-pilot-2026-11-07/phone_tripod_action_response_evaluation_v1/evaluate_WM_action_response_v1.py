"""Fixed warm5000/offline1000 H1 action-response evaluation, SOURCE only.

Default imports no Torch/native backend. All328 native records stay present.
Native .01-rad secants and model-only .0001-rad AD/FD are distinct outputs.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
SCHEMA='phone_tripod_fixed_WM_action_response_evaluation_protocol:1'
ROLES=['warm5000','offline1000_WM1100']
RULES=dict(contexts=8,records=328,side_records=320,axes=20,model_FD_epsilon_rad=.0001,device='cpu',cpu_threads=4,
    own_CPU_checkpoint_loads=2,WM_constructors=2,maximum_NN_FW_calls=20,action_jacobian_API_calls=16,
    Jacobian_output_dimension=274,Jacobian_method='torch.autograd.functional.jacobian reverse-mode vectorize=True:1forward+1batchedVJP carrying274 cotangent directions per context;16batchedVJP=4384 scalar reverse directions equivalent',
    endpoint_FD_scope='Native exactsame actualendpoints primary, modelsmallFD self-consistency only. No native infinitesimal/contact/H2/state-q Jacobian claim',
    triple_prediction='unavailable:22coverage heads are marginal; never multiply to fabricate simultaneous triple coverage',training_updates=0,native_calls=0,GPU=False)
def need(ok,msg):
    if not ok:raise ValueError(msg)
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Current absolute pin');return Path(v['path'])
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return float(v)
    return v
def write(path,v):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x',encoding='utf8') as f:json.dump(clean(v),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def npz(v):
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def metadata_preflight(path):
    p=read(path);need(p['schema']==SCHEMA and p['rules']==RULES and p['source']==pin(__file__),'Fixed new response protocol/source')
    need(type(p['CPU_run_authorized']) is bool and p['training_authorized'] is False and p['GPU_run_authorized'] is False,'CPU only/no training')
    for v in p['sources'].values():checked(v)
    for v in p['inputs'].values():
        if v is not None:checked(v)
    need(p['sources']['WM_trainer']['sha256']=='224a0c9f47e196c626e862aa52d46402a454a6d16f0b497d3720ae74e25ca539' and p['sources']['paired_trainer']['sha256']=='a1bcf9f820e00d7da4ac515dd81d45d39a2c59abd07c85319d1346db384b2439','Exact frozen phone model families')
    wp=read(checked(p['inputs']['warm_training_protocol']));bundle=read(checked(p['inputs']['task_bundle']))
    need(wp['sources']['trainer']==p['sources']['WM_trainer'] and wp['inputs']['geometry_catalog']==p['inputs']['geometry_catalog'] and bundle['model']==p['inputs']['model'],'Warm/current geometry and actual task/model')
    warm=read(checked(p['inputs']['warm_result']));wh=read(checked(p['inputs']['warm_health']))
    need(warm['iteration']==5000 and warm['checkpoint']==wh['checkpoint'] and wh['checks'] and all(q is True for q in wh['checks'].values()),'Own healthy fixed warm5000')
    checked(warm['checkpoint'])
    if p['inputs']['probe_report'] is not None:
        report=read(checked(p['inputs']['probe_report']));prp=read(checked(report['protocol']))
        need(report['schema']=='phone_tripod_heldout_action_response_result:1' and report['data_role']=='ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE' and report['permanent_training_eligible'] is False,'Heldout native data, never fit')
        need(report['task_bundle']==p['inputs']['task_bundle'] and prp['inputs']['model_mjb']==p['inputs']['model'] and prp['sources']['runner']==p['sources']['probe_runner'],'Native dataset actual task/model/source')
        if p['inputs']['probe_terminal'] is not None:
            tt=read(checked(p['inputs']['probe_terminal']));need(tt['returncode']==0 and tt['protocol']==report['protocol'] and tt['source']==p['sources']['probe_runner'],'Native terminated exit0/source')
        if p['inputs']['probe_saved_review'] is not None:
            rev=read(checked(p['inputs']['probe_saved_review']));need(rev['schema']=='phone_tripod_heldout_action_response_saved_review:1' and rev['producer_report']==p['inputs']['probe_report'] and rev['engineering_qualified'] is True and rev['checks'] and all(v is True for v in rev['checks'].values()),'Saved integrity review is not independent physical truth')
    pending=any(p['inputs'][k] is None for k in ['final_training_protocol','final_result','final_health','final_terminal','probe_report','probe_terminal','probe_saved_review'])
    if pending:need(p['CPU_run_authorized'] is False,'Pending actual fixedfinal/data forbids NN')
    else:
        r=read(checked(p['inputs']['final_result']));hh=read(checked(p['inputs']['final_health']));tp=read(checked(p['inputs']['final_training_protocol']));terminal=read(checked(p['inputs']['final_terminal']))
        need(r['paired_round']==1000 and r['shared_WM_updates']==1100 and r['status']=='COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION','Fixed offline1000, not best checkpoint')
        need(hh['checkpoint']==r['checkpoint'] and hh['paired_round']==1000 and hh['shared_WM_updates']==1100 and hh['checks'] and all(q is True for q in hh['checks'].values()),'Final serialized health')
        need(terminal['returncode']==0 and terminal['protocol']==p['inputs']['final_training_protocol'] and terminal['source']==tp['sources']['trainer'],'Actual final terminal')
        need(tp['sources']['trainer']==p['sources']['paired_trainer'] and tp['sources']['WM_trainer']==p['sources']['WM_trainer'] and tp['inputs']['geometry_catalog']==wp['inputs']['geometry_catalog']==p['inputs']['geometry_catalog'] and tp['inputs']['task_bundle']==p['inputs']['task_bundle'],'Exact final/warm frozen geometry/task/source lineage')
        checked(r['checkpoint']);report=read(checked(p['inputs']['probe_report']))
        need(report['schema']=='phone_tripod_heldout_action_response_result:1' and report['data_role']=='ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE','Heldout response dataset role, never fit')
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==H and not out.exists(),'Fresh output, no retry')
    return p,warm,out,pending

def load_probe(report):
    """Owner's actual saved schema is fixed before metadata protocol freeze."""
    rows=read(checked(report['branches']));need(len(rows)==328,'All328 planned rows, including unavailable')
    barrier=read(checked(report['baseline_barrier']));need(barrier['all8_passed'] is True and barrier['baseline_count']==8 and barrier['new_sides_executed_before_barrier']==0,'All8 baseline barrier before any side')
    fm=report['field_map']
    d=dict(x=np.full((328,58),np.nan),u=np.full((328,20),np.nan),y=np.full((328,58),np.nan),J=np.full((328,22,6),np.nan),Fn=np.full((328,22),np.nan),cov=np.full((328,22),np.nan),triple=np.full(328,np.nan),delta=np.full((328,52),np.nan),saved_head=np.full((328,178),np.nan),h=np.full(328,np.nan),n=np.zeros(328,np.int64),available=np.zeros(328,bool),known=np.zeros(328,bool))
    ids={r['branch_id']:i for i,r in enumerate(rows)};need(len(ids)==328,'Unique branch identities');bases={}
    for i,r in enumerate(rows):
        cid=int(r['context_id']);need(0<=cid<8,'Exact8contexts')
        if r['sign']==0:need(r['axis'] is None and cid not in bases,'One baseline percontext');bases[cid]=i
        if r['output'] is None:need(r['status']=='UNAVAILABLE','Unavailable keeps empty output, no invented source data');continue
        a=npz(r['output']);d['available'][i]=True;d['known'][i]=r['labels_known']
        for key,field in [('x','x58'),('u','u20'),('y','y58'),('J','impulse22x6'),('Fn','meanFn22'),('cov','coverage22'),('triple','triple_coverage'),('delta','delta52'),('saved_head','heads178')]:d[key][i]=a[fm[field]]
        need(bool(a[fm['labels_known']])==r['labels_known'] and bool(a[fm['complete_H']])==r['complete_H20ms'] and bool(a[fm['finite']])==r['finite'] and bool(a[fm['no_warning']])==r['no_warning'],'Exact saved row/mask ABI')
        d['h'][i]=r['dt_s'];d['n'][i]=r['n_substeps']
        need(np.isfinite(d['x'][i]).all() and np.isfinite(d['u'][i]).all() and d['h'][i]*d['n'][i]==.02,'Available actual current input20ms')
        if d['known'][i]:need(all(np.isfinite(d[k][i]).all() for k in ['y','J','Fn','cov','triple','delta']),'Known native H labels finite')
    need(set(bases)==set(range(8)) and all(d['available'][b] and d['known'][b] for b in bases.values()),'All8 complete baselines')
    baseline=np.array([bases[int(r['context_id'])] for r in rows],np.int64);side=np.array([i for i,r in enumerate(rows) if r['sign']!=0],np.int64)
    need(len(side)==320,'All20signsides percontext retained')
    for cid in range(8):need({(rows[i]['axis'],rows[i]['sign']) for i in side if rows[i]['context_id']==cid}=={(j,s) for j in range(20) for s in [-1,1]},'No axis or sign selection')
    for i in np.flatnonzero(d['available']):
        b=baseline[i];need(d['x'][i].tobytes()==d['x'][b].tobytes() and d['h'][i]==d['h'][b] and d['n'][i]==d['n'][b],'H1 samevisible current state/time; hidden remains unknown toWM')
        if rows[i]['sign']:
            j=rows[i]['axis'];off=np.arange(20)!=j;need(d['u'][i,off].tobytes()==d['u'][b,off].tobytes(),'Actual other controls unchanged')
    bound=report['actuator_bounds_source'];bounds=np.array(read(checked(bound['task_bundle']))[bound['key']],dtype=np.float64)
    need(bounds.shape==(20,2) and np.isfinite(bounds).all(),'Actual declared native actuator bounds')
    d.update(rows=rows,baseline=baseline,baseline_rows=np.array([bases[c] for c in range(8)],np.int64),sides=side,ctrl_bounds=bounds,pair_ids=np.array(report['canonical_pair_geom_ids']))
    return d

def infer(model,features,norm,physical,trainer,codec,torch,counters):
    # Explicit physicalF64 -> normalizeF64 -> NN FP32, same frozen route.
    normed=((features-norm['input_mean'])/norm['input_std']).float();raw=model(normed);counters['NN_FW']+=1
    head=raw[:,:178].double()*norm['head_std']+norm['head_mean'];y=codec.decode_physical_heads(features[:,:58],trainer.heads_from_physical(head),features[:,78]*features[:,79],physical['mass'],physical['I'],physical['g'])
    Fn=torch.nn.functional.softplus(raw[:,178:200]).double();cov=torch.sigmoid(raw[:,200:222].double())
    return dict(prediction58=y,head178=head,meanFn22=Fn,coverage22=cov,normalized366_FP32=normed,raw222_FP32=raw)
def tangent52(a,b,scalar):
    return np.r_[a[:43]-b[:43],scalar.log_world(a[43:52].reshape(3,3)@b[43:52].reshape(3,3).T),a[52:]-b[52:]]
def matrix_blocks(a):return dict(q=a[...,:20],qd=a[...,20:40],COM=a[...,40:43],world_dR=a[...,43:46],vCOM=a[...,46:49],omega=a[...,49:52])
def ratio_stats(error,native,stats):
    z=stats.basic(error);ref=stats.basic(native);return dict(error=z,native=ref,errorRMS_over_nativeRMS=(z['RMS']/ref['RMS']) if z['RMS'] is not None and ref['RMS'] not in (None,0.) else None,zero_response_prior_RMS=ref['RMS'])

def model_AD_FD(model,F,pred,d,norm,physical,trainer,codec,torch,counters):
    """16Jac API total, each reverse-vectorized274cotangents, not scalarAD.

    Baseline x/current geometry is action independent in H1. Its cached
    features stay constant; only actual control20 enters the derivative.
    Left-world local SO3 tangent around each model baseline is used for AD.
    """
    need(not torch.is_inference_mode_enabled(),'AD outside inference_mode')
    AD=np.full((8,274,20),np.nan);fd=np.full_like(AD,np.nan);valid=np.zeros((8,20),bool);du=np.full((8,20),np.nan);enc=np.full_like(du,np.nan)
    baseline_rows=d['baseline_rows'];lookup={int(r):i for i,r in enumerate(np.flatnonzero(d['available']))}
    baselineF=np.array([F[lookup[int(r)]] for r in baseline_rows]);baseR=pred['prediction58'][baseline_rows,43:52].reshape(8,3,3)
    for c in range(8):
        f=torch.tensor(baselineF[c],dtype=torch.float64);R0=torch.tensor(baseR[c],dtype=torch.float64)
        def function(u):
            feature=torch.cat([f[:58],u,f[78:]]).unsqueeze(0);v=infer(model,feature,norm,physical,trainer,codec,torch,counters);y=v['prediction58'][0];R=y[43:52].reshape(3,3)@R0.T
            local=.5*torch.stack([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
            return torch.cat([y[:43],local,y[52:],v['head178'][0],v['meanFn22'][0],v['coverage22'][0]])
        AD[c]=torch.autograd.functional.jacobian(function,f[58:78].clone().requires_grad_(True),vectorize=True,create_graph=False,strategy='reverse-mode').detach().numpy();counters['Jacobian_API']+=1;counters['batched_reverse_VJP']+=1;counters['reverse_cotangent_directions']+=274
    cases=[];ids=[];eps=.0001
    for c in range(8):
        for j in range(20):
            u=baselineF[c,58+j]
            if u-eps<d['ctrl_bounds'][j,0] or u+eps>d['ctrl_bounds'][j,1]:continue
            for sign in [-1,1]:v=baselineF[c].copy();v[58+j]+=sign*eps;cases.append(v);ids.append((c,j,sign))
    if cases:
        with torch.no_grad():v=infer(model,torch.tensor(np.array(cases),dtype=torch.float64),norm,physical,trainer,codec,torch,counters)
        values={k:x.detach().numpy() for k,x in v.items()};index={k:i for i,k in enumerate(ids)}
        for c in range(8):
            for j in range(20):
                if (c,j,-1) not in index:continue
                lo,hi=index[c,j,-1],index[c,j,1];den=cases[hi][58+j]-cases[lo][58+j];encoded=(float(values['normalized366_FP32'][hi,58+j])-float(values['normalized366_FP32'][lo,58+j]))*float(norm['input_std'][58+j]);du[c,j]=den;enc[c,j]=encoded
                state=tangent52(values['prediction58'][hi],values['prediction58'][lo],codec.scalar);vector=np.r_[state,values['head178'][hi]-values['head178'][lo],values['meanFn22'][hi]-values['meanFn22'][lo],values['coverage22'][hi]-values['coverage22'][lo]]
                # Retain raw finite-F64 denominator and separately flag NN collapse.
                fd[c,:,j]=vector/den;valid[c,j]=True
    return dict(AD274_per_u20=AD,AD_axis_finite=np.isfinite(AD).all(1),FD274_per_actual_du=fd,FD_bound_defined=valid,FD_actual_du_F64=du,FD_encoded_du_physical_F64=enc,FD_encoded_defined=valid&(enc!=0),FD_encoded_collapse=valid&(enc==0))

def run(protocol):
    p,warm,out,pending=metadata_preflight(protocol);need(p['CPU_run_authorized'] is True and not pending,'Actual Root final/data authorization')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out.mkdir();write(out/'protocol.json',p);start=time.perf_counter();phase='data';counts=dict(own_CPU_loads=0,WM_constructors=0,NN_FW=0,Jacobian_API=0,batched_reverse_VJP=0,reverse_cotangent_directions=0,native_GPU_optimizer_updates=0)
    try:
        report=read(checked(p['inputs']['probe_report']));d=load_probe(report)
        trainer=mod('response_fixed_WM',checked(p['sources']['WM_trainer']));codec=mod('response_fixed_codec',checked(p['sources']['structured_codec']));stats=mod('response_frozen_statistics',checked(p['sources']['statistics']));gap=mod('response_current_geometry',checked(p['sources']['directional']));geometry=gap.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']))
        final=read(checked(p['inputs']['final_result']));tp=read(checked(p['inputs']['final_training_protocol']));wp=read(checked(p['inputs']['warm_training_protocol']))
        loaded={role:torch.load(checked(v),map_location='cpu',weights_only=False) for role,v in [(ROLES[0],warm['checkpoint']),(ROLES[1],final['checkpoint'])]};counts['own_CPU_loads']=2
        ck=loaded[ROLES[0]];need(ck['schema']==trainer.ARTIFACT and ck['iteration']==5000 and ck['recipe']==wp['recipe'] and ck['sources']==wp['sources'] and ck['inputs']==wp['inputs'],'Exact warm5000 payload')
        fk=loaded[ROLES[1]];need(fk['schema']=='phone_tripod_sharedWM_A1A2_training_checkpoint:1' and fk['paired_round']==1000 and fk['shared_WM_updates']==1100 and fk['protocol']==p['inputs']['final_training_protocol'] and fk['recipe']==tp['recipe'] and fk['inputs']==tp['inputs'] and fk['sources']==tp['sources'],'Exact fixed final1000 payload')
        need(trainer.tree_equal(ck['normalizers'],fk['WM_normalizers'],torch) and trainer.tree_equal(ck['physical_context'],fk['physical_context'],torch),'Same frozen model normalizers/physical context')
        idx=np.flatnonzero(d['available']);F,cache=trainer.precompute_features(dict(x=d['x'][idx],u=d['u'][idx],h=d['h'][idx],n=d['n'][idx]),geometry,torch);rng0=trainer.rng(torch);all_metrics=[];response_metrics=[];invariance={}
        for role in ROLES:
            phase=role;ck=loaded[role];norm=ck['normalizers'] if role==ROLES[0] else ck['WM_normalizers'];physical=ck['physical_context'];state=ck['model_state'] if role==ROLES[0] else ck['shared_WM']
            with torch.random.fork_rng(devices=[]):model=trainer.make_model(torch)
            model.load_state_dict(state,strict=True);model.eval();model.requires_grad_(False);counts['WM_constructors']+=1;digest=trainer.tensor_digest(model.state_dict(),torch);ndigest=trainer.tensor_digest(norm,torch)
            with torch.no_grad():values=infer(model,torch.tensor(F,dtype=torch.float64),norm,physical,trainer,codec,torch,counts)
            pred={k:np.full((328,)+tuple(v.shape[1:]),np.nan,dtype=v.detach().numpy().dtype) for k,v in values.items()}
            for k,v in values.items():pred[k][idx]=v.detach().numpy()
            need(all(np.isfinite(v[idx]).all() for v in pred.values()),'All available predictions finite, no model-based filtering')
            truth_head=np.full((328,178),np.nan);known=d['known'];kk=np.flatnonzero(known)
            truth_head[kk]=stats.flat_head(codec.encode_observed_numpy(d['x'][kk],d['y'][kk],d['J'][kk],d['h'][kk]*d['n'][kk],physical['g'].numpy()))
            need(truth_head[kk].tobytes()==d['saved_head'][kk].tobytes(),'Native saved physical178 exact same frozen encoder')
            modelJ=pred['head178'][:,46:].reshape(328,22,6)
            cohorts={'ALL_KNOWN':known,'BASELINES':known&np.array([r['sign']==0 for r in d['rows']]),'ALL_SIDES':known&np.array([r['sign']!=0 for r in d['rows']])}
            for c in range(8):cohorts['CONTEXT_'+str(c)]=known&np.array([r['context_id']==c for r in d['rows']])
            for j in range(20):cohorts['AXIS_'+str(j)]=known&np.array([r['axis']==j for r in d['rows']])
            for key,values in [('family_id',[12,15]),('dt_s',[.0003125,.00015625]),('step_k',[0,3])]:
                for value in values:cohorts[key+'_'+str(value)]=known&np.array([d['rows'][d['baseline'][i]][key]==value for i in range(328)])
            for name,mask in cohorts.items():
                ii=np.flatnonzero(mask);all_metrics.append(dict(model=role,cohort=name,rows=len(ii),state=stats.state_stats(pred['prediction58'][ii],d['y'][ii],codec.scalar) if len(ii) else None,head178_blocks=[stats.basic((pred['head178'][ii]-truth_head[ii])[:,b]) for b in trainer.physical_blocks()],pair22=stats.pair_stats(modelJ[ii],d['J'][ii],pred['meanFn22'][ii],d['Fn'][ii],pred['coverage22'][ii],d['cov'][ii]) if len(ii) else None,true_simultaneous_triple_coverage=stats.basic(d['triple'][ii]),model_simultaneous_triple_prediction=None))
            s=d['sides'];b=d['baseline'][s];j=np.array([d['rows'][i]['axis'] for i in s]);valid=known[s]&known[b];du=d['u'][s,j]-d['u'][b,j]
            normdu=(pred['normalized366_FP32'][s,58+j].astype(np.float64)-pred['normalized366_FP32'][b,58+j].astype(np.float64))*norm['input_std'][58+torch.tensor(j)].numpy();encoded=valid&(normdu!=0)
            native178=truth_head[s]-truth_head[b];model178=pred['head178'][s]-pred['head178'][b];native52=np.full((320,52),np.nan);model52=np.full_like(native52,np.nan)
            for k in np.flatnonzero(valid):native52[k]=tangent52(d['y'][s[k]],d['y'][b[k]],codec.scalar);model52[k]=tangent52(pred['prediction58'][s[k]],pred['prediction58'][b[k]],codec.scalar)
            defined=valid&(du!=0);slopes={}
            for label,value,den,mask in [('native178_slope',native178,du,defined),('model178_slope',model178,du,defined),('model178_encoded_slope',model178,normdu,encoded),('native52_slope',native52,du,defined),('model52_slope',model52,du,defined)]:z=np.full_like(value,np.nan);z[mask]=value[mask]/den[mask,None];slopes[label]=z
            signalF=np.full((320,22),-1,np.int8);signalM=signalF.copy();object_signal=np.full(320,-1,np.int8)
            signalF[valid]=stats.signal_classes(d['J'][s[valid],:,:3],d['J'][b[valid],:,:3]);signalM[valid]=stats.signal_classes(d['J'][s[valid],:,3:],d['J'][b[valid],:,3:]);object_signal[valid]=stats.signal_classes(d['y'][s[valid],40:],d['y'][b[valid],40:])
            other_features=np.ones((320,366),bool);other_features[np.arange(320),58+j]=False
            need(pred['normalized366_FP32'][s][other_features&valid[:,None]].tobytes()==pred['normalized366_FP32'][b][other_features&valid[:,None]].tobytes(),'Exact actual NN offaxis normalized input rawbytes equality')
            central=[]
            for c in range(8):
                for axis in range(20):
                    minus=next(i for i,r in enumerate(d['rows']) if r['context_id']==c and r['axis']==axis and r['sign']==-1);plus=next(i for i,r in enumerate(d['rows']) if r['context_id']==c and r['axis']==axis and r['sign']==1);ok=bool(known[minus] and known[plus]);den=float(d['u'][plus,axis]-d['u'][minus,axis]) if ok else None
                    central.append(dict(context=c,axis=axis,minus_row=minus,plus_row=plus,defined=ok and den!=0,actual_denom_F64=den,native178=(truth_head[plus]-truth_head[minus]).tolist() if ok else None,model178=(pred['head178'][plus]-pred['head178'][minus]).tolist() if ok else None))
            for axis in [None,*range(20)]:
                mask=valid if axis is None else valid&(j==axis);response_metrics.append(dict(model=role,axis=axis,count=int(mask.sum()),state52={k:ratio_stats(matrix_blocks(model52-native52)[k][mask],matrix_blocks(native52)[k][mask],stats) for k in matrix_blocks(native52)},head178_blocks=[ratio_stats((model178-native178)[mask][:,ix],native178[mask][:,ix],stats) for ix in trainer.physical_blocks()]))
            ad=model_AD_FD(model,F,pred,d,norm,physical,trainer,codec,torch,counts)
            selfcheck=[]
            ranges=[('state_q',0,20),('state_qd',20,40),('state_COM',40,43),('state_local_world_rotation',43,46),('state_vCOM',46,49),('state_omega',49,52),('physical178',52,230),('meanFn22',230,252),('coverage22',252,274)]
            ranges=[(name,np.arange(lo,hi)) for name,lo,hi in ranges if name!='physical178']+[(name,52+ix) for name,ix in zip(['head_q_rad','head_qd_rad_s','head_COM_res_m','head_dR_rad','head_Jforce_Ns','head_Jmoment_Nms'],trainer.physical_blocks())]
            for name,ix in ranges:
                mask=ad['FD_bound_defined']&ad['AD_axis_finite'];aa=np.moveaxis(ad['AD274_per_u20'][:,ix],-1,1)[mask];ff=np.moveaxis(ad['FD274_per_actual_du'][:,ix],-1,1)[mask]
                error=stats.basic(ff-aa);ref=stats.basic(aa);selfcheck.append(dict(block=name,bound_available_axes=int(mask.sum()),encoded_collapse_axes=int(ad['FD_encoded_collapse'].sum()),error=error,AD=ref,relativeL2=float(np.linalg.norm(ff-aa)/np.linalg.norm(aa)) if np.linalg.norm(aa)!=0 else None,scope='Same-model finiteFP32 FD vs symbolic AD only; no native infinitesimal Jacobian or geometry-q validation'))
            np.savez_compressed(out/(role+'_all328_predictions.npz'),**pred,native_head178=truth_head,native_known=known,available=d['available'])
            np.savez_compressed(out/(role+'_all320_native_responses.npz'),side_row=s,baseline_row=b,axis=j,known=valid,actual_du_F64=du,raw_control_FP32_endpoint_du_F64=d['u'][s,j].astype(np.float32).astype(np.float64)-d['u'][b,j].astype(np.float32).astype(np.float64),encoded_du_physical_F64=normdu,encoded_defined=encoded,Actor_EMA_reachable_descriptor=np.array([d['rows'][i]['perturbation']['Actor_EMA_reachable_descriptor'] for i in s],bool),native178=native178,model178=model178,native52=native52,model52=model52,native_J22_response=d['J'][s]-d['J'][b],model_J22_response=modelJ[s]-modelJ[b],native_meanFn22_response=d['Fn'][s]-d['Fn'][b],model_meanFn22_response=pred['meanFn22'][s]-pred['meanFn22'][b],native_coverage22_response=d['cov'][s]-d['cov'][b],model_coverage22_response=pred['coverage22'][s]-pred['coverage22'][b],native_triple_coverage_response=d['triple'][s]-d['triple'][b],native_force_signal_class=signalF,native_moment_signal_class=signalM,native_object_signal_class=object_signal,**slopes)
            np.savez_compressed(out/(role+'_model_only_AD_FD.npz'),**ad);write(out/(role+'_model_only_AD_FD_statistics.json'),selfcheck);write(out/(role+'_all160_central_records.json'),central)
            invariance[role]=dict(weights=digest==trainer.tensor_digest(model.state_dict(),torch),normalizers=ndigest==trainer.tensor_digest(norm,torch),parameter_grad_None=all(not q.requires_grad and q.grad is None for q in model.parameters()))
        need(all(all(v.values()) for v in invariance.values()) and trainer.tree_equal(rng0,trainer.rng(torch),torch) and not torch.cuda.is_initialized(),'Weights/norm/RNG/CPU invariance')
        need(counts['NN_FW']<=20 and counts['Jacobian_API']==16 and counts['batched_reverse_VJP']==16 and counts['reverse_cotangent_directions']==4384,'Exact declared NN/AD bound')
        write(out/'endpoint_statistics.json',all_metrics);write(out/'finite_response_statistics.json',response_metrics);write(out/'all328_native_identities.json',d['rows'])
        r=dict(schema='phone_tripod_fixed_WM_action_response_evaluation_result:1',status='COMPLETED_FIXED_TWO_MODELS_ALL20_AXES_NO_SELECTION',protocol=pin(protocol),source=pin(__file__),counts=counts,roles=ROLES,native_known_rows=int(d['known'].sum()),unavailable_rows=int((~d['available']).sum()),available_unknown_or_partial_rows=int((d['available']&~d['known']).sum()),invariance=invariance,endpoint_statistics=pin(out/'endpoint_statistics.json'),finite_response_statistics=pin(out/'finite_response_statistics.json'),elapsed_s=time.perf_counter()-start,scope=RULES,physical_gradient_qualified=False,policy_benefit_qualified=False)
        write(out/'report.json',r);print(json.dumps(dict(report=pin(out/'report.json'),counts=counts)),flush=True);return 0
    except BaseException as e:write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),counts=counts,automatic_retry=False));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--protocol',type=Path,required=True);a.add_argument('--run',action='store_true');args=a.parse_args()
    if args.run:raise SystemExit(run(args.protocol))
    p,w,out,pending=metadata_preflight(args.protocol);print(json.dumps(dict(status='METADATA_PENDING_FINAL_OR_DATA' if pending else 'METADATA_READY_FOR_ROOT_REVIEW',Torch_imported='torch' in sys.modules,NN_native_calls=0,output=str(out))))
