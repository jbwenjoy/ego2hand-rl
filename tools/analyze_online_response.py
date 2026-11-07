"""Fixed offline1000 versus online300 on all permanently held-out responses.

CPU inference/input AD only. No fitting, simulator, checkpoint selection,
direction/axis filtering or normalizer updates. Reuses shared WM and decoder.
"""
from pathlib import Path
import argparse,json,os,sys,time,traceback
import numpy as np
from evaluate_online import preflight

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning import online_contract as C,world_model as W,runner as R
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime import state_codec as scalar


def load_probe(report,registry):
    C.need(report['data_role']=='ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE' and report['permanent_training_eligible'] is False,'Permanent heldout role')
    rows=C.read(registry.resolve_pin(report['branches']));C.need(len(rows)==328,'All328 rows')
    barrier=C.read(registry.resolve_pin(report['baseline_barrier']))
    C.need(barrier['all8_passed'] is True and barrier['baseline_count']==8 and barrier['new_sides_executed_before_barrier']==0,'Original all-baseline replay barrier')
    d={k:np.full(shape,np.nan) for k,shape in dict(x=(328,58),u=(328,20),y=(328,58),J=(328,22,6),Fn=(328,22),cov=(328,22),triple=(328,),saved_head=(328,178),h=(328,)).items()}
    d.update(n=np.zeros(328,np.int64),available=np.zeros(328,bool),known=np.zeros(328,bool));bases={};fm=report['field_map']
    C.need(len({r['branch_id'] for r in rows})==328,'Unique branch IDs')
    for i,row in enumerate(rows):
        cid=int(row['context_id']);C.need(0<=cid<8,'Eight fixed contexts')
        if row['sign']==0:C.need(cid not in bases and row['axis'] is None,'Unique baseline per context');bases[cid]=i
        if row['output'] is None:C.need(row['status']=='UNAVAILABLE','Unavailable row retained');continue
        with np.load(registry.resolve_pin(row['output']),allow_pickle=False) as z:
            for key,field in [('x','x58'),('u','u20'),('y','y58'),('J','impulse22x6'),('Fn','meanFn22'),('cov','coverage22'),('triple','triple_coverage'),('saved_head','heads178')]:d[key][i]=z[fm[field]]
            for field,value in [('labels_known',row['labels_known']),('complete_H',row['complete_H20ms']),('finite',row['finite']),('no_warning',row['no_warning'])]:C.need(bool(z[fm[field]])==value,'Saved mask '+field)
        d['available'][i]=True;d['known'][i]=row['labels_known'];d['h'][i]=row['dt_s'];d['n'][i]=row['n_substeps']
        C.need(np.isfinite(d['x'][i]).all() and np.isfinite(d['u'][i]).all() and d['h'][i]*d['n'][i]==.02,'Known current inputs')
        if d['known'][i]:C.need(all(np.isfinite(d[k][i]).all() for k in ('y','J','Fn','cov','triple','saved_head')),'Known finite native labels')
    C.need(set(bases)==set(range(8)) and all(d['available'][b] and d['known'][b] for b in bases.values()),'All eight complete baselines')
    baseline=np.array([bases[int(r['context_id'])] for r in rows]);sides=np.array([i for i,r in enumerate(rows) if r['sign']!=0]);C.need(len(sides)==320,'All320 sides')
    for cid in range(8):C.need({(rows[i]['axis'],rows[i]['sign']) for i in sides if rows[i]['context_id']==cid}=={(j,s) for j in range(20) for s in (-1,1)},'All20 axes/both signs')
    for i in np.flatnonzero(d['available']):
        b=baseline[i];C.need(d['x'][i].tobytes()==d['x'][b].tobytes() and d['h'][i]==d['h'][b] and d['n'][i]==d['n'][b],'Same actual H1 state/time')
        if rows[i]['sign']:
            axis=rows[i]['axis'];off=np.arange(20)!=axis;C.need(d['u'][i,off].tobytes()==d['u'][b,off].tobytes(),'Only named action coordinate differs')
    bounds_source=report['actuator_bounds_source'];bounds=np.array(C.read(registry.resolve_pin(bounds_source['task_bundle']))[bounds_source['key']])
    d.update(rows=rows,baseline=baseline,sides=sides,baseline_rows=np.array([bases[c] for c in range(8)]),bounds=bounds)
    return d


def infer(model,F,norm,physical,torch,counts):
    raw=model(W.normalize_features(F,norm));counts['NN_FW']+=1
    decoded=W.decode_output(raw,F[:,:58],F[:,78],F[:,79].long(),norm,physical)
    return dict(y=decoded['next_state58'],head=decoded['physical178'],Fn=decoded['meanFn22_N'].double(),cov=decoded['coverage_probability22_F64'],encoded=W.normalize_features(F,norm))


def tangent52(a,b):return np.r_[a[:43]-b[:43],scalar.log_world(a[43:52].reshape(3,3)@b[43:52].reshape(3,3).T),a[52:]-b[52:]]


def rms(a):return float(np.sqrt(np.mean(np.square(a)))) if np.asarray(a).size else None


def blocks52():return dict(q_rad=np.arange(20),qd_rad_s=np.arange(20,40),COM_m=np.arange(40,43),world_rotation_rad=np.arange(43,46),vCOM_m_s=np.arange(46,49),omega_rad_s=np.arange(49,52))


def response_stats(native,pred,known,axes):
    output=[]
    for axis in [None,*range(20)]:
        mask=known if axis is None else known&(axes==axis);entry=dict(axis=axis,rows=int(mask.sum()),blocks={})
        for block,idx in blocks52().items():
            ref=rms(native[mask][:,idx]);error=rms((pred-native)[mask][:,idx]);entry['blocks'][block]=dict(component_error_RMS=error,zero_response_prior_RMS=ref,error_over_zero_prior=error/ref if ref not in (None,0) else None)
        output.append(entry)
    return output


def ad_fd(model,F,pred,d,norm,physical,torch,counts):
    AD=np.full((8,274,20),np.nan);FD=np.full_like(AD,np.nan);valid=np.zeros((8,20),bool);encoded=np.full((8,20),np.nan)
    baseF=F[d['baseline_rows']];baseR=pred['y'][d['baseline_rows'],43:52].reshape(8,3,3)
    for context in range(8):
        f=torch.tensor(baseF[context],dtype=torch.float64);anchor=torch.tensor(baseR[context],dtype=torch.float64)
        def function(u):
            v=infer(model,torch.cat([f[:58],u,f[78:]]).unsqueeze(0),norm,physical,torch,counts);y=v['y'][0];rel=y[43:52].reshape(3,3)@anchor.T
            skew=.5*torch.stack([rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1]])
            return torch.cat([y[:43],skew,y[52:],v['head'][0],v['Fn'][0],v['cov'][0]])
        AD[context]=torch.autograd.functional.jacobian(function,f[58:78].clone().requires_grad_(True),vectorize=True,create_graph=False,strategy='reverse-mode').detach().numpy()
        counts['Jacobian_API']+=1;counts['reverse_cotangent_directions']+=274
    cases=[];identity=[];eps=1e-4
    for context in range(8):
        for axis in range(20):
            u=baseF[context,58+axis]
            if u-eps<d['bounds'][axis,0] or u+eps>d['bounds'][axis,1]:continue
            for sign in (-1,1):v=baseF[context].copy();v[58+axis]+=sign*eps;cases.append(v);identity.append((context,axis,sign))
    if cases:
        with torch.no_grad():values={k:v.detach().numpy() for k,v in infer(model,torch.tensor(np.array(cases),dtype=torch.float64),norm,physical,torch,counts).items()}
        lookup={key:i for i,key in enumerate(identity)}
        for context in range(8):
            for axis in range(20):
                if (context,axis,-1) not in lookup:continue
                lo,hi=lookup[context,axis,-1],lookup[context,axis,1];du=cases[hi][58+axis]-cases[lo][58+axis]
                FD[context,:,axis]=np.r_[tangent52(values['y'][hi],values['y'][lo]),values['head'][hi]-values['head'][lo],values['Fn'][hi]-values['Fn'][lo],values['cov'][hi]-values['cov'][lo]]/du
                encoded[context,axis]=(float(values['encoded'][hi,58+axis])-float(values['encoded'][lo,58+axis]))*float(norm['input_std'][58+axis]);valid[context,axis]=True
    stats=[]
    ranges=blocks52()|{name:52+idx for name,idx in zip(['head_q_rad','head_qd_rad_s','head_COM_res_m','head_dR_rad','head_Jforce_Ns','head_Jmoment_Nms'],W.physical_blocks())}|dict(meanFn22_N=np.arange(230,252),coverage22=np.arange(252,274))
    finite=np.isfinite(AD).all(1);mask=valid&finite
    for name,idx in ranges.items():
        a=np.moveaxis(AD[:,idx],-1,1)[mask];b=np.moveaxis(FD[:,idx],-1,1)[mask];den=np.linalg.norm(a)
        stats.append(dict(block=name,available_axes=int(mask.sum()),FD_minus_AD_RMS=rms(b-a),relative_L2=float(np.linalg.norm(b-a)/den) if den else None))
    return dict(AD274_per_u20=AD,FD274_per_u20=FD,bound_available=valid,AD_finite=finite,encoded_du_physical=encoded),dict(blocks=stats,encoded_collapse_axes=int((valid&(encoded==0)).sum()),scope='Same-model H1 input AD/FD consistency only. No native infinitesimal Jacobian or H2 geometry-q qualification.')


def run(args):
    path,c,registry,root,result,health,*_,out=preflight(args)
    C.need(registry.entries['action_response'].role=='ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE','Registry forbids fitting this data')
    probe=C.read(registry.resolve('action_response'));d=load_probe(probe,registry)
    C.need(d['available'].all(),'This fixed328-row diagnostic expects all saved branches; never replace unavailable data')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    from supermimic_research.runtime import structured
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    counts=dict(own_CPU_checkpoint_reads=0,WM_constructors=0,NN_FW=0,Jacobian_API=0,reverse_cotangent_directions=0,optimizer_steps=0,native_steps=0,GPU_calls=0)
    out.mkdir(parents=True);tick=time.perf_counter();phase='load'
    C.write(out/'intent.json',dict(tool=file_pin(__file__),result=file_pin(args.result),health=result['health'],checkpoint=result['checkpoint'],root_checkpoint=root['checkpoint'],probe=registry.pin('action_response'),all_axes=True,models=['offline1000','online300'],fitting=False,source_pins=C.source_pins()))
    try:
        old=torch.load(registry.resolve_pin(root['checkpoint']),map_location='cpu',weights_only=False);new=torch.load(registry.resolve_pin(result['checkpoint']),map_location='cpu',weights_only=False);counts['own_CPU_checkpoint_reads']=2
        C.validate_root_payload(old,root);C.validate_actor_payload(new,dict(kind='own_online',training_protocol=health['protocol']),health,root,c,C.source_pins())
        C.need(W.raw_tree_equal(old['WM_normalizers'],new['normalizers']) and W.raw_tree_equal(old['physical_context'],new['physical_context']),'Raw frozen model context')
        geometry=R._geometry(registry,c);F,cache=W.precompute_features(dict(x=d['x'],u=d['u'],h=d['h'],n=d['n']),geometry,chunk_rows=4)
        summaries=[];rng=W.capture_rng()
        for role,ck,norm in [('offline1000',old,old['WM_normalizers']),('online300',new,new['normalizers'])]:
            phase=role
            with torch.random.fork_rng(devices=[]):model=W.make_model(W.WorldModelConfig(hidden=(128,128)),device='cpu')
            counts['WM_constructors']+=1;model.load_state_dict(ck['shared_WM']);model.eval();model.requires_grad_(False);physical=ck['physical_context'];digest=W.tensor_digest(model.state_dict());norm_digest=W.tensor_digest(norm)
            with torch.no_grad():pred={k:v.detach().numpy() for k,v in infer(model,torch.tensor(F,dtype=torch.float64),norm,physical,torch,counts).items()}
            C.need(all(np.isfinite(v).all() for v in pred.values()),'No model-dependent finite row filtering')
            truth=structured.encode_observed_numpy(d['x'],d['y'],d['J'],d['h']*d['n'],physical['g'].numpy())
            truth_head=np.c_[truth['joint_delta40'],truth['COM_ballistic_residual3_m'],truth['world_rotation_delta3_rad'],d['J'].reshape(-1,132)]
            C.need(truth_head.tobytes()==d['saved_head'].tobytes(),'All native heads match saved encoder bytes')
            side=d['sides'];base=d['baseline'][side];axes=np.array([d['rows'][i]['axis'] for i in side]);known=d['known'][side]&d['known'][base]
            native52=np.full((320,52),np.nan);model52=np.full_like(native52,np.nan)
            for i in np.flatnonzero(known):native52[i]=tangent52(d['y'][side[i]],d['y'][base[i]]);model52[i]=tangent52(pred['y'][side[i]],pred['y'][base[i]])
            du=d['u'][side,axes]-d['u'][base,axes];encoded_du=(pred['encoded'][side,58+axes].astype(np.float64)-pred['encoded'][base,58+axes].astype(np.float64))*norm['input_std'][58+torch.tensor(axes)].numpy()
            native_J=d['J'][side]-d['J'][base];model_J=pred['head'][side,46:].reshape(-1,22,6)-pred['head'][base,46:].reshape(-1,22,6)
            other=np.ones((320,366),bool);other[np.arange(320),58+axes]=False
            C.need(pred['encoded'][side][other&known[:,None]].tobytes()==pred['encoded'][base][other&known[:,None]].tobytes(),'Off-axis normalized input bytes unchanged')
            zero=known&np.all(native52[:,40:43]==0,axis=1)
            endpoint_COM=rms((pred['y']-d['y'])[d['known'],40:43]);endpoint_rotation=np.array([np.linalg.norm(scalar.log_world(pred['y'][i,43:52].reshape(3,3)@d['y'][i,43:52].reshape(3,3).T)) for i in np.flatnonzero(d['known'])])
            head_stats=[dict(block=name,component_response_error_RMS=rms(((pred['head'][side]-pred['head'][base])-(truth_head[side]-truth_head[base]))[known][:,idx]),zero_response_prior_RMS=rms((truth_head[side]-truth_head[base])[known][:,idx])) for name,idx in zip(['q_rad','qd_rad_s','COM_res_m','dR_rad','Jforce_Ns','Jmoment_Nms'],W.physical_blocks())]
            ad,adstats=ad_fd(model,F,pred,d,norm,physical,torch,counts)
            with (out/(role+'_predictions.npz')).open('xb') as stream:np.savez_compressed(stream,**pred)
            with (out/(role+'_responses.npz')).open('xb') as stream:np.savez_compressed(stream,native52=native52,model52=model52,side_row=side,baseline_row=base,axis=axes,known=known,actual_du=du,encoded_du=encoded_du,native_J=native_J,model_J=model_J,native_cov=d['cov'][side]-d['cov'][base],model_cov=pred['cov'][side]-pred['cov'][base],native_meanFn=d['Fn'][side]-d['Fn'][base],model_meanFn=pred['Fn'][side]-pred['Fn'][base],native_triple=d['triple'][side]-d['triple'][base])
            with (out/(role+'_AD_FD.npz')).open('xb') as stream:np.savez_compressed(stream,**ad)
            stat=dict(model=role,all328_endpoint_COM_component_RMS_m=endpoint_COM,all328_endpoint_rotation_RMS_rad=rms(endpoint_rotation),
                all_axis_responses=response_stats(native52,model52,known,axes),head_response_blocks=head_stats,
                true_COM_exact_zero_sides=int(zero.sum()),phantom_COM_vector_RMS_m=float(np.sqrt(np.mean(np.sum(model52[zero,40:43]**2,axis=1)))) if zero.any() else None,
                encoded_du_collapse=int((known&(encoded_du==0)).sum()),model_AD_FD=adstats,simultaneous_triple_prediction=None,
                invariant_weights=digest==W.tensor_digest(model.state_dict()),invariant_norms=norm_digest==W.tensor_digest(norm),parameter_grad_None=all(p.grad is None for p in model.parameters()))
            C.need(stat['invariant_weights'] and stat['invariant_norms'] and stat['parameter_grad_None'],'No model or norm fitting during diagnostics')
            C.write(out/(role+'_statistics.json'),stat);summaries.append(stat)
        C.need(counts['NN_FW']==20 and counts['Jacobian_API']==16 and counts['reverse_cotangent_directions']==4384,'Declared batched input-AD diagnostic budget')
        C.need(W.raw_tree_equal(rng,W.capture_rng()) and not torch.cuda.is_initialized(),'Read-only RNG and CPU execution')
        report=dict(schema='supermimic_research.fixed_online_response_diagnostics:1',status='COMPLETE_FIXED_MODELS_ALL320_SIDES_NO_SELECTION',models=summaries,counts=counts,
            probe=registry.pin('action_response'),known_native_rows=int(d['known'].sum()),available_rows=int(d['available'].sum()),geometry_cache=cache,elapsed_s=time.perf_counter()-tick,
            physical_gradient_qualified=False,policy_benefit_qualified=False,scope='Repeated known-lineage heldout diagnostics, never fitting or checkpoint selection. Same-model AD/FD is not native-gradient validation; marginal coverage cannot certify joint triple contact.')
        C.write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),counts=counts)),flush=True);return report
    except BaseException as error:C.write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),counts=counts,automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--zero-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:run(a)
    else:
        *_,out=preflight(a);print(json.dumps(dict(status='RESPONSE_METADATA_READY',output=str(out),Torch_imported='torch' in sys.modules)))
