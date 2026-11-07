"""Fixed WM AD and finite responses against new same-context native branches.

No neural parameter updates. Primary comparison uses the same F64 command step;
the model's own AD/finite comparison uses its actually represented F32 step.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,os,sys,time
import numpy as np

H=Path(__file__).resolve().parent
BLOCKS={'q20':(0,20),'qd20':(20,40),'COM3':(40,43),'world_rotation3':(43,46),
        'COM_velocity3':(46,49),'world_omega3':(49,52),'coverage2':(52,54),
        'log_min_force2':(54,56),'log_endpoint_force2':(56,58),'log_nonpad_force1':(58,59)}
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x
def write(p,j):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(j),f,indent=2,allow_nan=False)
def load_arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def comparison(predicted,actual):
    p=np.asarray(predicted,dtype=np.float64);a=np.asarray(actual,dtype=np.float64)
    if not p.size or not np.isfinite(p).all() or not np.isfinite(a).all():return None
    pn=float(np.linalg.norm(p));an=float(np.linalg.norm(a));err=float(np.linalg.norm(p-a))
    return dict(features=int(p.size),prediction_L2=pn,native_or_reference_L2=an,error_L2=err,
        RMSE=float(np.sqrt(np.mean((p-a)**2))),relative_L2=None if an==0 else err/an,
        cosine=None if pn==0 or an==0 else float(np.dot(p,a)/(pn*an)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    cfg=read(args.protocol);assert cfg['schema']=='phone_full20_WM_native_secant_comparison_protocol:1'
    for r in [*cfg['inputs'].values(),*cfg['source_pins'].values()]:checked(r)
    assert cfg['source_pins']['runner']==pin(__file__) and cfg['parameter_updates_authorized'] is False
    assert cfg['checkpoint_iteration']==5000 and cfg['observable59_blocks']=={k:list(v) for k,v in BLOCKS.items()}
    out=Path(cfg['output']);assert not out.exists()
    if not args.run:print(json.dumps(dict(status='METADATA_ONLY_NO_TORCH_OR_MODEL',checkpoint=5000,states=12,sides=476)));return
    assert cfg['CPU_run_authorized'] is True
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
    trainer_path=checked(cfg['source_pins']['trainer'])
    spec=importlib.util.spec_from_file_location('frozen_phone_gradient_model',trainer_path)
    trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)
    ckpt=torch.load(checked(cfg['inputs']['checkpoint']),map_location='cpu',weights_only=False)
    assert ckpt['schema']==trainer.ARTIFACT_SCHEMA and ckpt['iteration']==5000 and ckpt['input_schema']==trainer.INPUT_SCHEMA
    model=trainer.make_model(ckpt['normalizers_F64'],torch);model.load_state_dict(ckpt['model_state'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    selection=read(checked(cfg['inputs']['selection']));op=load_arrays(checked(cfg['inputs']['operands']))
    native_report=read(checked(cfg['inputs']['native_report']));records=read(checked(native_report['records']))
    assert native_report['status']=='COMPLETE_BASELINE_MATCHED_FULL20_NATIVE_FINITE_ACTIONS'
    assert len(records)==488 and native_report['partial_branches']==0
    out.mkdir();write(out/'protocol.json',cfg);start=time.perf_counter()
    counts=dict(checkpoint_loads=1,model_constructors=1,NN_forward=0,action_AD_jacobians=0,NN_parameter_updates=0,native_steps=0,TEST_arrays_read=0)
    matrices={k:[] for k in ['native_secant_F64','model_secant_F64_denominator','model_secant_FP32_denominator',
        'model_AD_column','model_side_observable59','native_side_observable59','du64','du32','state_id','joint','branch_id']}
    model_bases=[];native_bases=[];jacobians=[];rows=[]
    std=op['TRAIN_observable59_population_std_F64'];nonconstant=op['TRAIN_observable59_nonconstant_bool']
    assert std.shape==(59,) and np.array_equal(nonconstant,std>0)
    for state in selection['states']:
        sid=state['state_id'];base=next(v for v in records if v['state_id']==sid and v['kind']=='baseline')
        native_base=load_arrays(checked(base['output']))['observable59'];native_bases.append(native_base)
        assert np.array_equal(native_base,op['baseline_observable59_F64'][sid])
        x=torch.tensor(op['source_state58_F64'][sid],dtype=torch.float32)
        u=torch.tensor(state['baseline_ctrl20_F64'],dtype=torch.float32)
        context=torch.tensor([state['native_dt_s'],state['substeps']],dtype=torch.float32)
        def function(control):
            raw=torch.cat((x,control,context)).unsqueeze(0)
            result=model(raw);counts['NN_forward']+=1
            return torch.cat((model.physical_delta(result)[0],result['coverage_logits2'][0].sigmoid(),result['logforce5'][0]))
        with torch.no_grad():model_base=function(u).double().numpy()
        model_bases.append(model_base)
        jac=torch.autograd.functional.jacobian(function,u,vectorize=True,create_graph=False).detach().double().numpy()
        counts['action_AD_jacobians']+=1;jacobians.append(jac)
        assert jac.shape==(59,20) and np.isfinite(jac).all()
        side_records=[v for v in records if v['state_id']==sid and v['kind']=='finite_action_side']
        for record in side_records:
            plan=record['perturbation'];j=plan['direction_joint'];du64=plan['actual_signed_du_F64_rad']
            branch=load_arrays(checked(record['output']));assert record['usable_observable59']
            native=branch['observable59'];changed=np.array(plan['actual_ctrl20_F64'])
            assert changed.tobytes()==branch['actual_control20'].tobytes()
            up=torch.tensor(changed,dtype=torch.float32);du32=float(up[j])-float(u[j])
            assert du32==plan['actual_signed_du_FP32_rad'] and abs(du64)>=1e-6
            with torch.no_grad():pred=function(up).double().numpy()
            ns=(native-native_base)/du64;ms64=(pred-model_base)/du64
            ms32=(pred-model_base)/du32 if du32!=0 else np.full(59,np.nan);ad=jac[:,j]
            assert np.isfinite(ns).all() and np.isfinite(ms64).all()
            blocks={}
            for name,(lo,hi) in BLOCKS.items():
                active=nonconstant[lo:hi]
                blocks[name]=dict(native_vs_model_FD= comparison(ms64[lo:hi],ns[lo:hi]),
                    model_AD_vs_own_FD=comparison(ad[lo:hi],ms32[lo:hi]),
                    normalized_native_vs_model_FD=comparison(ms64[lo:hi][active]/std[lo:hi][active],ns[lo:hi][active]/std[lo:hi][active]),
                    normalized_model_AD_vs_own_FD=comparison(ad[lo:hi][active]/std[lo:hi][active],ms32[lo:hi][active]/std[lo:hi][active]))
            rows.append(dict(state_id=sid,family_id=state['family_id'],dt_s=state['native_dt_s'],interval=state['interval_index'],
                source_branch=state['source_branch'],branch_id=record['branch_id'],joint=j,side=plan['side'],
                cohort='primary_original8' if j<8 else 'secondary_added12',du_F64=du64,du_FP32=du32,
                FP32_step_ratio=du32/du64,blocks=blocks,source_native_output=record['output']))
            for key,value in zip(matrices,[ns,ms64,ms32,ad,pred,native,du64,du32,sid,j,record['branch_id']]):matrices[key].append(value)
    assert len(rows)==476 and counts['NN_forward']==500 and counts['action_AD_jacobians']==12
    assert all(p.grad is None for p in model.parameters()) and not torch.cuda.is_initialized()
    summaries={}
    for cohort in ['primary_original8','secondary_added12','all20']:
        subset=rows if cohort=='all20' else [r for r in rows if r['cohort']==cohort]
        sums={}
        for block in BLOCKS:
            item={}
            for relation in ['native_vs_model_FD','model_AD_vs_own_FD','normalized_native_vs_model_FD','normalized_model_AD_vs_own_FD']:
                values=[r['blocks'][block][relation] for r in subset if r['blocks'][block][relation] is not None]
                item[relation]={}
                for metric in ['prediction_L2','native_or_reference_L2','error_L2','RMSE','relative_L2','cosine']:
                    v=[t[metric] for t in values if t[metric] is not None and math.isfinite(t[metric])]
                    item[relation][metric]=dict(defined_count=len(v),median=None if not v else float(np.median(v)),
                        mean=None if not v else float(np.mean(v)),p90=None if not v else float(np.quantile(v,.9)))
            sums[block]=item
        summaries[cohort]=dict(sides=len(subset),blocks=sums)
    npz=out/'all_side_arrays.npz'
    with npz.open('xb') as f:np.savez_compressed(f,**{k:np.asarray(v) for k,v in matrices.items()},
        model_baseline59=np.array(model_bases),native_baseline59=np.array(native_bases),model_action_AD12x59x20=np.array(jacobians),
        TRAIN_std59=std,TRAIN_nonconstant59=nonconstant)
    write(out/'side_metrics.json',rows)
    report=dict(schema='phone_full20_model_native_finite_response_comparison:1',status='COMPLETE_FIXED5000_SAME_CONTEXT_RESPONSES_AND_MODEL_AD',
        protocol=pin(args.protocol),counts=counts,sides=476,states=12,unavailable_sides=4,
        array_output=pin(npz),side_metrics=pin(out/'side_metrics.json'),summaries=summaries,
        extra_model_small_step_or_double_shadow=False,accuracy_or_physics_Jacobian_qualified=False,
        native_protocol_order_deviation='Per-state baseline then its perturbations; all12 baselines ultimately passed. Not all12-first as design prose.',
        scope='Finite-action agreement at actual F64 steps, model self-linearization at actual F32 steps. Large steps/contact switches and F32 quantization preclude interpreting native secants as analytic derivatives. No new weights or policy gain.',
        elapsed_s=time.perf_counter()-start)
    write(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],counts=counts,primary8_COM=summaries['primary_original8']['blocks']['COM3']['native_vs_model_FD'],
                          secondary12_COM=summaries['secondary_added12']['blocks']['COM3']['native_vs_model_FD'])),flush=True)

if __name__=='__main__':main()
