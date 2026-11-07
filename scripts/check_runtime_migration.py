"""Two fixed saved phone contexts: old/new CPU math and input-AD parity.

Only this validation script loads original frozen source with file specs.
Production runtime uses ordinary package imports and explicit data artifacts.
No NN module, native model, physics, learning or CUDA work is constructed.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def need(v,msg):
    if not v:raise ValueError(msg)

def old_module(name,record):
    p=Path(record['path']);need(pin(p)==record,'Frozen legacy source pin')
    spec=importlib.util.spec_from_file_location('validation_legacy_'+name,p);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def run(manifest):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
    need(not torch.cuda.is_initialized(),'CPU only')
    from supermimic_research.artifacts import ArtifactRegistry
    from supermimic_research.runtime import state_contract,state_codec,rotation,momentum,structured,geometry,directional,phone_task,ego_reference,native_reader,reader_replay
    newmods={m.__name__.rsplit('.',1)[-1]:m for m in [state_contract,state_codec,rotation,momentum,structured,geometry,directional,phone_task,ego_reference,native_reader,reader_replay]}
    registry=ArtifactRegistry(manifest)
    origins=json.loads((ROOT/'provenance/phone_runtime_origins.json').read_text())
    old={row['module'].rsplit('.',1)[-1]:old_module(row['module'].rsplit('.',1)[-1],row['origin']) for row in origins['modules']}
    report_path=registry.resolve('action_response');report=json.loads(report_path.read_text())
    rows=json.loads(registry.resolve_pin(report['branches']).read_text());selected=[rows[i] for i in [0,3]]
    need(all(v['kind']=='baseline' and v['status']=='EXECUTED' and v['labels_known'] for v in selected),'Mechanical baseline IDs0/3, not numerical selection')
    arrays=[]
    for rec in selected:
        with np.load(registry.resolve_pin(rec['output']),allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in z.files})
    with np.load(registry.resolve_pin(report['operands']),allow_pickle=False) as z:operands={k:z[k].copy() for k in z.files}
    protocol=json.loads(registry.resolve_pin(report['protocol']).read_text());physical=json.loads(registry.resolve_pin(protocol['inputs']['physical_parameters']).read_text())
    selection=json.loads(registry.resolve_pin(report['selection']).read_text())
    memory=[];actions=[]
    for rec in selected:
        with np.load(registry.resolve_pin(selection['states'][rec['context_id']]['source_trace']),allow_pickle=False) as z:
            memory.append(z['memory_before83_F64'].copy());actions.append(z['action_normalized20_F64'].copy())
    x=np.stack([a['source_state58_F64'] for a in arrays]);y=np.stack([a['target_H_state58_F64'] for a in arrays]);u=np.stack([a['actual_control20_F64'] for a in arrays])
    J=np.stack([a['impulse_source_BODY_moving_COM22x6_F64'] for a in arrays]);dt=np.array([v['dt_s'] for v in selected]);n=np.array([v['n_substeps'] for v in selected]);duration=dt*n
    ipos=np.array(physical['body_ipos_body_m']);g=np.array(physical['gravity_world_m_s2']);I=np.array(physical['I_body_3x3']);mass=physical['mass_kg']
    stats={};checks={};calls=dict(CPU_input_AD_grad_calls=0,NN_constructors=0,NN_forward=0,native_model_load=0,native_FK_forward_steps=0,optimizer_steps=0,CUDA_calls=0)
    def compare(name,a,b):
        if isinstance(a,torch.Tensor):
            need(isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and a.device.type==b.device.type=='cpu',name+' tensor interface')
            aa,bb=a.detach().numpy(),b.detach().numpy()
        elif isinstance(a,np.ndarray):
            need(isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype,name+' NumPy interface');aa,bb=a,b
        elif isinstance(a,dict):
            need(isinstance(b,dict) and set(a)==set(b),name+' dictionary keys')
            for key in a:compare(name+'.'+key,a[key],b[key])
            return
        elif isinstance(a,(list,tuple)):
            need(type(a)==type(b) and len(a)==len(b),name+' sequence')
            for k,(aa,bb) in enumerate(zip(a,b)):compare(name+'.'+str(k),aa,bb)
            return
        else:need(a==b,name+' literal');return
        if aa.dtype.kind in 'fc':
            need(np.array_equal(np.isnan(aa),np.isnan(bb)) and np.allclose(aa,bb,atol=1e-12,rtol=1e-12,equal_nan=True),name+' F64/explicitFP32 parity')
            finite=np.isfinite(aa)&np.isfinite(bb);err=float(np.max(np.abs(aa[finite]-bb[finite]))) if finite.any() else 0.
            stats[name]=dict(shape=list(aa.shape),dtype=aa.dtype.str,max_abs_error=err,nan_count=int(np.isnan(aa).sum()))
        else:need(np.array_equal(aa,bb),name+' exact discrete values')
    tensor=lambda a:torch.tensor(a,dtype=torch.float64,device='cpu')
    X,Y,U=tensor(x),tensor(y),tensor(u);TT=tensor(duration);II=tensor(I);GG=tensor(g);MM=tensor(mass)
    for i,rec in enumerate(selected):
        sid=rec['context_id'];q=operands['source_qpos27_F64'][sid];v=operands['source_qvel26_F64'][sid]
        compare(f'state58.{i}',old['state_codec'].actual_state(q,v,ipos),state_codec.actual_state(q,v,ipos))
    compare('inverse_state58',old['state_contract'].from_com_world_state(x,ipos),state_contract.from_com_world_state(x,ipos))
    olddelta=np.stack([old['state_codec'].delta52(a,b) for a,b in zip(x,y)]);newdelta=np.stack([state_codec.delta52(a,b) for a,b in zip(x,y)])
    compare('worldleft_delta52',olddelta,newdelta)
    compare('rotation_decode_numpy',old['rotation'].decode_numpy(x,olddelta,state_schema=rotation.STATE_SCHEMA,delta_schema=rotation.DELTA_SCHEMA),rotation.decode_numpy(x,newdelta,state_schema=rotation.STATE_SCHEMA,delta_schema=rotation.DELTA_SCHEMA))
    compare('Exp_world_actual_and_zero',old['rotation'].so3_exp_torch(torch.cat([tensor(olddelta[:,43:46]),torch.zeros((1,3))])),rotation.so3_exp_torch(torch.cat([tensor(newdelta[:,43:46]),torch.zeros((1,3))])))
    oldheads=old['structured'].encode_observed_numpy(x,y,J,duration,g);newheads=structured.encode_observed_numpy(x,y,J,duration,g);compare('structured_encode',oldheads,newheads)
    def heads_t(h):return {k:tensor(v) for k,v in h.items()}
    dec_old=old['structured'].decode_physical_heads(X,heads_t(oldheads),TT,MM,II,GG);dec_new=structured.decode_physical_heads(X,heads_t(newheads),TT,MM,II,GG)
    compare('structured_decode',dec_old,dec_new)
    compare('momentum_world',old['momentum'].momentum_residual_world(X,Y,tensor(J.sum(1)),MM,II,GG,TT),momentum.momentum_residual_world(X,Y,tensor(J.sum(1)),MM,II,GG,TT))
    checks['state_rotation_momentum_structured_cpu_outputs']=True
    catalog_path=registry.resolve('geometry_catalog');catalog=json.loads(catalog_path.read_text());metadata_path=registry.resolve_pin(catalog['provider_metadata']);metadata=json.loads(metadata_path.read_text())
    oldgap=old['directional'].DirectionalGapProvider(catalog_path)
    newgap=directional.DirectionalGapProvider(catalog_path,metadata_path=metadata_path,geometry_arrays_path=registry.resolve_pin(metadata['geometry_arrays']),direction_arrays_path=registry.resolve_pin(catalog['direction_arrays']))
    for label,method,args in [('poses','poses',(X,)),('features','features',(X,U,tensor(dt),torch.tensor(n,dtype=torch.int64)))]:compare('geometry_'+label,getattr(oldgap.provider,method)(*args),getattr(newgap.provider,method)(*args))
    compare('directional22',oldgap.features(X),newgap.features(X));checks['all_geometry_poses_features_directional22_and_masks']=True
    task_path=registry.resolve('phone_task');bundle=json.loads(task_path.read_text())
    gold_weighted=old_module('CHORD_weighted_golden',bundle['sources']['weighted_support']);gold_contacts=old_module('CHORD_contacts_golden',bundle['sources']['chord_contacts'])
    from chord_study import contact_geometry as current_weighted,contacts as current_contacts
    need(pin(current_weighted.__file__)['sha256']==bundle['sources']['weighted_support']['sha256'] and pin(current_contacts.__file__)['sha256']==bundle['sources']['chord_contacts']['sha256'],'Golden CHORD source bytes unchanged')
    weighted_path=Path(bundle['sources']['weighted_support']['path']).resolve();contacts_path=Path(bundle['sources']['chord_contacts']['path']).resolve()
    def golden_chord(name,path):
        p=Path(path).resolve()
        if p==weighted_path:return gold_weighted
        if p==contacts_path:return gold_contacts
        raise ValueError('Validation-only old loader received unexpected source')
    old['phone_task'].load_source=golden_chord
    tasks_old={};tasks_new={};MEM=tensor(np.stack(memory));ACT=tensor(np.stack(actions));task_pairs={}
    for mode in ['A1_tracking','A2_engineered_material_proxy']:
        ot=old['phone_task'].PhoneTask(task_path,mode=mode,geometry=oldgap);nt=phone_task.PhoneTask(task_path,mode=mode,geometry=newgap,reference_arrays_path=registry.resolve_pin(bundle['reference_arrays']),support_basis_path=registry.resolve_pin(bundle['basis']))
        tasks_old[mode]=ot;tasks_new[mode]=nt;om=ot.unflatten_memory(MEM);nm=nt.unflatten_memory(MEM)
        compare(mode+'.memory83',ot.flatten_memory(om),nt.flatten_memory(nm));compare(mode+'.obs222',ot.observation(X,om,oldgap),nt.observation(X,nm,newgap))
        oc,ou=ot.control_target(ACT,om);nc,nu=nt.control_target(ACT,nm);compare(mode+'.control',(oc,ou),(nc,nu))
        compare(mode+'.advance_memory83',ot.flatten_memory(ot.advance_memory(om,ACT,oc,ou)),nt.flatten_memory(nt.advance_memory(nm,ACT,nc,nu)))
        compare(mode+'.reward',ot.rewards(Y,om,ACT),nt.rewards(Y,nm,ACT));compare(mode+'.termination',ot.termination(Y,om),nt.termination(Y,nm))
        task_pairs[mode]=dict(memory83=np.stack(memory).tolist(),observation_dtype='explicitFP32 source behavior retained')
    checks['task_memory_control_obs222_A1_A2_reward_termination_CHORD']=True
    # Fixed scalar projection, q/u input AD only. No model/Actor is involved.
    def projected(mods,gap,tasks):
        xx=tensor(x).requires_grad_(True);uu=tensor(u).requires_grad_(True)
        features=gap.provider.features(xx,uu,tensor(dt),torch.tensor(n,dtype=torch.int64));dg=gap.features(xx)
        value=(features['pair_relative_translation_BODY_COM_m']*torch.linspace(.01,.02,66).reshape(22,3)).sum()
        value=value+(features['absolute_u20']*torch.linspace(.03,.04,20)).sum()+torch.where(dg['directional_gap_known'],dg['directional_gap_m'],0).sum()
        yy=mods['structured'].decode_physical_heads(xx,heads_t(newheads),TT,MM,II,GG)
        value=value+(yy[:,52:58]*torch.linspace(.001,.002,6)).sum()
        for mode,task in tasks.items():
            m=task.unflatten_memory(MEM);value=value+.01*task.rewards(yy,m,ACT)['total'].sum()
        grads=torch.autograd.grad(value,(xx,uu));calls['CPU_input_AD_grad_calls']+=1
        need(all(torch.isfinite(v).all() for v in grads),'Finite CPU input AD at both actual states')
        qdir=torch.linspace(-1.,1.,20);udir=torch.cos(torch.arange(20,dtype=torch.float64))
        return dict(scalar=value.detach(),q20_gradient=grads[0][:,:20],u20_gradient=grads[1],q_direction=(grads[0][:,:20]*qdir).sum(-1),u_direction=(grads[1]*udir).sum(-1))
    ad_old=projected(old,oldgap,tasks_old);ad_new=projected(newmods,newgap,tasks_new);compare('fixed_q_u_input_AD',ad_old,ad_new);checks['finite_fixed_q_u_scalar_input_AD_parity']=True
    need(all(m.__name__.startswith('supermimic_research.runtime.') for m in newmods.values()) and newgap.provider.__class__.__module__.startswith('supermimic_research.runtime.'),'New implementation uses ordinary package modules')
    need('mujoco' not in sys.modules and not torch.cuda.is_initialized(),'No native backend or CUDA initialization')
    checks['new_runtime_normal_import_no_old_source_selector']=True;checks['CPU_only_no_NN_native_learning']=True
    return dict(schema='phone_runtime_migration_validation:1',status='PASS_TWO_FIXED_REAL_CONTEXT_CPU_OUTPUT_AND_INPUT_AD_PARITY',checks=checks,passed=sum(checks.values()),total=len(checks),
        artifacts_manifest=pin(manifest),source_origins=pin(ROOT/'provenance/phone_runtime_origins.json'),selected_contexts=[dict(branch_id=r['branch_id'],context_id=r['context_id'],family_id=r['family_id'],dt_s=r['dt_s'],step_k=r['step_k'],output=r['output']) for r in selected],
        numeric_tolerance=dict(atol=1e-12,rtol=1e-12,scope='Migration parity only, not a physical accuracy threshold'),comparisons=stats,calls=calls,
        directional_calls=dict(old=oldgap.counts,new=newgap.counts),input_AD={key:val.detach().tolist() for key,val in ad_new.items()},
        runtime_sources={f.name:pin(f) for f in (ROOT/'src/supermimic_research/runtime').glob('*.py')},validation_source=pin(__file__),
        scope=['Mechanically fixed saved baselines0 and3 only, no result-based sample selection. Input physical impulses are existing saved labels, not newly reconstructed forces.',
        'Old frozen .py and CHORD file-spec loads occur only in this validation script/old module internals. New runtime imports ordinary package modules; no production source path selectors.',
        'CPU F64 math/geometry and input AD, plus source-explicit FP32 obs222 parity. No NN constructors/forward, native model/FK/forward/step, training or GPU.',
        'Structured decoder comparison is old versus new under the same momentum closure; it is not proof of exact native endpoint closure, correct physical gradients, holding, ego execution or generalization.'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=ROOT/'configs/artifacts.toml');parser.add_argument('--output',type=Path,default=ROOT/'provenance/runtime_validation.json');a=parser.parse_args()
    need(not a.output.exists(),'Use a fresh validation output; retain earlier outcomes')
    try:
        report=run(a.manifest)
    except BaseException as error:
        report=dict(schema='phone_runtime_migration_validation:1',status='FAILED_PRESERVED_NO_NATIVE_OR_LEARNING_RETRY',error=repr(error),traceback=traceback.format_exc(),validation_source=pin(__file__))
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');raise
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(dict(report=pin(a.output),passed=report['passed'],total=report['total'])))
