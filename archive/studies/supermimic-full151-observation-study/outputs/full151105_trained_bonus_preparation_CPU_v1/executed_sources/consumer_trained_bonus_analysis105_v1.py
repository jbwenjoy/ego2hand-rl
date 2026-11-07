"""CPU shadow of the exact trained geometric functions; default is metadata only.

No Torch/objective/WM import occurs until an explicitly authorized run. The
source-preregistered empirical secants are not a physical Jacobian certificate.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import zipfile
import numpy as np

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
PREREG = ROOT/'protocols/full151105_trained_geometric_bonus_secant_CPU_prereg_v1.json'
PREREG_SHA = '9d48ae798471638bb6bbc3e47d80112b185de68fd861e9eac5449286428413b7'
SCHEMA = 'full151105_trained_bonus_CPU_run:1'
ROLE = 'trained_bonus_physical_secant_vs_WM_AD'
KEYS = {'schema','role','protocol','output','CPU_run_authorized','cpu_threads','source_pins','cpu_ready'}
CHANNELS = ('raw_P','raw_S','BP','CS','A2')
PHASES = (0,49,99,149)
JOINTS = (0,4,8,12)
EPS = (.002,.001)
CAPTURE_KEYS = {'support','presence','minimum_signed_distance_m','projected_contact_com_o_m',
                'inward_normal_o','soft_contact_weight','sampled_signed_distance_m'}
OWN = {'consumer':ROOT/'src/full151_gradient/trained_bonus_analysis105_v1.py',
       'CLI':ROOT/'scripts/diagnose_trained_geometric_bonus_cpu_v1.py',
       'checker':ROOT/'scripts/check_trained_geometric_bonus_cpu_v1.py'}


def require(value,message):
    if not value: raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':digest(p)}


def checked(value):
    require(type(value) is dict and set(value)=={'path','sha256'},'Strict path/SHA pin required')
    require(type(value['path']) is str and Path(value['path']).is_absolute(),'Absolute pin required')
    h=value['sha256'];require(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Malformed SHA256')
    p=Path(value['path']).resolve(strict=True)
    require(p.is_file() and digest(p)==h,'Pin mismatch: '+str(p))
    return p


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def same_pin(a,b): return checked(a).samefile(checked(b)) and a['sha256']==b['sha256']


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def all_checks(d):
    c=d.get('checks');require(type(c) is dict and c and all(type(v) is bool and v for v in c.values()),'Actual checks must all pass')
    require(type(d.get('passed')) is int and type(d.get('total')) is int and d['passed']==d['total']==len(c),'Check counts mismatch')


def array_digest(values):
    h=hashlib.sha256()
    for name,a in sorted(values.items()):
        a=np.ascontiguousarray(a);h.update(name.encode());h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def validate_header(c):
    require(type(c) is dict and set(c)==KEYS,'Exact bonus CPU runplan keyset required')
    require(c['schema']==SCHEMA and c['role']==ROLE,'Wrong bonus runplan schema/role')
    require(type(c['CPU_run_authorized']) is bool,'Explicit native bool CPU authorization required')
    require(type(c['cpu_threads']) is int and c['cpu_threads']==1,'Frozen CPU threads1 required')
    require(type(c['output']) is str and Path(c['output']).is_absolute(),'Absolute output required')
    out=Path(c['output']).resolve()
    require(out.parent==ROOT/'outputs' and out.name.startswith('full151105_trained_bonus_CPU_'),'Scoped fresh bonus output required')
    require(not out.exists(),'No overwrite/retry of existing output')
    require(type(c['source_pins']) is dict and c['source_pins'],'Flat nonempty source table required')


def authorization_gate(c):
    require(c['CPU_run_authorized'] is True,'Root explicit CPU run authorization required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Fresh CUDA_VISIBLE_DEVICES=-1 process required')


def required_sources(protocol):
    """Resolve each required path once; no quadratic repeated Windows resolves."""
    result=dict(protocol['source_pins'])
    for method in ('P_ONLY','A2'):
        train=read(checked(protocol['inputs'][method+'_training_config']))
        for alias,value in train['source_pins'].items(): result[method+'__'+alias]=value
    actual=read(checked(protocol['inputs']['physical_config']))
    for alias,value in actual['source_pins'].items(): result['physical__'+alias]=value
    result.update({k:pin(v) for k,v in OWN.items()})
    return result


def declaration():
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Default must not import Torch/Warp')
    require(digest(PREREG)==PREREG_SHA,'Fixed preregistration drifted')
    return {'schema':'full151105_bonus_CPU_metadata_declaration:1','status':'DECLARED_NOT_EXECUTED',
            'runplan_schema':SCHEMA,'runplan_role':ROLE,'runplan_keys':sorted(KEYS),'protocol':pin(PREREG),
            'own_sources':{k:pin(p) for k,p in OWN.items()},'Torch_imported':False,'Warp_imported':False,
            'NPZ_opened':0,'checkpoint_loads':0,'objective_constructions':0,'WM_forwards':0,'geometry_forwards':0,'GPU_physics_Env':0}


def metadata_preflight(config_path):
    """Hash/JSON only: no tensor archives, models, or objective construction."""
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Use fresh default metadata process')
    c=read(config_path);validate_header(c)
    require(same_pin(c['protocol'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Wrong preregistration')
    p=read(checked(c['protocol']))
    require(p['schema']=='full151105_trained_geometric_bonus_secant_CPU_preregistration:1' and p['execution_authorized'] is False,'Wrong prereg source scope')
    inputs=p['inputs']
    for value in inputs.values():checked(value)
    have=set()
    for value in c['source_pins'].values():
        f=checked(value);have.add((str(f).casefold(),value['sha256']))
    needed=required_sources(p)
    require(all((str(checked(v)).casefold(),v['sha256']) in have for v in needed.values()),'Required immutable source closure missing')
    ready=read(checked(c['cpu_ready']))
    require(ready.get('schema')=='full151105_trained_bonus_CPU_contract_READY:1' and ready.get('status')=='PASS_PURE_CPU_CONTRACTS_NOT_REAL_SCORE','Wrong CPU fixture READY')
    require(same_pin(ready['protocol'],c['protocol']),'READY prereg mismatch')
    proof=read(checked(ready['cpu_report']));all_checks(proof)
    require(proof.get('schema')=='full151105_trained_bonus_CPU_contract_proof:1','Wrong pure checker proof')
    for k,path in OWN.items():
        require(same_pin(ready['source_pins'][k],pin(path)) and same_pin(proof['source_pins'][k],pin(path)),'READY/checker source differs: '+k)
    require(proof['scope']['actual_objective_constructs']==proof['scope']['actual_WM_loads']==proof['scope']['actual_full05_NPZ_reads']==0,'Fixture scope falsely elevated')
    health=read(checked(inputs['fixed_A2_health']));all_checks(health)
    require(health['method']=='A2' and health['iteration']==2000 and health['final_iteration_scope'] is True and same_pin(health['checkpoint'],inputs['fixed_A2_checkpoint']),'Actual finalA2 health mismatch')
    for method in ('P_ONLY','A2'):
        train=read(checked(inputs[method+'_training_config']))
        require(train['method']==method and train['role']=='primary_matched' and train['seed']==0 and same_pin(train['contact_binding'],inputs['objective_spec']),'Actual trained factory spec mismatch')
    spec=read(checked(inputs['objective_spec']))
    require(spec['schema']=='full151_declared_H123_objective_spec:2' and spec['CPU_authorized'] is True,'CPU factory spec unavailable')
    saved=read(checked(inputs['physical_saved_review']));all_checks(saved)
    require(saved['schema']=='full151105_full_serial_saved_independent:1' and saved['completed_branches']==152 and saved['baseline_count']==24 and saved['physical_gradient_qualified'] is False and saved['whole_hidden_state_qualified'] is False,'Wrong physical saved-review scope')
    require(same_pin(saved['config'],inputs['physical_config']) and same_pin(saved['producer_report'],inputs['physical_report']),'Wrong physical saved joins')
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Metadata imported science backend')
    return {'config':c,'config_pin':pin(config_path),'protocol':p,'required_sources':needed,
            'status':'PASS_CPU_METADATA_ONLY_NOT_SCORED','Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'model_loads':0,'objective_constructions':0}


def physical_layout():
    ids=np.concatenate((np.arange(8),np.repeat(np.arange(8),19)))
    return ids,np.asarray(PHASES,dtype=np.int64)[ids%4]


def pair_slots(s,j,ei):
    require(type(s) is int and 0<=s<8 and type(j) is int and 0<=j<4 and type(ei) is int and 0<=ei<2,'Frozen pair index required')
    minus=8+s*19+3+ei*8+j*2
    return minus,minus+1


def baseline_noise(donor,baseline):
    a=np.asarray(baseline,np.float64)
    require(a.shape==(3,) and np.isfinite(a).all() and np.isfinite(donor),'Finite three-baseline/donor component required')
    span=float(np.ptp(a));err=float(np.max(abs(a-float(donor))))
    scale=max(1.,abs(float(donor)),float(abs(a).max()))
    return {'range':span,'donor_error':err,'N':max(span,err),'scale':scale,'roundoff_proxy':8*2**-23*scale}


def component_floors(donors,baseline):
    require(set(donors)==set(baseline)==set(CHANNELS),'All raw/weighted channels required')
    noise={k:baseline_noise(donors[k],baseline[k]) for k in CHANNELS}
    f={'raw_P':max(1e-4,10*noise['raw_P']['N'],noise['raw_P']['roundoff_proxy']),
       'raw_S':max(1e-5,10*noise['raw_S']['N'],noise['raw_S']['roundoff_proxy']),
       'CS':max(1e-5,10*noise['CS']['N'],noise['CS']['roundoff_proxy'])}
    f['BP']=.1*f['raw_P']
    f['A2']=max(f['BP']+f['CS'],10*noise['A2']['N'],noise['A2']['roundoff_proxy'])
    return {'floors':f,'baseline_components':noise,'BP_representation_consistent':noise['BP']['N']<=f['BP'],
            'dt_weighted_floors':{k:.02*f[k] for k in ('BP','CS','A2')},'provenance':'NEW_PROPOSED prereg9d48 representational guard, not accumulated-error theorem'}


def classify_pair(minus,plus,floor):
    minus,plus=np.asarray(minus),np.asarray(plus)
    require(minus.shape==plus.shape==(2,) and np.isfinite(minus).all() and np.isfinite(plus).all(),'Two finite epsilon endpoint scores required')
    require(np.isfinite(floor) and floor>0,'Positive frozen component floor required')
    delta=plus.astype(np.float64)-minus.astype(np.float64)
    slopes=delta/(2*np.asarray(EPS,np.float64));signal=abs(delta)>floor
    rel=float(abs(slopes[0]-slopes[1])/max(abs(slopes[0]),abs(slopes[1]),1e-12))
    stable=bool(signal.all() and np.sign(slopes[0])==np.sign(slopes[1]) and rel<=.5)
    label='STABLE_NONFLAT' if stable else ('FLAT_OR_BELOW_NOISE' if not signal.all() else 'NONFLAT_UNSTABLE')
    return {'label':label,'stable_nonflat':stable,'numerators':delta.tolist(),'slopes':slopes.tolist(),'both_above_noise':bool(signal.all()),'relative_epsilon_disagreement':rel,'numerator_floor':float(floor)}


def compare_projection(slope,ad,floor,epsilon):
    require(epsilon in EPS and np.isfinite([slope,ad,floor]).all() and floor>0,'Finite component projection required')
    derivative_floor=floor/(2*epsilon);denom=max(abs(slope),abs(ad),derivative_floor)
    eligible=abs(slope)>derivative_floor
    return {'absolute_error':float(abs(ad-slope)),'normalized_error':float(abs(ad-slope)/denom),
            'denominator':float(denom),'physical_sign_eligible':bool(eligible),
            'sign_agreement':None if not eligible else bool(ad!=0 and np.sign(ad)==np.sign(slope)),
            'AD_over_physical':None if not eligible else float(ad/slope),
            'undefined_ratio_reason':None if eligible else 'PHYSICAL_NOT_ABOVE_COMPONENT_FLOOR'}


def comparisons(scores,gradients,d_eff,center_offset,clipped):
    require(set(scores)==set(gradients)==set(CHANNELS),'Five source-defined channels required')
    for k in CHANNELS:
        require(np.asarray(scores[k]).shape==(160,) and np.isfinite(scores[k]).all(),'All160 physical scores required')
        require(np.asarray(gradients[k]).shape==(8,20) and np.isfinite(gradients[k]).all(),'Per-state finite gradient20 required')
    require(d_eff.shape==center_offset.shape==(8,4,2,20) and clipped.shape==(8,4,2),'Direction ABI mismatch')
    require(np.isfinite(d_eff).all() and np.isfinite(center_offset).all() and clipped.dtype==np.bool_,'Finite directions/boolean clipped required')
    rows=[];classes=[];noise=[]
    for s in range(8):
        f=component_floors({k:scores[k][s] for k in CHANNELS},{k:scores[k][8+s*19:8+s*19+3] for k in CHANNELS})
        noise.append({'snapshot_id':s,**f})
        for channel in CHANNELS:
            for j,joint in enumerate(JOINTS):
                slots=[pair_slots(s,j,ei) for ei in range(2)]
                cl=classify_pair([scores[channel][m] for m,p in slots],[scores[channel][p] for m,p in slots],f['floors'][channel])
                classes.append({'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'channel':channel,'joint_index':joint,**cl})
                for ei,epsilon in enumerate(EPS):
                    g=np.asarray(gradients[channel][s],np.float64)
                    effective=float(g@d_eff[s,j,ei]);slope=cl['slopes'][ei]
                    row={'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'channel':channel,'joint_index':joint,
                         'epsilon_index':ei,'epsilon_rad':epsilon,'minus_physical_index':slots[ei][0],'plus_physical_index':slots[ei][1],
                         'physical_slope':slope,'WM_AD_effective':effective,'WM_AD_nominal_secondary':float(g[joint]),
                         'clipped':bool(clipped[s,j,ei]),'d_eff':d_eff[s,j,ei].tolist(),'center_offset':center_offset[s,j,ei].tolist(),
                         'center_offset_maxabs':float(abs(center_offset[s,j,ei]).max()),'physical_label':cl['label'],
                         'stable_nonflat':cl['stable_nonflat'],'component_representation_consistent':channel!='BP' or f['BP_representation_consistent'],
                         'numerator_floor':f['floors'][channel],**compare_projection(slope,effective,f['floors'][channel],epsilon)}
                    row['comparison_eligible']=row['component_representation_consistent']
                    if not row['comparison_eligible']:
                        row.update(sign_agreement=None,AD_over_physical=None,undefined_ratio_reason='BP_COMPONENT_REPRESENTATION_INCONSISTENT')
                    if channel in ('BP','CS','A2'):
                        row['dt_derived']={'physical_slope':.02*slope,'WM_AD_effective':.02*effective,'numerator_floor':.02*f['floors'][channel],'same_classification':True}
                    rows.append(row)
    require(len(rows)==320 and len(classes)==160,'All5channels x64rows/32groups retained')
    return rows,classes,noise


def summaries(rows):
    result=[]
    for channel in CHANNELS:
        for ei in range(2):
            allrows=[r for r in rows if r['channel']==channel and r['epsilon_index']==ei]
            for name,selected in [('all32',allrows),('stable_secondary',[r for r in allrows if r['stable_nonflat'] and r['component_representation_consistent']]),('stable_unclipped_secondary',[r for r in allrows if r['stable_nonflat'] and not r['clipped'] and r['component_representation_consistent']])]:
                n=len(selected)
                result.append({'channel':channel,'epsilon_rad':EPS[ei],'subset':name,'rows':n,
                  'sign_eligible':sum(r['physical_sign_eligible'] and r['comparison_eligible'] for r in selected),
                  'sign_agreement_count':sum(r['sign_agreement'] is True for r in selected),
                  'mean_absolute_error':None if not n else float(np.mean([r['absolute_error'] for r in selected])),
                  'RMS_absolute_error':None if not n else float(np.sqrt(np.mean([r['absolute_error']**2 for r in selected]))),
                  'clipped_rows':sum(r['clipped'] for r in selected),'dependent_groups_not_independent_samples':True})
    return result


class ProxyCapture:
    """One existing geometry forward, read-only output hook, always removed."""
    def __init__(self,proxy,batch): self.proxy=proxy;self.batch=batch;self.values=None;self.calls=0;self.handle=None
    def __enter__(self):
        def capture(module,args,result):
            import torch
            self.calls+=1;require(self.calls==1,'Unexpected extra geometry forward')
            require(type(result) is dict and set(result)==CAPTURE_KEYS,'Exact actual proxy output ABI required')
            b=self.batch;point=result['sampled_signed_distance_m']
            require(isinstance(point,torch.Tensor) and point.ndim==2 and point.shape[0]==b and point.shape[1]>0,'SDF point batch required')
            p=point.shape[1]
            shapes={'support':(b,64),'presence':(b,6),'minimum_signed_distance_m':(b,),
                    'sampled_signed_distance_m':(b,p),'soft_contact_weight':(b,p),
                    'projected_contact_com_o_m':(b,p,3),'inward_normal_o':(b,p,3)}
            saved={}
            for k,shape in shapes.items():
                t=result[k];require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==shape and bool(torch.isfinite(t).all()),'Finite CPU FP32 proxy output required: '+k)
                saved[k]=t.detach().clone().numpy().copy()
            self.values=saved
            return None
        self.handle=self.proxy.register_forward_hook(capture);return self
    def __exit__(self,typ,value,tb):
        self.handle.remove()
        if typ is None:require(self.calls==1 and self.values is not None,'Missing geometry capture')
        return False


def memory_for_phases(phases):
    import torch
    a=np.asarray(phases)
    require(a.ndim==1 and a.dtype.kind in 'iu' and ((0<=a)&(a<150)).all(),'Active preaction integer phases0..149 only')
    time_tensor=torch.tensor(a.copy(),dtype=torch.float32).reshape(-1,1)
    return {'start':torch.zeros_like(time_tensor),'time':time_tensor,'end':torch.full_like(time_tensor,151)}


def evaluate_objective(module,state,phases):
    import torch
    base=getattr(module,'base',module);memory=memory_for_phases(phases)
    before={k:v.clone() for k,v in memory.items()}
    with ProxyCapture(base.geometry_proxy,len(phases)) as capture:
        score,terms=module(state,memory)
    require(tuple(score.shape)==(len(phases),) and score.dtype==torch.float32 and bool(torch.isfinite(score).all()),'Finite per-row actual score required')
    require(all(torch.equal(memory[k],v) for k,v in before.items()),'Objective mutated preaction memory')
    for k in ('geometry_intent_penetration_prior','geometry_intent_support_match','geometry_intent_phase_weight'):
        require(k in terms and terms[k].shape==score.shape and bool(torch.isfinite(terms[k]).all()),'Actual terms missing or malformed')
    return score,terms,capture.values


def channels(p_score,p_terms,a_score,a_terms):
    import torch
    for name in ('geometry_intent_penetration_prior','geometry_intent_support_match','geometry_intent_phase_weight'):
        require(torch.equal(p_terms[name],a_terms[name]),'Actual eager P_ONLY/A2 components differ: '+name)
    require(bool((a_terms['geometry_intent_phase_weight']==1).all()),'This prereg requires source-active weight1; no silent generalization')
    return {'raw_P':a_terms['geometry_intent_penetration_prior'],'raw_S':a_terms['geometry_intent_support_match'],
            'BP':p_score,'CS':a_terms['geometry_intent_phase_weight']*a_terms['geometry_intent_support_match'],'A2':a_score}


def proxy_diagnostics(captured):
    d=captured['sampled_signed_distance_m'];w=captured['soft_contact_weight'];s=captured['support']
    return {'rows':int(len(d)),'sample_points_per_row':int(d.shape[1]),'basis':64,
            'minimum_sdf_m':d.min(-1).tolist(),'maximum_soft_weight':w.max(-1).tolist(),
            'sample_count_abs_sdf_le_1mm':(abs(d)<=.001).sum(-1).tolist(),
            'sample_count_abs_sdf_le_3mm':(abs(d)<=.003).sum(-1).tolist(),
            'support_live_count_gt_1e_3':(s>1e-3).sum(-1).tolist(),
            'scope':'All rows including dropped/far. Proxy sampled geometry, not native active contact or force; no eligibility filtering.'}


def _write(out,name,value):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def _npwrite(out,name,values):
    path=out/name;require(not path.exists(),'No NPZ overwrite');np.savez_compressed(path,**values)


def _tensor_digest(state):
    return array_digest({k:v.detach().cpu().contiguous().numpy() for k,v in state.items()})


def run_analysis(config_path):
    """Only Root's separate, exact authorized config may enter this path."""
    started=time.perf_counter();p=metadata_preflight(config_path);c=p['config'];authorization_gate(c)
    out=Path(c['output']);out.mkdir()
    counts={k:{'attempted':0,'returned':0} for k in ('strict_WM_loader','objective_factory','physical_objective_forward','WM_forward','predicted_objective_forward','u_VJP')}
    phase='archive';invariants={};receipts={}
    def call(name,fn,*args,**kwargs):
        counts[name]['attempted']+=1;value=fn(*args,**kwargs);counts[name]['returned']+=1;return value
    try:
        _write(out,'config.json',c);_write(out,'protocol.json',p['protocol']);_write(out,'source_inventory.json',c['source_pins'])
        with zipfile.ZipFile(out/'execution_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for i,v in enumerate(c['source_pins'].values()):z.write(checked(v),'%04d_%s'%(i,Path(v['path']).name))
        inputs=p['protocol']['inputs'];physical=checked(inputs['physical_arrays']).parent
        phase='saved_data_validation'
        from . import analysis105_v1 as old
        with np.load(checked(inputs['physical_arrays']),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        rows=read(checked(inputs['branch_index']));snaps=read(checked(inputs['snapshot_records']))
        saved=read(checked(inputs['physical_saved_review']))
        evidence={str(checked(v)).casefold():v for v in saved['evidence'].values()}
        def ev(path):
            v=evidence[str(Path(path).resolve(strict=True)).casefold()];return checked(v)
        producer_config=read(checked(inputs['physical_config']));proposal=read(checked(producer_config['proposal']))
        old._validate_data(a,rows,snaps,read(checked(inputs['baselines'])),proposal)
        old._validate_capacity(read(ev(physical/'capacity_ledgers.json')))
        lifecycle_c={'physical_run':str(physical),'artifact_pins':{'producer_config':inputs['physical_config']}}
        watch=Path(str(physical)+'_watchdog')
        old._validate_lifecycle(lifecycle_c,{'watch_state':read(ev(watch/'state.json')),'engineering_identity':read(ev(watch/'engineering_identity.json')),
            'post_terminal_CIM':read(ev(watch/'post_terminal_CIM.json')),'runtime':read(ev(physical/'runtime.json'))})
        before_arrays=array_digest(a)
        phase='CPU_import_and_loading'
        import torch
        require(not torch.cuda.is_initialized() and 'warp' not in sys.modules,'Fresh CPU-only/noWarp required')
        torch.set_num_threads(1)
        def rng_digest():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
        rng_before=rng_digest()
        from full151_native.bundle import load_bundle
        from full151_native.world_model_checkpoint import load_world_model_checkpoint
        from full151_intent.binding_v3 import build_contact_objective,build_penalty_only_objective
        bundle=load_bundle(checked(inputs['input_bundle']),expected_sha256=inputs['input_bundle']['sha256'])
        health_auditor=p['protocol']['source_pins']['health_auditor']
        original_load=torch.load;load_calls=[]
        def tracked_load(*args,**kwargs):load_calls.append(str(args[0]));return original_load(*args,**kwargs)
        torch.load=tracked_load
        try:
            wm,receipts['WM']=call('strict_WM_loader',load_world_model_checkpoint,checked(inputs['fixed_A2_checkpoint']),bundle,'A2',checked(inputs['fixed_A2_checkpoint']).parent.parent,checked(inputs['fixed_A2_health']),expected_health_auditor=health_auditor,expected_iteration=2000)
        finally:torch.load=original_load
        require(len(load_calls)==2 and all(Path(v).samefile(checked(inputs['fixed_A2_checkpoint'])) for v in load_calls),'Strict loader two internal checkpoint reads required')
        spec=read(checked(inputs['objective_spec']))
        penalty=call('objective_factory',build_penalty_only_objective,spec,device='cpu')
        objective=call('objective_factory',build_contact_objective,spec,device='cpu')
        modules={'WM':wm,'P_ONLY':penalty,'A2':objective}
        require(all(not m.training and all(not t.requires_grad for t in m.parameters()) for m in modules.values()),'Frozen eval-only modules required')
        require(_tensor_digest(penalty.base.state_dict())==_tensor_digest(objective.state_dict()),'Both actual factories must have identical registered constants')
        before_modules={k:_tensor_digest(m.state_dict()) for k,m in modules.items()}
        norm_before=_tensor_digest({k:v for k,v in wm.state_dict().items() if 'norm.' in k})
        receipts['P_ONLY']=penalty.metadata;receipts['A2']=objective.metadata
        _write(out,'loader_factory_receipts.json',receipts)
        ids,physical_phases=physical_layout();states=np.concatenate((a['donor_next_x'],a['branch_x_next']),axis=0)
        xt=torch.from_numpy(states.copy());x=torch.from_numpy(a['snapshot_x'].copy());u=torch.from_numpy(a['snapshot_u'].copy()).requires_grad_(True)
        before_tensors=_tensor_digest({'physical_x':xt,'x':x,'u':u})
        phase='physical_actual_geometry_scores'
        ps,pt,pcap=call('physical_objective_forward',evaluate_objective,penalty,xt,physical_phases)
        az,at,acap=call('physical_objective_forward',evaluate_objective,objective,xt,physical_phases)
        physical_channels={k:v.detach().numpy().copy() for k,v in channels(ps,pt,az,at).items()}
        _npwrite(out,'physical_scores.npz',physical_channels)
        _npwrite(out,'physical_proxy_P_ONLY.npz',pcap);_npwrite(out,'physical_proxy_A2.npz',acap)
        _write(out,'physical_proxy_diagnostics.json',proxy_diagnostics(acap))
        phase='one_fixed_WM_prediction'
        pred,logits=call('WM_forward',wm,x,u)
        pred_phases=np.asarray(PHASES*2,dtype=np.int64)
        pz,pterms,predpcap=call('predicted_objective_forward',evaluate_objective,penalty,pred,pred_phases)
        bz,bterms,predacap=call('predicted_objective_forward',evaluate_objective,objective,pred,pred_phases)
        predicted_channels=channels(pz,pterms,bz,bterms)
        _npwrite(out,'predicted_scores.npz',{k:v.detach().numpy().copy() for k,v in predicted_channels.items()})
        _npwrite(out,'predicted_proxy_P_ONLY.npz',predpcap);_npwrite(out,'predicted_proxy_A2.npz',predacap)
        _write(out,'predicted_proxy_diagnostics.json',proxy_diagnostics(predacap))
        _npwrite(out,'WM_prediction_before_AD.npz',{'predicted_x':pred.detach().numpy().copy(),'self15_logits':logits.detach().numpy().copy(),'snapshot_x':a['snapshot_x'],'actualu':a['snapshot_u']})
        phase='four_input_VJPs'
        grads={}
        for i,channel in enumerate(('BP','A2','CS','raw_P')):
            g,=call('u_VJP',torch.autograd.grad,predicted_channels[channel].sum(),u,retain_graph=i<3)
            require(g.shape==(8,20) and bool(torch.isfinite(g).all()),'Finite input gradient20 required')
            grads[channel]=g.detach().numpy().copy()
        grads['raw_S']=grads['CS'].copy() # Fixed declared source weights1; verified above.
        arithmetic={'BP_plus_support_vs_A2_maxabs':float(abs(physical_channels['BP']+physical_channels['CS']-physical_channels['A2']).max()),
          'gBP_vs_minus_point1gP_maxabs':float(abs(grads['BP']+.1*grads['raw_P']).max()),
          'gA2_vs_gCS_minus_point1gP_maxabs':float(abs(grads['A2']-grads['CS']+.1*grads['raw_P']).max()),
          'scope':'Arithmetic diagnostics, not replacing actual module primary values or certifying physical Jacobians'}
        _npwrite(out,'WM_predictions_gradients.npz',{'predicted_x':pred.detach().numpy().copy(),'self15_logits':logits.detach().numpy().copy(),
            **{'gradient_'+k:v for k,v in grads.items()},'snapshot_x':a['snapshot_x'],'actualu':a['snapshot_u'],
            'physical_phases':physical_phases,'predicted_phases':pred_phases,'target64_physical':objective.reference_support64[torch.from_numpy(physical_phases)].detach().numpy().copy(),
            'target64_predicted':objective.reference_support64[torch.from_numpy(pred_phases)].detach().numpy().copy()})
        phase='readonly_and_classification'
        invariants={'input_arrays_unchanged':array_digest(a)==before_arrays,'input_tensors_unchanged':_tensor_digest({'physical_x':xt,'x':x,'u':u})==before_tensors,
          'all_module_weights_and_buffers_unchanged':all(_tensor_digest(m.state_dict())==before_modules[k] for k,m in modules.items()),
          'normalizers_unchanged':_tensor_digest({k:v for k,v in wm.state_dict().items() if 'norm.' in k})==norm_before,
          'RNG_unchanged':rng_digest()==rng_before,'CUDA_uninitialized':not torch.cuda.is_initialized(),'Warp_not_imported':'warp' not in sys.modules,
          'no_parameter_gradients':all(t.grad is None for m in modules.values() for t in m.parameters())}
        require(all(invariants.values()),'CPU read-only invariance failed')
        for value in c['source_pins'].values():checked(value)
        for value in inputs.values():checked(value)
        require(pin(config_path)==p['config_pin'],'Runplan changed')
        comparisons_rows,classifications,noises=comparisons(physical_channels,grads,a['direction__d_eff'],a['direction__center_offset'],a['direction__clipped'])
        stats=summaries(comparisons_rows)
        for name,value in [('comparisons.json',comparisons_rows),('classifications.json',classifications),('component_noise.json',noises),('statistics.json',stats),('arithmetic_diagnostics.json',arithmetic)]:_write(out,name,value)
        with (out/'comparisons.csv').open('x',newline='',encoding='utf-8') as f:
            fields=['channel','snapshot_id','path','k','joint_index','epsilon_rad','physical_slope','WM_AD_effective','WM_AD_nominal_secondary','clipped','physical_label','numerator_floor','absolute_error','normalized_error','sign_agreement']
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(comparisons_rows)
        report={'schema':'full151105_trained_bonus_CPU_analysis:1','status':'COMPLETED_CPU_SHADOW_TRAINED_BONUS_SECANT_COMPARISON','config':p['config_pin'],'protocol':c['protocol'],
          'source_pins':c['source_pins'],'input_pins':inputs,'counts':counts,'actual_torch_load_calls':len(load_calls),'physical_scored_rows':160,
          'passive_MJB_constructor_scope':'3 source-accounted calls: strict Actor validation inside WMloader1 plus two objective factories1each; not separately instrumented, not physics. NumPy/SciPy sampling occurs inside the factories.',
          'WM_predicted_rows':8,'comparison_rows':320,'direction_groups_per_channel':32,'channels':list(CHANNELS),'invariants':invariants,
          'whole_hidden_state_qualified':False,'physical_gradient_qualified':False,'GPUAMP_training_gradient_reproduced':False,'Env_created':False,'GPU_used':False,'physics_steps':0,
          'scope':'CPU FP32shadow exact trained functions; pure sampled geometry on measured nextstate is not measured contact/force. All32 dependent groups/twoeps retained per channel; clipped orthogonal. Nsource1/Nseed1/Nego0. No originaltraining causalbenefit claim.',
          'elapsed_seconds':time.perf_counter()-started,'statistics':stats}
        _write(out,'report.json',report);_write(out,'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()})
        return report
    except BaseException as error:
        _write(out,'failure.json',{'schema':'full151105_trained_bonus_CPU_failure:1','status':'FAILED_CPU_PARTIAL_NOT_RETRIED','error':repr(error),'phase':phase,'counts':counts,
           'invariants_available':invariants,'config':p['config_pin'],'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'remaining_groups':'NOT_RUN; saved partial files preserved','elapsed_seconds':time.perf_counter()-started})
        raise
