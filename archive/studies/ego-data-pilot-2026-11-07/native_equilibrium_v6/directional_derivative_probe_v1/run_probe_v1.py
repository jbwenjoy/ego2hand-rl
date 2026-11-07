"""Preregistered at-most26 fresh native forward calls, no integration/search."""
from pathlib import Path
import hashlib,importlib.util,json,math,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def need(x,label):
    if not x:raise ValueError(label)
def plain(x):
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not math.isfinite(x):return {'nonfinite_diagnostic':repr(x)}
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(plain(x),f,indent=2,allow_nan=False)
def digest(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def stats(v):
    v=np.asarray(v);return {'maxabs':float(np.max(abs(v),initial=0)),'RMS':float(np.sqrt(np.mean(v*v))) if v.size else None}
def block_stats(v):return {k:stats(np.asarray(v)[s]) for k,s in [('COM',slice(0,3)),('moment',slice(3,6)),('joint',slice(6,26))]}


def main():
    cfg=read(HERE/'preregistration.json');need(pin(__file__)==cfg['source'],'source freeze')
    for v in cfg['input_pins'].values():need(pin(v['path'])==v,'pinned input/source '+v['path'])
    parent=read(cfg['input_pins']['native_report']['path']);pc=read(cfg['input_pins']['native_protocol']['path'])
    for v in [*pc['inputs'].values(),*pc['source_pins'].values()]:need(pin(v['path'])==v,'parent closure pin')
    rank=next(x for x in parent['seed_results'][0]['shortlist'] if x['rank']==0)
    x0=np.array(cfg['baseline_x46']);need(np.array_equal(x0,np.array(rank['x46'])),'fixed rank0 x46')
    with np.load(cfg['input_pins']['baseline_state']['path'],allow_pickle=False) as z:need(np.array_equal(x0,z['x46']),'rank0 raw state identity')
    actual=HERE/'actual_v1';actual.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('frozen_native_evaluator_for_directional_probe',cfg['input_pins']['native_evaluator']['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    started=time.perf_counter();E=module.NativeEquilibriumEvaluator.from_existing_pilot(max_heavy_queries=0,deadline_monotonic=None,digest_check_period=1)
    m=E.m;oid=E.oid;freej=int(m.body_jntadr[oid]);need(not np.any(m.dof_damping[20:26]) and not np.any(m.dof_frictionloss[20:26]) and m.jnt_stiffness[freej]==0,'same passive-free phone6')
    mass=float(m.body_mass[oid]);gmag=float(np.linalg.norm(m.opt.gravity));Mscale=mass*gmag*.077
    need(mass==.2 and abs(gmag-9.81)<1e-12,'frozen nominal physical scale')
    directions=[]
    for d in cfg['directions']:
        d=dict(d);index=d['x46_index'];bound=None
        if index<20:bound=m.jnt_range[index].copy()
        elif index>=26:bound=m.actuator_ctrlrange[index-26].copy()
        available=True;reason=None
        if bound is not None:
            # Every level must be valid; any touching/outside endpoint rejects
            # this complete direction rather than choosing smaller levels.
            available=all(bound[0]<x0[index]-h and x0[index]+h<bound[1] for h in d['absolute_h'])
            if not available:reason='At least one registered symmetric endpoint touches/exceeds a true native bound; entire direction unavailable'
        d.update(available=available,reason=reason,baseline_coordinate=float(x0[index]),true_bound=bound.tolist() if bound is not None else None)
        directions.append(d)
    planned=2+sum(2*len(d['absolute_h']) for d in directions if d['available']);need(planned<=26,'forward budget')
    records=[];counter={'forward_calls_started':0,'constraint_Jqacc_matvec_calls':0};error=None;modelaudit=None;attempt=None
    def evaluate(name,x,direction=None,h=None,sign=0):
        nonlocal attempt
        need(counter['forward_calls_started']<26,'hard26forward cap')
        need(time.perf_counter()-started<cfg['wall_budget_s'],'wall budget')
        counter['forward_calls_started']+=1;idx=len(records);attempt={'name':name,'x46':x.copy(),'direction':direction,'h':h,'sign':sign,'record_index':idx}
        s=E.evaluate(x[:20],x[20:23],x[23:26],x[26:]);raw={k:np.array(v).copy() for k,v in s['raw'].items()};d=E.last_data;n=int(d.nefc)
        # Save native raw operands before optional constraint diagnostics/math.
        fn=actual/f'probe_{idx:02d}_raw.npz';np.savez_compressed(fn,**raw)
        extra={k:np.asarray(getattr(d,k))[:n].copy() for k in ['efc_R','efc_D','efc_aref','efc_vel','efc_state']}
        Jq=np.empty(n)
        if n:E.mj.mj_mulJacVec(m,d,Jq,np.asarray(d.qacc));counter['constraint_Jqacc_matvec_calls']+=1
        extra['constraint_Jqacc']=Jq;extra['constraint_Jqacc_minus_aref']=Jq-extra['efc_aref']
        extra['warnings_number']=np.array(s['warnings_number']).copy();extra['warnings_lastinfo']=np.array(s['warnings_lastinfo']).copy()
        ef=actual/f'probe_{idx:02d}_constraint_extra.npz';np.savez_compressed(ef,**extra)
        contact_sig=[]
        for row,detail in zip(raw['contact_rows24'],s['native_contact_details']):
            contact_sig.append([*map(int,row[1:5]),int(detail['native_exclude']),float(detail['native_adhesion'])])
        signature={'generated_ordered_geom_dim_address_exclude_adhesion':contact_sig,
          'positive_normal_contact_indices':[i for i,row in enumerate(raw['contact_rows24']) if row[18]>0],
          'constraint_type_id_state_force_sign':[[int(t),int(i),int(st),int(np.sign(f))] for t,i,st,f in zip(raw['efc_type'],raw['efc_id'],extra['efc_state'],raw['efc_force'])]}
        finite=all(np.isfinite(v).all() for v in raw.values())
        residual=np.r_[raw['a_COM_world_m_s2']/gmag,raw['phone_I_world_3x3']@raw['alpha_world_rad_s2']/Mscale,raw['qacc26'][:20]/.1] if finite else None
        context={k:v for k,v in s.items() if k!='raw'};context.update(raw_npz=pin(fn),constraint_extra_npz=pin(ef),active_signature=signature,active_signature_sha256=digest(signature))
        cf=actual/f'probe_{idx:02d}_context.json';save(cf,context)
        rec={'probe_record_index':idx,'name':name,'direction':direction,'h':h,'sign':sign,'x46':x.copy(),'raw':raw,'extra':extra,'residual':residual,
             'raw_npz':pin(fn),'constraint_extra_npz':pin(ef),'context':pin(cf),'signature':signature,'signature_sha256':digest(signature),'unknown':s['unknown'],'native_probe_index':s['probe_index']}
        records.append(rec)
        if s['unknown'] or not finite:raise ValueError('Saved native unknown/nonfinite probe; no automatic retry')
        return rec
    try:
        evaluate('baseline1',x0);evaluate('baseline2',x0)
        for d in directions:
            if not d['available']:continue
            for h in d['absolute_h']:
                for sign in [-1,1]:
                    x=x0.copy();x[d['x46_index']]+=sign*h;evaluate(f"{d['name']}__h{h:g}__{sign:+d}",x,d['name'],h,sign)
    except Exception as e:error=repr(e)
    try:modelaudit={'passed':True,**E.audit_model()}
    except Exception as e:modelaudit={'passed':False,'error':repr(e)}
    analyses=[];repeat=None
    if len(records)>=2 and records[0]['residual'] is not None and records[1]['residual'] is not None:
        r0=records[0]['residual'];rr=records[1]['residual']
        keys=['qpos27','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26','a_COM_world_m_s2','alpha_world_rad_s2']
        repeat={'physical_bit_equal':{k:bool(np.array_equal(records[0]['raw'][k],records[1]['raw'][k])) for k in keys},
          'contact_physical_columns_bit_equal':bool(np.array_equal(records[0]['raw']['contact_rows24'][:,1:],records[1]['raw']['contact_rows24'][:,1:])),
          'active_signature_equal':records[0]['signature_sha256']==records[1]['signature_sha256'],
          'physical_residual_difference26':rr-r0,'difference_by_block':block_stats(rr-r0)}
        for d in directions:
            if not d['available']:continue
            levels=[]
            for h in d['absolute_h']:
                matches=[x for x in records if x['direction']==d['name'] and x['h']==h]
                if len(matches)!=2 or any(x['residual'] is None for x in matches):continue
                plus=next(x for x in matches if x['sign']==1);minus=next(x for x in matches if x['sign']==-1)
                rp,rm=plus['residual'],minus['residual'];forward=(rp-r0)/h;backward=(r0-rm)/h;central=(rp-rm)/(2*h)
                levels.append({'h':h,'direction_units':d['units'],'forward26':forward,'backward26':backward,'central26':central,
                  'forward_minus_backward26':forward-backward,'second_difference26':rp-2*r0+rm,
                  'central_by_block':block_stats(central),'asymmetry_by_block':block_stats(forward-backward),
                  'plus_signature_equals_baseline':plus['signature_sha256']==records[0]['signature_sha256'],'minus_signature_equals_baseline':minus['signature_sha256']==records[0]['signature_sha256'],
                  'plus_minus_signature_equal':plus['signature_sha256']==minus['signature_sha256']})
            changes=[]
            for big,small in zip(levels,levels[1:]):
                a,b=big['central26'],small['central26'];den=max(float(np.max(abs(a))),float(np.max(abs(b))))
                changes.append({'h_pair':[big['h'],small['h']],'central_difference26':b-a,'difference_by_block':block_stats(b-a),
                   'relative_infinity_difference':float(np.max(abs(b-a)))/den if den>0 else None})
            analyses.append({'direction':d['name'],'levels':levels,'across_h_changes':changes})
    metadata=[{k:v for k,v in rec.items() if k not in ['raw','extra','residual']}|{'physical_residual26':rec['residual']} for rec in records]
    report={'schema':'native_physical_directional_derivative_diagnostic:1','status':'COMPLETED_PREREGISTERED_DIRECTIONAL_DIAGNOSTIC' if error is None and len(records)==planned and modelaudit['passed'] else 'PARTIAL_OR_UNKNOWN_DIAGNOSTIC_PRESERVED',
       'preregistration':pin(HERE/'preregistration.json'),'source':pin(__file__),'input_pins':cfg['input_pins'],'baseline_x46':x0,
       'directions':directions,'registered_forward_count':planned,'wrapper_counts':counter,'native_counts':E.counts,'model_audit':modelaudit,
       'error':error,'failed_attempt':attempt if error else None,'elapsed_s':time.perf_counter()-started,'gravity_m_s2':gmag,'mass_kg':mass,'moment_scale_Nm':Mscale,'joint_acceleration_scale_rad_s2':.1,
       'records':metadata,'baseline_repeat':repeat,'directional_results':analyses,
       'scope':'Exactly fixed finite differences, no fidelity/geometry penalties, no pass/stability threshold, no h/seed/direction selection after values. Contact/constraint active signatures are descriptors, not global clearance or real force truth. Every raw array is copied from a fresh zero-speed native forward; separate JacVec is readonly on that same state. No integration/training/pose optimization/LP solve or hardware/grasp qualification.'}
    save(actual/'report.json',report);print(json.dumps({'report':pin(actual/'report.json'),'status':report['status'],'forward_count':E.counts['mj_forward_returned'],'directions':[{k:d[k] for k in ['name','available','reason']} for d in directions]}))


if __name__=='__main__':main()
