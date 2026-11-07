"""Saved base-point selection and exactly at most2 static evaluations, no optimizer."""
from pathlib import Path
import hashlib,json,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent;PILOT=ROOT.parents[1]


def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def plain(x):
    import numpy as np
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(plain(v),f,indent=2,allow_nan=False)


def main():
    import numpy as np
    from scipy.spatial.transform import Rotation
    cfg=json.loads((ROOT/'selection_protocol_v1.json').read_text(encoding='utf8'))
    if (ROOT/'report.json').exists():raise ValueError('Fresh preparation only')
    for p in [*cfg['inputs'].values(),*cfg['source_pins'].values()]:assert pin(p['path'])==p
    rows=[json.loads(s) for s in Path(cfg['inputs']['evaluation_log']['path']).read_text(encoding='utf8').splitlines()]
    candidates=[];eligible_A=[]
    for line,row in enumerate(rows,1):
        if row['seed']!=0 or row['stage'] not in ['A','B'] or row['purpose']!='base':continue
        below=row['fixed_material_below_line_dot_m2_s2'];moment=row['fixed_material_axis_torque_Nm'];finite=bool(np.isfinite(below) and np.isfinite(moment));stable=finite and below>=0
        candidate={'source_log_line_1based':line,'stage':row['stage'],'purpose':row['purpose'],'fixed_material_COM_below':stable,'fixed_material_below_dot_m2_s2':below,'abs_material_axis_torque_Nm':abs(moment),'driver_max_scaled_violation_INCLUDES_SUPPORT_AND_BELOW':row['max_scaled_constraint_violation'],'driver_search_constraints_pass':row['search_constraints_pass'],'physical_geometry_pass':row['physical_geometry_pass'],'actual_below_line_dot_m2_s2':row.get('actual_below_line_dot_m2_s2'),'current_support_status':row['current_support_status'],'saved_pad_gaps_m':row['pad_gaps_m'],'x26':row['x26']}
        candidates.append(candidate)
        if row['stage']=='A' and stable:eligible_A.append(candidate)
    write(ROOT/'all_seed0_AB_base_candidates.json',candidates)
    selected=[]
    if eligible_A:
        chosen=min(eligible_A,key=lambda x:(x['abs_material_axis_torque_Nm'],x['source_log_line_1based']))
        selected.append({'name':'seed0_A_stable_min_abs_material_axis','rule':'StablefixedmaterialCOMbelow>=0, minimumabsoluteaxisNm, originalfileorder tie. FDexcluded. No geometry-basedfilter.','source_log_line_1based':chosen['source_log_line_1based'],'saved_source_row':chosen,'x26':np.asarray(chosen['x26'])})
    designated=json.loads(Path(cfg['inputs']['seed1_result']['path']).read_text(encoding='utf8'))
    with np.load(cfg['inputs']['seed1_state']['path'],allow_pickle=False) as a:rawseed1=a['qpos27'].copy()
    source_row=designated['selected_evaluation'];selected.append({'name':'root_designated_seed1_final_selected_state','rule':'Rootexplicitlydesignatedknownsavedendpointbeforefuturepolish/holding; not minerrorselection or independentunseenvalidation.','saved_source_row':source_row,'source_state':cfg['inputs']['seed1_state'],'x26':np.asarray(source_row['x26'])})
    write(ROOT/'selection_manifest_v1.json',{'schema':'two_saved_polish_seed_selection:1','protocol':pin(ROOT/'selection_protocol_v1.json'),'all_seed0_AB_base_candidate_count':len(candidates),'eligible_A_count':len(eligible_A),'selected':selected,'max_static_evaluations':2,'no_future_result_selection':True})
    sys.path.insert(0,str(PILOT/'retarget_gravity_v3'))
    from evaluator_v3 import StaticRetargetEvaluator
    started=time.perf_counter();E=StaticRetargetEvaluator.from_existing_pilot(max_heavy_queries=200,deadline_monotonic=started+30);results=[]
    try:
        for choice in selected:
            x=choice['x26'];s=E.evaluate(x[:20],x[20:23],x[23:],exact_all=False)
            for row in s['phone_geometry_rows']:
                if row['status']=='DISJOINT_AABB_CERTIFICATE':
                    new=E._describe_pair({'kind':'convex','v':E.phone_vertices},E._shape(row['geom_id'],s['phone_R_world'],s['phone_p_world_m']),phone=True,force_exact=True)
                    identity={k:row[k] for k in ['geom_id','name','is_pad']};row.clear();row.update(identity);row.update(new)
            if choice['name']=='root_designated_seed1_final_selected_state' and not np.array_equal(s['qpos27'],rawseed1):
                # Quaternion conversion can choose the equivalent global sign;
                # raw comparison is saved, with pose equivalence separately checked.
                same=np.array_equal(s['qpos27'][:23],rawseed1[:23]) and abs(float(s['qpos27'][23:27]@rawseed1[23:27]))>=1-1e-14
                if not same:raise ValueError('Designatedseed1state mismatch')
            search=E.constraint_vector(s,pad_gap_window_m=(50e-6,150e-6),nonpad_buffer_m=50e-6,self_buffer_m=50e-6);physical=E.constraint_vector(s,pad_gap_window_m=(0.,200e-6),nonpad_buffer_m=0.,self_buffer_m=0.)
            pads=[r for r in s['phone_geometry_rows'] if r['is_pad']];nonpads=[r for r in s['phone_geometry_rows'] if not r['is_pad']];worst_nonpad=min(nonpads,key=lambda x:x['lower_bound_m']);worst_self=min(s['self_geometry_rows'],key=lambda x:x['lower_bound_m'])
            state_path=ROOT/(choice['name']+'_state.npz');np.savez_compressed(state_path,qpos27=s['qpos27'],qvel26=np.zeros(26),ctrl20=np.zeros(20),x26=x,geom_xpos=E.d.geom_xpos.copy(),geom_xmat=E.d.geom_xmat.copy(),site_xpos=E.d.site_xpos.copy(),body_xpos=E.d.xpos.copy(),body_xquat=E.d.xquat.copy(),phone_R_world=s['phone_R_world'],phone_p_world=s['phone_p_world_m'])
            result={'name':choice['name'],'selection':choice,'state':s,'search_constraints':search,'physical_constraints':physical,'geometry_search_violation_m':float(np.maximum(-search['values_m'],0).max()),'physical_violation_m':float(np.maximum(-physical['values_m'],0).max()),'pad_signed_gaps_m':[r['exact_gap_m'] for r in pads],'worst_nonpad':worst_nonpad,'worst_self':worst_self,'actual_pair_statuses':[c['pairing']['status'] for c in s['contacts']],'support_operands_available':s['support_operands']['available'],'saved_state':pin(state_path),'polish_interpretation':'Freezingphoneposepreservesfixedmaterialgravityline; q20onlymayclosepads, but requiresallhardgeometry+uniquecontact requalification. Notyetfeasibleorholding.'}
            write(ROOT/(choice['name']+'_evaluation.json'),result);results.append(result)
        report={'schema':'fixed_phone_polish_two_seed_static_preparation:1','status':'COMPLETED_SAVED_SELECTION_AND_TWO_STATIC_EVALUATIONS_NOT_OPTIMIZATION','protocol':pin(ROOT/'selection_protocol_v1.json'),'source':pin(__file__),'selection_manifest':pin(ROOT/'selection_manifest_v1.json'),'all_seed0_AB_base_candidates':pin(ROOT/'all_seed0_AB_base_candidates.json'),'selected_count':len(selected),'counts':E.counts,'no_LP_optimization_physics_training_GPU':True,'elapsed_seconds':time.perf_counter()-started,'seeds':[{'name':r['name'],'fixed_material_axis_Nm':r['state']['fixed_material_line_gravity_torque_Nm'],'pad_gaps_m':r['pad_signed_gaps_m'],'geometry_search_violation_m':r['geometry_search_violation_m'],'physical_violation_m':r['physical_violation_m'],'actual_pair_statuses':r['actual_pair_statuses'],'support_available':r['support_operands_available'],'worst_nonpad_name':r['worst_nonpad']['name'],'worst_nonpad_gap_m':r['worst_nonpad']['exact_gap_m'],'worst_self_names':r['worst_self']['names'],'worst_self_lower_or_gap_m':r['worst_self']['lower_bound_m'],'unknown_count':len(r['state']['unknown']),'saved_state':r['saved_state']} for r in results]}
        report['outputs']={p.name:pin(p) for p in ROOT.iterdir() if p.is_file() and p.name!='report.json'};write(ROOT/'report.json',report);print(json.dumps(plain({'report':pin(ROOT/'report.json'),'seeds':report['seeds'],'counts':E.counts})))
    except BaseException as error:
        write(ROOT/'failure.json',{'error':repr(error),'completed_states':len(results),'counts':E.counts,'optimization_physics_training_GPU':0});raise


if __name__=='__main__':main()
