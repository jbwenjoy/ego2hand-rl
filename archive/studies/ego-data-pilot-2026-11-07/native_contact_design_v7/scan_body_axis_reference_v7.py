"""Finite body-Z contact-relocation hypotheses, not faithful ego tracking.

Same fixed wrist/model/q20; move phone root by declared body-axis offsets.
Each has one fresh native probe and independent final22/189 geometry. LPs use
all generated designated-pad points without positive-force sampling filters.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from evaluator_contact_design_v7 import NativeContactDesignEvaluator
from native_wrench_lp_v7 import solve_native_wrench

ROOT=Path(__file__).resolve().parent;P=ROOT.parent
OFFSETS_M=[0.,.0245,.049]


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def checked(x):
    if pin(x['path'])!=x:raise ValueError('Changed input '+x['path'])
    return Path(x['path'])


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    return x


def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(x),f,indent=2,allow_nan=False)


def lp_from_probe(probe):
    cd=probe['contact_design'];contacts=cd['phone_hand_contacts']
    if not cd['available'] or not contacts:return None,'No known generated pad contacts'
    if (any(c['finger_group'] not in ('finger1','finger2')
          or c['hand_geom_name'] not in ('right_finger1_tip_sensor_col','right_finger2_tip_sensor_col')
          or c['native_condim']!=3 or c['native_efc_address']<0 or c['native_exclude']!=0 for c in contacts)
        or {c['finger_group'] for c in contacts}!={'finger1','finger2'}):
        return None,'All generated hand contacts must be active designated pads on both fingers; others retained as rejection'
    pd=cd['PD_raw']
    if not all(np.all(pd[k]) for k in ('actuator_ctrllimited20','actuator_forcelimited20','joint_actfrclimited20')):
        raise ValueError('Explicit original limited flags')
    if pd['native_disabled_actuator_group_bitmask']!=0:raise ValueError('No disabled actuators')
    if not pd['direct_unit_joint_mapping_verified_by_frozen_factory']:raise ValueError('Original unit joint mapping')
    b=np.asarray(pd['qfrc_bias26'])[:20]-np.asarray(pd['qfrc_passive26'])[:20]
    return solve_native_wrench(points_world=[c['position_world_m'] for c in contacts],
      frames_rows=[c['frame_rows_normal_t1_t2_world'] for c in contacts],
      force_on_object_sign=[c['force_on_object_sign'] for c in contacts],
      contact_dims=[c['native_condim'] for c in contacts],friction5=[c['native_friction5'] for c in contacts],
      J_hand_3x20=[c['hand_Jp3x20_world'] for c in contacts],finger_groups=[c['finger_group'] for c in contacts],
      COM_world=cd['phone_COM_world_m'],mass_kg=probe['model_configuration']['phone_mass_kg'],
      robot_bias_minus_passive20=b,pd_gain20=pd['PD_gain20'],pd_offset20=pd['PD_offset20'],
      ctrl_range20x2=pd['actuator_ctrlrange20x2'],actuator_force_range20x2=pd['actuator_forcerange20x2'],
      joint_force_range20x2=pd['joint_actfrcrange20x2'],normal_cap_per_finger_N=3.),None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true')
    a=ap.parse_args();p=read(a.protocol)
    if p['schema']!='fixed_wrist_bodyZ_contact_relocation_protocol:7' or p['bodyZ_offsets_m']!=OFFSETS_M:raise ValueError('Frozen hypotheses')
    for rec in p['inputs'].values():checked(rec)
    for rec in p['source_pins'].values():checked(rec)
    if p['source_pins']['runner']!=pin(__file__):raise ValueError('Source binding')
    fixture=read(checked(p['inputs']['native_J_contract']))
    analytic=read(checked(p['inputs']['analytic_LP_contract']))
    if (fixture['passed']!=87 or fixture['total']!=87 or any(v is not True for v in fixture['checks'].values())
        or analytic['passed']!=4 or analytic['total']!=4 or any(v is not True for v in analytic['checks'].values())):
        raise ValueError('Prior contracts')
    out=Path(p['output_path'])
    if out.exists():raise ValueError('No overwrite/retry')
    if not a.run:print(json.dumps({'status':'METADATA_ONLY_NO_MODEL','fixed_initial_probes':3,'maximum_additional_ctrl_probes':3}));return
    if p['CPU_run_authorized'] is not True:raise ValueError('Explicit CPU binding')
    out.mkdir();write(out/'protocol.json',p);e=None;rows=[];phase='load';start=time.perf_counter();LP_calls=0
    try:
        parent=read(checked(p['inputs']['native_search_report']))
        seed=next(s for s in parent['seed_results'] if s['seed']==0)
        c=next(c for c in seed['shortlist'] if c['rank']==0);x=np.asarray(c['x46'])
        with np.load(checked(p['inputs']['initial_state']),allow_pickle=False) as z:
            if not np.array_equal(z['x46'],x):raise ValueError('Exact seed binding')
        e=NativeContactDesignEvaluator.from_existing_pilot(max_heavy_queries=5000,digest_check_period=1)
        if not np.array_equal(e.m.opt.gravity,np.array([0.,0.,-9.81])):raise ValueError('Original robot gravity')
        R0=np.asarray(c['native']['raw']['phone_body_R_world']);p0=np.asarray(c['native']['raw']['phone_root_world_m'])
        targets=read(checked(p['inputs']['fixed_human_material_targets']))['preparation']
        target={i+1:np.asarray(t['target_projected_phone_convex_object_m']) for i,t in enumerate(targets)}
        for index,delta in enumerate(OFFSETS_M):
            phase='reference_'+str(index);t=x[20:23]+e.Rp.T@R0@np.array([0.,0.,delta])
            probe=e.evaluate(x[:20],t,x[23:26],x[26:])
            write(out/('reference_'+str(index)+'_initial_probe.json'),probe)
            geometry=e.final_geometry(x[:20],t,x[23:26],pad_gap_window_m=[-.0001,.0002])
            lp,reason=lp_from_probe(probe)
            if lp is not None:LP_calls+=1
            material=[]
            for contact in probe['contact_design'].get('phone_hand_contacts') or []:
                point=(np.asarray(contact['position_world_m'])-probe['raw']['phone_root_world_m'])@probe['raw']['phone_body_R_world']
                finger=contact['finger_index']
                material.append({'native_contact_index':contact['native_contact_index'],'finger_index':finger,
                  'point_in_phone_body_m':point,'original_human_material_target_body_m':target.get(finger),
                  'material_distance_to_original_target_m':None if finger not in target else float(np.linalg.norm(point-target[finger]))})
            eligible=bool(not probe['unknown'] and geometry['geometry_gate_pass']
              and lp is not None and lp['hard_bounds_pass'] and lp['strict_force_moment_pass'])
            command_probe=None;state=None
            if eligible:
                phase='inverse_control_probe_'+str(index)
                command_probe=e.evaluate(x[:20],t,x[23:26],lp['proposed_ctrl20'])
                state=out/('reference_'+str(index)+'_state.npz')
                np.savez_compressed(state,qpos27=command_probe['raw']['qpos27'],qvel26=command_probe['raw']['qvel26'],
                  ctrl20=command_probe['raw']['ctrl20'],x46=np.r_[x[:20],t,x[23:26],lp['proposed_ctrl20']])
            row={'index':index,'bodyZ_offset_m':delta,'world_translation_delta_m':R0@np.array([0.,0.,delta]),
              'translation_palm_m':t,'phone_R_world_unchanged':True,'robot_q20_unchanged':True,
              'probe':probe,'geometry':geometry,'LP':lp,'LP_unavailable_reason':reason,
              'source_material_contact_diagnostics':material,'ideal_geometry_PD_candidate':eligible,
              'proposed_control_native_probe':command_probe,'candidate_state':None if state is None else pin(state),
              'faithful_ego_reference_or_holding_qualified':False}
            rows.append(row);write(out/('reference_'+str(index)+'.json'),row)
            print(json.dumps({'index':index,'offset_mm':delta*1000,'geometry':geometry['geometry_gate_pass'],
              'native_contacts':probe['native_contact_count'],'LP_loss':None if lp is None else lp['scaled_loss'],
              'strict_ideal_candidate':eligible}),flush=True)
        audit=e.audit_model();audit['passed']=True
        write(out/'report.json',{'schema':'fixed_wrist_bodyZ_contact_relocation_result:7','protocol':pin(a.protocol),
          'status':'COMPLETED_FIXED_THREE_REFERENCE_HYPOTHESES','references':[{'index':r['index'],
            'bodyZ_offset_m':r['bodyZ_offset_m'],'result':pin(out/('reference_'+str(r['index'])+'.json')),
            'ideal_geometry_PD_candidate':r['ideal_geometry_PD_candidate'],'candidate_state':r['candidate_state']} for r in rows],
          'candidate_count':sum(r['ideal_geometry_PD_candidate'] for r in rows),'counts':e.counts,'LP_calls':LP_calls,
          'model_audit':audit,'wall_seconds':time.perf_counter()-start,'new_training_physics_steps':0,
          'scope':'Geometric contact relocation hypotheses on real ego phone asset, original wrist/model/q20. Human material target deviations explicitly reported. Ideal force proposals only; no native soft-equilibrium, faithful ego tracking, policy or holding qualification.'})
    except BaseException as err:
        write(out/'failure.json',{'phase':phase,'error':repr(err),'completed_references':len(rows),
          'counts':None if e is None else e.counts,'LP_calls':LP_calls,'automatic_retry':False});raise


if __name__=='__main__':main()
