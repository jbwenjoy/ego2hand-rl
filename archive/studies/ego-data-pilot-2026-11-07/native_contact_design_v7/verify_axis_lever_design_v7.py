"""Verify one conditional COM-axis design with real native contact geometry.

Recomputes all contact positions/J/PD operands at the moved phone pose, then
tests strict force balance plus the original moment tolerance. No steps here.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from evaluator_contact_design_v7 import NativeContactDesignEvaluator
from scan_body_axis_reference_v7 import pin,read,write,checked,lp_from_probe
from strict_support_lp_v7 import solve_strict_support

ROOT=Path(__file__).resolve().parent


def raw_equal(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return bool(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true')
    a=ap.parse_args();p=read(a.protocol)
    if p['schema']!='conditional_axis_reference_native_verification_protocol:7':raise ValueError('Schema')
    for rec in [*p['inputs'].values(),*p['source_pins'].values()]:checked(rec)
    if p['source_pins']['runner']!=pin(__file__):raise ValueError('Frozen runner')
    design=read(checked(p['inputs']['conditional_axis_design']))
    dr=design['result'];delta=dr['delta_bodyZ_m']
    if delta is None or not np.isfinite(delta) or not 0<=delta<=.075:raise ValueError('One known fixed conditional-axis candidate')
    if (dr['solver_status']!=0 or dr['all_solution_values_finite'] is not True
        or dr['equality_scaled_max_abs']>1e-8 or dr['max_inequality_violation']>1e-8
        or any(min(dr[k])< -1e-8 for k in ('torque_slack_lower20_Nm','torque_slack_upper20_Nm','ctrl_lower_slack20_rad','ctrl_upper_slack20_rad'))):
        raise ValueError('Numerically valid conditional design, not status alone')
    out=Path(p['output_path'])
    if out.exists():raise ValueError('No overwrite/retry')
    if not a.run:print(json.dumps({'status':'METADATA_ONLY_NO_MODEL','delta_bodyZ_m':delta,'native_forward_max':3,'geometry_max':1,'LP_max':2}));return
    if p['CPU_run_authorized'] is not True:raise ValueError('Explicit CPU run binding')
    out.mkdir();write(out/'protocol.json',p);e=None;LP_calls=0;start=time.perf_counter();phase='load'
    try:
        old=read(checked(p['inputs']['native_search_report']))
        oldc=next(c for s in old['seed_results'] if s['seed']==0 for c in s['shortlist'] if c['rank']==0)
        x=np.asarray(oldc['x46']);R0=np.asarray(oldc['native']['raw']['phone_body_R_world'])
        with np.load(checked(p['inputs']['initial_state']),allow_pickle=False) as z:
            if not raw_equal(z['x46'],x):raise ValueError('Same original seed')
        e=NativeContactDesignEvaluator.from_existing_pilot(max_heavy_queries=3000,digest_check_period=1)
        if not np.array_equal(e.m.opt.gravity,[0.,0.,-9.81]):raise ValueError('Original robot gravity')
        t=x[20:23]+e.Rp.T@R0@np.array([0.,0.,delta])
        phase='fresh_relocated_geometry'
        probe=e.evaluate(x[:20],t,x[23:26],x[26:]);write(out/'initial_probe.json',probe)
        targets=read(checked(p['inputs']['fixed_human_material_targets']))['preparation']
        target={i+1:np.asarray(v['target_projected_phone_convex_object_m']) for i,v in enumerate(targets)}
        material=[]
        for con in probe['contact_design'].get('phone_hand_contacts') or []:
            point=(np.asarray(con['position_world_m'])-probe['raw']['phone_root_world_m'])@probe['raw']['phone_body_R_world']
            finger=con['finger_index']
            material.append({'native_contact_index':con['native_contact_index'],'finger_index':finger,
              'point_in_phone_body_m':point,'original_human_material_target_body_m':target.get(finger),
              'material_distance_to_original_target_m':None if finger not in target else float(np.linalg.norm(point-target[finger]))})
        geometry=e.final_geometry(x[:20],t,x[23:26],pad_gap_window_m=[-.0001,.0002]);write(out/'geometry.json',geometry)
        lp,reason=lp_from_probe(probe)
        if lp is not None:LP_calls+=1
        write(out/'minimax_LP.json',{'LP':lp,'unavailable_reason':reason})
        strict=None;native=[];repeat={};entry={};state=None
        if geometry['geometry_gate_pass'] and lp is not None and lp['hard_bounds_pass']:
            phase='strict_moment_box_support_LP';pd=probe['contact_design']['PD_raw']
            strict=solve_strict_support(lp,pd_gain20=pd['PD_gain20'],pd_offset20=pd['PD_offset20'],
              ctrl_range20x2=pd['actuator_ctrlrange20x2'],moment_tolerance_Nm=1e-4,force_tolerance_N=1e-8)
            LP_calls+=1;write(out/'strict_support_LP.json',strict)
            if strict['strict_force_moment_pass'] and strict['hard_bounds_pass']:
                for i in range(2):
                    phase='proposed_control_fresh_repeat_'+str(i)
                    v=e.evaluate(x[:20],t,x[23:26],strict['proposed_ctrl20']);native.append(v)
                    write(out/('command_probe'+str(i)+'.json'),v)
                for key in ('qpos27','qvel26','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26','a_COM_world_m_s2','alpha_world_rad_s2'):
                    repeat[key]=raw_equal(native[0]['raw'][key],native[1]['raw'][key])
                repeat['physical_contact_rows_without_probe_id']=raw_equal(np.asarray(native[0]['raw']['contact_rows24'])[:,1:],np.asarray(native[1]['raw']['contact_rows24'])[:,1:])
                v=native[0];ag=v['native_contact_aggregates']
                entry={'native_known_no_warning':not v['unknown'] and not native[1]['unknown'],
                  'both_pads_loaded':all(ag[k]['normal_force_signed_N']>.1 for k in ('thumb_pad','index_pad')),
                  'initial_each_pad_at_most3N':all(ag[k]['normal_force_signed_N']<=3. for k in ('thumb_pad','index_pad')),
                  'no_native_nonpad_table_force':all(ag[k]['normal_force_signed_N']<=1e-8 for k in ('phone_nonpad','phone_table')),
                  'requested_ctrl_legal_and_unclamped':raw_equal(v['PD']['requested_ctrl20'],v['PD']['effective_ctrl20']),
                  'all_fresh_physical_repeats_bit_equal':all(repeat.values()),
                  'complete_independent_geometry':geometry['geometry_gate_pass'],
                  'strict_ideal_force_moment_and_real_PD':strict['strict_force_moment_pass'] and strict['hard_bounds_pass']}
                state=out/'candidate_state.npz'
                np.savez_compressed(state,qpos27=v['raw']['qpos27'],qvel26=v['raw']['qvel26'],ctrl20=v['raw']['ctrl20'],
                  x46=np.r_[x[:20],t,x[23:26],v['raw']['ctrl20']])
        audit=e.audit_model();audit['passed']=True
        known=bool(entry and all(entry.values()))
        write(out/'report.json',{'schema':'conditional_axis_reference_native_verification_result:7',
          'status':'COMPLETED_ONE_REAL_REFERENCE_VERIFICATION','protocol':pin(a.protocol),
          'delta_bodyZ_m':delta,'source_conditional_design':p['inputs']['conditional_axis_design'],
          'phone_translation_palm_m':t,'robot_q20':x[:20],'phone_rotvec_palm_rad':x[23:26],
          'native_initial_probe':pin(out/'initial_probe.json'),'geometry':pin(out/'geometry.json'),
          'minimax_LP':pin(out/'minimax_LP.json'),'strict_support_LP':None if strict is None else pin(out/'strict_support_LP.json'),
          'proposed_control_native_probes':[pin(out/('command_probe'+str(i)+'.json')) for i in range(len(native))],
          'fresh_repeat_raw_bit_equal':repeat,'entry_gates':entry,'holding_diagnostic_eligible':known,
          'candidate_state':None if state is None else pin(state),'model_audit':audit,'counts':e.counts,'LP_calls':LP_calls,
          'world_translation_delta_m':R0@np.array([0.,0.,delta]),
          'original_ego_material_reference_displacement_hypothesis_m':delta,
          'source_material_contact_diagnostics':material,
          'faithful_ego_or_holding_or_policy_qualified':False,'new_native_steps_training_GPU':0,
          'wall_seconds':time.perf_counter()-start,
          'scope':'Actual same fixed model/wrist/q20 at a new phone body-axis reference; newly generated contacts/true PD recomputed. Ideal support and repeated native preload only. Original human material targets remain unchanged; large displacement is an engineering reference adaptation, not faithful ego tracking or holding.'})
    except BaseException as err:
        write(out/'failure.json',{'phase':phase,'error':repr(err),'counts':None if e is None else e.counts,
          'LP_calls':LP_calls,'automatic_retry':False});raise


if __name__=='__main__':main()
