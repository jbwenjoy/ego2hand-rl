"""Additional saved-only q0/hash/clock/LP-witness verification; no simulation."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/motion_reference_diagnostic_v1'
INTENT=Path('D:/workspace/supermimic-observation-contact-intent-study')
GROUND=Path('D:/workspace/supermimic-observation-warmseed-grounding-study')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    r=read(OUT/'report.json');z=np.load(OUT/'motion_reference_arrays.npz',allow_pickle=False);mat=np.load(OUT/'saved_q0_cone_matrices.npz',allow_pickle=False);checks={};residuals={};identities={}
    checks['all_input_hashes_exact']=all(sha(p['path'])==p['sha256'] for p in r['input_pins'].values())
    checks['primary24_checks_pass']=all(r['checks'].values())
    for name in ('baseline','midpoint'):
        d=GROUND/f'outputs/h123_rest_{name}_v1'
        with np.load(d/'raw_states.npz',allow_pickle=False) as a:
            checks[name+'_actual_pose_saved_every4_bitexact']=np.array_equal(z[name+'__actual_object_pose_w'],a['object_pose_w'][::4])
            checks[name+'_raw_clock_saved_every4_bitexact']=np.array_equal(z[name+'__raw_time_s'],a['raw_time_s'][::4])
            checks[name+'_clock_error_only_recorded_not_retimed']=np.array_equal(z[name+'__raw_minus_reference_clock_s'],a['raw_time_s'][::4]-z['reference_time_s'])
        checks[name+'_no_new_grade']=r['physical_comparisons'][name]['new_motion_grade']=='NOT_ASSIGNED_POSTHOC_EXPLORATORY_ONLY' and read(d/'evaluation.json')['rest_hold_pass'] is False
    for name,d in (('h23_baseline',GROUND/'outputs/warmseed_rest_qref_v1'),('h123_baseline',GROUND/'outputs/h123_rest_baseline_v1'),('h123_midpoint',GROUND/'outputs/h123_rest_midpoint_v1')):
        hypothesis,variant=name.split('_');candidate=INTENT/f'outputs/frame0_{hypothesis}_seeded_{variant}_v3/initialization_candidate.npz'
        planpath=d.with_name(d.name+'_watchdog')/'plan_snapshot.json';plan=read(planpath);e=read(d/'evaluation.json')
        with np.load(candidate,allow_pickle=False) as c,np.load(d/'raw_states.npz',allow_pickle=False) as a:
            checks[name+'_q0_candidate_cast_bitexact']=np.array_equal(a['joint_rad'][0].astype(np.float32),c['joint_pos_rad'][0].astype(np.float32))
            checks[name+'_pose0_candidate_cast_bitexact']=np.array_equal(a['object_pose_w'][0].astype(np.float32),c['mapped_object_pose_w'][0].astype(np.float32))
        checks[name+'_plan_candidate_evaluation_binding']=plan['candidate']['sha256']==sha(candidate) and e['plan_sha256']==sha(planpath)
        identities[name]={'candidate_sha256':sha(candidate),'plan_sha256':sha(planpath),'original_evaluation_sha256':sha(d/'evaluation.json')}
    preload=read(INTENT/'outputs/seeded_preload_cpu_v1/designed_target_u0.json')
    with np.load(INTENT/'outputs/frame0_h23_seeded_baseline_v3/initialization_candidate.npz',allow_pickle=False) as c:
        checks['nativeCPU_q32_H23_q0_cast_binding']=np.array_equal(np.array(preload['q0_FP32_rad'],dtype=np.float32),c['joint_pos_rad'][0].astype(np.float32))
        checks['blocked_B_target_still_not_created']=preload['feasible'] is False and preload['u0_rad'] is None
    for name,c in r['saved_q0_cone_screens'].items():
        if c['status']!='SAVED_CONE_SCREEN_COMPLETE':
            checks[name+'_UNKNOWN_not_false']=c['dynamic_feasible'] is None and c['static_feasible'] is None
            continue
        A=mat[name+'__native_wrench_matrix_world']
        for condition,v in c['conditions'].items():
            if not v['feasible_in_native_object_wrench_screen']:continue
            w=np.array(v['edge_weights_N']);target=np.array(c['static_demand_world_N_Nm'] if condition=='static' else c['dynamic_demand_world_N_Nm']);delta=A@w-target
            prefix=name+'_'+condition;checks[prefix+'_nonnegative_weights']=w.min()>=-1e-10;checks[prefix+'_force_balance']=np.abs(delta[:3]).max()<1e-8;checks[prefix+'_COM_torque_balance']=np.abs(delta[3:]).max()<1e-9
            residuals[prefix]={'force_max_abs_N':float(np.abs(delta[:3]).max()),'torque_max_abs_Nm':float(np.abs(delta[3:]).max()),'sum_normal_candidate_N':float(w.sum())}
    result={'status':'SAVED_BINDINGS_AND_WITNESSES_PASS' if all(checks.values()) else 'SAVED_VERIFICATION_FAIL','checks':{k:bool(v) for k,v in checks.items()},'actual_initial_state_identity_bindings':identities,'successful_LP_witness_residuals':residuals,'report_sha256':sha(OUT/'report.json'),'verifier_sha256':sha(__file__),'new_contact_queries':0,'new_physics_steps':0,'GPU_used':False,'scope':'Saved-array and mathematical allocation checks only. Original static grades are unchanged; no posthoc motion grade or runtime feasibility is assigned.'}
    (OUT/'saved_bindings_and_witness_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'checks_passed':sum(result['checks'].values()),'checks_total':len(checks),'failed':[k for k,v in checks.items() if not v]},indent=2));raise SystemExit(not all(checks.values()))

if __name__=='__main__':main()
