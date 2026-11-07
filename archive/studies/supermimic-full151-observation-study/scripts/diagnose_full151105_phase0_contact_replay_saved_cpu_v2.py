"""Describe saved A2/0 repeat differences without a new gate or native call."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151105_gradient_empirical_gradient05_v2'
OUT=ROOT/'outputs/full151105_A2_phase0_contact_repeat_diagnostic_v2'
def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def magnitude(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return float(np.abs(a-b).max()) if a.size else 0.
def signature(s):return {'raw_wp_contact':s['raw_wp_contact'],'raw_efc_type':s['raw_efc_type'],'contacts_without_force':[dict((k,v) for k,v in c.items() if k!='force6') for c in s['contacts']]}
def main():
    OUT.mkdir(exist_ok=False)
    samples=[json.loads(x) for x in (RUN/'physical_samples.jsonl').read_text().splitlines()]
    rewards=[json.loads(x) for x in (RUN/'reward_snapshots.jsonl').read_text().splitlines()]
    truth=[s for s in samples if s['path']=='A2' and s['mode']=='donor' and s['phase']==0]
    truth_reward=next(q for q in rewards if q['path']=='A2' and q['mode']=='donor' and q['k']==0)
    assert len(truth)==4
    comparisons=[];changes=[]
    statefields=('pre_qpos','pre_qvel','post_qpos','post_qvel','ctrl','actuator_force')
    for bid in (76,77,78):
        branch=[s for s in samples if s['branch_id']==bid];q=next(q for q in rewards if q['branch_id']==bid);assert len(branch)==4
        steps=[]
        for t,s in zip(truth,branch):
            changed_contacts=[]
            assert len(t['contacts'])==len(s['contacts'])
            for ci,(c,b) in enumerate(zip(t['contacts'],s['contacts'])):
                different=[k for k in c if c[k]!=b[k]]
                if different:
                    item={'branch_id':bid,'substep':s['substep'],'pre_time_s':s['pre_time'],'post_time_s':s['post_time'],'contact_index':ci,'geom_pair':c['geom'],'efc_address':c['efc_address'],'changed_fields':different,'truth_force6':c['force6'],'repeat_force6':b['force6'],'force6_maxabs_difference':magnitude(c['force6'],b['force6']),'repeat_first_component_float64_hex':float(b['force6'][0]).hex(),'repeat_first_component_float64_bits_le_hex':np.asarray(b['force6'][0],np.float64).tobytes().hex(),'nonzero_difference_is_F64_subnormal':bool(0<magnitude(c['force6'],b['force6'])<np.finfo(np.float64).tiny)}
                    changed_contacts.append(item);changes.append(item)
            steps.append({'substep':s['substep'],'state_control_actuator_maxdiff':{k:magnitude(t[k],s[k]) for k in statefields},'raw_efc_force_exact':t['raw_efc_force']==s['raw_efc_force'],'raw_efc_type_exact':t['raw_efc_type']==s['raw_efc_type'],'geometry_order_signature_truth':digest(signature(t)),'geometry_order_signature_repeat':digest(signature(s)),'full_contacts_truth_sha256':digest(t['contacts']),'full_contacts_repeat_sha256':digest(s['contacts']),'measured_HO_force_world_maxdiff_N':magnitude(t['measured_hand_object']['force_w_N'],s['measured_hand_object']['force_w_N']),'measured_HO_direct_torque_world_maxdiff_Nm':magnitude(t['measured_hand_object']['torque_at_contact_w_Nm'],s['measured_hand_object']['torque_at_contact_w_Nm']),'summed_HO_force_world_maxdiff_N':magnitude(t['hand_object_force_on_object_w_N'],s['hand_object_force_on_object_w_N']),'summed_HO_COMmoment_world_maxdiff_Nm':magnitude(t['hand_object_moment_about_COM_w_Nm'],s['hand_object_moment_about_COM_w_Nm']),'changed_contacts':changed_contacts})
        comparisons.append({'branch_id':bid,'steps':steps,'reward_capture_differences':{k:magnitude(truth_reward[k],q[k]) for k in ('physics','contact','torque','raw_found','raw_sensordata','matched_post20_GPU_terms_FP32','matched_post20_GPU_score_FP32','native15_GPU_object_rate_terms_secondary','native15_GPU_object_rate_score_secondary')}})
    r={'schema':'full151105_A2_phase0_contact_repeat_diagnostic:1','status':'DESCRIPTIVE_SAVED_DIFFERENCES_NO_NEW_GATE','inputs':{'samples':pin(RUN/'physical_samples.jsonl'),'rewards':pin(RUN/'reward_snapshots.jsonl'),'full_saved_audit':pin(ROOT/'outputs/full151105_full_serial_saved_review_v1/report.json'),'producer_source':pin(ROOT/'src/full151_gradient/serial_probe105_v2.py')},'auditor':pin(Path(__file__)),'snapshot':'A2/0','baseline_row_ids':[76,77,78],'comparisons':comparisons,'summary':{'changed_contact_records':len(changes),'changed_substep':1,'changed_contact_index':3,'max_force6_difference_N':max(x['force6_maxabs_difference'] for x in changes),'geometry_order_signature_all_equal':all(s['geometry_order_signature_truth']==s['geometry_order_signature_repeat'] for b in comparisons for s in b['steps']),'raw_efc_force_and_type_all_equal':all(s['raw_efc_force_exact'] and s['raw_efc_type_exact'] for b in comparisons for s in b['steps']),'state_control_actuator_and_reward_differences_all_zero':all(all(v==0 for v in s['state_control_actuator_maxdiff'].values()) for b in comparisons for s in b['steps']) and all(all(v==0 for v in b['reward_capture_differences'].values()) for b in comparisons),'total_HO_force_and_COMmoment_all_equal':all(s['summed_HO_force_world_maxdiff_N']==s['summed_HO_COMmoment_world_maxdiff_Nm']==0 for b in comparisons for s in b['steps'])},'interpretation':['The earlier full audit diagnostic combined contact payload equality with efc equality; its False does not mean efc differed. Here raw efc force/type, contact geometry/order/positions/normals/distances, all pre/post states, ctrl and actuator force are exactly equal.','Only contact index3 at the first5ms substep has normal component0 in the donor versus a finite negative F64 subnormal in all three repeats. The corresponding per-contact world-force vector changes at that subnormal scale; direct torque and summed force/COM moment are unchanged at saved arithmetic precision.','The source allocates force6 as F64 np.zeros then calls native mj_contactForce. These records alone do not identify whether the subnormal originates in decoding/host representation or an internal numerical path. No new call or causal diagnosis was made.','Task-score and six-state-channel repeatability are empirically exact at this input. Full raw contact payload is not byte-identical. This distinction preserves the original baseline gates and supplies no new threshold, failure reclassification, hidden-state certificate or HO-force-gradient qualification.'],'new_models_Env_physics_WM_GPU':False}
    assert not any(n.split('.')[0] in ('torch','mujoco','mujoco_warp','warp','mjlab','dex_supermimic') for n in sys.modules)
    (OUT/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':pin(OUT/'report.json'),'summary':r['summary']},indent=2))
if __name__=='__main__':main()
