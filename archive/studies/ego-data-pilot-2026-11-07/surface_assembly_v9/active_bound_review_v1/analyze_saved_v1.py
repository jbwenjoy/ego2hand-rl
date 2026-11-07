"""All four saved finals: hard-model vs algorithm-trust margins, no solve/FK."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent.parent


def pin(f):
    f=Path(f).resolve();b=f.read_bytes()
    return {'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))


def bits(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    rules=read(HERE/'rules.json')
    if (HERE/'report.json').exists():raise FileExistsError('fresh saved review only')
    for rec in [rules['source'],*rules['inputs'].values()]:assert pin(rec['path'])==rec
    report=read(rules['inputs']['producer_report']['path']);protocol=read(report['protocol']['path'])
    assert pin(report['protocol']['path'])==report['protocol']
    cfg=protocol['config'];oldpin=protocol['inputs']['old_native_report'];assert pin(oldpin['path'])==oldpin
    old=read(oldpin['path']);orig=next(s for s in old['seed_results'] if s['seed']==0)['shortlist'][0]
    q0=np.array(orig['x46'][:20]);xb=np.array(orig['x46'][20:26])
    interface=read(rules['inputs']['joint_names_interface']['path']);names=interface['joint_order']
    results=[];all_rows=[];pins=[];all_hard=None
    for seed in report['seed_results']:
        for final in seed['finals']:
            rec=final['result'];assert pin(rec['path'])==rec;pins.append(rec)
            c=read(rec['path']);v=c['candidate'];x=np.array(v['x14']);native=c['native_probe']
            q=np.array(native['raw']['qpos27'][:20]);pd=native['contact_design']['PD_raw']
            hard=np.array(pd['joint_range20x2']);assert hard.shape==(20,2)
            assert pd['direct_unit_joint_mapping_verified_by_frozen_factory']
            assert np.array_equal(np.asarray(pd['actuator_trnid20x2'])[:,0],np.arange(20))
            assert bits(q[:8],x[:8]) and bits(q[8:],q0[8:])
            if all_hard is None:all_hard=hard.copy()
            else:assert bits(all_hard,hard)
            eps=cfg['q_planner_margin_rad'];trust=cfg['q_trust_rad']
            qlo=np.maximum(hard[:8,0]+eps,q0[:8]-trust)
            qhi=np.minimum(hard[:8,1]-eps,q0[:8]+trust)
            lo_t=xb[:3]-cfg['palm_translation_trust_m'];hi_t=xb[:3]+cfg['palm_translation_trust_m']
            lo_r=xb[3:]-cfg['rotvec_coordinate_trust_rad'];hi_r=xb[3:]+cfg['rotvec_coordinate_trust_rad']
            rows=[]
            for j in range(8):
                hlo,hhi=hard[j];alo,ahi=qlo[j],qhi[j];value=q[j]
                active=lambda margin:bool(margin<=rules['near_bound_q_rad'])
                row={'seed':seed['seed'],'rank':final['rank'],'index':j,'joint':names[j],
                    'q_rad':float(value),'source_q_rad':float(q0[j]),'delta_source_rad':float(value-q0[j]),
                    'true_hard_range_rad':[float(hlo),float(hhi)],'algorithm_range_rad':[float(alo),float(ahi)],
                    'hard_lower_margin_rad':float(value-hlo),'hard_upper_margin_rad':float(hhi-value),
                    'algorithm_lower_margin_rad':float(value-alo),'algorithm_upper_margin_rad':float(ahi-value),
                    'lower_boundary_source':'true_hard_plus_planner_margin' if hlo+eps>=q0[j]-trust else 'source_minus_q_trust',
                    'upper_boundary_source':'true_hard_minus_planner_margin' if hhi-eps<=q0[j]+trust else 'source_plus_q_trust',
                    'near_true_hard_lower':active(value-hlo),'near_true_hard_upper':active(hhi-value),
                    'near_algorithm_lower':active(value-alo),'near_algorithm_upper':active(ahi-value),
                    'proposed_next_algorithm_range_rad':[float(hlo+eps),float(hhi-eps)]}
                rows.append(row);all_rows.append(row)
            phone=[]
            for kind,values,low,high,tol in [('palm_translation_m',x[8:11],lo_t,hi_t,rules['near_bound_translation_m']),
                                           ('palm_rotvec_rad',x[11:14],lo_r,hi_r,rules['near_bound_rotvec_rad'])]:
                for j in range(3):phone.append({'coordinate':kind,'axis':'xyz'[j],'value':float(values[j]),
                    'lower':float(low[j]),'upper':float(high[j]),'lower_margin':float(values[j]-low[j]),
                    'upper_margin':float(high[j]-values[j]),'near_lower':bool(values[j]-low[j]<=tol),
                    'near_upper':bool(high[j]-values[j]<=tol)})
            lp=c['current_native_LP'];minimum=float(min(np.min(lp['torque_slack_lower20']),np.min(lp['torque_slack_upper20'])))
            result={'seed':seed['seed'],'rank':final['rank'],'case_pin':rec,'joint_rows':rows,'phone_box_rows':phone,
                'rest12_bit_identical_source':True,'all_q_inside_true_hard_bounds':bool(np.all(q>=hard[:,0]) and np.all(q<=hard[:,1])),
                'assembly_gate':v['assembly_gate'],'final_geometry_gate':c['final_geometry']['geometry_gate_pass'],
                'current_native_LP_scaled_loss':lp['scaled_loss'],'current_native_LP_minimum_torque_slack_Nm':minimum,
                'current_native_LP_finger_normal_N':dict(zip(lp['group_names'],lp['group_normal_N'])),
                'strict_native_LP_solver_status':None if c['strict_native_support_LP'] is None else c['strict_native_support_LP']['solver_status'],
                'holding_diagnostic_eligible':c['holding_diagnostic_eligible']}
            results.append(result)
    assert len(results)==4 and len(all_rows)==32
    assert not any(x in sys.modules for x in ('mujoco','torch','warp','mujoco_warp'))
    newq=np.c_[all_hard[:8,0]+cfg['q_planner_margin_rad'],all_hard[:8,1]-cfg['q_planner_margin_rad']]
    out={'schema':'saved_v9_active_bound_author_review:1','rules':pin(HERE/'rules.json'),'source':pin(__file__),
        'producer_report':rules['inputs']['producer_report'],'producer_protocol':report['protocol'],'old_native_report':oldpin,
        'cases':results,'all32_joint_rows':all_rows,'case_pins':pins,'counts':{'saved_final_JSON_cases':4,'new_model_FK_LP_step_NN_GPU_calls':0},
        'next_range_proposal':{'q8_range_rad':newq.tolist(),'joint_order8':names[:8],
            'phone_palm_translation_box_m':np.c_[lo_t,hi_t].tolist(),'phone_palm_rotvec_box_rad':np.c_[lo_r,hi_r].tolist(),
            'keep_phone_boxes_if_not_active':True,'fixed_remaining_q12':q0[8:].tolist(),
            'scope':'Remove only artificial source+-qtrust for q8; preserve actual hard bounds plus original numerical planner margin, same10 assembly constraints/geometry/scene. Existing feasible finals can initialize a new explicitly bounded local search, not a global feasibility claim.'},
        'interpretation':'Near bounds are numerical descriptors under fixed unit-specific tolerances, not new physical accuracy/qualification gates. Slack at a witness does not prove removing that constraint can never help elsewhere; no unique causal attribution.'}
    (HERE/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':pin(HERE/'report.json'),'cases':[{'seed':x['seed'],'rank':x['rank'],
        'near_q':[{'j':r['index'],'q':r['q_rad'],'lower':r['lower_boundary_source'] if r['near_algorithm_lower'] else None,
                    'upper':r['upper_boundary_source'] if r['near_algorithm_upper'] else None} for r in x['joint_rows'] if r['near_algorithm_lower'] or r['near_algorithm_upper']],
        'near_phone':[r for r in x['phone_box_rows'] if r['near_lower'] or r['near_upper']],
        'native_LP_loss':x['current_native_LP_scaled_loss'],'tau_min_slack':x['current_native_LP_minimum_torque_slack_Nm'],
        'Fn':x['current_native_LP_finger_normal_N']} for x in results],'next_q8_ranges':newq.tolist()}))


if __name__=='__main__':main()
