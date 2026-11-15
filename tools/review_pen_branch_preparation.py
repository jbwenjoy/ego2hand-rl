"""Independent branch source/target, role, actual control and scale readback."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.pen_branch_profiles import branch_profiles
from supermimic_research.learning.pen_interventions import first_raw_action
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    p=json.loads(a.prepared.read_text());r=json.loads(verify_file_pin(p['inputs']['collection']).read_text())
    efc=json.loads(verify_file_pin(p['inputs']['EFC_review']).read_text());old=json.loads(verify_file_pin(p['inputs']['old_prepared']).read_text())
    for pin in p['sources']+r['sources']+r['task_sources']:verify_file_pin(pin)
    cfg=tomllib.loads(verify_file_pin(r['config']).read_text());directions,profiles=branch_profiles(cfg)
    expected={(v['name'],dt) for v in profiles for dt in cfg['dt']}
    if {(c['name'],c['dt']) for c in r['cases']}!=expected or len(r['cases'])!=256:raise ValueError('Exact declared source cohort')
    proposal=json.loads(verify_file_pin(r['proposal']).read_text());idx=np.load(verify_file_pin(proposal['artifacts']['selected']))
    qref=np.load(verify_file_pin(proposal['clip']))['reference_qpos']
    data={role:dict(np.load(verify_file_pin(pin))) for role,pin in p['outputs'].items()}
    items=[]
    for case_id,c in enumerate(r['cases']):
        z=np.load(verify_file_pin(c['trace']));physical=np.load(verify_file_pin(c['GPU_physical_context']))
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids,dofs=m.jnt_qposadr[jids],m.jnt_dofadr[jids]
        aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids])
        body=m.body('object/cube').id;joint=m.body_jntadr[body];qa,da=m.jnt_qposadr[joint],m.jnt_dofadr[joint]
        d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        K=len(z['actions20']);n=round(.02/c['dt']);k=c['anchor'];lo,hi=k*n,(k+1)*n
        target_role='NEW_TRAIN' if c['role']=='TRAIN' else 'CHECK';packet=data[target_role]
        ids=np.flatnonzero(packet['case_id']==1000+case_id)
        reached=k<K
        local=dict(named_indices=np.array_equal(qids,idx['qids']) and np.array_equal(dofs,idx['dofs']) and np.array_equal(aids,idx['aids']),
            declared_single_interval=len(ids)==int(reached),original_clock=np.array_equal(z['pre_action_phase'],np.arange(1,K+1)) and len(z['post_qpos'])==K*n,
            state_carry=np.array_equal(z['pre_qpos'][1:],z['post_qpos'][:-1]) and np.array_equal(z['pre_qvel'][1:],z['post_qvel'][:-1]),
            first_terminal=not np.any(z['terminated_truncated'][:-1]) and (bool(z['terminated_truncated'][-1].any()) or K==c['planned_intervals']),
            no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
            role_direction_contract=c['direction_seed']==cfg[c['role'].lower()+'_direction_seed'] and c['supervision']=='BRANCH_WM_DYNAMICS_NONEXPERT_NOT_IMITATION')
        raw=np.tile((2*idx['delta20']).astype(np.float32),(K,1))
        if reached:raw[k]=first_raw_action(idx['delta20'],directions[c['role']][c['direction']],c['epsilon'],c['sign'])
        local['declared_raw_actions']=np.array_equal(raw,z['actions20'])
        memory=idx['delta20'].astype(np.float32);targets=[]
        lower=np.maximum(physical['jnt_range'][0,jids,0],physical['actuator_ctrlrange'][0,aids,0]);upper=np.minimum(physical['jnt_range'][0,jids,1],physical['actuator_ctrlrange'][0,aids,1])
        for phase,action in zip(z['pre_action_phase'],raw):
            memory=np.float32(.5)*memory+np.float32(.5)*np.clip(np.float32(.5)*action,-.6,.6)
            targets.append(np.clip(qref[int(phase)]+memory,lower,upper))
        control_error=float(abs(np.repeat(targets,n,axis=0)-z['post_ctrl'][:,aids]).max())
        local['original_EMA_actual_PD']=control_error<2e-7
        errors=None
        if reached:
            row=ids.item();e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt']);w=np.load(verify_file_pin(e['wrenches']))
            def state(q,v):
                R=Rotation.from_quat(q[qa+3:qa+7][[1,2,3,0]]).as_matrix()
                return np.r_[q[qids],v[dofs],q[qa:qa+3],R[:,0],R[:,1],v[da:da+3],R@v[da+3:da+6]]
            source=state(z['pre_qpos'][lo],z['pre_qvel'][lo]);end=state(z['post_qpos'][hi-1],z['post_qvel'][hi-1])
            rot=lambda s:np.column_stack([s[43:46],s[46:49],np.cross(s[43:46],s[46:49])])
            R0,R1=rot(source),rot(end);mean=w['world_COM_wrench'][lo:hi].mean(0)
            Fn=np.zeros(6);nonhand=np.zeros(n)
            for rawrow in z['raw_contact_tick_plus23']:
                tick,g,h=map(int,rawrow[:3])
                if not lo<=tick<hi:continue
                ob=m.geom_bodyid[[g,h]]==body
                if ob.sum()!=1:continue
                name=m.geom(h if ob[0] else g).name or '';force=max(0.,rawrow[18])
                if name.startswith('robot/'):
                    digits=[j for j in range(1,6) if f'/right_finger{j}_' in name];Fn[digits[0]-1 if digits else 5]+=force/n
                else:nonhand[tick-lo]+=force
            warm=d.qacc_warmstart if lo==0 else z['post_qacc'][lo-1]
            x=np.r_[source,warm,z['post_ctrl'][lo,aids],c['dt']]
            y=np.r_[end[:43]-source[:43],Rotation.from_matrix(R1@R0.T).as_rotvec(),end[49:]-source[49:],z['post_qacc'][hi-1],R0.T@mean[:3],R0.T@mean[3:],Fn]
            I=np.array(c['physics']['inertia_BODY']);mass=c['physics']['mass'];gravity=np.array(c['physics']['gravity'])
            head=y.copy();head[46:52]=np.r_[mass*(end[49:52]-source[49:52])-.02*(mean[:3]+mass*gravity),R1@I@R1.T@end[52:55]-R0@I@R0.T@source[52:55]-.02*mean[3:]]
            errors=[float(abs(x-packet['x'][row]).max()),float(abs(y-packet['y'][row]).max()),float(abs(head-packet['heads'][row]).max())]
            valid=bool(np.all(np.linalg.norm(w['force_residual_N'][lo:hi],axis=1)<=1e-4) and np.all(np.linalg.norm(w['torque_residual_Nm'][lo:hi],axis=1)<=1e-5) and np.all(np.linalg.norm(w['linear_impulse_residual_N_s'][lo:hi],axis=1)<=5e-7) and not np.any(z['overflow'][lo:hi]) and np.all(nonhand<=1e-9) and np.linalg.norm(y[43:46])<np.pi-.001)
            local.update(source_state_warm_target=errors[0]<1e-12,source_BODY_wrench_Fn6_target=errors[1]<1e-12,
                observed_impulse_defect=errors[2]<1e-12,numerical_mask=bool(packet['valid'][row])==valid,
                branch_control_index=int(packet['control_index'][row])==k)
        items.append(dict(case_id=case_id,name=c['name'],dt=c['dt'],checks={k:bool(v) for k,v in local.items()},errors=errors,control_error=control_error))
    old_train=dict(np.load(verify_file_pin(old['outputs']['TRAIN'])));train=data['TRAIN'];scales=np.load(verify_file_pin(p['scales']));heads=np.load(verify_file_pin(p['head_scales']));valid=train['valid']
    global_checks=dict(old_TRAIN_prefix_exact=all(np.array_equal(v,train[k][:len(v)]) for k,v in old_train.items()),
        new_TRAIN_suffix_exact=all(np.array_equal(v,train[k][len(old_train[k]):]) for k,v in data['NEW_TRAIN'].items()),
        unchanged_SEQUENCE_CHECK=p['outputs']['SEQUENCE_CHECK']==old['outputs']['CHECK'],
        TRAIN_only_input_mean=np.array_equal(scales['x_mean'],train['x'][valid].mean(0)),
        TRAIN_only_output_mean=np.array_equal(scales['y_mean'][:84],train['y'][valid,:84].mean(0)),
        TRAIN_only_defect_mean=np.array_equal(heads['head_mean'][46:52],train['heads'][valid,46:52].mean(0)),
        six_normal_zero_means=np.array_equal(scales['y_mean'][84:],np.zeros(6)),
        no_full_input_overlap=not bool({x.tobytes() for x in train['x']}&{x.tobytes() for x in data['CHECK']['x']}))
    if a.output.exists():raise ValueError('Fresh independent branch review')
    a.output.mkdir();result=dict(schema='pen_branch_preparation_saved_review:1',prepared=file_pin(a.prepared),source=file_pin(__file__),cases=items,
        checks=global_checks,passed=sum(sum(c['checks'].values()) for c in items)+sum(global_checks.values()),total=sum(len(c['checks']) for c in items)+len(global_checks),
        neural_updates=0,native_integration_steps=0,scope='Independent branch interval indexing, named state/world angular chart, source-BODY wrench/Fn6 and defects, original raw/EMA control and role/TRAIN-only means. Not a physical derivative or Actor accuracy gate.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),passed=result['passed'],total=result['total'])))
    if result['passed']!=result['total']:raise ValueError('Preserve failed branch readback; no training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
