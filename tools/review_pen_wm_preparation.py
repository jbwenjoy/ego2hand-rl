"""Independent indexed state, held control, source-frame wrench and split review."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    if args.output.exists():raise ValueError('Fresh independent pen preparation review')
    p=json.loads(args.prepared.read_text());r=json.loads(verify_file_pin(p['run']).read_text());efc=json.loads(verify_file_pin(p['EFC_review']).read_text())
    for pin in p['sources']+r['sources']+r['task_sources']:verify_file_pin(pin)
    cfg=tomllib.loads(verify_file_pin(r['config']).read_text());proposal=json.loads(verify_file_pin(r['proposal']).read_text());idx=np.load(verify_file_pin(proposal['artifacts']['selected']))
    packed={role:dict(np.load(verify_file_pin(pin))) for role,pin in p['outputs'].items()};items=[]
    def state(q,v,c):
        qa,da=c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr'];R=Rotation.from_quat(q[qa+3:qa+7][[1,2,3,0]]).as_matrix()
        return np.concatenate([q[idx['qids']],v[idx['dofs']],q[qa:qa+3],R[:,:2].T.flatten(),v[da:da+3],R@v[da+3:da+6]])
    for caseid,c in enumerate(r['cases']):
        z=np.load(verify_file_pin(c['trace']));a=packed[c['role']];rows=np.flatnonzero(a['case_id']==caseid)
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt']);w=np.load(verify_file_pin(e['wrenches']));n=round(.02/c['dt']);body=c['model_layout']['object_body']
        Fn=np.zeros((len(z['post_qpos']),6))
        for raw in z['raw_contact_tick_plus23']:
            tick,g,h=map(int,raw[:3]);ob=m.geom_bodyid[[g,h]]==body
            if ob.sum()!=1:continue
            name=m.geom(h if ob[0] else g).name or ''
            if name.startswith('robot/'):
                groups=[j for j in range(1,6) if f'/right_finger{j}_' in name];group=groups[0]-1 if groups else 5;Fn[tick,group]+=max(0.,raw[18])
        errors=np.zeros(5)
        for k,row in enumerate(rows):
            lo,hi=k*n,(k+1)*n;source=state(z['pre_qpos'][lo],z['pre_qvel'][lo],c);target=state(z['post_qpos'][hi-1],z['post_qvel'][hi-1],c)
            R0=np.column_stack([source[43:46],source[46:49],np.cross(source[43:46],source[46:49])]);R1=np.column_stack([target[43:46],target[46:49],np.cross(target[43:46],target[46:49])])
            mean=w['world_COM_wrench'][lo:hi].mean(0);bodyw=np.concatenate([R0.T@mean[:3],R0.T@mean[3:]])
            x=np.concatenate([source,d.qacc_warmstart if lo==0 else z['post_qacc'][lo-1],z['post_ctrl'][lo,idx['aids']],[c['dt']]])
            y=np.concatenate([target[:43]-source[:43],Rotation.from_matrix(R1@R0.T).as_rotvec(),target[49:]-source[49:],z['post_qacc'][hi-1],bodyw,Fn[lo:hi].mean(0)])
            I=np.array(p['physics']['inertia_BODY']);mass=p['physics']['mass'];g=np.array(p['physics']['gravity'])
            defect=np.r_[mass*(target[49:52]-source[49:52])-.02*(mean[:3]+mass*g),R1@I@R1.T@target[52:55]-R0@I@R0.T@source[52:55]-.02*mean[3:]]
            errors=np.maximum(errors,[abs(x-a['x'][row]).max(),abs(y-a['y'][row]).max(),abs(defect-a['heads'][row,46:52]).max(),abs(y[78:84]-a['y'][row,78:84]).max(),abs(y[84:]-a['y'][row,84:]).max()])
        plan=np.load(verify_file_pin(c['action_plan']))['raw_actions'];seeds=cfg['train_action_seeds'] if c['role']=='TRAIN' else cfg['check_action_seeds']
        checks=dict(named_indices_state_and_input=errors[0]<1e-12,full_target=errors[1]<1e-12,observed_defect=errors[2]<1e-12,
                    source_BODY_wrench=errors[3]<1e-12,all_six_contact_channels=errors[4]<1e-12,
                    role_seed_contract=c['action_seed'] in seeds,executed_predeclared_plan=np.array_equal(z['actions20'],plan[:len(z['actions20'])]),
                    complete_control_intervals=len(rows)==len(z['actions20']),nonexpert_role=c['supervision']=='WM_DYNAMICS_NONEXPERT_NOT_IMITATION')
        items.append(dict(case_id=caseid,role=c['role'],seed=c['action_seed'],dt=c['dt'],checks={k:bool(v) for k,v in checks.items()},errors=errors.tolist()))
    scales=dict(np.load(verify_file_pin(p['scales'])));hs=dict(np.load(verify_file_pin(p['head_scales'])));t=packed['TRAIN'];valid=t['valid']
    global_checks=dict(TRAIN_only_input_mean=np.array_equal(scales['x_mean'],t['x'][valid].mean(0)),TRAIN_only_output_mean=np.array_equal(scales['y_mean'][:84],t['y'][valid,:84].mean(0)),
        TRAIN_only_defect_mean=np.array_equal(hs['head_mean'][46:52],t['heads'][valid,46:52].mean(0)),Fn6_zero_mean=np.array_equal(scales['y_mean'][84:],np.zeros(6)),
        source_quality_receipts=efc['passed']==efc['total'],role_cases_disjoint=not bool(set(packed['TRAIN']['case_id'])&set(packed['CHECK']['case_id'])))
    args.output.mkdir();result=dict(schema='pen_WM_preparation_saved_review:1',prepared=file_pin(args.prepared),source=file_pin(__file__),cases=items,checks=global_checks,
        passed=sum(sum(c['checks'].values()) for c in items)+sum(global_checks.values()),total=sum(len(c['checks']) for c in items)+len(global_checks),neural_updates=0,simulation_integration_steps=0,
        scope='Independent object-first/world angular state,20ms target/warm/control, source-BODY wrench/Fn6 and observed defect, role/scale readback. Nonexpert dynamics only; not expert success, predictive/physical gradient accuracy or new trajectory generalization.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],total=result['total'])))
    if result['passed']!=result['total']:raise ValueError('Preserve failed independent data review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
