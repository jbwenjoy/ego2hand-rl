"""Saved LP primals checked against native EFC Jacobian unit columns."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    import mujoco
    run,reference,out=args.run.resolve(),args.reference.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((run/'report.json').read_text());bind=json.loads((reference/'binding.json').read_text())
    for pin in report['sources']:verify_file_pin(pin)
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));records=[];unit_queries=0;primal_count=0
    for rec in report['records']:
        z=np.load(verify_file_pin(rec['raw']));model.opt.timestep=rec['source_dt_s'];d=mujoco.MjData(model)
        d.qpos[:]=z['qpos27'];d.qvel[:]=0;d.ctrl[:]=z['ctrl20'];mujoco.mj_forward(model,d)
        oid=bind['mug_body_id'];R=d.xmat[oid].reshape(3,3);COM=d.xipos[oid];root=d.xpos[oid]
        W=[];L=[]
        for con in rec['contacts']:
            current=d.contact[con['contact_index']]
            if list(map(int,current.geom))!=con['geom_ids'] or current.dim!=3:raise ValueError('Native contact order/identity changed')
            for ray in range(4):
                coefficient=np.zeros(d.nefc);coefficient[current.efc_address+ray]=1.
                generalized=np.zeros(model.nv);mujoco.mj_mulJacTVec(model,d,generalized,coefficient);unit_queries+=1
                # Free-joint rotation entries are BODY axes about the root.
                # Move their world torque to COM before comparing support columns.
                F=generalized[20:23];T=R@generalized[23:26]-np.cross(COM-root,F)
                W.append(np.r_[F,T]);L.append(-generalized[:20])
        W=np.array(W).reshape(-1,6).T;L=np.array(L).reshape(-1,20).T
        w_error=float(np.max(np.abs(W-z['wrench6xN']))) if W.size else 0.
        l_error=float(np.max(np.abs(L-z['joint_load20xN']))) if L.size else 0.
        primals=[]
        checks=[('cone6D',rec['cone6D'],None,False),('force_only',rec['force_only'],None,True)]
        for domain in rec['domains']:
            checks.extend([(domain['name']+'/fixed',domain['feasibility'],domain,False),
                           (domain['name']+'/maximum',domain['maximum'],domain,False)])
        for name,answer,domain,force_only in checks:
            if not answer['success']:continue
            witness=answer['witness'];lam=np.array(witness['ray_coefficients_N'])
            target=z['target6'].copy()
            if name.endswith('/maximum'):target=np.r_[0,0,answer['support_weight_N'],0,0,0]
            residual=W@lam-target
            valid=bool(np.min(lam)>=-1e-9 and np.max(np.abs(residual[:3]))<1e-8 and (force_only or np.max(np.abs(residual[3:]))<1e-9))
            margin=None
            if domain is not None:
                tau=np.array(domain['offset'])+L@lam;lo,hi=np.array(domain['torque_interval']);f=np.array(domain['free_friction'])
                margin=float(np.minimum(tau-(lo-f),(hi+f)-tau).min());valid&=margin>=-1e-8
            primals.append(dict(name=name,passed=valid,force_residual_N=residual[:3].tolist(),moment_residual_Nm=residual[3:].tolist(),torque_margin_Nm=margin));primal_count+=1
        if w_error>1e-10 or l_error>1e-10 or not all(x['passed'] for x in primals):raise ValueError('Native EFC/LP primal comparison failed')
        records.append(dict(operand=rec['raw'],native_wrench_column_max_error=w_error,native_joint_load_column_max_error=l_error,primals=primals))
    expected_attempts=8*len(report['records']);expected_calls=sum(8 for r in report['records'] if r['contacts'])
    checks=dict(all_native_EFC_column_and_primal_checks=True,expected24_snapshots=len(records)==24,
        attempted_LPs_are_not_confused_with_solver_invocations=report['LP_attempts']==expected_attempts and report['linprog_invocations']==expected_calls,
        actual_no_integration=report['integration_steps']==0,one_full6D_feasible_pose=sum(r['cone6D']['success'] for r in report['records'])==1)
    out.mkdir(parents=True);result=dict(schema='mug_support_native_EFC_review:1',source=file_pin(__file__),run=file_pin(run/'report.json'),checks=checks,
        passed=sum(checks.values()),total=len(checks),records=records,primal_witnesses_checked=primal_count,
        native_forward=24,native_EFC_unit_column_queries=unit_queries,integration_steps=0,neural_updates=0,
        scope='Independent native EFC-Jacobian columns and saved feasible primal equations. Does not independently prove solver infeasibility/optimality, compliance reachability, dynamic stability or source contact force truth. Unit vectors are read-only probes, never applied to efc_force or the simulator.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=checks,primal_witnesses=primal_count,unit_queries=unit_queries)))
    if not all(checks.values()):raise RuntimeError('Retain failed support review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
