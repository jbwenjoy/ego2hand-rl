"""Independent GPU EFC -> local force -> world/BODY wrench/generalized audit."""
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from supermimic_research.artifacts import file_pin,verify_file_pin


def review_saved_warp_contacts(report_path,output,expected_cases):
    args=SimpleNamespace(report=Path(report_path),output=Path(output))
    if args.output.exists():raise ValueError('Fresh receipt review')
    r=json.loads(args.report.read_text());args.output.mkdir();cases=[]
    actual={(c.get('name',c['controller']),c['dt']) for c in r['cases']}
    if actual!=set(expected_cases) or len(r['cases'])!=len(expected_cases):raise ValueError('Explicit unique expected cohort')
    for pin in r['sources']:verify_file_pin(pin)
    for c in r['cases']:
        if c.get('contact_label_schema')!='GPU_constraint_indexed_no_adhesion:1':raise ValueError('Explicit corrected label schema')
        z=np.load(verify_file_pin(c['trace']));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));p=np.load(verify_file_pin(c['GPU_physical_context']))
        raw=z['raw_contact_tick_plus23'];addresses=z['GPU_contact_efc_addresses'];ranges=z['GPU_efc_ranges'];flat=z['GPU_efc_force_flat'];N=len(z['post_qpos']);ticks=raw[:,0].astype(int)
        if m.npair:raise ValueError('Explicit pair friction requires separate captured pair metadata')
        if len(ranges)!=N+1 or ranges[0]!=0 or ranges[-1]!=len(flat) or len(addresses)!=len(raw):raise ValueError('EFC/contact receipt ranges')
        local=np.zeros((len(raw),6));valid=True
        for i,row in enumerate(raw):
            tick=int(row[0]);dim=int(row[3]);count=1 if dim==1 else (2*(dim-1) if int(m.opt.cone)==0 else dim);ids=addresses[i,:count];nefc=ranges[tick+1]-ranges[tick]
            if row[4]!=addresses[i,0]:raise ValueError('Source GPU address relabeled')
            if ids[0]<0:continue
            if np.any((ids<0)|(ids>=nefc)):valid=False;continue
            edges=flat[ranges[tick]+ids].astype(np.float64)
            if int(m.opt.cone)==0 and dim>1:
                # Independently sum each edge pair and its signed tangent.
                for j in range(dim-1):
                    plus,minus=edges[2*j:2*j+2];local[i,0]+=plus+minus
                # These tasks have no explicit pairs or interval material DR.
                # Reconstruct contact friction from saved actual GPU geom tables
                # and native priority/max rules, independently of reader output.
                g,h=map(int,row[1:3]);fg,fh=p['geom_friction'][0,[g,h]]
                friction=fg if m.geom_priority[g]>m.geom_priority[h] else fh if m.geom_priority[h]>m.geom_priority[g] else np.maximum(fg,fh)
                mu=friction[[0,0,1,2,2]]
                for j in range(dim-1):local[i,j+1]=(edges[2*j]-edges[2*j+1])*mu[j]
            else:local[i,:dim]=edges
        label_error=float(abs(local-raw[:,18:24]).max()) if len(raw) else 0.
        body=c['model_layout']['object_body'];qa=c['model_layout']['object_qposadr'];da=c['model_layout']['object_dofadr'];dt=c['dt'];mass=float(p['body_mass'][0,body])
        g1,g2=raw[:,1].astype(int),raw[:,2].astype(int);o1,o2=m.geom_bodyid[g1]==body,m.geom_bodyid[g2]==body;mask=o1^o2
        row=raw[mask];ti=ticks[mask];sign=np.where(o2[mask],1.,-1.);frame=row[:,9:18].reshape(-1,3,3)
        F=sign[:,None]*np.einsum('nji,nj->ni',frame,local[mask,:3]);T=sign[:,None]*np.einsum('nji,nj->ni',frame,local[mask,3:])+np.cross(row[:,6:9]-z['pre_object_COM'][ti],F)
        world=np.zeros((N,6));np.add.at(world,ti,np.c_[F,T])
        R=Rotation.from_quat(z['pre_qpos'][:,qa+3:qa+7][:,[1,2,3,0]]).as_matrix()
        BODY=np.c_[np.einsum('nji,nj->ni',R,world[:,:3]),np.einsum('nji,nj->ni',R,world[:,3:])]
        # Both controlled objects have COM at their free-body origin.
        if np.any(np.array(c['model_layout']['COM_BODY'])!=0):raise ValueError('Explicit zero-COM free-body force identity')
        force_error=world[:,:3]-z['post_qfrc_constraint'][:,da:da+3]
        torque_error=BODY[:,3:]-z['post_qfrc_constraint'][:,da+3:da+6]
        dv=z['post_qvel'][:,da:da+3]-z['pre_qvel'][:,da:da+3]
        momentum=mass*dv-dt*(world[:,:3]+mass*m.opt.gravity)
        legacy=z['legacy_host_local_force6'];adhesion=z['legacy_host_adhesion'];finite=np.isfinite(legacy).all(axis=1)&np.isfinite(adhesion)
        legacy_error=np.full(len(raw),np.nan);legacy_error[finite]=legacy[finite,0]-local[finite,0]+adhesion[finite]
        old_difference=np.max(abs(legacy[finite]-local[finite]),axis=1) if finite.any() else np.zeros(0)
        max_force=float(np.linalg.norm(force_error,axis=1).max());max_torque=float(np.linalg.norm(torque_error,axis=1).max());max_impulse=float(np.linalg.norm(momentum,axis=1).max())
        checks=dict(valid_source_indices=valid,local_labels_exact=label_error<1e-10,
            generalized_force=max_force<1e-4,generalized_torque=max_torque<1e-5,linear_impulse=max_impulse<5e-7,
            copied_host_difference_explained=bool(np.nanmax(abs(legacy_error),initial=0.)<1e-9),
            all_joint_actuators=bool(np.all(m.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT))))
        out=args.output/f'{c.get("name",c["controller"])}_{c["dt"]}.npz'
        np.savez_compressed(out,world_COM_wrench=world,object_BODY_COM_wrench=BODY,force_residual_N=force_error,torque_residual_Nm=torque_error,linear_impulse_residual_N_s=momentum)
        cases.append(dict(task=c['task'],controller=c['controller'],name=c.get('name',c['controller']),dt=c['dt'],checks=checks,raw_contacts=len(raw),
            local_label_max_error=label_error,max_force_residual_N=max_force,max_torque_residual_Nm=max_torque,max_linear_impulse_residual_N_s=max_impulse,
            changed_host_contact_rows=int(np.count_nonzero(old_difference>1e-7)),nonfinite_host_rows=int(np.count_nonzero(~finite)),
            max_legacy_normal_plus_adhesion_error=float(np.nanmax(abs(legacy_error),initial=0.)),wrenches=file_pin(out)))
    checks=dict(expected_cases=len(cases)==len(expected_cases),all_receipt_checks=all(all(c['checks'].values()) for c in cases))
    result=dict(schema='corrected_warp_contact_receipts_review:2',source=file_pin(__file__),run=file_pin(args.report),checks=checks,cases=cases,
        passed=sum(sum(c['checks'].values()) for c in cases),total=sum(len(c['checks']) for c in cases),neural_updates=0,native_integration_steps=0,
        scope='Actual saved GPU constraint indices/forces, copied-host adhesion difference, moving-COM wrenches, generalized force/torque and free-object linear impulse. Not task success, measured material calibration, or WM gradient validation.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result))
    if not all(checks.values()):raise ValueError('Preserve failed corrected contact review')
    return result
