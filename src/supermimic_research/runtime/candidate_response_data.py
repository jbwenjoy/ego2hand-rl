"""Fresh parent-split native data with independent execution of every candidate."""
import gc
import json
import subprocess
import time
import tomllib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import freeze_sources,verify_source_snapshot
from .controlled_evaluation import write
from .response_parent_split import parent_roles
from .saved_native_plant import saved_native_plant
from .saved_dynamic_arrays import restore_dynamic_arrays
from .gpu_data_snapshot import ResidentDataSnapshot
from .gpu_model_snapshot import capture_model
from .warp_response_recorder import record_response
from ..mechanics.warp_packet_decode import decode_saved_case
from ..mechanics.candidate_response_targets import packet_xy
from ..mechanics.pen_replay_fidelity import compare_response
from ..mechanics.visited_source_review import graph_trace
from ..mechanics.impulse_codec import close_velocities
from ..learning.pen_response import fit_pen_scales,INPUT_SCHEMA,OUTPUT_SCHEMA,HEAD_SCHEMA
ROOT=Path(__file__).resolve().parents[3]


def main(a):
    import warp as wp
    import mujoco_warp as mw
    import torch
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());rp=(cp.parent/cfg['parents']).resolve();r=json.loads(rp.read_text())
    roles=parent_roles(r['response_parent_split']);gates=tomllib.loads((cp.parent/cfg['gates']).read_text())
    if (r['response_data_role']!='FRESH_WM_PARENTS' or cfg['source_role']!=r['response_data_role'] or cfg['repeats']!=3 or cfg['candidates']!=57
            or cfg['horizon_s']!=.02 or not cfg['audit_all_candidates'] or cfg['validity']!='ALL_REPEATS_PHYSICAL_AND_GRAPH_FIDELITY_AND_REPEAT_GATES'
            or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')):raise ValueError('Frozen fresh-data contract')
    keys={(c['name'],c['dt'],k) for c in r['cases'] for k in roles};observed={(p['mode'],p['dt'],p['k']) for p in r['response_rows']}
    if not observed<=keys:raise ValueError('No undeclared parent roles')
    for key in ['source_receipt','installed_source_receipt','engine_source_receipt']:verify_source_snapshot(json.loads(verify_file_pin(r[key]).read_text()))
    a.output.mkdir();wp.init();wp.set_device('cuda:0');torch.set_num_threads(4);started=time.perf_counter()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt=freeze_sources(sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[ROOT/'tools/collect_pen_candidate_data.py'],a.output/'source_snapshot',ROOT,revision)
    write(a.output/'source_receipt.json',receipt);installed_root=Path(mw.__file__).resolve().parent
    installed=freeze_sources(sorted(p for p in installed_root.rglob('*.py') if not p.name.endswith('_test.py') and not p.name.startswith('test_')),
        a.output/'installed_source_snapshot',installed_root,'installed-source-not-git');write(a.output/'installed_source_receipt.json',installed)
    intent=dict(schema='pen_candidate_response_preparation:1',config=file_pin(cp),parents=file_pin(rp),source_receipt=file_pin(a.output/'source_receipt.json'),
        installed_source_receipt=file_pin(a.output/'installed_source_receipt.json'),sources=[e['snapshot'] for e in receipt['records']],input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,head_schema=HEAD_SCHEMA,
        role_contract=r['response_parent_split'],neural_updates=0,Actor_updates=0,scope='New actual parent sources; every57 candidate at3 repeats independently executed and EFC decoded. All failed rows retained, shared whole-action-group validity mask, TRAIN-only scales. Simulator-response data, not experts/ego/gradient/speedup.')
    write(a.output/'intent.json',intent);rows=[];parent_rows=[];packed={'TRAIN':[],'CHECK':[]};physics=None;native_ticks=0
    for case_id,c in enumerate(r['cases']):
        records=[p for p in r['response_rows'] if p['mode']==c['name'] and p['dt']==c['dt']]
        if not records:continue
        first_source=dict(np.load(verify_file_pin(records[0]['source'])))
        m,dm,data,reconstruction=saved_native_plant(c,first_source,observed_layout=c['observed_Data_layout'])
        model=np.load(verify_file_pin(c['fixed_model_arrays']));_,scalars,hashes=capture_model(dm)
        body,qa,da=(c['model_layout'][key] for key in ['object_body','object_qposadr','object_dofadr']);n=round(.02/c['dt'])
        iq=model['body_iquat'][0,body];R=Rotation.from_quat(iq[[1,2,3,0]]).as_matrix()
        current=dict(mass=float(model['body_mass'][0,body]),inertia_BODY=(R@np.diag(model['body_inertia'][0,body])@R.T).tolist(),gravity=model['opt.gravity'][0].astype(float).tolist())
        if physics is None:physics=current
        if current!=physics:raise ValueError('One exact saved physical profile')
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)]);qids=m.jnt_qposadr[jids];dofs=m.jnt_dofadr[jids]
        aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids]);physical=np.load(verify_file_pin(c['GPU_physical_context']))
        was_gc=gc.isenabled();gc.disable()
        try:
            with wp.ScopedCapture() as capture:mw.step(dm,data)
        finally:
            if was_gc:gc.enable()
        step_graph=capture.graph
        for k in sorted({p['k'] for p in records}):
            group=sorted([p for p in records if p['k']==k],key=lambda p:p['repeat']);role=roles[k];parent_id=len(parent_rows)
            if len(group)!=3 or any(p['role']!=role for p in group):raise ValueError('All repeats belong to original parent role')
            source=dict(np.load(verify_file_pin(group[0]['source'])));restore_dynamic_arrays(data,source);snapshot=ResidentDataSnapshot(data);wp.synchronize()
            if not snapshot.audit(data)['all_array_bytes_equal']:raise ValueError('Full observed source mismatch')
            with wp.ScopedCapture() as capture:snapshot.restore(data)
            restore_graph=capture.graph;folder=a.output/f'parent_{parent_id:03d}';folder.mkdir();local=[];xs=[];ys=[];hs=[];packets={};worlds={};normals={}
            for rep,p in enumerate(group):
                d=dict(np.load(verify_file_pin(p['trace'])))
                if d['Fn'].shape!=(57,n,6) or d['Warm'].shape!=(57,n,26):raise ValueError('Versioned GPU Fn6/actual warm labels required')
                for j in range(57):
                    wp.capture_launch(restore_graph)
                    packet=record_response(m,dm,data,step_graph,aids,d['PD_targets'][j],n,instrument=False);native_ticks+=n
                    packet_path=folder/f'R{rep}_A{j:02d}.npz';np.savez_compressed(packet_path,**packet)
                    efc,decoded=decode_saved_case(c,packet,m,physical);W=decoded['world_COM_wrench'];Fn=np.zeros((n,6));nonhand=np.zeros(n)
                    for contact in packet['raw_contact_tick_plus23']:
                        tick,g,h=int(contact[0]),int(contact[1]),int(contact[2]);ob=m.geom_bodyid[[g,h]]==body
                        if ob.sum()!=1:continue
                        name=m.geom(h if ob[0] else g).name or '';normal=max(0.,float(contact[18]))
                        if name.startswith('robot/'):
                            digit=next((index-1 for index in range(1,6) if name.startswith(f'robot/right_finger{index}_')),5);Fn[tick,digit]+=normal
                        else:nonhand[tick]+=normal
                    metrics,fidelity=compare_response(graph_trace(d,j),packet,d['W'][j].astype(float),W,qids,dofs,qa,da,gates)
                    x,y,head=packet_xy(source,packet,W,Fn,qids,dofs,aids,qa,da,c['dt'],physics)
                    contacts=packet['raw_contact_tick_plus23'];penetration=max(0.,float(-contacts[:,5].min())) if len(contacts) else 0.;self_pen=0.
                    for contact in contacts:
                        if all((m.geom(int(g)).name or '').startswith('robot/') for g in contact[1:3]):self_pen=max(self_pen,-float(contact[5]))
                    ranges=m.jnt_range[jids];q=packet['post_qpos'][:,qids]
                    excursion=max(0.,float((ranges[:,0]-q).max()),float((q-ranges[:,1]).max()))
                    own=dict(source_qv_exact=np.array_equal(packet['pre_qpos'][0],source['qpos'][0]) and np.array_equal(packet['pre_qvel'][0],source['qvel'][0]),
                        PD_exact=np.array_equal(packet['post_ctrl'][:,aids],np.broadcast_to(d['PD_targets'][j],(n,20))),no_aid=not np.any(packet['post_qfrc_applied']) and not np.any(packet['post_xfrc_applied']),
                        capacity=not np.any(packet['overflow']),finite_targets=bool(np.isfinite(np.r_[x,y,head]).all()),no_nonhand_contact=bool(np.all(nonhand<=1e-9)),
                        Fn_graph_agreement=bool(np.max(abs(Fn-d['Fn'][j]))<=gates['EFC_force_N']),rotation_chart=np.linalg.norm(y[43:46])<np.pi-.001,
                        actual_next_warm=np.array_equal(y[52:78],packet['post_qacc_warmstart'][-1]))
                    record=dict(parent_id=parent_id,parent_key=p['parent_key'],role=role,mode=c['name'],dt=c['dt'],k=k,repeat=rep,candidate=j,
                        packet=file_pin(packet_path),source=p['source'],prediction=p['trace'],EFC_checks={key:bool(val) for key,val in efc['checks'].items()},
                        fidelity=fidelity,fidelity_metrics=metrics,checks={key:bool(val) for key,val in own.items()},
                        base_valid=bool(all(efc['checks'].values()) and all(fidelity.values()) and all(own.values())),
                        warm_graph_max_abs=float(abs(d['Warm'][j]-packet['post_qacc_warmstart']).max()),Fn_graph_max_N=float(abs(Fn-d['Fn'][j]).max()),
                        impulse_max_N_s=efc['max_linear_impulse_residual_N_s'],geometry=[penetration,self_pen,excursion],predicted_geometry=d['pen'][j].astype(float).tolist(),
                        source_episode_task_qualified=c['summary']['qualified'])
                    local.append(record);xs.append(x);ys.append(y);hs.append(head);packets[(rep,j)]=packet;worlds[(rep,j)]=W;normals[(rep,j)]=Fn
            valid=np.zeros(171,dtype=bool)
            for j in range(57):
                spread_checks=[]
                for rep in [1,2]:
                    _,checks=compare_response(packets[(rep,j)],packets[(0,j)],worlds[(rep,j)],worlds[(0,j)],qids,dofs,qa,da,gates)
                    spread_checks.append(dict(repeat=rep,checks=checks,Fn6=float(abs(normals[(rep,j)]-normals[(0,j)]).max())<=gates['EFC_force_N']))
                group_valid=all(local[rep*57+j]['base_valid'] for rep in range(3)) and all(all(e['checks'].values()) and e['Fn6'] for e in spread_checks)
                for rep in range(3):
                    index=rep*57+j;valid[index]=group_valid;local[index].update(group_valid=group_valid,repeat_spread=spread_checks)
            block=dict(x=np.array(xs),y=np.array(ys),heads=np.array(hs),valid=valid,parent_id=np.full(171,parent_id),candidate=np.tile(np.arange(57),3),
                repeat=np.repeat(np.arange(3),57),phase=np.full(171,k),dt=np.full(171,c['dt']),geometry=np.array([e['geometry'] for e in local]))
            np.savez_compressed(folder/'targets.npz',**block);rows.extend(local);packed[role].append(block)
            parent_rows.append(dict(parent_id=parent_id,key=group[0]['parent_key'],role=role,mode=c['name'],dt=c['dt'],k=k,source=group[0]['source'],
                observations=c['observed_Data_layout'],reconstruction=reconstruction,targets=file_pin(folder/'targets.npz'),rows=171,valid=int(valid.sum())))
            write(a.output/'progress.json',dict(parents=parent_rows,attempted_rows=len(rows),valid_rows=sum(p['valid'] for p in parent_rows),native_ticks=native_ticks,wall_s=time.perf_counter()-started))
            print(json.dumps(dict(parent=parent_id,role=role,k=k,dt=c['dt'],valid=int(valid.sum()),rows=171,native_ticks=native_ticks,wall_s=time.perf_counter()-started)),flush=True)
        _,after_scalars,after_hashes=capture_model(dm)
        if scalars!=after_scalars or hashes!=after_hashes:raise ValueError('Audit changed model')
    data={role:{key:np.concatenate([block[key] for block in blocks]) for key in blocks[0]} for role,blocks in packed.items()}
    train=data['TRAIN'];scales,head_scales=fit_pen_scales(train['x'][train['valid']],train['y'][train['valid']],train['heads'][train['valid']])
    checks=dict(parent_roles_disjoint=not bool(set(data['TRAIN']['parent_id'])&set(data['CHECK']['parent_id'])),excluded_phases=all(p['k'] not in [0,4] for p in parent_rows),
        all_declared_parents=len(parent_rows)==len(keys),minimum_TRAIN_valid=int(train['valid'].sum())>=cfg['minimum_TRAIN_valid'],minimum_CHECK_valid=int(data['CHECK']['valid'].sum())>=cfg['minimum_CHECK_valid'])
    outputs={};closure={}
    for role,z in data.items():
        D=lambda v:torch.tensor(v,dtype=torch.float64)
        decoded=close_velocities(D(z['x']),D(z['heads']),D(physics['mass']),D(physics['inertia_BODY']),D(np.linalg.inv(physics['inertia_BODY'])),D(physics['gravity']),D(z['heads'][:,46:52])).numpy()
        closure[role]=float(abs(decoded-z['y']).max());checks[role+'_target_defect_identity']=closure[role]<1e-8
        checks[role+'_finite']=all(np.isfinite(value).all() for value in z.values());checks[role+'_Fn6_nonnegative']=bool(np.all(z['y'][:,84:]>=0))
        path=a.output/(role+'.npz');np.savez_compressed(path,**z);outputs[role]=file_pin(path)
    np.savez_compressed(a.output/'scales.npz',**scales);np.savez_compressed(a.output/'head_scales.npz',**head_scales)
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(installed,live=True)
    result=dict(**intent,parents_detail=parent_rows,rows=rows,outputs=outputs,physics=physics,scales=file_pin(a.output/'scales.npz'),head_scales=file_pin(a.output/'head_scales.npz'),
        checks={key:bool(val) for key,val in checks.items()},passed=int(sum(checks.values())),total=len(checks),closure_errors=closure,native_audit_ticks=native_ticks,
        parent_primary_candidate_ticks=sum(c['candidate_physics_ticks'] for c in r['cases']),parent_actual_ticks=sum(c['physics_steps'] for c in r['cases']),
        parent_extra_response_ticks=r['extra_response_candidate_ticks'],labeled_candidate_ticks=r['response_candidate_ticks_including_parent_primary'],
        attempted_rows=len(rows),valid_rows=sum(p['valid'] for p in parent_rows),collector_wall_s=time.perf_counter()-started)
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[result['passed'],result['total']],rows={role:[len(z['x']),int(z['valid'].sum())] for role,z in data.items()},audit_ticks=native_ticks)))
    if not all(checks.values()):raise ValueError('Preserve incomplete preparation; no training')
