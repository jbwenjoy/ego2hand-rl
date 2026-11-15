"""Zero-integration collision-only queries at the four recorded branch input poses."""
import argparse
import copy
import gc
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.controlled_evaluation import write
from supermimic_research.runtime.gpu_model_snapshot import capture_model,restore_model_arrays,model_leaves
from supermimic_research.runtime.saved_dynamic_arrays import restore_dynamic_arrays
from supermimic_research.runtime.gpu_data_snapshot import ResidentDataSnapshot


def main(a):
    import mujoco_warp as mw
    import warp as wp
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());rp=(cp.parent/cfg['replay']).resolve();r=json.loads(rp.read_text())
    if (cfg['mode']!='STATE_PLUS_VELOCITY_SUPPORT' or cfg['dt']!=.005 or cfg['anchor']!=9 or cfg['tick']!=3 or cfg['pair']!=[1,29]
            or cfg['poses']!=['ORIGINAL_ACTUAL','INSTRUMENTED_R0','INSTRUMENTED_R1','INSTRUMENTED_R2'] or cfg['repeats']!=5
            or cfg['pipeline']!='KINEMATICS_PLUS_COLLISION_ONLY' or a.output.exists()):raise ValueError('Exact recorded branch-pose diagnostic')
    a.output.mkdir();wp.init();wp.set_device('cuda:0');stream=wp.Stream(device='cuda:0')
    receipt=freeze_sources(sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[Path(__file__)],a.output/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(a.output/'source_receipt.json',receipt)
    original=json.loads(verify_file_pin(r['original_run']).read_text());c=next(c for c in original['cases'] if c['name']==cfg['mode'] and c['dt']==cfg['dt'])
    source=dict(np.load(verify_file_pin(c['planning'][9]['source'])));z=np.load(verify_file_pin(c['trace']));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
    for key,value in dict(np.load(verify_file_pin(c['GPU_physical_context']))).items():getattr(m,key)[:]=value[0]
    values=dict(np.load(verify_file_pin(c['fixed_model_arrays'])));scalars=json.loads(verify_file_pin(c['fixed_model_scalars']).read_text())
    with wp.ScopedStream(stream):
        dm=mw.put_model(m);restore_model_arrays(dm,values,scalars)
        data=mw.make_data(m,nworld=source['qpos'].shape[0],naconmax=len(source['contact.geom']),njmax=source['efc.force'].shape[1]);restore_dynamic_arrays(data,source)
        _,reconstructed_scalars=model_leaves(data)
        was_gc=gc.isenabled();gc.disable()
        try:
            with wp.ScopedCapture(stream=stream) as capture:
                mw.kinematics(dm,data)
                mw.collision(dm,data)
        finally:
            if was_gc:gc.enable()
    graph=capture.graph;g,h=cfg['pair'];poses=[];records=[]
    cpu=copy.copy(m)
    if np.shares_memory(cpu.geom_pos,m.geom_pos):raise ValueError('Detached CPU geometry query model required')
    geometry_fields=[]
    for key in ['body_pos','body_quat','body_ipos','body_iquat','jnt_pos','jnt_axis','geom_pos','geom_quat','geom_size','mesh_vert','qpos0']:
        if key not in values:continue
        target=getattr(cpu,key);value=values[key]
        if value.shape==(1,*target.shape):value=value[0]
        if value.shape!=target.shape:raise ValueError(f'CPU geometry field shape mismatch: {key}')
        target[:]=value;geometry_fields.append(key)
    for name in cfg['poses']:
        if name=='ORIGINAL_ACTUAL':
            q,v=z['pre_qpos'][39],z['pre_qvel'][39];state=z['initial_native_state'];input_pin=c['trace']
        else:
            repeat=int(name[-1]);entry=next(e for e in r['cases'] if e['mode']==cfg['mode'] and e['dt']==cfg['dt'] and e['k']==9 and e['repeat']==repeat)
            p=np.load(verify_file_pin(entry['trace']));q,v=p['pre_qpos'][3],p['pre_qvel'][3];state=p['initial_native_state'];input_pin=entry['trace']
        host=mujoco.MjData(cpu);mujoco.mj_setState(cpu,host,state,mujoco.mjtState.mjSTATE_INTEGRATION);host.qpos[:]=q;host.qvel[:]=v
        mujoco.mj_fwdPosition(cpu,host);segment=np.zeros(6);distance=mujoco.mj_geomDistance(cpu,host,g,h,.01,segment)
        folder=a.output/name;folder.mkdir()
        with wp.ScopedStream(stream):
            restore_dynamic_arrays(data,source);data.qpos.assign(q.astype(np.float32)[None]);data.qvel.assign(v.astype(np.float32)[None])
            initial=ResidentDataSnapshot(data);wp.synchronize_stream(stream)
            np.savez_compressed(folder/'input.npz',qpos=q,qvel=v,initial_native_state=state,CPU_segment=segment,CPU_geom_xpos=host.geom_xpos[[g,h]],CPU_geom_xmat=host.geom_xmat[[g,h]])
            for repeat in range(cfg['repeats']):
                initial.restore(data);before=initial.audit(data);wp.capture_launch(graph,stream=stream);wp.synchronize_stream(stream)
                count=int(data.nacon.numpy()[0]);ncollision=int(data.ncollision.numpy()[0])
                if count>data.naconmax:raise ValueError('Collision capacity overflow')
                fields={key:getattr(data.contact,key).numpy()[:count].copy() for key in ['geom','dist','pos','frame','worldid','type']}
                pair_mask=np.all(np.sort(fields['geom'],axis=1)==[g,h],axis=1)
                path=folder/f'collision_R{repeat}.npz';np.savez_compressed(path,**fields,geom_xpos=data.geom_xpos.numpy()[0,[g,h]],geom_xmat=data.geom_xmat.numpy()[0,[g,h]])
                records.append(dict(pose=name,repeat=repeat,source_array_bytes_exact=before['all_array_bytes_equal'],static_scalar_equal=before['scalar_equal'],
                    nacon=count,ncollision=ncollision,naconmax=data.naconmax,capacity_valid=ncollision<=data.naconmax,
                    pair_contacts=int(pair_mask.sum()),pair_distances_m=fields['dist'][pair_mask].astype(float).tolist(),trace=file_pin(path),
                    CPU_GPU_geom_position_max_m=float(abs(data.geom_xpos.numpy()[0,[g,h]]-host.geom_xpos[[g,h]]).max())))
        poses.append(dict(name=name,input=input_pin,input_values=file_pin(folder/'input.npz'),CPU_unfiltered_signed_distance_m=float(distance),
            max_input_qpos_difference_from_original_actual=float(abs(q-z['pre_qpos'][39]).max()),
            scope='Recorded kinematic input override on restored source arrays; not a complete original dynamic state. CPU query is unfiltered direct distance, not a CPU contact-force or filtered-contact truth oracle.'))
    after,after_scalars,_=capture_model(dm);model_exact=after_scalars==scalars and all(after[k].tobytes()==values[k].tobytes() for k in values)
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_collision_only_branch_probe:1',config=file_pin(cp),replay=file_pin(rp),source_receipt=file_pin(a.output/'source_receipt.json'),
        pair=cfg['pair'],pair_names=[m.geom(g).name,m.geom(h).name],original_model_arrays=c['fixed_model_arrays'],original_model_scalars=c['fixed_model_scalars'],
        model_all_arrays_scalars_exact=model_exact,Data_static_scalars=reconstructed_scalars,Data_static_role='RECONSTRUCTED_NOT_ORIGINAL',
        CPU_geometry_table_fields=geometry_fields,poses=poses,records=records,collision_queries=len(records),native_integrations=0,neural_updates=0,
        installed_sources=[file_pin(Path(mw.kinematics.__code__.co_filename)),file_pin(Path(mw.collision.__code__.co_filename)),file_pin(Path(mw.make_data.__code__.co_filename))],
        scope='Fixed4 recorded inputs x5 repeats, kinematics+collision only, no integration or constraint solve. No force labels read from stale EFC buffers. CPU distance is a geometric comparator under explicit GPU-table geometry, not proof GPU geometry/filtering is identical or a validated replacement simulator.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),model_exact=model_exact,
        poses=[dict(name=p['name'],CPU_distance=p['CPU_unfiltered_signed_distance_m'],GPU_contacts=[e['pair_contacts'] for e in records if e['pose']==p['name']]) for p in poses])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
