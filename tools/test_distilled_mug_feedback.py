"""Evaluate frozen residual feedback actors on known and fresh physical parameter combinations."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch
from distill_mug_witness_feedback import h3,plant,VARIANTS,MugResidualActor,observation,TRAIN
from test_mug_witness_mpc import context
from qualify_mug_window import NativeStrictWindow
from repair_mug_path_geometry import LIMITS


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    distill=h3.file_pin(root/'mug_feedback_distill_v1/report.json')
    report=h3.physics.read_json(distill);pins=h3.physics.read_json(report['frozen'])['weights']
    modelpin=h3.file_pin(root/'mug_nominal_tracking_gains_v1/report.json')
    fit=h3.physics.read_json(modelpin);cfg=h3.physics.read_json(fit['intent']);inputs=cfg['inputs']
    asset=h3.verify_file_pin(inputs['model']);bind=h3.physics.read_json(inputs['binding']);old=h3.physics.load_arrays(inputs['reference'])
    qualification=h3.file_pin(root/'mug_window_native_check_v1/report.json')
    kernel_source=h3.physics.read_json(h3.physics.read_json(qualification)['intent'])['build']
    lib=h3.verify_file_pin(h3.physics.read_json(kernel_source)['helper'])
    previous=h3.physics.read_json(h3.file_pin(root/'mug_tracking_feedback_v1/report.json'))['rows']
    torch.set_num_threads(1);nets={}
    for a,pin in pins.items():
        ck=torch.load(h3.verify_file_pin(pin),map_location='cpu',weights_only=True)
        n=MugResidualActor(*ck['dims']);n.load_state_dict(ck['state']);nets[a]=n.eval()
    sampler=qmc.LatinHypercube(d=4,seed=8551);grid=qmc.scale(sampler.random(12),[.95,.95,.90,-.001],[1.05,1.05,1.10,.001])
    conditions={**VARIANTS,**{f'generated_{i:02}':v.tolist() for i,v in enumerate(grid)}}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(distilled=distill,nominal_model=modelpin,kernel_qualification=qualification,
        code=h3.file_pin(__file__),seed=8551,conditions=conditions,scope=
        'Frozen paired students, open-loop control, six source singles, six earlier nonsource/nominal and 12 fresh four-parameter Latin hypercube cases per clip. Primary comparison: both full nominal runs preserved and strictly positive median paired gain in BOTH clips over frozen commands on generated cases. All decisions omit true perturbation and solver memory. One CPU network forward per action, no lookahead.'))
    rows=[]
    for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
        f=h3.physics.load_arrays(fit['clips'][str(clip)]['data'])
        item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows'];z=h3.physics.load_arrays(item['reference']);T=z['body']
        pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
        nominal=mujoco.MjModel.from_binary_path(str(asset))
        for case,(mass,mu,tau,offset) in conditions.items():
            m=plant(asset,bind,mass,mu,tau);P,python=context(m,bind,pos,rot);native=NativeStrictWindow(P,lib)
            def initial():
                d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=old['feedforward_PD_target20'][0]
                qa=P.chart.qa;d.qpos[qa:qa+3]+=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])]).apply([offset,0,0])
                mujoco.mj_forward(m,d);return d
            for arm in ('open','pose','wrench'):
                d=initial();last=np.zeros(20);qs=[];metrics=[];cmds=[];wall=0.
                for i,cmd0 in enumerate(f['cmd']):
                    before=perf_counter()
                    if arm=='open':cmd=cmd0
                    else:
                        ctx,e=observation(nominal,f['state'][i],P.state(d),cmd0,last)
                        if abs(e).max()<1e-10:e[:]=0.
                        with torch.inference_mode():delta=nets[arm](torch.from_numpy(ctx),torch.from_numpy(e)).numpy()
                        cmd=np.clip(cmd0+delta,*nominal.actuator_ctrlrange.T)
                    wall+=perf_counter()-before;last=cmd-cmd0
                    val=native.step(d,i,cmd);metrics.append(val);qs.append(d.qpos.copy());cmds.append(cmd.copy())
                    if np.any(val>LIMITS):break
                passed=len(metrics)-int(np.any(metrics[-1]>LIMITS))
                again=initial()
                for i,c in enumerate(cmds):
                    np.testing.assert_array_equal(python.step(again,i,c),metrics[i]);np.testing.assert_array_equal(again.qpos,qs[i])
                if arm=='open' and case in VARIANTS:
                    gold=next(r for r in previous if r['arm']=='open' and r['condition']==case and r['clip']==clip)
                    assert passed==gold['passing']
                if case=='nominal':
                    np.testing.assert_array_equal(qs,f['state'][1:,1:1+m.nq]);np.testing.assert_array_equal(cmds,f['cmd'])
                trace=out/f'{clip}_{case}_{arm}.npz';np.savez_compressed(trace,ctrl=cmds,qpos=qs,metrics=metrics)
                rows.append(dict(clip=clip,condition=case,arm=arm,train_condition=case in TRAIN,passing=passed,
                    full=passed==len(f['cmd']),last=metrics[-1].tolist(),decisions=len(metrics),decision_wall_s=wall,
                    trace=h3.file_pin(trace)))
            h3.write_json(out/'progress.json',rows)
            print(clip,case,[r['passing'] for r in rows[-3:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
