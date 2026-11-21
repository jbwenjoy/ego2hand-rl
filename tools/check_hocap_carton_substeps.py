"""Frozen 30 Hz G04_2 reference-controller time-step check, with independent native action replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_hocap_carton import evaluate
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.ego.hocap_carton_control import replay
from supermimic_research.strict_json import write_json


def read(path): return json.loads(path.read_bytes())


def run(source_report,out):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        kernel.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000):
            raise OSError("Could not set BELOW_NORMAL_PRIORITY_CLASS for this CPU study")
    out.mkdir(exist_ok=False,parents=True)
    original=read(source_report)
    src=verify_file_pin(original['source_report'])
    initial_model=verify_file_pin(original['mjb'])
    old_eval=read(source_report.parent/'phase_evaluation.json')
    protocol=verify_file_pin(old_eval['protocol'])
    assert old_eval['passed_numeric_development_gates']
    with np.load(verify_file_pin(original['trace']),allow_pickle=False) as z:
        wrists=z['target_wrist']; fingers=z['target_finger_q']; times=z['source_nominal_time_s']
        obj=z['world_from_carton_source']; human=z['source_right_hand21']
        original_states={k:z[k] for k in ('sim_qpos','sim_qvel','substep_wrist_generalized_wrench','substep_finger_ctrl')}
    fps=original['nominal_fps']
    assert fps==30 and np.max(abs(np.diff(fingers,axis=0))/np.diff(times)[:,None])<30.000001
    np.testing.assert_allclose(np.diff(times),1/fps,atol=1e-12,rtol=0)
    variants={}
    for sub in (20,40,80):
        folder=out/f"substeps_{sub}";folder.mkdir()
        model=mujoco.MjModel.from_binary_path(str(initial_model))
        timestep=float(model.opt.timestep)
        assert timestep==1/(fps*40)
        model.opt.timestep=1/(fps*sub)
        scene=initial_model if sub==40 else folder/'scene.mjb'
        if sub!=40: mujoco.mj_saveModel(model,str(scene),None)
        # Reverting just dt in the new model must recover the entire original serialized MJB.
        test=mujoco.MjModel.from_binary_path(str(scene));test.opt.timestep=timestep
        with TemporaryDirectory() as temp:
            recovered=Path(temp)/'same.mjb';mujoco.mj_saveModel(test,str(recovered),None)
            same=file_pin(recovered)['sha256']==original['mjb']['sha256']
            if not same: raise ValueError('MJB round trip changed an additional field')
        variants[sub]=dict(folder=folder,scene=scene,wall=[],bitwise_model_except_dt=same)
    # Three deterministic timings per configuration, rotating the order. The controller targets
    # are interpolated inside the 30 Hz interval by the unmodified archived control law.
    for repetition,order in enumerate(((40,20,80),(80,40,20),(20,80,40))):
        for sub in order:
            entry=variants[sub]
            model=mujoco.MjModel.from_binary_path(str(entry['scene']))
            start=perf_counter()
            trace=replay(model,wrists,fingers,times,original['wrist'])
            entry['wall'].append(perf_counter()-start)
            if len(trace['substep_finger_ctrl'])!=(len(times)-1)*sub:raise ValueError('Substep count changed')
            if repetition==0:
                trace.update(source_nominal_time_s=times,target_finger_q=fingers,target_wrist=wrists,
                             world_from_carton_source=obj,source_right_hand21=human)
                archive=entry['folder']/'trace.npz';np.savez_compressed(archive,**trace)
                if sub==40:
                    for key,value in original_states.items():np.testing.assert_array_equal(trace[key],value)
                entry['native_q_sha']=hashlib.sha256(trace['sim_qpos'].tobytes()).hexdigest()
                rep=entry['folder']/'report.json'
                write_json(rep,dict(mjb=file_pin(entry['scene']),trace=file_pin(archive),
                    source_report=original['source_report'],wrist=original['wrist'],table=original['table'],
                    mass_kg=original['mass_kg'],nominal_fps=fps,physics_dt_s=float(model.opt.timestep),
                    substeps_per_frame=sub))
            else:
                if hashlib.sha256(trace['sim_qpos'].tobytes()).hexdigest()!=entry['native_q_sha']:
                    raise ValueError('Native CPU timing repeat changed trajectory')
    results=[]
    nominal_path=variants[40]['folder']/'trace.npz'
    with np.load(nominal_path,allow_pickle=False) as z:reference=z['world_from_carton_sim']
    for sub,d in variants.items():
        rep=d['folder']/'report.json';scored=d['folder']/'phase_evaluation.json'
        evaluate(rep,src,protocol,scored)
        result=read(scored)
        with np.load(d['folder']/'trace.npz',allow_pickle=False) as z:pose=z['world_from_carton_sim']
        position=1000*np.linalg.norm(reference[:,:3,3]-pose[:,:3,3],axis=1)
        angle=np.rad2deg((Rotation.from_matrix(reference[:,:3,:3]).inv()*Rotation.from_matrix(pose[:,:3,:3])).magnitude())
        if sub==40:
            for k,v in result['metrics'].items():np.testing.assert_allclose(v,old_eval['metrics'][k],rtol=0,atol=0)
            assert result['gates']==old_eval['gates']
        results.append(dict(substeps=sub,gate_report=file_pin(scored),mjb_identical_except_timestep=d['bitwise_model_except_dt'],
            total_integrations=(len(times)-1)*sub,source_frames=len(times),outer_rate_hz=fps,simulation_hz=fps*sub,
            replay_wall_s=d['wall'],median_wall_s=float(np.median(d['wall'])),passed=result['passed_numeric_development_gates'],
            failed_gates=[k for k,v in result['gates'].items() if not v],metrics=result['metrics'],phases=result['phases'],
            pairwise_vs_40=dict(position_rms_mm=float(np.sqrt(np.mean(position**2))),position_max_mm=float(position.max()),
                rotation_rms_deg=float(np.sqrt(np.mean(angle**2))),rotation_max_deg=float(angle.max()))))
    paper=dict(scope='Native MuJoCo CPU dt sensitivity on one known G04_2 development trajectory. Wrist and finger knots 30 Hz; feedback and interpolated drive still applied once per physics substep as in the original controller.',
            parent=file_pin(source_report),protocol=file_pin(protocol),tool=file_pin(__file__),
            walltime_scope='Native control integration including per-substep tracing, excluding model load, serialization and independent score. Three repetitions at balanced order, serial low-priority on CPU.',
            unchanged_40_control_commands_states_metrics_and_gates=True,results=results)
    write_json(out/'summary.json',paper)
    for r in results:
        m=r['metrics']
        print(r['substeps'],r['passed'],r['failed_gates'],r['median_wall_s'],
              m['position_rms_mm'],m['rotation_rms_deg'],m['peak_lift_mm'],m['endpoint_error_mm'],
              m['supported_airborne_intervals'],m['rerun_max_state_error'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('original',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();run(args.original.resolve(),args.output.resolve())
