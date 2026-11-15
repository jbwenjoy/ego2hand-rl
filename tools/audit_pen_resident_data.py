"""Actual full-data roundtrip preflight, not a plant-replay qualification."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case,write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.runtime.gpu_data_snapshot import ResidentDataSnapshot
from supermimic_research.runtime.gpu_model_snapshot import capture_model


def main(a):
    import torch
    import warp as wp
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());bp=(cp.parent/cfg['base_config']).resolve();base=tomllib.loads(bp.read_text())
    pp=(cp.parent/cfg['proposal']).resolve();proposal=json.loads(pp.read_text());selected=np.load(verify_file_pin(proposal['artifacts']['selected']))
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified snapshot preflight')
    torch.set_num_threads(4);wp.set_device('cuda:0');a.output.mkdir();registry=ArtifactRegistry(bp.parent/base['artifact_manifest'])
    sources=[file_pin(__file__)]+[file_pin(ROOT/p) for p in ['src/supermimic_research/runtime/gpu_data_snapshot.py',
        'src/supermimic_research/runtime/gpu_model_snapshot.py','src/supermimic_research/runtime/controlled_evaluation.py',
        'src/supermimic_research/runtime/pen_gpu_initialization.py']]
    intent=dict(schema='actual_pen_dynamic_array_roundtrip:1',config=file_pin(cp),base_config=file_pin(bp),proposal=file_pin(pp),sources=sources,
        neural_updates=0,declared_extra_native_probe_ticks=4,
        scope='2 initial-source one-physics-step repeats per native dt, restored full WP Data before untouched actual reference episode. Not later-source/20ms plant fidelity, derivative labels, policy or task qualification.')
    write(a.output/'intent.json',intent);cases=[];receipts=[]
    for dt in [.005,.0025]:
        folder=a.output/f'actual_{dt}'
        def initialize(env):
            receipt=initialize_and_query_pen(env,folder,selected['source_qpos'],selected['qpos'],selected['delta20'],selected['target_BODY'],support_all_robot_contacts=True)
            if env.sim.step_graph is None or not env.sim.use_cuda_graph:raise ValueError('Same native captured-step graph required')
            wp.synchronize();snapshot=ResidentDataSnapshot(env.sim.wp_data);wp.synchronize()
            before=snapshot.audit(env.sim.wp_data);_,scalars,model_before=capture_model(env.sim.wp_model)
            np.savez_compressed(folder/'full_initial_dynamic_arrays.npz',**snapshot.host_values())
            write(folder/'dynamic_scalars.json',snapshot.scalars);repeats=[]
            try:
                for repeat in range(2):
                    snapshot.restore(env.sim.wp_data);wp.synchronize()
                    with wp.ScopedDevice(env.sim.wp_device):wp.capture_launch(env.sim.step_graph)
                    wp.synchronize();changed=snapshot.audit(env.sim.wp_data)
                    state={k:getattr(env.sim.wp_data,k).numpy().copy() for k in ['qpos','qvel','ctrl','qacc','qacc_warmstart','time','qfrc_applied','xfrc_applied']}
                    np.savez_compressed(folder/f'probe_{repeat}.npz',**state)
                    snapshot.restore(env.sim.wp_data);wp.synchronize();restored=snapshot.audit(env.sim.wp_data)
                    repeats.append(dict(repeat=repeat,changed_arrays=changed['differing_arrays'],restored=restored,
                        active_finite=bool(all(np.isfinite(v).all() for v in state.values())),
                        no_aid=not np.any(state['qfrc_applied']) and not np.any(state['xfrc_applied']),probe=file_pin(folder/f'probe_{repeat}.npz')))
            finally:
                snapshot.restore(env.sim.wp_data);wp.synchronize()
            final=snapshot.audit(env.sim.wp_data);_,after_scalars,model_after=capture_model(env.sim.wp_model)
            checks=dict(initial_exact=before['all_array_bytes_equal'] and before['scalar_equal'],
                physical_step_changes_state=all(bool(r['changed_arrays']) for r in repeats),
                all_repeat_restores=all(r['restored']['all_array_bytes_equal'] and r['restored']['scalar_equal'] for r in repeats),
                final_full_data_exact=final['all_array_bytes_equal'] and final['scalar_equal'],
                model_unchanged=model_before==model_after and scalars==after_scalars,
                active_finite=all(r['active_finite'] for r in repeats),no_aid=all(r['no_aid'] for r in repeats))
            record=dict(dt=dt,arrays=final['arrays'],scalar_leaves=final['scalar_leaves'],checks=checks,repeats=repeats,
                initial_arrays=file_pin(folder/'full_initial_dynamic_arrays.npz'),initial_scalars=file_pin(folder/'dynamic_scalars.json'))
            receipts.append(record);write(folder/'snapshot_review.json',record)
            if not all(checks.values()):raise ValueError('Restore preflight fails; no actual continuation')
            return dict(**receipt,resident_dynamic_snapshot=file_pin(folder/'snapshot_review.json'))
        c=evaluate_case('hand2-pen-spin','DATA_SNAPSHOT_PREFLIGHT',None,{**base,'physics_dt':dt,
            'numerical_options':cfg['numerical_options'],'numerical_device_override':True},bp.parent,registry,folder,
            initializer=initialize,constant_action=2*selected['delta20'])
        c['name']='RESTORED_BASELINE';cases.append(c)
    for pin in sources:verify_file_pin(pin)
    write(a.output/'report.json',dict(**intent,cases=cases,snapshot_receipts=receipts,passed=sum(sum(r['checks'].values()) for r in receipts),total=sum(len(r['checks']) for r in receipts)))
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),passed=sum(sum(r['checks'].values()) for r in receipts),total=sum(len(r['checks']) for r in receipts),arrays=[r['arrays'] for r in receipts])))


if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False):main(p.parse_args())
