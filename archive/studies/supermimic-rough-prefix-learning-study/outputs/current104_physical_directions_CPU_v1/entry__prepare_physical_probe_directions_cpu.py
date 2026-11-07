"""Freeze outcome-blind absolute-target perturbations; no simulated branch."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import checked,read,pin
from rough_prefix.interior_directions import prepare_interior_directions


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    plan=read(args.plan)
    if plan['schema']!='rough_physical_probe_direction_CPU_preparation:1':
        raise ValueError('unregistered preparation schema')
    for v in plan['pins'].values(): checked(v)
    if plan['pins']['entry']!=pin(__file__): raise ValueError('changed entry')
    if plan['pins']['direction_source']!=pin(ROOT/'src/rough_prefix/interior_directions.py'):
        raise ValueError('changed direction implementation')
    if plan['GPU_authorized'] is not False or plan['physical_branches_authorized'] is not False:
        raise ValueError('CPU preparation cannot authorize physics')
    if plan['path_order']!=['A0','A1L','P_ONLY','A2'] or plan['phase_order']!=[0,4,8,12]:
        raise ValueError('unregistered source/phase selection')
    if (plan['seed'],plan['direction_count'],plan['epsilons'],plan['guard_ulps'])!=(2026110401,8,[.002,.001],8):
        raise ValueError('unregistered perturbation definition')
    output=Path(plan['output']).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()): raise ValueError('unowned output')
    if not args.run:
        print(json.dumps(dict(status='CPU_PREFLIGHT_ONLY',physics_steps=0,GPU_used=False)))
        return
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    for k,v in plan['pins'].items():
        source=checked(v)
        if source.suffix!='.npz' and source.suffix!='.mjb':
            (output/(k+'__'+source.name)).write_bytes(source.read_bytes())
    (output/'preparation_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    import mujoco
    # Metadata load only: no MjData, reader, environment, forward, collision, or step.
    model=mujoco.MjModel.from_binary_path(str(checked(plan['pins']['nominal_model'])))
    jids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
    aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids])
    lo=np.maximum(model.jnt_range[jids,0],model.actuator_ctrlrange[aids,0]).astype(np.float32)+np.float32(.020)
    hi=np.minimum(model.jnt_range[jids,1],model.actuator_ctrlrange[aids,1]).astype(np.float32)-np.float32(.020)
    centers=[];state_rows=[]
    rng_before=np.random.get_state()
    with np.load(checked(plan['pins']['saved_predictions']),allow_pickle=False) as saved:
        for name in plan['path_order']:
            events=read(checked(plan['pins'][name+'_control_events']))
            targets=np.asarray([e['actualu'] for e in events],np.float32)
            if targets.shape!=(15,1,20) or not np.array_equal(targets[:,0],saved[name+'__u']):
                raise ValueError('saved targets differ from original physical events: '+name)
            for k in plan['phase_order']:
                centers.append(targets[k,0].copy())
                state_rows.append(dict(path=name,phase=k,nominal_u=targets[k,0].tolist()))
    arrays=prepare_interior_directions(np.stack(centers),lo,hi,seed=plan['seed'],
        count=plan['direction_count'],epsilons=plan['epsilons'],guard_ulps=plan['guard_ulps'])
    for i,row in enumerate(state_rows):
        row.update(free_coordinate_count=int(arrays['free_mask'][i].sum()),
            free_coordinate_indices=np.flatnonzero(arrays['free_mask'][i]).tolist(),
            valid_pair_count=int(arrays['valid_pair'][i].sum()),potential_pair_count=16,
            status='PREPARED_NOT_SIMULATED' if arrays['free_mask'][i].any() else 'NOT_RUN_NO_FREE_COORDINATE')
    rng_after=np.random.get_state()
    if not all(np.array_equal(x,y) for x,y in zip(rng_before,rng_after)):
        raise AssertionError('global NumPy RNG mutated')
    for v in plan['pins'].values(): checked(v)
    np.savez_compressed(output/'directions_and_FP32_targets.npz',**arrays)
    report=dict(schema='rough_physical_probe_direction_CPU_result:1',status='PREPARED_NOT_PHYSICALLY_QUALIFIED',
        plan=pin(args.plan),rows=state_rows,direction_instances=128,
        potential_FD_pairs=256,valid_domain_pairs=int(arrays['valid_pair'].sum()),
        zero_dimensional_states=int((~arrays['free_mask'].any(-1)).sum()),
        seed=plan['seed'],guard_ulps=plan['guard_ulps'],epsilons=plan['epsilons'],
        directions_shared_across_three_WM=True,selection_uses_model_outputs=False,
        selection_uses_physical_perturbation_outputs=False,targets_clipped_or_resampled=False,
        source_and_input_pins_unchanged=True,global_NumPy_RNG_unchanged=True,
        MuJoCo_nominal_model_metadata_read=True,env_created=False,physics_steps=0,
        physics_forward_or_collision_calls=0,WM_forwards=0,GPU_used=False,
        Torch_or_Warp_imported=any(x in sys.modules for x in ('torch','warp','mujoco_warp')),
        dataset=pin(output/'directions_and_FP32_targets.npz'),wall_seconds=time.perf_counter()-started,
        scope='Symmetric interior-coordinate subspace only. Fresh donor targets must equal archived centers bitexact; otherwise fail setup, never regenerate masks. Active-bound face-normal/one-sided directions and complete action Jacobian remain untested.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','direction_instances','potential_FD_pairs','valid_domain_pairs','zero_dimensional_states','physics_steps','WM_forwards','GPU_used')}))


if __name__=='__main__':main()
