"""CPU-only wiring checks for the secondary ±4 mm disturbance protocol, without a scored policy."""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp

from probe_flash_wuji_replay import rebuilt
from evaluate_flash_wuji_quality import source_truth
from supermimic_research.runtime.flash_quality import COMMANDS,STATES,repeat_control_trajectory
from supermimic_research.runtime.flash_disturbance import bounded_pulse,closed_run,describe,onset_frame,pulse_series


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    wp.config.kernel_cache_dir=str(args.warp_cache);wp.init();wp.set_device("cpu")
    def fresh():return rebuilt(args.parquet,args.checkpoint,True)
    protocol=json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    truth,_,frames=source_truth(args.parquet)
    formal_onset=onset_frame(truth,protocol)
    assert formal_onset is not None
    # Truncate a short development fixture independently of the formal intervention schedule.
    # Force f1 for the test only so both signs are delivered even by the zero-residual baseline.
    harness_onset=1; horizon=6
    env=fresh()
    def zero(e,_):return wp.zeros(e.action_input.shape,dtype=wp.float32,device=e.device)
    nom,stop=closed_run(env,zero,horizon,harness_onset,0,protocol,bypass=True)
    sham,_=closed_run(fresh(),zero,horizon,harness_onset,0,protocol)
    for k in [*STATES,*(f'command_{key}' for key in COMMANDS)]:np.testing.assert_array_equal(nom[k],sham[k])
    hand=env.scene.layout.hand("right");dof=hand.wrist_pos_dof_ids[0];ob=env.scene.objects[0].bodies[0].body_id
    sub=env.config.sim.substeps;limit=env.config.action.wrist_pos_clip
    output=[]
    for direction in [-1,1]:
        tr,terminal=closed_run(fresh(),zero,horizon,harness_onset,direction,protocol)
        computed,deltas=pulse_series(nom['command_joint_target_pos'],env.scene.robot_reference.joint_target[:,dof],
                                     dof,limit,harness_onset,direction,protocol,sub)
        # Zero policy uses identical source-relative wrist position targets in each run.
        np.testing.assert_allclose(tr['command_joint_target_pos'][:,dof],computed[:,dof],atol=1e-7,rtol=0)
        np.testing.assert_array_equal(tr['applied_pulse_m'],deltas)
        assert np.count_nonzero(deltas)==2*sub and np.all(np.sign(deltas[deltas!=0])==direction)
        np.testing.assert_allclose(abs(deltas[deltas!=0]),.004,atol=2e-7)
        only_position=dict(nom,command_joint_target_pos=computed)
        re,rc,ol,tstop=repeat_control_trajectory(fresh,only_position,{'failed':True},len(frames))
        ol['applied_pulse_m']=deltas
        for k in ('joint_q','joint_qd'): assert np.isfinite(ol[k]).all()
        for key in COMMANDS:
            np.testing.assert_array_equal(ol['command_'+key],only_position['command_'+key])
        assert np.array_equal(nom['command_joint_f'],ol['command_joint_f'])
        assert np.max(abs(ol['command_joint_f'][:,:6]))<=4.001
        rec,_=describe(tr,nom,truth,frames,hand.palm_frame.body_id,ob,harness_onset,sub,30.,protocol,terminal)
        np.testing.assert_allclose(bounded_pulse(direction*(limit-.0008),0,limit,direction*.004),direction*.0008,atol=1e-9)
        # Exercise the actual remaining command allowance and emitted force saturation
        # on an isolated near-limit fixture. The primary/source target remains intact.
        near=nom["command_joint_target_pos"].copy()
        refx=env.scene.robot_reference.joint_target[:,dof]
        sl=slice(harness_onset*sub,(harness_onset+2)*sub)
        near[sl,dof]=np.repeat(refx[harness_onset:harness_onset+2],sub)+direction*(limit-.0008)
        clipped,limited=pulse_series(near,refx,dof,limit,harness_onset,direction,protocol,sub)
        np.testing.assert_allclose(limited[sl],direction*.0008,atol=1e-7)
        actuator_force=[]
        def measured_scene():
            instrumented=fresh();native_step=instrumented.solver.step
            def measured(*args):
                native_step(*args)
                actuator_force.append(float(instrumented.solver.mjw_data.qfrc_actuator.numpy()[0,dof]))
            instrumented.solver.step=measured
            return instrumented
        repeat_control_trajectory(measured_scene,dict(nom,command_joint_target_pos=clipped),
                                  {"failed":True},len(frames))
        force=np.array(actuator_force);cap=55./np.sqrt(3.)
        signed_peak=float((direction*force[sl]).max())
        assert abs(force).max()<=cap+1e-4 and signed_peak>=cap-.01
        # In the actual protocol the short fixture ends before the original source lift.
        early=dict(nom,applied_pulse_m=np.zeros(len(deltas)))
        unseen,_=describe(early,nom,truth,frames,hand.palm_frame.body_id,ob,formal_onset,sub,30.,protocol,tstop)
        assert unseen['recovery_state']=='unexposed'
        output.append(dict(direction=direction,delivered_substeps=int(np.count_nonzero(deltas)),delta_m=float(deltas[harness_onset*sub]),
                 original_formal_schedule_status=unseen['recovery_state'],lateral_clip_m=direction*.0008,
                 measured_saturated_lateral_force_N=float(force[sl].max() if direction>0 else force[sl].min()),
                 max_physical_wrist_displacement_mm=rec['peak_paired_wrist_displacement_mm'],
                 open_loop_all_other_control_channels_preserved=True))
    report=dict(scope='CPU development wiring fixture; six intervals, zero residual, test-only onset at interval 1. No policy recovery result.',
                source_formal_onset_frame=int(frames[formal_onset]),fixture_onset_frame=1,substeps_per_interval=sub,
                unchanged_zero_pulse_native_states_and_commands=True,actual_object_wrench_verified_zero_per_substep=True,
                measured=output)
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('parquet','checkpoint','protocol','out','warp-cache'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
