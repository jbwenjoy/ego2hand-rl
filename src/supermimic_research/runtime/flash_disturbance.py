"""Secondary source-timed wrist target pulses; all perturbations act through the hand."""
import numpy as np
from scipy.spatial.transform import Rotation

from .flash_quality import SolverRecorder


def onset_frame(source_pose, protocol):
    height = source_pose[:,2,3] - source_pose[0,2,3]
    peak = float(height.max())
    if peak < protocol["minimum_source_peak_lift_m"]:
        return None
    ids = np.flatnonzero((height[1:] >= protocol["source_lift_fraction"]*peak) & (np.diff(height)>0))+1
    if not len(ids) or int(ids[0])+protocol["pulse_intervals"] >= len(height):
        return None
    return int(ids[0])


def bounded_pulse(current, reference, limit, requested):
    """Apply only the available headroom in the requested direction."""
    if requested >= 0:
        return min(requested, max(0., reference+limit-current))
    return -min(-requested, max(0., current-(reference-limit)))


def pulse_series(commands, reference_x, dof, limit, onset, direction, protocol, substeps):
    result=commands.copy()
    deltas=np.zeros(len(result))
    if onset is not None and direction:
        lo=onset*substeps
        hi=min((onset+protocol["pulse_intervals"])*substeps,len(result))
        for s in range(lo,hi):
            deltas[s]=bounded_pulse(result[s,dof],reference_x[s//substeps],limit,direction*protocol["amplitude_m"])
            result[s,dof]+=deltas[s]
    return result,deltas


def closed_run(env, action_fn, horizon, onset, direction, protocol, *, bypass=False):
    """Run a policy on fresh observations; an exogenous pulse never changes the policy buffer."""
    rec=SolverRecorder(env)
    observer=env.solver.step
    hand=env.scene.layout.hand("right")
    dof=hand.wrist_pos_dof_ids[0]
    limit=float(env.config.action.wrist_pos_clip)
    if not np.isfinite(limit) or limit<=0:raise ValueError("Invalid wrist position allowance")
    ref=env.scene.robot_reference.joint_target[:,dof]
    stepcount=0;applied=[];actions=[]
    def inject(s0,s1,control,contacts,dt):
        nonlocal stepcount
        frame=stepcount//env.config.sim.substeps
        delta=0.
        if not bypass and onset is not None and direction and onset<=frame<onset+protocol["pulse_intervals"]:
            original=control.joint_target_pos.numpy().copy()
            delta=bounded_pulse(original[dof],ref[frame],limit,direction*protocol["amplitude_m"])
            if delta:
                temp=original.copy()
                temp[dof]+=delta
                control.joint_target_pos.assign(temp)
                try:
                    observer(s0,s1,control,contacts,dt)
                finally:
                    control.joint_target_pos.assign(original)
            else:
                observer(s0,s1,control,contacts,dt)
        else:
            observer(s0,s1,control,contacts,dt)
        applied.append(delta);stepcount+=1
    env.solver.step=inject
    stop=dict(failed=False,reason="end of source",termination_causes=[])
    for i in range(horizon):
        if int(env.timestep.numpy()[0])!=i or np.any(env.object_control_scale.numpy()):
            raise ValueError("Wrist pulse evaluation requires continuous unassisted source time")
        action=action_fn(env,i);actions.append(action.numpy().copy())
        env.step(action);rec.end_frame()
        if env.done.numpy()[0] or env.truncation.numpy()[0]:
            stop=dict(failed=bool(env.done.numpy()[0]) or i+1<horizon,
                      reason="native episode ended",termination_causes=[name for name,arr in
                         zip(env.termination.cause_names,env.termination.cause_arrays) if arr.numpy()[0]])
            break
    rec.close()
    trace=rec.materialize()
    trace["actor_action"]=np.stack(actions)
    trace["applied_pulse_m"]=np.array(applied)
    return trace,stop


def describe(trace, nominal, source, frames, hand_body, object_body, onset, substeps, fps, protocol, stop):
    n=len(trace["body_q"]);base_n=len(nominal["body_q"]);paired=min(n,base_n)
    body=trace["body_q"];base=nominal["body_q"]
    a,b=body[:paired,object_body],base[:paired,object_body]
    rotation=np.rad2deg((Rotation.from_quat(a[:,3:]).inv()*Rotation.from_quat(b[:,3:])).magnitude())
    translation=np.linalg.norm(a[:,:3]-b[:,:3],axis=1)
    hand_delta=body[:paired,hand_body,:3]-base[:paired,hand_body,:3]
    obj_delta=a[:,:3]-b[:,:3]
    obj=body[:,object_body]
    err=np.linalg.norm(obj[:,:3]-source[:n,:3,3],axis=1)
    rot_err=np.rad2deg((Rotation.from_quat(obj[:,3:]).inv()*Rotation.from_matrix(source[:n,:3,:3])).magnitude())
    pulse=trace["applied_pulse_m"]
    planned=protocol["pulse_intervals"]*substeps
    visited=0 if onset is None else max(0,min(len(pulse)-onset*substeps,planned))
    delivered=int(np.count_nonzero(abs(pulse)>1e-8))
    at_full=int(np.isclose(abs(pulse),protocol["amplitude_m"],atol=1e-6,rtol=0).sum())
    answer=dict(end_source_frame=int(frames[n-1]),simulated_intervals=n-1,requested_source_intervals=len(frames)-1,
       terminal=stop,common_frames=paired,planned_pulse_substeps=planned,visited_pulse_substeps=visited,
       delivered_pulse_substeps=delivered,full_amplitude_substeps=at_full,fully_exposed=at_full==planned,
       maximum_abs_command_delta_mm=float(1000*np.max(abs(pulse),initial=0.)),
       signed_mean_command_delta_on_visited_substeps_mm=float(1000*pulse.sum()/visited) if visited else None,
       prefix_source_position_rms_mm=float(1000*np.sqrt(np.mean(err**2))),
       prefix_source_rotation_rms_deg=float(np.sqrt(np.mean(rot_err**2))))
    series=dict(object_delta_xyz_m=obj_delta,wrist_delta_xyz_m=hand_delta,object_rotation_delta_deg=rotation,
                source_position_error_m=err,source_rotation_error_deg=rot_err)
    if onset is None or not delivered:
        return dict(answer,recovery_state="not_applicable" if onset is None else "unexposed"),series
    if paired<=onset+1:
        return dict(answer,recovery_state="no_paired_postpulse_observation"),series
    first=onset+1;end_pulse=onset+protocol["pulse_intervals"]
    last=min(paired-1,end_pulse+protocol["post_pulse_observation_intervals"])
    delta=translation[first:last+1]; angles=rotation[first:last+1]
    deviation=(translation>protocol["recovery_translation_m"]) | (rotation>protocol["recovery_rotation_deg"])
    answer.update(prepulse_object_position_delta_mm=float(1000*translation[onset]),
       prepulse_object_rotation_delta_deg=float(rotation[onset]),
       peak_paired_object_deviation_mm=float(1000*delta.max()),
       peak_paired_rotation_deviation_deg=float(angles.max()),
       peak_paired_wrist_displacement_mm=float(1000*np.linalg.norm(hand_delta[first:last+1],axis=1).max()),
       peak_paired_object_deviation_frame=int(frames[first+int(delta.argmax())]),
       postpulse_observed_intervals=max(0,last-end_pulse),matched_window_complete=last>=end_pulse+protocol["post_pulse_observation_intervals"])
    dep=np.flatnonzero(deviation[first:last+1])+first
    prebad=bool(deviation[onset])
    answer["already_outside_recovery_band_at_onset"]=prebad
    answer["return_latency_s"]=None
    if at_full!=planned:
        answer["recovery_state"]="partial_exposure"
    elif prebad:
        answer["recovery_state"]="nominal_diverged_before_pulse"
    elif not len(dep):
        answer["recovery_state"]="no_departure" if answer["matched_window_complete"] else "censored_no_observed_departure"
    else:
        k=protocol["consecutive_return_frames"];amend=None
        for j in range(max(int(dep[0])+1,end_pulse),last-k+2):
            if not deviation[j:j+k].any():
                amend=j;break
        answer["first_departure_frame"]=int(frames[int(dep[0])])
        if amend is None:
            answer["recovery_state"]="not_recovered" if answer["matched_window_complete"] else "censored_after_departure"
        else:
            answer["recovery_state"]="recovered"
            answer["return_latency_s"]=(amend-end_pulse)/fps
            answer["return_frame"]=int(frames[amend])
            answer["departed_again_in_observed_window"]=bool(deviation[amend+k:last+1].any())
    return answer,series
