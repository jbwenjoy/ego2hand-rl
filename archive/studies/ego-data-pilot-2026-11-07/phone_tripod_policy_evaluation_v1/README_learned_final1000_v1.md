# Fixed paired1000 native Actor evaluation

`evaluate_learned_actors_v1.py --protocol <absolute path>` is metadata only.
`learned_CPUFalse_v1.json` has four explicitly pending final inputs:
`training_final_protocol`, `final_result`, `final_health`, `final_terminal`.
The actual entry refuses pending files, checkpoint100, incomplete counts or
nonzero training exit. Root will bind the same fixed paired1000/WM1100 result
and current artifact/source pins, authorize CPU/NN/native execution and use a
fresh output. No current source preparation executed Torch or MuJoCo.

`--run` performs one trusted own CPU checkpoint load, restores both `stateactor`
state dictionaries and exactly checks each serialized Actor norm. The pinned
upstream factory validates the actual imported Actor/common source before
construction. Two factory calls construct two Actors and two unused Critics;
there is zero Critic forward, zero WM construction/forward, zero backward or
optimizer and zero GPU. CUDA is masked before importing Torch. Actor CPU
inference has one B1 forward per issued20ms command, at most400 across4 cases.

Each arm uses the same actual initial qpos27/zero qvel/ctrl20, model and frozen
native kernel at h=.0003125 and .00015625. The complete four-case budget is
38400 native steps, with4 original reset forwards; there are no observer
forwards, state resets during a case or external forces. The previously
qualified `TaskController` feeds current COM/world state58 and obs222 into the
Actor. It is wrapped with the training source's `CanonicalMemoryTask`, keeping
all four floating history vectors F64. Future object commands remain allowed
task inputs; true future contacts/forces and model predictions are not inputs.
Observation/control semantics are mode-independent, so A1's common task
observation/control implementation is used for both restored policies without
constructing or evaluating the A2 engineered support reward proxy.

Every abort, original null-tail metric, full native gate, legacy two-pad gate
and dual-dt incomplete result is retained. Tracking adds a separate101-frame
descriptor at integer POST ticks k*n (k0 initialized state). It reports COM
distance and full SO3 geodesic angle against the immutable command, never
indexes from accumulated native floating time, fills an aborted tail or
stretches the command. A phone frozen at its actual initial COM/BODY rotation
is an analytical tracking comparator only; its full101 command error and its
error on the policy's available POST frames are both reported. It is not a
physical baseline rollout. No new tracking-success threshold is introduced,
and the existing holding5mm gate is not repurposed as tracking success.

The interface `load_final_actors(protocol_path, torch)` returns the two CPU
Actors and a load/count receipt; its protocol remains this fixed evaluation
schema. Other acquisition stages need their own explicit protocol binding,
not a renamed evaluation protocol. Output includes controller records,
native traces/summaries, masked tracking arrays/statistics, all4 case manifest
and final report or a preserved failure receipt. No automatic retries.

This evaluates a matched offline shared-WM pilot. It does not establish
completed online SuperMimic/CHORD, unseen-state/ego generalization, engineered
proxy contact truth, or policy benefit before the actual physical results.
