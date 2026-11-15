# Registered readback and fixed-model response assessment v1

Freeze before evaluating old DIRECT/DIRECT_CV forward/JVPs on the new cohort.
Probe data remain CHECK/NEVER_FIT. Verify its complete declared population,
source bytes/phase/actual PD/held control/no-aid/clock/mocap/geometry and decoder
force/torque/impulse checks. Preserve invalid rows. Independently rotate decoded
mean WORLD wrench by source qpos BODY,compare cached-frame labels separately.

Verify all404 arrays and all model scalar leaves against the old WM training
parent's actual fixed GPU model at dt.0025,not just mass/inertia or XML name.
Also enforce model input/output/family schemas and actual physics dictionaries.
If mismatch exists,report it and do not call the model matched to this plant.
Use checkpoint TRAIN scales without fitting on this cohort. No optimization.

Report all repeated plus-minus contrasts,3 baseline spreads,and both finite
amplitudes for object12,wrench6 and Fn6. Compare native FD(.002) versus FD(.005)
with relative difference<=.25 and cosine>=.9 as a PREDECLARED local consistency
screen. Both amplitudes must exceed their observed repeat-spread envelope. That
screen remains finite-amplitude/direction-specific,not proof of a derivative.

At each source,compute model JVP w.r.t ACTUAL PD along the same2 directions,
not raw controls or a nominal torque tensor. Compare against repeated native
central contrasts divided by2epsilon. Raw->PD local scale.25 is checked in the
collector. Candidate model screen: qualified rows,matched physics,local native
consistency,observed signal,and model normalized AD relative error<1(zero-
response comparator) with cosine>=.9. Report every failed screen and aggregate
relative error; a few passing directions do not qualify a full Jacobian/model.
No Actor use is authorized by this diagnostic alone. Record no neural updates.
