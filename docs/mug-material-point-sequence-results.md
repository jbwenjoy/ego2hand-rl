# HOT3D mug contact trajectory comparison

On the 61 annotated source frames of Aria clip 1884, seven contact material points from the initial robot grasp were tracked through the 101-sample object reference. These are simulated contact hypotheses. Human motion supplied the five relative fingertip displacement objectives. All three articulated arms retained the same initial grasp, source clock, nominal mug, 2 s duration and original preload for the geometry comparison.

| Trajectory | Max reference hand/mug penetration (mm) | Nominal wrench solutions also within 0.1 mm, of 97 | First simulated COM error above 10 mm |
| --- | ---: | ---: | ---: |
| Fingertip displacement | 5.60 | 1 | 0.30 s |
| Fingertips and contact material points | 2.45 | 4 | 0.12 s |
| Plus active gap penalty and 5 rad/s joint-target cap | 0.081 | 74 | 0.32 s |

The 5 rad/s value is an experimental target slew cap. Nine of the 100 bounded transitions used an optimizer iterate at the evaluation budget; the 101 reported states include the fixed initial pose. Constant-grasp baseline native qpos reproduced exactly. Full-run COM RMS for the bounded arm was 734 mm, versus 426 mm for fingertip tracking alone; late loss and table interaction are included.

A second comparison held this last trajectory fixed and solved a minimum-change joint preload with equal and opposite contact wrenches and actuator force/control bounds. Twenty-nine of 97 nominal intervals had feasible solutions. Other intervals used the original preload. The baseline trajectory replayed exactly. With the new servo targets the first 10 mm crossing was 0.34 s and the first 10° crossing 0.36 s (baseline 0.32/0.34 s); COM RMS over 2 s was 389 mm. Geometry and quasi-static wrench admission remain insufficient for sustained dynamic support. Reference penetration differs from penetration during execution. These numbers are for one annotated clip and nominal simulator mechanics.

[Material-point arm](../runs/mug_temporal_material_ik_v1/report.json) · [bounded arm](../runs/mug_gap_material_ik_v1/report.json) · [torque-bounded feedforward](../runs/mug_temporal_feedforward_v1/report.json)
