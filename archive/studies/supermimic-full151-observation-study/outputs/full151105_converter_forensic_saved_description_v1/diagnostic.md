# Converter arithmetic forensics: author-side saved description

This is saved-output rendering, not Contact’s independent mathematics audit. No conversion/model/query/scoring is repeated. The original dense attempt FAILED its output-scaled8eps32 endpoint gate before factory/scoring; that FAIL remains unchanged. This new arithmetic-only condition cannot grant dense-scoring eligibility.

|Set|Block|Unit|Failed elements|Failed rows|Maximum absolute difference|
|---|---|---|---:|---:|---:|
|452 endpoints|q|rad|0|0|0|
|452 endpoints|qd|rad/s|0|0|0|
|452 endpoints|position|m|0|0|0|
|452 endpoints|rotation|matrix entry|0|0|4.76837158e-07|
|452 endpoints|linear_velocity|m/s|0|0|0|
|452 endpoints|angular_velocity|rad/s, reader/wrist frame|5|5|5.24520874e-06|
|160 anchors|q|rad|0|0|0|
|160 anchors|qd|rad/s|0|0|0|
|160 anchors|position|m|0|0|0|
|160 anchors|rotation|matrix entry|0|0|0|
|160 anchors|linear_velocity|m/s|0|0|0|
|160 anchors|angular_velocity|rad/s, reader/wrist frame|0|0|2.38418579e-07|

All452 endpoint and160 anchor feature rows are shown, not only failures. q/qd use exactFP32 bits, including signedzero; their numeric difference being0 does not override a bit mismatch. Other components retain the ORIGINAL bound8eps32·max(1,|actual|,|reference|). Black path dividers are source grouping, not a continuous time axis across restores.

All1808 solverPRE pose comparisons and all2260 conditioning descriptors are retained. State58 omega is reader/wrist-frame angular velocity: raw freejoint BODYomega → objectrotation toWORLD → inversepalmrotation. It is neither untouched bodyomega nor generic Euler-angle velocity. Position, rotation entries, linear velocity and angular velocity use different units and have no combined mixed-unit score.

The F64 shadow is SECONDARY arithmetic on the same saved inputs, not replacement physical state or reference truth. Operand absolute sums and cancellation ratios are descriptors, not a proved numerical bound or newly accepted tolerance. Different F64 summation orders can matter nearzero. A correlation with cancellation does not uniquely prove roundoff as the cause, does not eliminate wrongframe/mapping/phase, and does not uniquely implicate TF32. Both CPU and recorded GPU differences from the shadow are shown.

The forensic producer reports only passiveMJB1/nativeState2260 and diagnostics, with factory/score/WM/Actor/AD/optimizer/Env/physics0. RNG/in-memory invariance flags remain source-backed runtime evidence. Existing sources/config/failure/data are immutable; originalFAILED gate is not rescued by a successful diagnostic serialization or any newer replay flags.

![all452_endpoint_old_rule.png](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v1/all452_endpoint_old_rule.png)

![all160_anchor_old_rule.png](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v1/all160_anchor_old_rule.png)

![all1808_solver_pose_old_rule.png](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v1/all1808_solver_pose_old_rule.png)

![all2260_conditioning_secondary.png](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v1/all2260_conditioning_secondary.png)

- [all_old_component_diagnostics.csv](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v1/all_old_component_diagnostics.csv)
- [all2260_conditioning_by_block.csv](D:/workspace/supermimic-full151-observation-study/outputs/full151105_converter_forensic_saved_description_v1/all2260_conditioning_by_block.csv)
