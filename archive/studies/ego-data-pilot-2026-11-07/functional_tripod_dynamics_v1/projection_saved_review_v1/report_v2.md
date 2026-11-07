Saved tripod diagnostic review:14/14. All12 cases aborted; no complete2s/tail. Prefixduration is NOT stable holding duration.

|Geometry|Control|dt(ms)|Failure prefix(s)|Prefix all3 Fn>.1 coverage|Jz / downward gravity impulse|
|---|---|---:|---:|---:|---:|
|exact_source|no_preload_ctrl_equal_q|0.3125|0.1159375|0|0.212893|
|exact_source|no_preload_ctrl_equal_q|0.15625|0.11578125|0|0.210455|
|exact_source|normal_1N_each_pad|0.3125|0.12|0.0520833|0.245018|
|exact_source|normal_1N_each_pad|0.15625|0.12|0.046875|0.239194|
|exact_source|normal_3N_each_pad|0.3125|0.1121875|0.0807799|0.343268|
|exact_source|normal_3N_each_pad|0.15625|0.11265625|0.0804438|0.367273|
|source1mm|no_preload_ctrl_equal_q|0.3125|0.125625|0|0.350399|
|source1mm|no_preload_ctrl_equal_q|0.15625|0.12609375|0|0.355066|
|source1mm|normal_1N_each_pad|0.3125|0.123125|0|0.326699|
|source1mm|normal_1N_each_pad|0.15625|0.12328125|0|0.329571|
|source1mm|normal_3N_each_pad|0.3125|0.1153125|0|0.235886|
|source1mm|normal_3N_each_pad|0.15625|0.115625|0|0.237769|

Impulse uses every signed solverPRE world contact force×dt once, including non-designated/table contacts. Ratios describe each aborted prefix, not momentum closure or a holding certificate. Initially exact thumb gap197.041um has Fn0; source1mm gaps34.347/198.048/13.803um have all Fn0 under all3 controls. Initial absent contact is only one issue: later contacts are intermittent, and upward impulse remains below the full gravity impulse. Newthree-pad categories do not turn raw force into success; oldtwo-pad categories correctly count geom41 as nonpad.

All6 controls are within actual control/actuator/joint-force bounds; saved PD reconstruction maxerror2.22e−16Nm. Entire12prefixes/gates/PREclassification/6845steps/6136contactreads are reproduced. The first audit13/14 was an auditor batch-dot rounding mismatch near quaterniondot1; oldreport/source retained, report_v2 uses original per-row dot formula without changing any gate.

Next bounded diagnostic should require native actual3pad loading atentry plus all original nonpad/self/control bounds, then address sustained wrench/COM support with explicitly bounded control. A small negativegap target is only an engineering proposal, neither a unique explanation nor a guaranteed remedy. Do not blindly increase normalforce, enforce zeroqacc, change materials, or add object assistance.
