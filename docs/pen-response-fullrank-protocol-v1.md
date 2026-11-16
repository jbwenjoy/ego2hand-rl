# Full20D physical responses and reserved validation — protocol v1

Registered before new source execution/collection or fitting. Old four-arm
models failed physical-response CHECK gates. Rank4/20 and response-loss scale
are leads,not proven sole causes. This cohort changes coverage and histories;
do not attribute its aggregate outcome to rank alone.

Use maintained collector/recorder/native4/decoder/packer. New schema3 probe and
preparation2 retain explicit TRAIN or VALIDATE_NEVER_FIT roles;old324 CHECK
and old816 TRAIN keep bytes,roles,splits and masks. No old native state is
rebased across source versions. This is data-only:zero optimizer/Actor updates.

Generate twenty unit orthonormal PD directions by PCG64 Gaussian20x20,QR of
its transpose,positive R diagonal,then transpose Q to rows. Rank20,Gram I and
condition1 are verified before simulation. TRAIN seed9401;validation9402.
First four TRAIN rows form a declared matched rank4 response-loss subset.
They are not the old Gaussian4 directions;old data are not overwritten.

TRAIN:two frozen final64 actors x prefix-noise seeds8841/8842,sigma.01,
uninterrupted phase1 execution,source phases12/24/36/48/64/72. Complete all
four current LIVE W1 producer episodes before any restore. Pre-materialize71
raw-noise draws per seed;use mean actor plus paired exogenous noise through
the original raw clamp/scale/EMA/PD pipeline. No post-start reset or assistance.
Only actor/norm weights are initialized using full404 W1 Model/scalar and
joint/config/observation/action checks against the frozen W1 context. Historical
CHECK context contributes only Model compatibility,zero states/response labels.

Reserved validation:separate fresh executions,seeds8861/8862,65 noise draws,
phases18/42/66,independent direction seed9402. Role remains
VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT. Never relabel it as TRAIN or
the old CHECK. The same original reference and fixed actor seeds limit it to
development validation,not original-trajectory/final-test/ego generalization.

Per source:two base repeats +20 directions x2 amplitudes(.002/.005rad) x2
signs x2 repeats =162 branches. Alternate sign order by repeat. Submit
base_raw+4*sign*epsilon*direction;reject clipping and require actual held
PD20=baseline_PD+sign*epsilon*direction within2e-7rad. Restore every branch
STRICT with full native/Python/RNG/observation readback,then eight2.5ms ticks.
Do not enable the unresolved read-locked epoch prototype or change physics.

Source gates:nonterminal,eight previous samples,object/self/joint penetration
2mm/.2mm/.001rad,nonhand<=.1N,mean hand normal>.1N,finite,metric errors0,
no aids,unclipped base. Retain every unavailable/excluded source/branch with
no replacement. Source/case declared denominators stay TRAIN24/3888 and
validation12/1944. If all available,native work TRAIN2272+31104=33376,
validation2080+15552=17632,total51008;actual lower work must be reported.

Retain all original raw native/EFC/warmstart/PD/geometry/time packets and exact
sources. Independently decode and pack state55+warm26+actualPD20+dt1 to
delta52+nextWarm26+BODY-moving-COM wrench6+Fn6. Original force1e-4N,torque1e-5Nm,
linear impulse5e-7Ns,no-overflow/no-aid/nonhand/chart/geometry gates are unchanged.
All failed individual gates and rows stay in preparation with an explicit mask;
do not loosen tolerances to make full rank pass. Report per-source qualified
direction rank/conditioning after masks,both amplitudes and repeat spread.

No fit/scales from validation. After preparation,freeze a separate experiment
comparing rank4 vs rank20 response pairs on the SAME full20 TRAIN point bank,
same point draws/counts,initial NN,valid-TRAIN scales,batch/update budget and
response weighting. Only pair-direction eligibility differs in that ablation;
it tests supervised response-direction coverage,not the cost of collecting
fewer point samples. Scales/weights/forward anchors are separate ablations.
Report native/preparation/pair/point/QA/GPU/wall costs,including failed rows.
Evaluate reserved validation only at predeclared final budgets;no early stopping
or selection based on its labels. Old324 CHECK stays NEVER_FIT/developmental.
Physical/contact/action-gradient gates must pass before imagined Actor use.

Cache32 integration and prior timing remain separately evidenced. This cohort
uses default LIVE W1;it makes no new GPU1 cache,training speed,main-policy or
video claim. Full controlled task,multitrajectory,seeds,ego assets and hardware
goals remain active. No driver/CUDA/package changes or external publication.
