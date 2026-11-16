# Matched rank4/rank20 response supervision — fit protocol v1

Registered during data collection,before fitting or any model evaluation on
the new reserved validation labels. No fit begins until both collectors have
terminal passing reports,independent physical preparation/data review passes,
and TRAIN-only common products/leakage-negative checks pass. Do not reinterpret
study/data3 as model-family2 (which elsewhere means DIRECT_VS_DIRECT_CV).

Extend the maintained pen_contrast_fit/update/checkpoint machinery;do not clone
physics/networks/trainers into another study repository. Six fresh arms:
POINT_DIRECT/POINT_IMPULSE,RANK4_DIRECT/RANK4_IMPULSE,RANK20_DIRECT/RANK20_IMPULSE.
All use PenResponseWM,family schema1,DIRECT/IMPULSE heads,one common initial
network(init7411),same new full20 valid TRAIN point bank,point draws(seed7411),
TRAIN-only point/head scales,and exactly2000 live updates per arm.

Use width512,batch256,AdamW lr3e-4/decay1e-5,clip1,auxiliary.1. Response arms
also sample32 pairs/update with independent RNG seed7751;rank4/rank20 select
their explicit eligible pair-index tables from the SAME complete TRAIN pair
bank. Point streams/counts are identical across all arms. Both response ranks
perform64 extra endpoint forwards/update;point arms do not. Equal updates/
point draws are not equal FLOPs or wall costs. Record each cost separately.

Retain the original response objective:mean normalized finite-contrast squared
error over object12 plus BODY-moving-COM wrench6,using point y_scale and
weight.1. Fn6 remains point-supervised and separately evaluated,not an added
unreported response loss. Target derivative normalization/weight/forward-anchor
changes are NOT part of this rank-only experiment;register them separately.
This comparison isolates supervised pair-direction coverage within a full20
point bank,not sample-collection cost or pure rank effects versus the old bank.

Save initial weights and checkpoints100/2000. At100 require finite NN/Adam/
gradients,exact model/optimizer/point+pair sampler/RNG reload,and actual next
update clone parity. Continue the resumed user goal through2000 after a healthy
check;historical first-checkpoint pause was already served and explicitly
resumed. Charge clone-only QA updates separately (2 clones x6 arms x2 health
checkpoints=24 QA updates) from12000 live updates. No native or Actor steps.

Only the final fixed2000 budget evaluates reserved VALIDATE. No validation
labels enter fit/scales/model choice/early stopping;curve100/500/1000 uses
TRAIN diagnostics only. Report full and unchanged physically qualified masks,
forward object/warm/wrench/Fn6 normalized errors,per-source ranks and physical
two-amplitude/repeat consistency screens. Compare actual finite responses and
AD-JVPs in the validation direction basis at the exact source states,with the
existing <=.25 amplitude-relative/>=.9 cosine and observed-spread gates.
Candidate model gradient screen remains relative error<1 and cosine>=.9;
zero-response=1 comparator and per-pair/raw counts remain visible. Passing
some local screens is not a full20 smooth Jacobian or Actor readiness claim.

Report all arms,including deteriorated forward results. Same original reference,
fixed actor training seed and held noisy histories are DEVELOPMENT validation,
not independent trained seeds,whole-original-reference/final-test or ego
generalization. Old324 CHECK stays NEVER_FIT and is not a training input.
Physical Fn6/wrench/action-gradient reliability must qualify before imagined
Actor optimization. No main64 policy/video promotion or driver/CUDA changes.
