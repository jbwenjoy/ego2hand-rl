# Matched saved-interval soft-geometry supervision V1

Keep original new-backend dataset/physics/roles/masks/sources. All3202 valid TRAIN
points and779 complete physically qualified TRAIN pairs supply supervision;
all1591 valid development rows/386 qualified pairs and all1440 declared groups
remain evaluation denominators. This validation is already inspected,NEVER_FIT,
not final test. No model-dependent row selection or use of the scalar stability
screen to fit.67 impulse failures and empty/source-clipped groups remain retained.

Three DIRECT arms,each2000 AdamW updates,point256/pair32,sampler7411/pair7751,
width512,lr.0003,decay.00001,clip1,original point/auxiliary(.1) and object12+
wrench6 finite-response losses(weight1) unchanged. These DIRECT controls isolate
the added path signal;not a new all-family comparison or validation-selected
architecture winner. All three use the same frozen original initial90-output
NN weights plus same zero-initialized scalar path head;no trained checkpoint
warmstart or optimizer migration. Existing physical/normalization buffers reused
exactly. All point/pair batches match. All arms calculate both added loss terms,
including zero weights;no added batch or data for path arms.

Path head shares trunk and predicts nonnegative native interval geometry LOSS,
the sum of log1p((maximum/physical-limit)^2) across4 path channels. It does not
predict distances,endpoint geometry,hard-failure probabilities or error bits.
In this observed dataset only object penetration is nonzero;other channels/hard
flags have no positive coverage. Do not claim risk qualification from low loss.
Use softplus(linear(hidden))*TRAIN_mean/log2;zero head starts approximately at
TRAIN mean. Point loss divides by max(TRAIN standard deviation,1e-6). Contrast
loss divides by max(RMS of eligible TRAIN pair FD,.001*point_scale). Per-PD FD
uses2epsilon on float64 labels;TRAIN unit fitting1 production+1 invalid-TRAIN
isolation call at registration,0 in training/evaluation. No new90-output fit.

RESPONSE_ONLY path point/response weights0/0;PATH_POINT1/0;
PATH_POINT_RESPONSE1/1. Unused baseline path head receives zero gradients and
owned Adam cursors,remaining exactly zero. Its learned-loss prediction is a
constant;it supplies no PD gradient. All data remain on CUDA. Hot updates have
no CPU loss reads/evaluation/geometry recalculation. Capture logs at100/2000,
checkpoint/fullmodel+Adam+RNG+samplers/next-update health at100/2000;live6000
optimizer steps+12 clone QA steps,0native/Actor. Report actual partial costs on
failure. Finite/frozen-buffer/byte-equivalent restore and next-step tests required.
Continue the already authorized finite run after healthy100 checkpoint;ETA uses
last50 CUDA events and excludes future QA/IO/evaluation. Devicewide GPU samples
are not isolated kernel utilization or an overall speedup benchmark.

At final2000 only,evaluate all3 arms,forward/path errors,model-own RAW AD/FD and
original native OBSERVABLE/FULL reward contrasts/screens. Scores add predicted
soft geometry penalty(-.25*loss) to maintained endpoint/mean-wrench residual/
support proxy. Hard geometry/error/phase failures remain unmodeled,not silently
estimated from this scalar. On these observed rows recorded hardflags0,so FULL
native comparisons are meaningful but do not establish full-domain reward/risk
coverage. Keep all invalid-center counts. Verify baseline point/response code
against maintained DIRECT math and prior owned results before new claims.

One-step EMA fixed RAW chain includes actual-PD base value,controller diagonal
and direct RAW action cost;no straight-through offsets. Model own FD denominator
8PD-epsilon,original .25/.9 two-amplitude/two-repeat native screen;candidate
relative error<1/sign>=.9 is descriptive,not Actor admission. Preserve full and
stable-screen denominators. No validation-selected checkpoint,loss weight or
TRAIN renormalization. Prior arrays can supply verified actual RAW actions from
frozen analysis because source/context/rows must match exactly.

Independent review must recompute saved score/gradient projections,all1440
group identities/FD/aggregation and unchanged model/Adam metadata/initialization.
No Actor,task,ego or speedup admission follows a green health/AD-own-FD check.
Original M0–M5,A1/A2,multistep native reward improvement,7.36s no-aid task,
multi-reference/seed total costs,ego assets/gravity/dynamics/RGB and hardware
contracts remain. No driver/CUDA/installed source edits or publication.
