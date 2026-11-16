# New-backend WM components, fixed six-arm comparison V1

Freeze this design before new WM optimization. Registration requires completed
same-backend raw TRAIN/validation, independent preparation/combined physical
review and backend_pen_response_common_data:1. Verify their exact pins/receipts,
all masks and TRAIN-only reconstruction. No old dataset relabeling, checkpoint
initialization or fixed historical count. Register actual inputs and TRAIN units
after dataset products exist; current protocol/source can be prepared earlier.

Six fresh arms:POINT,RESPONSE,CONTACT,each DIRECT/IMPULSE. POINT uses maintained
state/head supervised loss. RESPONSE adds object12 and source-BODY wrench6
finite-contrast loss. CONTACT adds Fn6 loss to the same terms. Response weight1,
point weight1,auxiliary weight.1 are fixed;no validation-driven weight search.
All six use width512,same initial NN bytes,point batch256/seed7411,AdamW.0003/
weight-decay.00001/clip1,2000 updates each. Four response arms use identical
32-pair batches/seed7751 from all physically eligible20-direction TRAIN pairs.
No response stability or amplitude screen selects training pairs. Shared units
are eligible new TRAIN derivative RMS with original.001*point-scale floor.
This isolates response components,not a rank4/rank20 ablation or equal FLOPs.
Original multi-reference,independent-seed and full A1/A2 requirements remain.

Use maintained PenResponseWM and update_contrast/update_impulse math. Create
owned backend_pen_response_arm:1 checkpoints;new restoration validates exact
backend/dataset/config/units/pair identities before construction,all frozen
buffers before loading,and full Adam cursors/samplers/RNG. Never convert to an
old arm schema. Old loaders must reject these arms. Models have no Actor/native
runtime. All TRAIN arrays and batches stay on CUDA;no per-update geometry or host
health scalar checks. Checks and logging occur at declared boundaries.

Save/CPU-readback and exact clone-next-update checks at100 and2000. Both clones
take one QA update per arm,live arms remain unchanged. Live12000 Adam plus24
clone-only updates=12024 on a successful complete run;failed/partial work must
be retained separately. Record startup/reconstruction/source checks,CUDA event
intervals,checkpoint/QA/IO and device-wide utilization without claiming clean
performance or acceleration. Count each TRAIN normalizer/unit reconstruction
in registration and run;none uses validation.

TRAIN metrics at100/500/1000/2000;validation metrics only at final2000,never for
health,selection,normalization,early stopping or fitting. Validation remains
already-inspected development on the same controlled reference/fixed actor seed,
not an independent final test,ego sequence or policy seed generalization.
Keep original physical masks/empty groups and all model results. Later physical
response/secant/JVP comparisons must use same point units and the registered
amplitude/repeat screens;do not admit an Actor from finite NN/checkpoint health.

No automatic pause:the historical first-checkpoint pause was already served and
the user explicitly resumed. Continue this registered budget after healthy100.
No installed source,driver/CUDA,physics,gates or completed trial edits.

Final evaluation configuration/source is pinned in the model registration before
updates. Preserve every declared validation group,including empty groups with
null metrics;never invent zero response for unavailable data. Compute saved-state
PD-direction JVPs at observed baseline inputs only,all20 basis directions,all six
models. Report forward MSE,finite model contrasts and JVP-vs-native contrasts in
common new TRAIN point units. Existing two-amplitude/repeat physical screens keep
their frozen old descriptive units and cannot filter TRAIN pairs. A candidate
local AD direction requires original pair masks,valid saved baseline,physical
screen,relativeL2<1 and cosine>=.9,unchanged from the original model thresholds.
Report ALL_PHYSICAL_VALID and FROZEN_PHYSICAL_SCREEN denominators,missing bases,
zero-effect norms and all failures;undefined relative ratios remain null. This
does not establish a full Jacobian,successful reward gradient,Actor or task.
