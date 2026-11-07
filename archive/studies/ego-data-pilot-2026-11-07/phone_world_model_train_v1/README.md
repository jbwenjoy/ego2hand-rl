# Fresh phone world-model training source

**SOURCE DRAFT — no network construction, training or data-array execution has
been authorized for this source yet.** The runtime entry point is
`train_phone_wm_cpu.py --protocol ABSOLUTE.json [--run]`; without `--run` it checks
only JSON/hashes. With no protocol it prints the declaration. Root must first
bind the actual audited acquisition, source review and fresh output.

`train_phone_wm.py` exposes `declaration()`, `metadata_preflight(path)`,
`load_partition(pin, allowed_families)`, `fit_normalizers(train)`,
`make_model(normalizers, torch)`, and `run_training(protocol_path)`.

Input is **80 raw physical features**: COM/world state58, actual absolute u20 in
rad, native integration h in seconds, and requested substep count64/128. H is
always20ms. The model uses a fresh128×128 SiLU MLP. Its59 outputs comprise a
normalized physical world-left delta52, two coverage logits and five nonnegative
log1p-force predictions (softplus), ordered min-pad2/last-PRE-pad2/nonpad-max1.
State decoding calls the frozen phone `A.decode_torch`, never the old pen chart
or the adapter's raw78 callback. No old checkpoint, normalizer, xi46, dynamics278,
self15 target, pen reward or weight is loaded.

TRAIN-only F64 population mean/std determine immutable F32 input/delta/logforce
buffers. Declared std floors are numerical conditioning choices, not dynamics
accuracy gates. Last-layer weights start at zero; biases give TRAIN delta mean,
coverage mean and logforce mean. Coverage initialization alone clips probability
to[.001,.999] to keep logits finite; input commands and labels are never clipped.

The fixed objective is the sum of three terms with weights1:

1. mean squared normalized physical delta52 error;
2. mean BCE against soft per-pad coverage fractions;
3. mean squared log1p-force error divided by TRAIN logforce std.

Coverage is native solver-PRE positive normal aggregate `Fn > .1N`, over the
interval. Min forces use all PRE substeps. Endpoint forces are last solver PRE
at H−h, whereas the next state is POST H. The nonpad maximum excludes designated
pads and table by the acquisition contract; table and signed raw force evidence
remain in the acquisition, not invented targets. Fresh h/n inputs make the two
discretizations explicit. The loss adds no inherited native pen loss and no
assumed zero-defect physics law.

The acquisition's frozen36/6/6 family split includes both dt values and every
interval in a family. Raw split NPZs retain partial rows/NaN targets and explicit
eligibility masks. The loader uses only the dataset audit's eligibility flag,
requires complete H and finite used values, and saves every used/excluded raw ID
with a reference to pinned detailed reasons. It opens TRAIN+VALIDATION only;
TEST is hash-bound and remains unopened. Aliases crossing frozen splits block
metadata admission. Validation does not fit normalization or select a checkpoint;
the fixed final update is5000.

Recipe: CPU4/FP32, seed1729, full eligible TRAIN batch, AdamW lr.001/wd0,
betas(.9,.999)/eps1e−8/foreachFalse, gradient norm clip1, exactly5000 updates;
checkpoints250/1000/5000. The source records raw named postclip gradients tagged
PRE optimizer I, weights/Adam tagged AFTER I, actual optimizer steps, dataset and
protocol pins, immutable byte-length/SHA log prefixes, and Python/NumPy/Torch raw
RNG states. It writes atomic state-dict checkpoints, not pickled model objects.
At each checkpoint it performs validation and one own CPU readback, checks all
state/Adam/RNG values and exact same validation predictions, then writes health
and elapsed/ETA receipts. Planned budget is5000 train FW/backward/steps plus3
validation and3 readback FW; no TEST FW, old weight load, native model or physics.

At healthy250+ETA Root must complete the first-checkpoint handoff and pause the
Goal. The explicitly authorized process may continue its fixed5000-update budget
in the background; this source neither changes Goal status nor waits for user
approval. A failed check preserves the checkpoint/failure and stops, with no retry.

These are local simulated family rollouts around two mappings of one source
frame, with assumed physical parameters and unobserved solver history. No Markov,
independent ego, real-force accuracy, robust holding, Actor or CHORD benefit claim
follows from fitting or from a healthy checkpoint.
