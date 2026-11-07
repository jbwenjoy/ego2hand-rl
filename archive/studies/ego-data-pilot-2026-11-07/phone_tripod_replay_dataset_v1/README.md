# New tripod replay / pair22 saved dataset adapter

`make_protocol_v1.py --describe` declares the interface before data exist.
After a real collector terminal, Root supplies its report and terminal to the
protocol maker. `export_dataset_v1.py --protocol PATH` is JSON/hash-only;
`--run` requires explicit saved-only authorization. There is no native, FK,
Torch/NN, optimizer, old-data-fitting or physics route in this exporter.

The producer is `phone_tripod_task_acquisition_result:1`. Every source raw
trace and decision stays frozen. The adapter reads both exact pins, checks
duplicate fields, derives typed train/monitor NPZ files, and retains an all-row
archive, source identities, receipts, partial masks and observed-prefix labels.
Family0..11 are TRIPOD_TRAIN,12..15 TRIPOD_MONITOR; paired dt share family.
Same loaded initial state is a known-state lineage, not independent ego data.

The pinned causal exporter `decode_window` is reused unchanged for native
force6/frame sign and moving-COM torque. Both force and direct torque rotate
via frame.T; force on geom2 is positive and on geom1 negative. Lever torque is
added once. All signed phone contacts are retained in impulse sums, including
zero/subthreshold Fn. The two blocks then rotate into the frozen source BODY
axes. PositiveFn means/coverage and generated-contact coverage are separate.
Triple simultaneous Fn>.1N coverage comes from actual per-tick15/28/41 sums,
not the product or minimum of marginal coverage. The original two-pad Reader
summary remains separately preserved.

The pure structured NumPy encoder is reused by AST extraction of its function
only, avoiding import of its Torch decoder. The scalar world-left52 codec is
pinned. Observed nonzero momentum defects are saved; they are never forced
to zero. These future-H labels are targets, not current Actor/WM inputs.

Only complete finite no-warning intervals receive H impulse/meanFn/coverage
and178-head targets. A partial row keeps raw observed prefix and its separate
observed-prefix impulse; targetH/deltaH/nextmemoryH/nextobsH remain NaN or
knownFalse. Mechanical post-memory bookkeeping is saved under a separate key.
No raw row is silently removed. FP32 observations stay FP32; producer F64
observation aliases are explicitly cast values, not F64 observation arithmetic.

WM eligibility includes all physical completeH rows, including task-failure
endpoints. Actor starts require pre-task-boundary alive. The first recorded
control-boundary POST-H loss is absorbing; later starts are excluded. Native
per-tick threshold events are independent diagnostics, not a replacement MDP
terminal. Collection0.4s truncation may bootstrap and is not the2s reference
timeout. No native two-second holding success is inferred from this dataset.

Input aliases use the complete after-action300-state SHA, actual u raw bytes,
dt/n, model physics digest and integration signature. Cross-split conflicts
block dataset qualification and training without deleting/reassigning rows.
Alias-connected family groups are reported. The control segment is discovered
by exact before/after/u/memory byte agreement, never a guessed vector slice.

The positive exported report schema is
`phone_tripod_replay_training_dataset:1`, status
`COMPLETED_SAVED_TRIPOD_REPLAY22_LABEL_DATASET`, with `engineering_qualified`,
checks, `partition_arrays`, `family_split`, and a complete `field_map` for the
trainer. This is a self-checked export, not an independent saved audit. A later
review may use `phone_tripod_replay_dataset_saved_review:1` and bind its
`dataset_report` to this exact report. No future audit/hash is prefilled.

Historical fresh-final12, old CHECK/TEST and old TRAIN array values are not
opened by this adapter. A trainer may separately mix the explicitly permitted
old684 TRAIN rows for WM only; that is a new training recipe, not this export.
