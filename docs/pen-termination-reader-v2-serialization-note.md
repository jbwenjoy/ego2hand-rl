# Explicit saved-reader V2 correction; native experiment unchanged

The original native comparator A completed120 queries/960 integrations with
7/7 checks. Its V1 saved reader completed all arithmetic but returned
3,373/3,493: exactly120 metadata_audit fields failed. Preserve that complete
failed report at runs/pen_termination_comparator_review_v1/A/report.json and
its original source/registration/log. No A native query was repeated.

The source snapshot contains support_weight_state_path as the typed tuple
(("attr","support_weight"),). The audit is JSON, which serializes tuples as
lists. V1 directly compared the loaded JSON list against the typed tuple.
All original/imported typed tree digests, exact snapshot metadata, whole
numerical/Python inventory, actual action/PD, packet and force/impulse checks
passed. This is a reader representation error, not a relaxed physics gate.

The explicit V2 reader and pen_termination_audit_json_v2 helper compare only
the two audit metadata representations with the exact strict-JSON encoding of
expected typed metadata. They never normalize the saved native/Python/RNG
trees, typed snapshot metadata, source tree digests, imported digests, numerical
checks or physical thresholds. Two focused CPU regression tests pass, including
wrong-value/missing-field rejection and preservation of the original tuple.
The full601 CPU tests for the native implementation had already passed.

V2 has its own registration and output paths. The original immutable worker
configuration and producer retain the V1 reader-registration reference used at
launch. V2 explicitly validates that legacy reference and records its own
registration and version2. It consumes the same unmodified native reports.
Process B is launched only after V2 review A passes, then independently reviewed
with V2. Summary code explicitly requires both V2 reviews and retains the V1
failure and its cost. A new source file does not rewrite earlier source receipts.

    python tools/review_pen_termination_comparator_v2.py --config configs/pen_termination_comparator_v1.toml --register
    python tools/review_pen_termination_comparator_v2.py --config configs/pen_termination_comparator_v1.toml --replica A

Then the unchanged native worker with --replica B and the V2 reader with
--replica B. The native total cap remains1,920 and optimizer/normalizer budgets
remain zero. The V2 analytical registration checks are not saved-data success;
inspect each final review report before claiming it passes.
