# Research plan v1.33 — TRAIN responses decoded; validation continues

Preserve all original M0–M5 and v1.32 requirements. Goal ACTIVE/incomplete.

1. Matched two-arm2000 training/final9 evaluations remain TERMINAL exit0.
   No task benefit shown: R3.48–3.50s,S2.04–2.08s,both0/3 strict7.36s.
   Original weights/gates/checkpoints and non-ego video remain immutable.
2. Original TRAIN response collector session2528 is TERMINAL exit0. All24
   declared sources available,20 source-qualified;4 R sources at64/72 from
   both seeds excluded by unclipped-mean gate. All3888 declared cases retained,
   3240 executed. Producer2272+branches25920=28192 actual/attempted native,0Adam.
3. Independent TRAIN preparation session12907 is TERMINAL exit0:69/69 overall
   checks,3202/3240 valid rows. Keep38 failed linear-impulse rows masked under
   original5e-7Ns tolerance;64762/64800 row flags pass. All960 groups retained,
   779 eligible rank20 pairs,158 rank4. Empty/invalid groups are not replaced.
4. Reserved validation collector session44504 started only after TRAIN terminal;
   live at latest actual handle poll. Do not restart on observer timeout. On
   terminal success run exact registered independent preparation,then combined
   registered data review. On failure retain original exception/raw context.
5. New backend_response_dataset numeric adapter and production preparation CLI
   have6 targeted isolation/rejection tests;495 full tests pass. They explicitly
   require new schemas/roles/backend,keep empty masks/identities and dynamic
   counts,and use valid new TRAIN only. Production registration correctly denies
   missing independent review before creating outputs;success path not exercised.
6. After actual combined review:register configs/pen_backend_common_data_v1.toml
   via tools/prepare_backend_pen_dataset.py --register,then execute without that
   flag;inspect products/negative-label isolation and exact eligible-group lists.
   No production normalization/model fitting yet. Synthetic test normalizations
   are not new labels. Old scales remain descriptive only in coverage reports.
7. Subsequent WM fitting needs an explicit new-backend model adapter and separate
   fixed-budget registration. Never relabel old datasets/checkpoints or reuse old
   fixed3866/1935/949 counts as new evidence. Keep development validation NEVER_FIT;
   no Actor until physical contact/response/gradient admission. Retain independent
   A1/A2,multi-reference/seed total costs,source gravity/noninertial mapping/assets/
   mass-inertia-friction/RGB calibration and hardware contract. No source/driver/
   CUDA edits or publication. Goal ACTIVE/incomplete.

See pen-backend-train-response-results-2026-11-16.md and
pen-backend-dataset-readiness-results-2026-11-16.md.
