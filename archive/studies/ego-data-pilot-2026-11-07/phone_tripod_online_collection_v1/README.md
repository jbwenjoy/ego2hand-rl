# Matched online collection, new stage

`collect_round(protocol_path)` runs actual CPU Actor inference and the already
qualified `PhoneReaderTaskBridge`. No WM/Critic forward, training, optimizer,
online normalization update, native observer forward or auxiliary object force
is introduced. Defaults inspect source/JSON pins only. `prepare_protocol_v1.py`
can form CPUFalse while final paired1000 checkpoint/health are pending; actual
execution requires the own AFTER1000/WM1100 artifact and its executed protocol.

Two arms use the same exact probe030 qpos/zeroqvel/u0, four deterministic named
profiles, two native timesteps and at most 20 complete 20ms intervals per
episode. Each arm has maximum 160H/15360 native steps (both arms 30720). Failed
or partial episodes stop without refilling the budget. Each control is the
bounded convex combination `.95 * Actor + .05 * fixed_profile` before the same
frozen F64 EMA/true-range task mapping. There is no clipping. This is local
known-initial-lineage simulation exploration, not independent ego validation.

Every reset and native interval is persisted and fsynced by the original raw
sink before replay/label use. Current Actor inputs are obs222 computed from
current state/history/reference/current geometry; contact impulses, force and
coverage over the future H are labels only. Source-before-action,
after-action and POST-target integration300 receipts remain pinned. Partial
prefixes remain raw; their H targets, next obs/memory and physical labels stay
unknown/NaN and unfit. Boundary task failure is absorbing; collection20H
truncation can bootstrap. Any native tick threshold diagnostic does not replace
the frozen boundary MDP rule. These0.4s acquisitions do not establish2s holding.

`dataset_report.json` schema `phone_tripod_online_common_replay_dataset:1`
provides the canonical field_map, common_pool pin, two arm pins, exact aliases
and identities. Rows also carry arm/profile/global-episode IDs, Actor FP32
action and mixed F64 action, H2-start masks, and both A1/A2 real reward values.
`load_common_pool(dataset_report_pin, reward_mode=...)` returns complete arrays,
eligible row indices and the mode-specific reward key to a future separately
authorized matched updater. It neither deduplicates aliases nor fits a model.
Online family IDs0..3 are scoped by the new protocol, not the old offline16
families. The frozen label function's internal family guard is delegated only
for mathematical reuse; returned provenance explicitly says ONLINE_TRAIN.

Read-only Actor state, raw mixed-dtype norm buffers, host normalizer constants
and RNG are checked before/after; actual first actions/controls and source
aliases are reported rather than presumed distinct. Source weights/norms are
loaded once from the own paired checkpoint; two existing factory calls make
Actor2 plus unused Critic2. No NN or native calls were performed by delivery.
