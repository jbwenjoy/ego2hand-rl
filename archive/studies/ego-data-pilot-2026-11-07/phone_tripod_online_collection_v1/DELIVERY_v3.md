Current multi-round source: `collect_online_v3.py`, with the pure JSON/payload
validator `actor_lineage_v3.py`. Frozen v1/v2 sources and outputs are retained.
The actual Actor/raw/native loop and canonical label/replay semantics remain
those delivered in v2. No collection or learning was run by this delivery.

The v3 protocol uses `collection_cycle` 1..3 and exactly:

```
actor_input = {kind, stage_id, checkpoint, health, training_protocol}
```

`stage_id` identifies the supplied **previous completed** checkpoint cycle,
not the upcoming acquisition cycle. Cycle1 uses `kind=offline1000,stage_id=0`,
strict AFTER1000/WM1100 and the actual final resume protocol. Cycles2/3 accept
only `phone_tripod_online_matched_checkpoint:1` with `stage_cycle=stage_id`,
`update_in_cycle=100`, and corresponding own positive saved health. A first
healthy10 checkpoint is a resume ancestor only, never a next-collection input.
Online-only global updates are100*(stage-1)+update; total-lineage paired updates
add1000 and shared-WM updates add1100. No optimizer is loaded by the collector.

Inputs retain the offline provenance and replace deprecated `paired_*` with
`offline_seed_protocol` plus `root_training_protocol`, `root_result`,
`root_health`, `root_terminal`. These four root pins bind the same actual final
experiment as the seed, including its actual resume protocol. Payload sources,
inputs and recipe equal the supplied executed training protocol before any NN
constructor. Every parent-health hop strictly decreases global updates,
preserves recipe/source/root/norm identity, and reaches the exact offline root.
Different successful resume protocol files can occur in that same frozen
experiment. Actor raw weights/norm/host and RNG checks before/after collection
remain in force. Root-normalizer equality is delegated to the pinned health and
source chain; no second root checkpoint is independently loaded here.

`prepare_protocol_v3.make_round_protocol(...)` returns a fresh round contract;
`write_round_protocol(p,path)` saves and metadata-validates it. The default
`protocol_CPUFalse_v3.json` has pending root/Actor inputs and cannot execute.
Root's separately reviewed loop can copy that template and supply the four
root pins plus the exact Actor endpoint. Each output must be a fresh immediate
child directory, e.g. `stage_cycle01_run1`; no pool is overwritten or recollected
on resume. Each pool's global episode IDs0..15 are unique within that pool;
when joining pools use `(poolSHA, original_episode_id)` to reindex and preserve
the original identities, never treat repeated numeric IDs as the same episode.

The seven `collector_sources` keys are `task`, `reader_replay`, `directional`,
`imagination`, `Actor_Critic`, `Actor_common`, `paired_trainer`. The last remains
the frozen offline trainer's CanonicalMemoryTask wrapper. New loop source is
its own `trainer`; its `collector` source separately binds v3.

The pure contract8/8 checks JSON routing, synthetic parent headers, three
meaningful rejects and preserved function ASTs. Its tagged `.bin` files are
not real checkpoints and were never deserialized. It provides no actual health,
Actor inference, native rollout or learning qualification. The new experiment
must stop after its first real healthy10/ETA as specified by Root; this source
delivery does not authorize starting it or advancing another stage.
