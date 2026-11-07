The current executable source is `collect_online_v2.py`; v1 and its metadata
protocol remain unchanged. `prepare_protocol_v2.py` forms CPUFalse. The pure
contract checks AST/metadata and analytic profile bounds only, without NN or
native execution. It does not qualify a future Actor rollout.

After the actual paired1000 process completes, run the JSON-only binder:

```
python bind_final_result_v1.py --final-result FINAL_RESULT --health FINAL_HEALTH --terminal FINAL_TERMINAL
```

The binder writes a **fresh CPUFalse** contract, including the actual executed
resume protocol, rather than retaining the initial training protocol. If the
frozen result schema has no `protocol`, it explicitly derives the pin from the
same successful terminal. It never deserializes checkpoint weights. Root can
review and create a separately pinned CPUTrue protocol for `collect_online_v2.py
--protocol ... --run` after the independent final evaluation.

Callable APIs:

- `collect_round(protocol_path)`: CPU-only read-only matched Actor collection;
  durable raw, canonical labels and a new common pool; no learning.
- `load_common_pool(dataset_report_pin, reward_mode=...)`: NumPy raw and
  canonical views, typed masks and per-mode real reward key.
- `make_common_replay(..., ReplayBuffer, device, torch)`: actually populates the
  supplied upstream replay class from that pinned pool. Native targets stay
  raw before the buffer's `nan_to_num`; eligibility masks are retained. Global
  episode IDs prevent arm/profile boundary crossings. This still performs no
  fitting, NN forward or optimizer call.

Weights, Actor norms, old dataset splits, old offline1000 counts and physical
parameters remain unchanged. A later learning loop must use its own authorized
recipe, matched update budget and health/stop checks. Source delivery does not
constitute a completed online learning loop.
