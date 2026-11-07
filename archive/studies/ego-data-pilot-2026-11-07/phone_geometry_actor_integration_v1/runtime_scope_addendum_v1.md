# v2 runtime boundary

Use `phone_task_v2.py`, `reader_replay_adapter_v2.py` and
`phone_actor_imagination_v2.py` with `protocol_CPUFalse_v2.json`.
The v1 sources and metadata result are retained. v2 adds strict20ms/integer
memory clocks, verified actual imported Actor/common source before NN
construction, explicit singleton/batched inputs, and explicit FP32 CPU or
`cuda:index` factory support. No CUDA/factory/NN call was run by preparing it.

The factory forks CPU RNG plus only the selected CUDA device, initializes on
CPU in declared FP32, and explicitly moves to the requested device. The
parameter-free test does not instantiate this factory or prove CUDA runtime.

The online bridge uses the frozen Reader's acquisition safety bounds: drift
0.15m, angle90deg, penetration5mm. Its task stop is a POST20ms descriptor.
Neither is the original tickwise holding evaluator (50mm/45deg abort and all
two-second/tail/contact/penetration gates). Training collection through this
bridge must not be reported as an original2s evaluation. A later separately
frozen native policy evaluator must use the original per-native-tick gates,
all failures retained, both native dt values and new object-reference errors.

An8-edge design-cone unit-support fingerprint is not the native condim3
four-edge pyramidal solver. Nominalmu=.7 and the same-material/fixed-normal
proxy remain assumptions. Defined A2 objective/proxy masks do not make actual
contact, normal, force, physical action gradient or holding knowledge true.

This runtime contract uses F64 physical state/clock/absolute-control memory and
FP32 Actor output. `initial_memory(like=FP32)` is not this declared runtime and
would round the baseline. Flattened replay memory is F64; F32 action-history
values are preserved numerically, but subsequent arithmetic may differ from
online mixed-dtype history. Canonicalizing history arithmetic is a future
training integration requirement, not proven bit parity by this source test.
