# Bounded dynamic feature JVP check

After forward GPU qualification and the registered timing runs,validate the
feature implementation's derivatives. This does not tune a model or change the
existing performance/physical criteria. No core code edits after timed source.

Use actual CUDA32-world feature layout and current fixed-cache/reference data,
with an external shallow feature clone whose qpos/qvel/raw/EMA inputs are pure
Torch values. Never replace WarpBridge allocations. Vary phases1/48/120/368,
two deterministic tangent patterns per phase. Compare LIVE and CACHED policy161
plus critic188 output and JVP with respect to qpos,qvel,raw action and EMA only.
Mount/reference/phase are fixed inputs;mobile-wrist derivatives are not claimed.
Use rtol0,forward atol2e-7 and JVP atol2e-6,record exact equality and actual errors.
Keep source inputs and cache digests unchanged;zero native/optimizer work.

This qualifies numerical feature transformation derivatives,NOT world-model
action gradients,contact derivatives,imagined Actor optimization or task/ego
success. All previous failed WM/physical gates remain. Preserve any failed
attempt and do not replace physical validation with feature calculus.
