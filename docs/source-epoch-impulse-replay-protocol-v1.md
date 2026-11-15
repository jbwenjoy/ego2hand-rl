# Registered replay of the retained impulse counterexample

The original epoch benchmark/readback remains6157/6159: one linear impulse gate
fails in READ_LOCKED_EPOCH,RESIDUAL_P030,D2,eps.005,sign+,repeat0(case69).
Its270 cross-mode observed endpoint envelopes pass. Do not erase/relabel that
failed row or relax the5e-7Ns gate. Replay this exact retained source/action,
not a newly generated or favorable source.

Use30 repeats each of STRICT_A,STRICT_B and READ_LOCKED_EPOCH,rotating mode order
with repeat index. STRICT_A/B call the identical default strict implementation
as a negative control. Full source/task/byte/observation checks remain. Save all
90 controls x8 native steps=720 physical integrations,0 updates. One strict
warmup restore adds0 native steps. No new solver/asset/dt/tolerance changes.

Use the same recorder plus a read-only anomaly observer. If free-object linear
impulse computed from recorded qvel and generalized constraint force exceeds
5e-7Ns at a native tick,retain additional actual Data arrays including mass
matrix,smooth/constraint force and acceleration. The observation happens AFTER
that tick;extra reads can affect later scheduling and are not a proof of cause.
All physical gates still come from independent raw-EFC decoding. Preserve every
failure in every mode; do not claim failure rates from this selected case are
population estimates or a deterministic replay guarantee.
