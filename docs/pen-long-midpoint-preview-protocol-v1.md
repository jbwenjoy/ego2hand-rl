# Fixed midpoint video diagnostic

Register RESIDUAL iteration1000 before its checkpoint exists. Run exactly one
fresh LIVE W1 mean-policy episode with the same weights-only/full-plant/strict
gates and recorder used by final long evaluation. Stop at the first terminal
or at full7.36s. Keep its actual outcome even if it looks worse. This is a
single intermediate diagnostic; do not select checkpoints, stop training,
change budget/gates or use its performance to alter this trial. SUPPORT has
not completed its budget and is not compared. The final registered2000-round
nine-case evaluation and repeat0 comparison video are still required.

Render two camera angles of this one saved trajectory at20x slow playback.
Both panels must say they are camera views of the same RESIDUAL1000 rollout.
Save every2.5ms pose without interpolation, terminal stills explicitly labeled.
Measure independent actual native integrations and zero optimizer work. The
video remains non-ego, controlled reference, perfect-pose simulation. Preserve
all source/input pins and a failure record on exceptions; never overwrite or
replace the old64 video or final long-evaluation artifacts.
