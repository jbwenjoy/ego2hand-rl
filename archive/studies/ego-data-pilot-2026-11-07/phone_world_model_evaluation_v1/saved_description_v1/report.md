# Fixed final5000 phone WM evaluation

The once-only CPU evaluation completed with one checkpoint load, one constructor,
114 one-step predictions and120 recursive predictions (11 neural calls total).
No gradients, optimizer, native model or physics were run. All120 TEST rows remain,
including6 masked partial targets. Fixed5000 was chosen before TEST; no earlier
checkpoint was evaluated on TEST.

| One-step ALL114 | COM RMS mm | Full SO3 RMS degrees |
|---|---:|---:|
| WM final5000 | 2.252 | 1.440 |
| TRAIN mean | 9.246 | 3.642 |
| Constant velocity | 1.653 | 0.874 |
| Continuous ballistic COM | 0.766 | 0.874 |

The strong kinematic baselines prevent calling the learned model a one-step pose
winner. The learned coverage Brier is .03421 versus TRAIN mean .11820, while its
nonpad-force maximum RMSE is30.568N versus mean2.345N. Force channels are distinct
PRE summaries; neither Brier nor pose accuracy establishes true contact forces.

True coverage groups contain63 no-pad,16 one-pad and35 both-pad-observed intervals.
The learned COM RMS is2.717/1.827/1.306mm respectively; ballistic is.130/.666/1.296mm.
Both-pad coverage means each pad was seen during the interval, not simultaneous
support. No-pad coverage does not exclude weak, nonpad or table contacts.

Recursive learned RMS at H1/H2/H5/H9 is .665/1.448/6.919/18.554mm and
.954/1.909/5.495/13.660 degrees, all12 targets. H10 has only6 genuine targets:
29.550mm/12.880 degrees; the other6 partial targets are not fabricated. H10 must
not be interpreted as a monotonic curve on the same population. All predictions
were finite and nearSO3. Raw true-joint excursions for both predictions and
available native targets are listed separately in report.json, without clipping.

Saved validation total loss worsens1.135→1.915→2.459 at250/1000/5000, while its
coverage Brier worsens.00732→.01258→.02497. All5000 training log rows and all three
validation checkpoints are retained, with no TEST checkpoint selection.

This is local simulated heldout-family evaluation around two mappings of one ego
frame under assumed physical parameters. Hidden solver/history state is absent.
There is no independent ego dynamics, robust holding, true-Jacobian, Actor or
CHORD benefit qualification.
