| q bias (rad) | Policy | Bounds | Progress actual / goal (deg) | Endpoint error (deg) | Position max (mm) | Shaft max (deg) | No drop | Task success |
|---|---|---|---|---|---|---|---|---|
| 0.00 | A0 | pass | 21.98 / 63.94 | 41.95 | 145.41 | 75.99 | pass | fail |
| 0.00 | A1 zero-shot | pass | 57.85 / 63.94 | 6.08 | 10.51 | 19.26 | pass | pass |
| 0.03 | A0 | pass | 31.69 / 63.94 | 32.25 | 133.16 | 79.53 | pass | fail |
| 0.03 | A1 zero-shot | fail | 63.25 / 63.94 | 0.69 | 11.49 | 20.12 | pass | fail |
| 0.06 | A0 | pass | 17.72 / 63.94 | 46.21 | 46.49 | 77.29 | pass | fail |
| 0.06 | A1 zero-shot | fail | 67.99 / 63.94 | 4.06 | 12.14 | 19.08 | pass | fail |

All actual initial q/qdot/object pose/velocities/wrist/initial target arrays are exactly equal. Frozen criteria change only in reference NPZ binding. Four new cases retain full 3s raw contact512; historical bias0 A0 retains aggregate contact counts only. GPU released; no training/adaptation.
