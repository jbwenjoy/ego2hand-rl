# Saved contact regime: author description

This is an author summary of all 608 branch PRE samples, 2,831 original contact records, 152 restored intervals, eight snapshots and 64 epsilon pairs. It is not an independent audit. Every snapshot appears below; no flat-response, force-positive or old stable-group selection was applied.

| Path | k | PRE | Generated | HO / secondary+ | HH | object–nonhand / secondary+ | p/R-flat perturbations | unchanged u endpoints | flat p/R pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0 | 76 | 190 | 190 / 152 | 0 | 0 / 0 | 7/16 | 0/16 | 3/8 |
| A0 | 49 | 76 | 152 | 152 / 152 | 0 | 0 / 0 | 4/16 | 0/16 | 2/8 |
| A0 | 99 | 76 | 380 | 380 / 380 | 0 | 0 / 0 | 8/16 | 0/16 | 4/8 |
| A0 | 149 | 76 | 304 | 304 / 304 | 0 | 0 / 0 | 8/16 | 0/16 | 4/8 |
| A2 | 0 | 76 | 114 | 114 / 38 | 0 | 0 / 0 | 1/16 | 0/16 | 4/8 |
| A2 | 49 | 76 | 171 | 0 / 0 | 171 | 0 / 0 | 16/16 | 2/16 | 8/8 |
| A2 | 99 | 76 | 760 | 0 / 0 | 152 | 608 / 608 | 16/16 | 2/16 | 8/8 |
| A2 | 149 | 76 | 760 | 0 / 0 | 152 | 608 / 608 | 16/16 | 2/16 | 8/8 |

The table counts repeated generated contact records, not unique persistent contacts or independent observations. Each snapshot contains 19 separate four-substep intervals. “Secondary+” means the original decoded normal force exceeds 1e-6 N and the saved constraint-type bit, world, raw-slot and host-address guards pass. It does not replace the primary generated-geometry count.

A2 k49 has no generated hand–object or object–nonhand pair across its 76 PRE samples. Its 171 generated records are hand–hand. A2 k99 and k149 also have no generated hand–object records, but each has 608 object–nonhand records, all secondary-positive under the saved evidence rule, plus 152 hand–hand records. Thus the late p/R encoding-flat observations cannot all be described as contact-free object motion.

At k99 and k149, the original ordered native pairs are (0,2), `terrain` ↔ `object/A_bottom`, and (0,3), `terrain` ↔ `object/A_top`; their body names are `terrain` ↔ `object/cube`. Each pair contributes 304 records per snapshot. These are exact saved names, not a newly certified static-environment or support-contact model. The same snapshots contain hand pairs (53,66), `robot/right_finger4_link3_col` ↔ `robot/right_finger5_link3_col`, and (14,27), `robot/right_finger1_link3_col` ↔ `robot/right_finger2_link3_col`. At k49 the hand pairs are (55,70), `robot/right_finger4_link4_col` ↔ `robot/right_finger5_tip_sensor_col`, and (18,29), `robot/right_finger1_tip_sensor_col` ↔ `robot/right_finger2_link4_col`. Full native body names and all other snapshots’ pairs are retained in the JSON table.

All 16 perturbed endpoints at each of A2 k49, k99 and k149 have zero saved physical p/R centered response; all eight plus/minus p/R pairs are numerically flat. Only two perturbed endpoints per snapshot have unchanged full20 control. The other 42 of the 48 late endpoints therefore have a nonzero recorded control change despite flat object p/R encoding. This is a finite, FP32-encoded, one-interval observation for the tested coordinates and amplitudes. Hand-state responses, finite resolution, contact constraints and hidden state remain separate issues; this table does not identify a unique cause or establish a zero physical Jacobian.

Absence of generated candidates is not global clearance. Force6 values were decoded by the original producer; contact dimension is missing for a complete independent pyramid decode. Stored world force and COM moment were only copied by the producer and are not recomputed here. Host addresses and raw WP slots remain different address spaces. No model, state converter, FK, MJB, world wrench calculation, AD, new physics or training is executed by this description.
