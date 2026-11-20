# First shared vs separate physical PPO continuation on two pen clips

Both arms warm-started from residual checkpoint 2781 with frozen empirical normalization. All 32 alternating rollout batches per profile used matched sampling-noise arrays and PPO permutation seeds, 65,536 environment steps, 524,288 physical integrations, and 1,024 Adam updates. In the shared arm the Actor, Critic and optimizer all share identity. In the separate arm each clip owns a complete copy. Each first full checkpoint was read back and restored into two newly constructed W32 runtimes. These results concern a single short continuation and two development clips, not a full independent-seed A1/A2 comparison.

| Clip | Valid samples / 32,768 shared | Valid samples / 32,768 separate |
|---|---:|---:|
| 001 | 25,372 | 32,170 |
| 003 | 4,228 | 29,618 |

The last shared 003 batch had 2,047 geometric terminations out of 2,048. The stored native rollout independently confirms 2,047 source-phase-1 samples and 2,046 immediate terminations from phase 1. Native position and shaft termination and engine overflow were absent. The separate 003 arm had 1,806 valid samples in its last batch. Neither reached the second half in training. The checkpoint restoration passed despite subsequent learning deterioration.

| Frozen native policy | 001 valid seconds (2 repeats) | 003 valid seconds (2 repeats) |
|---|---:|---:|
| Parent | 7.36, 7.36 | 0.02, 0.02 |
| Shared | 0.04, 0.04 | 0, 0 |
| Separate 001 | 4.84, 4.86 | 1.14, 1.14 |
| Separate 003 | 1.44, 1.44 | 0.28, 0.28 |

All 16 strict evaluations failed. Geometry-stopped cases are censored at the first failed control. The two clips were both previously used in development. Longer run survival of the 001 specialist on 003 than of the 003 specialist is a result of these checkpoints, not evidence that training on 003 is generally harmful.

A separate first-action swap held the complete initial W32 state fixed for each test with four paired 32-world Gaussian noise batches per condition. On 003, the shared mean gave zero geometry-valid samples in 128 attempts regardless of whether the shared or parent standard deviations were used. The parent mean with shared standard deviations gave 74/128; the unmodified parent gave 93/128. The 003 specialist mean gave 128/128 with either its own or the parent standard deviations. This localizes the start failure mainly to the changed mean on that state, without identifying why the training updates drove it there.

Next isolate conflicting policy updates from critic/Adam sharing and measure cross-clip action-mean drift during training. The same-source safety and strict no-aid gates, original PPO parent and both independent baselines should remain available for direct comparison. [Training](../runs/pen_multiclip_ppo_v1/report.json) · [independent saved-state audit](../runs/pen_multiclip_ppo_review_v1/report.json) · [native evaluation](../runs/pen_multiclip_ppo_native_v1/report.json) · [distribution swap](../runs/pen_multiclip_distribution_v1/report.json).
