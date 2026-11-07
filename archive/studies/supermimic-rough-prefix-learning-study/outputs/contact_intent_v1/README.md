# Prefix16 declared contact-intent input

This is a runnable, frozen **robot kinematic intent hypothesis** for the reconstructed R prefix, not a human-contact annotation or a force estimate. It binds source frames `[200,216)` to `H123_cubic_minimax_restoration_filtered_prefix16_v1`, candidate SHA `6364a6e262f7fbedac049d2d9e789b2c9077cbd6ef7656ef2e9b394161bab81a`.

The producer reads only the already saved R `fit16` native geometry, the exact new16 q/object/wrist bridge, the H123 material-hypothesis binding, and known model/context assets. It does not query a new collision state, optimize, step physics, read H0/T0 policy contacts, or consume upstream robot labels. The legacy151 target archive supplies only the unchanged 512-direction basis; no legacy support or contact target is reused.

There are 96 saved hand-object contact slots across 16 frames. All 16 frames have available, bounded-overlap declared geometry and phase3. `contact_known=true` means that this declared robot-geometry hypothesis is usable, **not that human contact is observed**. Human contact and confidence remain unknown in all16 frames. No frame is `known_no_contact`; native absence and padded slots cannot establish release or positive clearance. Missing reference fingers do not forbid stabilizing contacts by other fingers.

Targets are object-centric union-support descriptors of binary geometric contacts: force-on-object inward normals, moments about the object COM, one radius division, unit diagnostic friction rays (`mu=.1`, 8 edges plus normal), fixed512 basis. They are not the native `.7` friction/torsion cone, contact-force amplitude, or physical wrench balance. Saved witness points can be penetrating contact midpoints. Their role is a frozen shaping target.

The unchanged vendored objective uses 64 fixed stride8 directions out of512 and 64 surface samples per enabled hand geometry (21 geoms, 1344 points, mount excluded). A CPU model instance copies fixed actual-context `geom_size`, `geom_pos`, and `geom_friction`; it is never uploaded to physics. The normalizer remains the frozen nominal-MJB enclosing radius `0.1290454076850486 m`, and COM is the same MJB body COM. Full512 remains the independent diagnostic basis.

```python
import json
from rough_prefix.contact_binding import (
    build_contact_objective, build_penalty_only_objective,
)

with open("D:/workspace/supermimic-rough-prefix-learning-study/outputs/contact_intent_v1/objective_spec.json", encoding="utf-8") as f:
    spec = json.load(f)
a2 = build_contact_objective(spec, device="cpu")
p_only = build_penalty_only_objective(spec, device="cpu")
bonus_unscaled, terms = a2(x1_state58, memory)
```

The caller must add `.02 * bonus_unscaled` **exactly once, in imagination only**. This preparation does not test a trainer hook, alter native collection/reward/WM loss, or authorize training.

The A2 formula remains `phase_weight * support_match - .1 * penetration_prior`. Phase weights are `[0,0,1,1,.5]`. Penetration is always active: `(max(relu(-sample_sdf-.003))/.001)^2`. P_ONLY directly returns `-.1 * penetration_prior` from the same frozen objective; it does not subtract support from a combined reward. All-unknown targets therefore reduce A2 to P_ONLY, including state gradients. Actualnew16 has nonzero declared support, so this particular A2 input is not degenerate to P_ONLY.

CPU evidence is [the28-check receipt](../contact_binding_cpu_v1/report.json). It covers real16 construction, finite nonzero state gradients, phase/frame masks, allunknown and deep-penetration negatives, P_ONLY value/gradient identity, CPU outerAMP, target/basis/COM reconstruction, wrong hash/hyper/source rejection, RNG preservation, and pureTorchFK versus saved native points (max difference `1.57635e-8 m`). New collision/query/force/physics calls and CUDA/Warp use were all zero.

The failed local contact-proxy/action-gradient qualification and compiled-precision qualification are retained. Correct local geometry gradients and interface checks do not establish useful dynamics gradients, native contact-force prediction, a grasp, or physical tracking success. This single redesigned fixed-palm prefix is not independent motion or real ego-data generalization. The input is ready for a separately reviewed pilot hook and engineering checks; GPU/training and learning benefit are untested here.

Reproduction uses the existing CPU interpreter with `CUDA_VISIBLE_DEVICES=-1`: run `scripts/build_prefix_contact_targets_cpu.py --output NEWDIR`, then `scripts/check_prefix_contact_binding_cpu.py --spec NEWDIR/objective_spec.json --output NEWCHECKDIR`. Both reject existing output directories. Source and target hashes are in `objective_spec.json`, `binding_manifest.json`, and the independent READY manifest.
