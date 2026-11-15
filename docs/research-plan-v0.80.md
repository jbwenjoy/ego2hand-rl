# Native-only CCD numerical sensitivity after the stage-wrapper limit

Current progress: all20 fixed recorded inputs contain physical pair1/29 in
the native NXN candidate list. Native staged broad/narrowphase exactly matches
the previous contact-presence/distance results. The pen is CYLINDER(5), finger
mesh MESH(7), not a discrete box/mesh pair. Six analytic fixtures pass24/24.
The separate GJK/EPA wrapper still fails10/20 target contact comparisons even
after fresh Geom structures; before/after cache fields remain[-1,-1], so the
cache-contamination explanation is not supported. Mixed130/160 checks remain
failed, and separate wrapper values cannot certify exact native internals.

Use the unchanged native kinematics+collision pipeline, without the wrapper,
to test an explicitly declared2x2 COLLISION numerical-policy factorial:

1. CCD iteration cap50 versus100 and device ccd_tolerance1e-6 versus1e-7.
   Keep every other404 model array/195 scalar leaf, fixed geometry, material,
   mass/inertia, all four recorded inputs, Data layout and no-solve/no-integration
   pipeline fixed. Explicitly log device override, original and resulting values
   and exact changed-field lists. Installed backend/driver/CUDA stay untouched.
2. All4 cells x4 input labels x5 exact-reset repeats =80 native collision queries,
   ZERO physics integrations/force solves/neural steps. Four labels are two
   distinct qpos images, not independent trials. Baseline must exactly match
   earlier20 native results; preserve failures and all cells, no favorable choice.
3. Report pair presence, signed distances, total contacts/broadphase count and
   overflow. No force labels read from old EFC buffers. Negative CPU direct
   distance stays a limited geometric comparator, not task truth. Any recovered
   contact at these local poses is sensitivity evidence, not a complete repair,
   reliable gradient, stable task, calibrated real dynamics or ego success.
4. If a numerical cell appears promising, require broader native contact-domain
   and repeated20ms/controller verification at ORIGINAL task/replay/force gates
   before adopting it. Save fresh observed Data scalars/layout/import evidence.
   Old lost imports and reconstructed capacities remain explicitly limited.
   No gain/weight sweep, hidden geometry replacement or longer horizon follows
   merely from these four labels. Full Goal remains ACTIVE/incomplete.

The original learned SuperMimic/CHORD/save-resume, fair support-policy and
multi-trajectory cost comparisons, contact/gradient validation, cube/pen task
evaluation, true ego input/asset/dynamics and hardware contracts remain intact.
