Penalty-only follow-up: DESIGN ONLY, NOT RUN

No training entry is connected. No GPU, Torch forward, optimizer, physics or
policy evaluation has been run in this workspace. The generator/validator uses
only the Python standard library. The wrapper is an implementation candidate.

Existing frozen A1L/A2 files in D:/workspace/supermimic-chord-study are read-only
inputs. They are not changed or relabeled by this proposal. The proposal JSON
schema is deliberately not accepted by the current train_controlled_pen.py.

Unscaled imagined bonuses (external control dt=.02 applied exactly once):
 A1L:   0
 P_ONLY:-.1 * P
 A2:    phase_weight * support_match - .1 * P
 P = square(max(relu(-sample_sdf-.003))/.001), exactly the frozen A2 term.

The new wrapper calls the unchanged A2 forward and directly selects its P
tensor. It does not subtract attraction from the combined A2 bonus and does
not duplicate the SDF/penetration mathematics. Only support coefficient1->0.
No coefficients, thresholds, points, reference, physics, controller or native
world-model/collection reward are tuned. Penalty remains active in every phase.

P_ONLY-A1L identifies the penalty increment. A2-P_ONLY identifies support's
increment conditional on that penalty. No support-only arm is included, so a
complete2x2 interaction or unconditional support benefit is not identified.

Proposed primary: fresh training seeds1,2,3, 2000 iterations each, same128env,
WM4000/auxfailure4000/actor1800/critic7200 and648192 real transitions per run.
Existing same-seed A1L/A2 are immutable comparators. P_ONLY runs, if approved,
would be serial on one GPU. All original physical task/contact gates remain.
The follow-up is exploratory. Its rationale followed the pilot; this concrete
proposal is frozen after existing A1L/A2 n3 outcomes were known, before P_ONLY
training. It is not retroactive preregistration.

The minimal three new runs use historical A1L/A2 controls. Method/time/cache
history is therefore not newly randomized across all three arms. Disclose this
batch limitation. A fully contemporaneous randomized three-arm experiment would
require a separate approved nine-run budget; that is not proposed for launch.

Before any launch: review proposal.json and wrapper, add a distinct P_ONLY
entry/audit/queue in this workspace, prove CPU value/gradient/RNG isolation,
then approve bounded real-CUDA smoke. No current file is a training command.

Reproduce metadata validation (does not run the wrapper):
 D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe scripts/make_penalty_only_proposals.py --check-only
