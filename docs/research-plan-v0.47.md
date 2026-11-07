# Research plan v0.47: support-conditioned source correspondence

2026-11-08. Previous turn was progress: matched material-mode results and source
patch diagnostics were committed as92a0f18. This turn creates and checks actual
material-point/normal and whole-hand collision-aware proposals. One new static
pose and three small engagement variants qualify geometry/source-candidate
correspondence, but no6D native frozen-cone support. No new neural training,
checkpoint or dynamic integration. The Goal remains active; the user's first
healthy new neural checkpoint pause rule has not been triggered.

## Evidence that changes the next action

-36 independent finger material-anchor solves:34 pass exploratory patch gates,
  but all6 composed whole-hand poses collide. Whole-hand correction clears one
  of two source hypotheses. The nonnegative-face-offset source subset retains
  a failed result, with thumb distance4.816mm and one component intrusion209um.
- The full30-candidate pose clears all compiled3025 pairs and places three
  material anchors within.371–.644mm with normal dots.948–.987. Actual native
  forces are initially zero; material-point correspondence is not contact.
- Three small engagement versions all clear full geometry and match actual
  native contacts to the uncertain source regions. Fn>.1N for each group, but
  only3D force balance is feasible. Frozen native6D cone support still fails.
  Three checked numerical separating witnesses independently support this
  local result; not global or dynamic infeasibility or measured human force.
- Saved readback includes36 material metrics,11 fresh full geometry reruns and
  44 direct EFC columns; full library104tests pass. Two failed reviewer attempts
  are retained. Correct actual pair count is2816+189+20=3025, not the3026/21
  wording in some prior reports; actual saved pair lists/filters were unchanged.

## Next executable work

1. Condition new local proposals on6D support, not just source position/normal.
   Use checked separating directions to rank legitimate opposing contact
   candidates and their moment arms. A candidate ray that breaks one separator
   is only a proposal hint, not sufficient support or native attainability.
2. Before changing the task, compare the same uncertain source candidate
   envelope under current robot gravity versus conditionally mapped source
   gravity. Record these as counterfactual diagnostics with explicit frames;
   do not alter simulation gravity or claim source world calibration. The
   fixed-wrist palm mapping can change the physical task. If preserving source
   intent conflicts with support, expose that conflict rather than relabeling
   an easier pose as source-correct retargeting.
3. Explore alternate material patches/normal directions and bounded hand/object
   placements with explicit source deviations, fixed robot root, original point
   contact prior and full geometry. Do not harden uncertain30/21-point masks into
   force truth, silently enlarge them, loosen gates, inflate friction/spin or
   keep adding preload. The present anchors/searches do not exhaust morphology.
4. Require actual native three-group support and legal actuator commands, then
   unchanged complete2s/two-dt no-aid gates. Static solver/geometry success and
   cone support do not certify compliance or full native dynamics. Separate any
   material/mass/contact-time sensitivity from algorithm comparisons.
5. Rebuild61-frame BODY/COM/controller observations and declared nonuniform-clock
   to50Hz resampling; retain independent RGB and sequence-gravity requirements.
   Collect fresh explicitly role-separated TRAIN interventions only for useful
   physical tasks, never fit old328 heldouts. Verify action/contact/physical
   gradients, object-frame support objectives and matched A1/A2 with total cost.
6. On the first healthy NEW neural checkpoint, verify model/optimizers/RNG/sampler
   and saved-state restoration, estimate ETA, then pause as requested. These
   bounded mechanical least-squares searches are not that neural training.

## Full objective is still required

Controlled fixed-wrist Hand2 cube/pen physical loading/playback, short learning,
save/resume and common no-object-aid evaluation; validated object-frame wrench
targets and same-budget A1/A2 benefit; trustworthy WM action/contact gradients;
geometry, measured/unmeasured dynamics, gravity, frame/data/controller contracts;
independent RGB ego reconstruction and full trajectories; multiple trajectories
and shared-versus-per-trajectory total costs; hardware version/joints/zeros/gains/
latency/observations before deployment. Static mug geometry, annotation FK,
changed physics or training-set results cannot replace these requirements.
Continue in the unified repository, preserve all failures/provenance, and do not
change drivers/CUDA, pause without the standing condition, or mark this Goal done.

Evidence: [patch and support results](mug-patch-proposals-2026-11-08.md),
`provenance/mug_patch_proposals/summary.json`, prior v0.46 and unchanged baselines.
