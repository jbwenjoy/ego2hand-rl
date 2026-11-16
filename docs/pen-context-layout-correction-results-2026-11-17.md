# Actual Newton context layout and prospective replay version2

The full54 native-screened experiment is unchanged. The following correction
concerns only the not-yet-executed diagnostic observer/reader. No diagnostic
native steps, optimizer updates or normalizer fits were performed.

The frozen V1 observer incorrectly allocated context grad/Mgrad/search all at
physical nv=26. The installed `_create_solver_context` instead uses model.nv_pad
for grad and allocates Mgrad only for CG. Reading all ten original source
snapshots establishes nv=26,nv_pad=28,solver=Newton2. On CPU, the actual pinned
factory with each saved numeric allocation envelope produces grad(1,28),
Mgrad(1,0),search(1,26). All52 preflight checks pass at
`runs/pen_context_layout_v2_preflight/report.json`, SHA256
`2ea848fc773c426024835fb92f05a3a2d6b828be60395d7863fd5f0c590fd729`.
These envelopes are not native Model/Data instances; the test invokes only
allocation code, never solve/forward/step or a CUDA graph.

The first V2 draft guessed32 for nv_pad using a hypothetical envelope. The
actual-source preflight rejected that guess before creating its result
directory. Preserve `runs/pen_context_layout_v2_preflight.log`, its earlier
proxy-only test log `runs/pen_context_layout_v2_tests591_v1.log`, and Git revision
618ab9c. Those tests were not evidence of the saved model's layout. The corrected
28-wide implementation was tested and committed before any V2 registration;
its full591 tests pass in11.014s in
`runs/pen_context_layout_v2_tests591_v2.log`. The first failed CPU preflight and
both test histories cost zero native integrations. No earlier success report
is overwritten or promoted to GPU qualification.

V2 retains full28-wide raw gradients and the native empty Mgrad arrays. It
skips copies/alias checks for empty buffers rather than fabricating an Mgrad
value. Force arithmetic consumes the26 physical entries, verifies both padded
entries remain zero, and leaves all raw copies unchanged. Actual CPU factory,
empty-buffer copying, padding rejection, force-prefix and scope-cleanup tests
pass5/5. A separate reader implements its force/impulse arithmetic independently
and reuses the frozen canonical raw/EFC packet decoder with that reuse disclosed.
Its analytic registration checks pass5/5; actual saved-data review is pending.

Existing V1 source/registrations stay immutable and unused for native replay.
V2 uses a documented shared-src binding scope around the unchanged V1 control
orchestration, preserving the __main__.main.<locals>.step source ABI. It records
entry/exit binding identity, restores bindings on failure, and saves an adapter
attestation plus byte-identical thin-entry snapshot. The V2 reader requires
successful worker attestations. The preflight and correct591-test pins are
prerequisites, not optional notes.

- Worker V2 registration SHA256:
  `12dae2a85e8c3f7f3510d6640d6a9b87544ad49e6dd9dfc2dabed4e93bcbf048`.
- Reader V2 registration SHA256:
  `6b9703148f2cb3b6fe0cff7b70ccfa9b0888c4c6418de404aed8cecde421dfe1`.
- Actual default-entry pending guard5/5, SHA256
  `57843fd7b7f8e57bba1ac074b1d79b445bdfe723f434fb50e0281fab3c437e71`:
  no simulator imports attempted, bindings restored, no A/B output created.

The120-query/960-native total budget and both fresh processes are unchanged.
Full54 producer and final V3 review remain mandatory terminal prerequisites.
Use only these explicit V2 entry points after closure:

    python tools/run_pen_solve_exit_replay_v2.py --config configs/pen_solve_exit_replay_v2.toml --replica A
    python tools/review_pen_solve_exit_saved_v2.py --config configs/pen_solve_exit_saved_review_v2.toml --replica A

Then B with the same commands and --replica B. First queries count as GPU
qualification inside the same budget. Preserve failed/partial actual work and
stop for an explicit new version if a contract fails. No installed file, solver
value, physics/gate/integrator/tolerance/backend or driver/CUDA was modified.
No failing-tick private data, solver root cause, model/Actor/ego admission or
whole-goal completion is established by this CPU correction.
