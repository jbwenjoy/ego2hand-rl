# Research plan v1.11 — actual CPU/GPU profile;empty-query trial negative

Preserve the complete v1.10 goal and M0–M5 requirements. Previous turnfe4a621
executed the four-arm WM study and retained failed gradients. This turn profiles
real32-world PPO and tests one opt-in query optimization,not a fabricated win.

1. Archive failed warmup JSON attempt,monitored/cross-thread profile,isolated
   profile,matched BASE/EMPTY comparison,raw CPU+CUDA trace and independent
   reviews. Total450560 native integrations/864 diagnostic PPO updates;no main
   policy promotion.366 tests,collector contracts and review37/37 pass;initial
   GPU arithmetic18/19 is retained with corrected19/19 grouping. Original
   geometry/phase/no-aid/overflow/source/reset guards remain.
2. EMPTY_SIGNALS removes64 empty queries but median collect1.77684->1.78945s,
   ratio.99295 and empirical numerical/reset equivalence is incomplete. Keep
   UPSTREAM default;do not call the prototype an acceleration or relax gates.
   GPU trace records196447 collection kernels;five CCD groups360ms summed,
   each698 times matching512 native+64 control+2*61 reset forwards. Instrumented
   collect~5.97s includes profiling overhead,not clean performance or occupancy.
3. NEXT implement the measured fixed-reference/frame cache opportunities in
   shared src:the reward still recomputes world reference64 times despite an
   existing table;policy188/critic126 repeatedly form fixed wrist matrices.
   Preserve exact normalization order (policy normalizes before quat_to_mat;
   critic calls quat_to_mat directly),dynamic joints/object/velocities/actions,
   invalid quaternion NaNs and used-reference timing. Extend pointer/digest/
   immutable ownership guards for any new cached tensors. No duplicated physics.
4. Validate CPU and actual GPU all369 phases/all32 worlds,nonzero origins,
   live dynamic changes,cache corruption and public pending rejection. New
   same-source fixture can compare live/cached modes without historical state
   migration. Separate one-control mechanism parity from long-rollout chaotic
   variation;save all failures/clean timings/reset counts. Profile a matching
   main-actor weights-only W32 context if it passes full404 Model/scalar/contract
   checks;do not assume the W1 helper or20 physical fields prove W32 equivalence.
5. Then investigate repeated upstream/adapter reset forward/observation work.
   Coalescing needs explicit command/derived-state/observation-cache epoch and
   random-consumption proof before removing any refresh. Do not blindly delete
   698 collisions or change solver/CCD/geometry. Larger batches/world counts
   need matched total samples/updates/cost and derived Model/numerical checks.
   No driver/CUDA/installed-package edits;profiled waits are not additive costs.
6. Continue full20D local excitation and isolated TRAIN-only derivative loss
   normalization/weight/forward-anchor experiments. Keep the inspected324 CHECK
   NEVER_FIT and reserve new histories/whole-original-reference final holdouts.
   Four-arm models remain unqualified for Actor optimization;rank4/20 and~99%
   response-loss fraction are leads,not sole causal explanations. Preserve8
   TRAIN rejects,strict/epoch solver cases and observer variability.
7. Full cube/pen task/evaluation/resume,contact support efficacy,genuine A1/A2,
   multitrajectory/multiseed total-cost results,ego collision/scale/frame/inertia/
   friction admission and hardware subversion/joints/zero/gains/latency/observed
   state contracts remain part of the goal. Main64 policy/video is unchanged.

Goal ACTIVE/incomplete. Historical first-checkpoint pause was served/resumed;
no new pause request or external blocker introduced.
