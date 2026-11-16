# Explicit context-layout correction v2, before native replay

Preserve both v1 collector/reader registrations and sources. Neither v1 native
replica has run. Installed `_create_solver_context` allocates Newton grad at
nv_pad=32, Mgrad at width0 (CG-only), and search at physical nv=26. The v1
observer allocated all three at26 and would reject actual layouts. Its analytic
copy tests missed this factory rule. This is a definite implementation defect,
not a native solver fault, replay failure or evidence of physical accuracy.

V2 copies the entire32-wide raw gradient and native empty Mgrad. Empty buffers
retain their actual shape and skip alias/copy operations; no zero-valued Mgrad
is fabricated. Force arithmetic uses only the26 actual DOFs and verifies the
six padded gradient entries remain zero as initialized by the pinned factory.
Raw copies retain their full32-wide representation. No solver/state/gradient
values, integrator, tolerance, gates, backend or installed file is changed.

The shared-src explicit scope selects the V2 observer/epoch helper while
executing the immutable v1 worker orchestration with runpy as __main__, keeping
the existing owned-source callable ABI. Binding identity is checked/restored
on exit and recorded separately. The V2 reader similarly adapts only its
physical gradient view before the separately implemented arithmetic. Original
raw copies are untouched. An adapter attestation and byte-identical thin-entry
snapshot accompany each new registration/replica/review. Future complete V2
evidence must include and verify these attestations; the old root report alone
does not identify the adapter.

All ten conditions, schedules, two fresh A/B processes,120 queries/960 total
native cap, zero updates, terminal full54 prerequisites and original saved-
review requirements remain. CPU registration/layout tests are not CUDA graph
qualification. No v1 native work is retried or omitted because none was run.
Preserve failed/partial V2 results and use another explicit version if needed.
