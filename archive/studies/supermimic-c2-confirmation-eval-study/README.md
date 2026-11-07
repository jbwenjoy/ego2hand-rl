# C2_020 prospective confirmation evaluation

The measured final cohort is V3: `configs/c2-confirmation-evaluation-v3.json`
and `scripts/run_c2_portfolio_queue_v3.py`. All27jobs completed on2026-11-03
at12:09:55Shanghai, with the1500s scheduling guard and45s child timers.
Read-only default preflight is available through that V3script. Its output is
`outputs/confirmation_evaluation_v3`; existing output directories cannot be
overwritten. The original interfaces below remain preserved as V1 history.

The separate qualification passed. The initial V2queue stopped on a metadata
rename PermissionError after the first full physical rollout; that attempt is
engineering INVALID and its remaining26jobs are NOT_RUN. V3 adds bounded
atomic-metadata I/O handling to the worker and its imported supervisor writer;
physics, Actor, filter, thresholds and the frozen27job schedule are unchanged.
No process or physics retry is introduced. All attempts remain separate.

`outputs/control_cost_audit_v3/report.json` verifies the full27actual-u/EMA/t0/
capacity/source contracts. It records289.156s V3whole-driver wall, with27.188s
qualification and15.266s failed-V2 engineering walls separately. These are
wall-time scopes, not measured GPU compute. Task/COM512 and training-seed
hierarchy are independently audited by the frozen confirmation analysis.

This separate cohort evaluates the nine new final2000 checkpoints trained with
C2_020 on seeds4,5,6. It follows the schedule frozen before that training: three
fresh processes per checkpoint, evaluationseed0,27jobs. Trainingseed N is3 per
method; process repeats are nested and are not additional trainingseeds.

`scripts/evaluate_pen_checkpoint_103.py` explicitly validates the original
format103, complete native config and controller/source/reference identities.
It preserves the original deterministic residual Actor and loads its state
strictly. It neither rewrites103 as3 nor enables the training adapter. The
single external filter therefore matches the trained20mrad controller without
double filtering. Outer hard/actuator limits and immutable reference signatures
are also checked against the actual evaluation environment.

The wrapper factory AST matches the already measured common20mrad portfolio.
Original t0 and EMA memory remain unchanged;150controlintervals use one filter
per process_actions and original4×5ms physics. The final post150 sticky overflow
read must be0, with one world and no resets. This narrow capacity condition is
not a dynamics or solver accuracy guarantee. Task gates and COM512 diagnostics
are unchanged and all task results are retained.

Default CPU preflight:

```powershell
& D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe scripts/run_c2_portfolio_queue.py
& D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe scripts/run_format103_qualification.py
```

GPU execution requires external authorization. The qualification command adds
`--run`, invokes only the fixed first schedule job under the shared GPU lock
and45s process-tree cap, and stores it in `outputs/format103_qualification_v1`.
It is kept separate from the27job confirmation and is never reused or selected
by task quality. The full queue uses independent45s child timers and a600s
scheduling/stop-request budget with45s cleanup reserve, no retries, and stops
only on engineering failure. These are practical timers, not OS hard realtime.

CPU evidence and exact source/plan hashes are in `outputs/preparation_v1` and
`outputs/checkpoint103_contract_cpu_v1`. Previously frozen training, native,
original evaluator and old5/20mrad workspaces are read-only inputs.
