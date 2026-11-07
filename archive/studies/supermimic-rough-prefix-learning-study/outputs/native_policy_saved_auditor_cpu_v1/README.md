# Native policy saved-only audit

The frozen checker `scripts/audit_native_policy_saved_cpu.py` constructs no Env, native data/model instance, optimizer, physics step or geometry query. All reconstruction is from saved arrays/JSON, pinned model geometry ID metadata and the serialized Actor tensors on CPU.

Stable receipt schema is `rough_native_policy_saved_CPU_audit:1`. Top-level `engineering_qualified` and `checks` describe recorded input/coverage/control/source/capacity integrity. `native_policy_pass` separately preserves the original numerical quality gates; a quality failure does not make a complete, valid measurement an engineering failure. Missing runs are `NOT_RUN`; partial/nonfinite/overflow records remain unknown rather than quality FAIL.

The checker independently reconstructs 15 baseline+native residual EMA+hard/C2 targets, checks all four substeps per target, verifies prephase k→postphase k+1 and terminal observation15, and counts **15** target differences including startup ctrl→first learned target. It compares full initial q/qvel/object native twist/mocap/ctrl to qualified A0 and checks the declared initialization, including the existing 1e-6 body-angular API arithmetic tolerance.

Physical tracking remains the original16 canonical observations, with61-state secondary tracking diagnostics, all61 actual joint/drop checks, and60 solved penetration/load/contact measurements. Contact force signs, object-frame points/normals and direct torsion are reconstructed from raw contact rows; COM wrench uses the saved **preintegration solver body pose/COM**, while q/qd is the postintegration state. This is not an exact continuous dynamics closure or a force-prediction qualification.

Actual8 after-copy and after-set-constant readbacks, reference/model/source ZIP, checkpoint/Actor digest, supervisor identity, unchanged-normalizer assertions, and one sticky capacity read after step60/no-reset are cross-bound. Checkpoint Actor bytes are independently hashed. No post-evaluation Actor state_dict or155-dimensional observations were saved: the post-eval digest/all-tensor equality are source-enforced saved evidence, not an independent replay of Actor inference.

The23-check receipt here is explicitly synthetic and not a physical outcome. The first actual A1L audit separately passed38 engineering checks at `outputs/policy_saved_audit_a1l_v2/report.json`; its physical verdict stayed FAIL on the shaft criterion. No GPU evaluation was repeated for this checker.

```powershell
& D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe `
 D:/workspace/supermimic-rough-prefix-learning-study/scripts/audit_native_policy_saved_cpu.py `
 --config ACTUAL_CONFIG.json --run-dir ACTUAL_RUN `
 --a0-dir D:/workspace/supermimic-rough-prefix-learning-study/outputs/native_a0_v2 `
 --output NEW_REVIEW_DIR
```

Each engineering smoke/evaluation remains one seed0 checkpoint and one declared process. The audit does not establish method benefit, complete motion success, a qualified contact dynamics model or real ego-data generalization.
