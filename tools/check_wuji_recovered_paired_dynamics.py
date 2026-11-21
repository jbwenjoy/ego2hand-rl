"""Exercise the recovered-transfer scheduler, provenance and cost accounting on invented participants."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import run_wuji_recovered_paired_dynamics as paired
from heldout_wuji_cost import summarize
from run_wuji_heldout import artifact, digest, read, store


def record(subject, kind, step, method="shared_wm", amount=4.0, outcome=None):
    return dict(subject=subject, method=method, kind=kind, steps=step, card=method is not None,
                known_wall_s=amount, unknown_cost=False, successful_job=True, outcome=outcome)


def successful_job(root, key, payload):
    folder = root / "jobs" / key
    folder.mkdir(parents=True)
    store(folder / "result.json", payload)
    pin = artifact(folder / "result.json")
    store(folder / "attempt_001.json", dict(success=True, receipt=pin))
    return pin


class SyntheticJobs:
    def __init__(self, root):
        self.root = root
        self.repo = Path(paired.__file__).resolve().parents[1]
        self.ledger = []
        self.commands = []

    def entries(self):
        return self.ledger

    def local(self, script, *opts):
        return [script, *map(str, opts)]

    def one(self, key, subject, method, kind, card, build, result_file, *, steps):
        dest = self.root / key
        dest.mkdir()
        cmd = list(map(str, build(dest)))
        self.commands.append((subject, kind, steps, cmd))
        row = record(subject, kind, steps, method)
        assert card and steps > 0
        if kind == "adapt":
            checkpoint = dest / "checkpoints"
            checkpoint.mkdir()
            (checkpoint / f"policy_{steps}.safetensors").write_text(key)
            store(dest / result_file, dict(learner_steps=steps, role="per_reference_adaptation"))
            source = Path(cmd[cmd.index("--dynamics-transfer") + 1])
            store(dest / "warm_dynamics_binding.json", dict(manifest_sha256=digest(source)))
            row["checkpoint_health"] = dict(step=steps, resume_usable=True, actor_usable=True)
        elif kind == "evaluate":
            actor = Path(cmd[cmd.index("--checkpoint") + 1])
            contract = Path(cmd[cmd.index("--contract") + 1])
            store(dest / result_file, dict(checkpoint_sha256=digest(actor), contract_sha256=digest(contract),
                                           status="failed", force_validation=dict(validated=False)))
            row["outcome"] = dict(status="failed", full_repeated=False)
        else:
            raise AssertionError("Unexpected stress invocation on a deliberately unexposed fixture")
        self.ledger.append(row)
        return dest, row


def test(root, *, unscorable=False, corrupt=False):
    out = root / "new"
    original = SimpleNamespace(root=root / "primary")
    recovery = root / "recovery"
    for folder in (out, original.root, recovery):
        folder.mkdir(parents=True)
    actor = root / "recovered_actor"
    actor.write_text("Synthetic recovered policy")
    manifest = root / "paired_model"
    manifest.write_text("Synthetic same-step model")
    people = ["fixture_a", "fixture_b", "fixture_c"]
    steps = [104640, 209280, 313920, 418560]
    base = dict(subjects=[dict(subject=s) for s in people], worlds=192, steps=steps, budget=steps[-1],
                teachers={}, teacher_health={"shared_wm": dict(usable=False, status="invalid_nominal_actor")},
                protocol=dict(primary=people[0], supplementary=people[1:]), upfront_exclusive_card_s={"shared_wm": 100.0},
                upfront_exclusive_card_s_range={"shared_wm": [100., 103.]},
                canonical_init="fixture_initial", upstream="/fixture/upstream", physics={"overrides": ["physics.fixed=x"]})
    registered = dict(decisions={"shared_wm": dict(selected_pin=artifact(actor), selection_cpu_wall_s=2.,
                                                  status="selected_latest_finite_actor")})
    plan = paired.recovered_plan(base, registered)
    assert not base["teacher_health"]["shared_wm"]["usable"] and plan["teacher_health"]["shared_wm"]["usable"]
    assert plan["teachers"]["shared_wm"] == artifact(actor)
    common, prior, stages = [], [], {}
    for s in people:
        pq = root / (s + ".parquet")
        cf = root / (s + ".contract")
        pq.write_text(s + " source")
        cf.write_text(s + " reference")
        status = "unscorable" if unscorable and s == people[-1] else "scorable"
        stages[s] = dict(parquet=artifact(pq), contract=artifact(cf), source_status=status)
        common.append(record(s, "prepare", None, None, 3.))
        if status == "unscorable":
            continue
        certs = [successful_job(original.root, f"{s}_cert_{device}", dict(fixture=s, device=device))
                 for device in ("cpu", "cuda0")]
        score = dict(checkpoint_sha256="wrong" if corrupt else digest(actor), contract_sha256=digest(cf),
                     parquet_sha256=digest(pq), status="failed", force_validation=dict(certificates=certs))
        zero = dict(status="failed", full_repeated=False)
        for kind, suffix in (("evaluate", "eval_0"), ("disturbance", "disturb_0")):
            key = f"{s}_shared_wm_{suffix}"
            receipt = successful_job(recovery / "recovery_engine", key, score)
            if kind == "disturbance":
                store(Path(receipt["path"]).parent / "heldout_pairing.json",
                      dict(heldout_report_sha256=prior[-1]["receipt"]["sha256"], stress_report_sha256=receipt["sha256"]))
            prior.append(dict(record(s, kind, 0, outcome=zero), key=key, receipt=receipt))
        for step in steps:
            prior.append(record(s, "adapt", step, amount=5.))
            prior.append(record(s, "evaluate", step, outcome=zero))
    store(original.root / "cost_and_outcomes.json", dict(entries=common))
    base_control = dict(plan, protocol=dict(plan["protocol"], adaptation=[paired.CONTROL]),
                        upfront_exclusive_card_s={paired.CONTROL: 100.},
                        teacher_health={paired.CONTROL: plan["teacher_health"]["shared_wm"]})
    ctrl = summarize(base_control, common + [dict(e, method=paired.CONTROL) for e in prior],
                     {s: x["source_status"] for s, x in stages.items()})["methods"][paired.CONTROL]
    previous = dict(recovery_methods={paired.CONTROL: {"standalone": ctrl}}, unique_recovery_jobs=prior,
                    actual_costs=dict(original_plus_recovery_plus_distinct_teachers_s_range=[1000., 1003.], unknown_costs=False))
    extras = dict(source_teacher=artifact(actor), dynamics_manifest=artifact(manifest), recovered_report=previous, primary_entries=common,
                  export_cpu_s=6., final_warm_report={"actual_costs": dict(supplement_new_s_known=1., unknown_costs=False)})
    jobs = SyntheticJobs(out)
    store(out / "execution_inputs.json", dict(fixture=True))
    with patch.object(paired, "prepare", return_value=(extras, plan, original, jobs, stages)):
        paired.execute(SimpleNamespace(recovery=recovery, mode="run"), verified=None)
    report = read(out / "supplement_outcomes.json")
    n = 2 if unscorable else 3
    assert len(jobs.commands) == 8 * n
    assert report["reused_zero_shot_reports"] == n and len(report["reused_zero_shot_entries"]) == n * 2
    assert report["standalone"]["assigned"] == 3 and report["standalone"]["unscorable"] == 3 - n
    assert report["standalone"]["target_branches_accounted"]
    assert report["standalone"]["teacher_inspection_cpu_s_once"] == 2.
    assert report["standalone"]["pretraining_exclusive_card_s_range"] == [100., 103.]
    assert sum(row["common_cost_components_s"]["paired_dynamics_cpu_share"]
               for row in report["standalone"]["subjects"].values()) == 6.
    cost = report["actual_costs"]
    inc = n * 8 * 4.
    assert cost["original_and_all_supplements_plus_distinct_teachers_s_range"] == [1001. + inc, 1004. + inc]
    assert cost["new_target_gpu_s_known"] == cost["new_target_job_s_known"] == inc
    assert all(cost[k] == 0 for k in ("new_zero_shot_jobs", "new_source_preparation_jobs", "new_teacher_search_jobs",
               "new_dynamics_extraction_jobs", "new_pretraining_jobs"))
    for s in people[:n]:
        rows = [(step, cmd) for subject, kind, step, cmd in jobs.commands if subject == s and kind == "adapt"]
        assert [v[0] for v in rows] == steps
        for i, (_, cmd) in enumerate(rows):
            assert cmd[cmd.index("--worlds-per-reference") + 1] == "192"
            assert cmd[cmd.index("--dynamics-transfer") + 1] == str(manifest)
            assert cmd[cmd.index("--wm-fit-horizon") + 1] == "3" and "physics.fixed=x" in cmd
            assert cmd[cmd.index("--curriculum-horizon-steps") + 1] == str(steps[-1])
            assert f"reset.seed={42 + people.index(s)}" in cmd
            if i == 0:
                assert "--resume-replay" not in cmd and "--resume-state" not in cmd
                assert cmd[cmd.index("--adapt-from-actor") + 1] == str(actor)
                assert cmd[cmd.index("--shared-initial-state") + 1] == "fixture_initial"
            else:
                assert "--adapt-from-actor" not in cmd and "--shared-initial-state" not in cmd
                for name in ("replay", "state"):
                    path = Path(cmd[cmd.index("--resume-" + name) + 1])
                    assert path == out / f"{s}_{paired.ARM}_adapt_{steps[i-1]}" / "checkpoints" / f"{name}_{steps[i-1]}.safetensors"
        assert report["paired_stage_status"][s][0]["same_evaluation_reused"]
    return dict(assigned=3, scorable=n, zero_shot_reports_reused=n, target_segments=4*n,
                target_scores=4*n, extra_jobs_at_zero=0, GPU_used=False)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        all_ok = test(path / "all")
        incomplete = test(path / "unscorable", unscorable=True)
        try:
            test(path / "switched_actor", corrupt=True)
        except ValueError as exc:
            assert "Registered identities differ" in str(exc)
        else:
            raise AssertionError("Mismatched recovered zero-shot identity accepted")
    print(json.dumps(dict(three_assignments=all_ok, one_unscorable=incomplete, mismatched_zero_shot_rejected=True)))
