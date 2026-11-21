"""Count validated source clips and exclusive-card costs from a round-robin trial ledger."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_bytes())


def seconds(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid charged duration")
    return value


def fold(plan, rows, *, verify=True):
    references = [r["sequence"] for r in plan["references"]]
    if not references or len(set(references)) != len(references):
        raise ValueError("Planned source sequences must be distinct")
    source_hashes = {r["sequence"]:r.get("sha256") for r in plan["references"]}
    checkpoint_hashes = {}
    methods = {}
    segments, failed = set(), {}
    for row in rows:
        stream = row["stream"]
        key = (stream, row["round"])
        if key in segments:
            raise ValueError(f"Duplicate scored segment {key}")
        segments.add(key)
        method = "single" if stream.startswith("single") else stream
        if method not in ("single","shared_mf","shared_wm"):
            raise ValueError(stream)
        if method == "single" and stream not in {f"single{i}" for i in range(len(references))}:
            raise ValueError("Unknown single-reference stream")
        expected = [references[int(stream[len("single"):])]] if method == "single" else references
        if set(row["source_sequences"]) != set(expected):
            raise ValueError(f"Source bank mismatch in {key}")
        if sorted(c["sequence"] for c in row["checks"]) != sorted(expected):
            raise ValueError(f"Incomplete or duplicate score in {key}")
        if method not in methods:
            methods[method] = dict(train_s=0., eval_s=0., failed_s=0., coordinator_s=0., unknown_costs=[], warm_s=0.,
                                   sampled_steps=0, references={}, events=[], first_qualified_card_s=None,
                                   first_qualified_known_lower_bound_s=None)
        m = methods[method]

        def charge_failures(attempts):
            for record in attempts:
                path = record["attempt"]
                item = (method, record.get("elapsed_s"), record.get("returncode"))
                if path in failed:
                    if failed[path] != item:
                        raise ValueError(f"Inconsistent duplicated attempt {path}")
                    continue
                failed[path] = item
                if record.get("elapsed_s") is None:
                    m["unknown_costs"].append(path)
                else:
                    m["failed_s"] += seconds(record["elapsed_s"])
                    if record.get("finished") is False:
                        m["unknown_costs"].append(path)

        charge_failures(row.get("failed_train_attempts", []))
        m["train_s"] += seconds(row["train_all_in_s"])
        # Charge any unallocated segment-wide overhead up front. Its position within this
        # segment is unresolved; this gives a conservative upper time for its first pass.
        m["coordinator_s"] += seconds(row.get("coordinator_overhead_s_this_invocation",0.))
        m["warm_s"] += seconds(row["useful_warm_s"])
        m["sampled_steps"] += int(row["native_added_steps"])
        for check in row["checks"]:
            seq = check["sequence"]
            if type(check["passed"]) != bool or not check["gates"] or any(type(v)!=bool for v in check["gates"].values()):
                raise ValueError("Invalid evaluation flags")
            if check["passed"] != all(check["gates"].values()):
                raise ValueError(f"Overall success disagrees with gates: {key} {seq}")
            if verify:
                raw = read(check["report"])
                checkpoint = row["checkpoint"]
                if checkpoint not in checkpoint_hashes:
                    with Path(checkpoint).open("rb") as f:
                        checkpoint_hashes[checkpoint] = hashlib.file_digest(f,"sha256").hexdigest()
                if (raw["checkpoint_sha256"] != checkpoint_hashes[checkpoint]
                        or raw["parquet_sha256"] != source_hashes[seq]):
                    raise ValueError(f"Wrong checkpoint or reference in {key} {seq}")
                if (raw["protocol_sha256"] != plan["protocol_sha256"] or raw["sequence"] != seq
                        or raw["gates"] != check["gates"] or raw["passed_numeric_development_gates"] != check["passed"]):
                    raise ValueError(f"Evaluation receipt mismatch in {key} {seq}")
                if raw["physical_intervals"] != raw["requested_intervals"] and check["passed"]:
                    raise ValueError("An incomplete physical rollout cannot pass")
                raw_repeat = raw.get("physical_repeat",{"status":"not_recorded"}).get("status")
                if raw_repeat != check.get("physical_repeat",{"status":"not_recorded"}).get("status"):
                    raise ValueError(f"Physical repeat status disagrees with receipt in {key} {seq}")
            charge_failures(check.get("failed_eval_attempts", []))
            m["eval_s"] += seconds(check["eval_all_in_s"])
            clock = m["train_s"] + m["eval_s"] + m["failed_s"] + m["coordinator_s"]
            stats = m["references"].setdefault(seq, dict(checks=0, last_checked_s=None, first_passing_check=None,
                                                         first_physical_repeat_pass=None,repeat_statuses=[]))
            repeat_status = check.get("physical_repeat",{"status":"not_recorded"}).get("status","not_recorded")
            stats["repeat_statuses"].append(repeat_status)
            if repeat_status == "passed" and stats["first_physical_repeat_pass"] is None:
                stats["first_physical_repeat_pass"] = dict(previous_checked_s=stats["last_checked_s"],
                    verified_at_s=None if m["unknown_costs"] else clock, known_cost_at_check_s=clock,checkpoint=row["checkpoint"])
            if check["passed"] and stats["first_passing_check"] is None:
                stats["first_passing_check"] = dict(previous_checked_s=stats["last_checked_s"],
                                                     verified_at_s=None if m["unknown_costs"] else clock,
                                                     known_cost_at_check_s=clock, checkpoint=row["checkpoint"])
                if m["first_qualified_known_lower_bound_s"] is None:
                    m["first_qualified_known_lower_bound_s"] = clock
                    m["first_qualified_card_s"] = None if m["unknown_costs"] else clock
            stats["checks"] += 1
            stats["last_checked_s"] = None if m["unknown_costs"] else clock
            count = sum(v["first_passing_check"] is not None for v in m["references"].values())
            repeated = sum(v["first_physical_repeat_pass"] is not None for v in m["references"].values())
            m["events"].append(dict(card_s=clock, cost_is_lower_bound=bool(m["unknown_costs"]),
                                    sequence=seq, qualified_distinct=count, passed=check["passed"],
                                    physical_repeat_distinct=repeated,physical_repeat_status=repeat_status))
    all_methods = {"single","shared_mf","shared_wm"} == set(methods)
    common = min((m["events"][-1]["card_s"] for m in methods.values()), default=0.) if all_methods else None
    if any(m["unknown_costs"] for m in methods.values()):
        common = None
    for m in methods.values():
        count = m["events"][-1]["qualified_distinct"]
        clock = m["events"][-1]["card_s"]
        m["planned_clips"], m["evaluated_clips"], m["qualified_clips"] = len(references), len(m["references"]), count
        m["exclusive_card_s_known"] = clock
        m["gpu_h_per_qualified_clip"] = clock/3600/count if count and not m["unknown_costs"] else None
        m["physical_repeat_qualified_clips"] = m["events"][-1]["physical_repeat_distinct"]
        m["passed_at_common_budget"] = (max((e["qualified_distinct"] for e in m["events"] if e["card_s"] <= common),
                                            default=0) if common is not None else None)
    return dict(scored_segments=len(segments), methods=methods, common_scored_card_seconds=common,
                scope="Closed training and scoring segments plus their recorded failed attempts. "
                      "Distinct clips are counted once. Startup and validation are charged; "
                      "setup/intake and ongoing/unlogged attempts are outside this ledger. A null cost indicates no verified denominator or incomplete cost.")


def unscored_processes(plan, rows):
    """Expose late failures and completed subprocess costs even when no segment ledger was published."""
    bulk = Path(plan["bulk"])
    closed = set()
    for row in rows:
        closed.add(Path(row["measured"]["path"]).name)
        closed.update(Path(a["attempt"]).name for a in row.get("failed_train_attempts",[]))
        for c in row["checks"]:
            closed.add(Path(c["report"]).parent.name)
            closed.update(Path(a["attempt"]).name for a in c.get("failed_eval_attempts",[]))
    records, incomplete = {}, {}
    for f in sorted(bulk.glob("*.stdout.log")):
        name=f.name.removesuffix(".stdout.log")
        if name in closed:
            continue
        match=re.match(r"(?:eval_)?r\d+_(single\d+|shared_mf|shared_wm)(?:_|$)",name)
        if not match:
            continue
        stream=match.group(1)
        method="single" if stream.startswith("single") else stream
        receipt=bulk/(name+".process.json")
        if receipt.exists():
            p=read(receipt)
            if p.get("elapsed_s") is not None and "returncode" in p:
                records.setdefault(method,[]).append(dict(attempt=name,seconds=seconds(p["elapsed_s"]),
                                                         returncode=p["returncode"],receipt=str(receipt)))
                continue
        incomplete.setdefault(method,[]).append(name)
    return records, incomplete


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tranche",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    path=a.tranche/"ledger.json"
    ledger=path.read_bytes() if path.exists() else b"[]"
    plan,rows=read(a.tranche/"plan.json"),json.loads(ledger)
    result=fold(plan,rows)
    unscored, pending=unscored_processes(plan,rows)
    result["unscored_process_receipts"]=unscored
    result["unscored_attempts_without_terminal_receipts"]=pending
    result["shared_plan_and_snapshot_s"]=plan.get("plan_and_snapshot_wall_s")
    for name,m in result["methods"].items():
        total=m["exclusive_card_s_known"]+sum(c["seconds"] for c in unscored.get(name,[]))
        m["exclusive_card_s_including_unscored_known"]=total
        n=m["qualified_clips"]
        m["gpu_h_per_qualified_clip_including_unscored"]=(total/3600/n if n and not pending.get(name)
                                                        and not m["unknown_costs"] else None)
        repeated=m["physical_repeat_qualified_clips"]
        m["gpu_h_per_physical_repeat_clip_including_unscored"]=(total/3600/repeated if repeated
                                         and not pending.get(name) and not m["unknown_costs"] else None)
    result["ledger_sha256"]=hashlib.sha256(ledger).hexdigest()
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
