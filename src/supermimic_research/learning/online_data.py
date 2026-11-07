"""Canonical online pool identities and sampling, without collection or learning.

Reports are consumed in their explicit chronological order. Physical aliases
across different reports remain separate weighted observations. Only identity
columns are re-indexed; masks, partial NaNs and per-mode rewards are retained.
Torch is not imported here. Explicit feature preparation may invoke the shared
pure-tensor encoder; mixed sampling uses the caller's Torch API/generator.
"""
from __future__ import annotations

import json
import numpy as np
from . import data as D

MODES = ("A1_tracking", "A2_engineered_material_proxy")
SCHEMA = "phone_tripod_online_common_replay_dataset:1"
ORIGIN_FIELDS = ("_source_pool_index", "_source_pool_sha256", "_source_row_id", "_source_episode_id")


def _masks(canonical):
    n = len(canonical["row_id"])
    for key in ["WM_eligible", "Actor_start_eligible", "complete_H"]:
        if canonical[key].shape != (n,) or canonical[key].dtype != np.bool_:
            raise ValueError("Typed canonical mask: " + key)
    wm, actor, complete = (canonical[k] for k in ["WM_eligible", "Actor_start_eligible", "complete_H"])
    if np.any(wm & ~complete) or np.any(actor & ~wm):
        raise ValueError("Partial/unknown rows cannot become fit or Actor starts")
    return wm, actor, complete


def actor_indices(canonical, horizon=2):
    """Imagined starts, not two-native-window ground-truth eligibility."""
    if type(horizon) is not int or not 1 <= horizon <= 100:
        raise ValueError("Positive integer imagination horizon up to100")
    wm, actor, _ = _masks(canonical)
    step = canonical["step_k"]
    if step.shape != wm.shape or step.dtype.kind not in "iu" or np.any((step < 0) | (step > 100)):
        raise ValueError("Exact integer reference step0..100")
    return np.flatnonzero(actor & wm & (step <= 100 - horizon))


def merge_pools(reports, registry):
    """Read only online TRAIN common pools and preserve all rows and rewards.

    Return canonical/rewards/mapping/selection. Reserved `_source_*` arrays
    provide provenance to the selected physical cache, never model features.
    A repeated report digest is an error, not silently deduplicated. Distinct
    reports may contain equal x/u/full states and are intentionally retained.
    """
    if not isinstance(reports, (list, tuple)) or not reports:
        raise ValueError("At least one chronological online pool report")
    digests = [(p["bytes"], p["sha256"]) for p in reports]
    if len(set(digests)) != len(digests):
        raise ValueError("Duplicate pool report content; no implicit reweighting")
    parts, mapping, episode_map = [], [], {}
    rewards = {mode: [] for mode in MODES}
    rows_total = 0
    for pool_index, report_pin in enumerate(reports):
        path = registry.resolve_pin(report_pin)
        report = json.loads(path.read_text(encoding="utf8"))
        if report.get("schema") != SCHEMA:
            raise ValueError("Only online common TRAIN pool schema is accepted")
        view = D.read_partition(path, role="train", for_training=True, registry=registry)
        c = {k: v.copy() for k, v in view.canonical.items()}
        if any(k in c for k in ORIGIN_FIELDS):
            raise ValueError("Raw report collides with reserved origin columns")
        n = len(c["row_id"])
        if not n:
            raise ValueError("Empty online pool; no refill or partial promotion")
        for key in ["row_id", "episode_id"]:
            if c[key].shape != (n,) or c[key].dtype.kind not in "iu":
                raise ValueError("Typed original identity: " + key)
        if len(np.unique(c["row_id"])) != n:
            raise ValueError("Original row IDs must be unique within each pool")
        wm, actor, _ = _masks(c)
        if set(report["mode_reward_keys"]) != set(MODES):
            raise ValueError("Both A1/A2 rewards required for the common rows")
        for mode in MODES:
            r = view.raw_arrays[report["mode_reward_keys"][mode]].copy()
            if r.shape != (n,) or r.dtype.kind != "f" or not np.isfinite(r[actor & wm]).all():
                raise ValueError("Known finite per-mode reward at every eligible start: " + mode)
            if rewards[mode] and r.dtype != rewards[mode][0].dtype:
                raise ValueError("Reward dtype differs between pools; no implicit cast")
            rewards[mode].append(r)
        original_rows, original_eps = c["row_id"].copy(), c["episode_id"].copy()
        global_eps = np.empty(n, dtype=np.int64)
        for i, (rid, eid) in enumerate(zip(original_rows, original_eps)):
            identity = (report_pin["sha256"], int(eid))
            if identity not in episode_map:
                episode_map[identity] = len(episode_map)
            global_eps[i] = episode_map[identity]
            mapping.append(dict(global_row_id=rows_total+i, global_episode_id=int(global_eps[i]), pool_index=pool_index, pool=dict(report_pin), original_row_id=int(rid), original_episode_id=int(eid)))
        c["row_id"] = np.arange(rows_total, rows_total+n, dtype=np.int64)
        c["episode_id"] = global_eps
        c.update(_source_pool_index=np.full(n, pool_index, dtype=np.int64), _source_pool_sha256=np.full(n, report_pin["sha256"], dtype="U64"), _source_row_id=original_rows, _source_episode_id=original_eps)
        if parts:
            if c.keys() != parts[0].keys():
                raise ValueError("Canonical field sets differ between pools")
            for key in c:
                if c[key].dtype != parts[0][key].dtype or c[key].shape[1:] != parts[0][key].shape[1:]:
                    raise ValueError("Canonical field dtype/shape differs; no implicit cast: " + key)
        parts.append(c); rows_total += n
    canonical = {k: np.concatenate([c[k] for c in parts]) for k in parts[0]}
    wm, _, complete = _masks(canonical)
    selection = dict(pool_reports=[dict(p) for p in reports], total_rows=rows_total, total_episodes=len(episode_map), WM_included_global_row_ids=canonical["row_id"][wm].tolist(), WM_excluded_global_row_ids=canonical["row_id"][~wm].tolist(), partial_global_row_ids=canonical["row_id"][~complete].tolist(), Actor_H2_global_row_ids=canonical["row_id"][actor_indices(canonical, 2)].tolist(), raw_rows_retained=True, physical_alias_policy="Retained across distinct reports; no deduplication or hidden weight change")
    return dict(canonical=canonical, rewards={m: np.concatenate(v) for m, v in rewards.items()}, mapping=mapping, selection=selection)


def physical_rows(canonical):
    """Explicit WM selection plus a receipt linking cache indices to sources."""
    wm, _, _ = _masks(canonical); ii = np.flatnonzero(wm)
    if not len(ii):
        raise ValueError("No complete known WM rows")
    fields = dict(x="x58", u="u20", y="y58", h="h", n="n", J="impulse22x6", Fn="meanFn22", cov="coverage22")
    rows = {target: canonical[source][ii] for target, source in fields.items()}
    source_ids = []
    for cache_index, i in enumerate(ii):
        row = dict(cache_index=cache_index, global_row_id=int(canonical["row_id"][i]), global_episode_id=int(canonical["episode_id"][i]))
        if all(k in canonical for k in ORIGIN_FIELDS):
            row.update(pool_index=int(canonical["_source_pool_index"][i]), pool_sha256=str(canonical["_source_pool_sha256"][i]), original_row_id=int(canonical["_source_row_id"][i]), original_episode_id=int(canonical["_source_episode_id"][i]))
        source_ids.append(row)
    return rows, dict(included_global_row_ids=canonical["row_id"][ii].tolist(), excluded_global_row_ids=canonical["row_id"][~wm].tolist(), source_ids=source_ids, selection="Explicit WM_eligible only; no finite-value filtering")


def prepare_pool(merged, geometry, physical, *, device="cpu", chunk_rows=4):
    """Prepare the WM cache only; caller owns the resident real Actor bank."""
    rows, receipt = physical_rows(merged["canonical"])
    cache, scope = D.prepare_physical_rows(rows, geometry, physical, device=device, chunk_rows=chunk_rows)
    return dict(**merged, WM_cache=cache, WM_selection=receipt, geometry_cache=scope, Actor_indices=actor_indices(merged["canonical"], 2))


def mixed_WM_batch(online, old, tripod, generator, torch):
    """Return (batch, provenance), with CPU RNG draws in fixed4/2/2 order."""
    sources = [("online", online, 4), ("old_original", old, 2), ("old_tripod", tripod, 2)]
    keys = online.keys()
    for _, rows, _ in sources:
        if rows.keys() != keys or "x" not in rows or len(rows["x"]) < 1:
            raise ValueError("Three nonempty physical caches with identical fields")
        for key in keys:
            if len(rows[key]) != len(rows["x"]) or rows[key].shape[1:] != online[key].shape[1:] or rows[key].dtype != online[key].dtype or rows[key].device != online[key].device:
                raise ValueError("Same cache field shape/dtype/device: " + key)
    selected, provenance, start = [], {}, 0
    for name, rows, count in sources:
        cpu_indices = torch.randint(len(rows["x"]), (count,), generator=generator, device="cpu")
        provenance[name] = dict(local_indices=cpu_indices.tolist(), output_range=[start, start+count], available_rows=len(rows["x"]))
        indices = cpu_indices.to(rows["x"].device)
        selected.append({k: rows[k][indices] for k in keys}); start += count
    return {k: torch.cat([v[k] for v in selected]) for k in keys}, provenance
