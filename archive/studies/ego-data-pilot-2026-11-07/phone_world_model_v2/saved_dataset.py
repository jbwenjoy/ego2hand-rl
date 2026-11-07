"""Read-only phone322 loader: preserve H2 branches, boundaries and receipt aliases.

No random split or independent-trajectory claim is available in this prefix.
No training/normalizer fit is performed. Metadata mode never imports Torch/NPZ.
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

DATASET_SCHEMA = "phone_saved_H20ms_transition_dataset:1"
REPORT_SCHEMA = "phone_saved_transition_dataset_build_report:1"
STATE_SCHEMA = "phone_COM_world_state58:1"
PHYSICAL_FIELDS = {
    "dq20_rad": [0, 20], "dqd20_rad_s": [20, 40], "dp_COM_world_m": [40, 43],
    "Log_Rnext_RcurT_world_rad": [43, 46], "dv_COM_world_m_s": [46, 49],
    "domega_world_rad_s": [49, 52],
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    p = Path(path).resolve(strict=True)
    b = p.read_bytes()
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def checked(record):
    p = Path(record["path"])
    require(p.is_absolute(), "Pinned path must be absolute")
    actual = pin(p)
    require(actual["bytes"] == record["bytes"] and actual["sha256"] == record["sha256"], "Saved input pin changed")
    return p.resolve(strict=True)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metadata_preflight(report_path, *, expected_report_sha256, overlap_audit_path, expected_overlap_sha256):
    """JSON/hash only; verify fixed saved report, schema and original alias receipt."""
    report_pin = pin(report_path)
    require(report_pin["sha256"] == expected_report_sha256, "Dataset report identity differs")
    report = read(report_path)
    require(report.get("schema") == REPORT_SCHEMA and report.get("status") ==
            "COMPLETE_ALL322_SAVED_TRANSITIONS_NOT_DYNAMICS_OR_MARKOV_QUALIFICATION", "Wrong saved report")
    outputs = report["outputs"]
    require(set(outputs) == {"transitions322_F64.npz", "sample_identities322.json", "source_boundary_cohorts7.json",
                             "branch_H2_context161.json", "schema.json"}, "Exact original output membership required")
    paths = {k: checked(v) for k, v in outputs.items()}
    schema = read(paths["schema.json"])
    require(schema.get("schema") == DATASET_SCHEMA and schema.get("state_schema") == STATE_SCHEMA
            and schema.get("physical_delta52_is_native_WM_chart52") is False
            and schema.get("physical_delta52_fields") == PHYSICAL_FIELDS,
            "Physical-world delta52 schema required; old native chart52 is incompatible")
    overlap_pin = pin(overlap_audit_path)
    require(overlap_pin["sha256"] == expected_overlap_sha256, "Alias receipt identity differs")
    overlap = read(overlap_audit_path)
    require(overlap.get("schema") == "phone_transition_cross_cohort_receipt_alias_audit:1"
            and overlap.get("all322_samples_retained") is True and overlap.get("split_assigned") is False
            and overlap.get("deduplication_applied") is False, "Original complete no-split alias receipt required")
    for key, expected in (("dataset_report", Path(report_path)),
                          ("sample_identities", paths["sample_identities322.json"]),
                          ("NPZ", paths["transitions322_F64.npz"])):
        require(checked(overlap["inputs"][key]).samefile(expected), "Alias receipt does not bind this dataset")
    return {"report": report, "schema": schema, "paths": paths, "overlap": overlap,
            "dataset_report_pin": report_pin, "overlap_receipt_pin": overlap_pin,
            "arrays_loaded": 0, "models_training_physics": 0}


def alias_components(identities, controls):
    """Return exact receipt/control alias rows and boundary-connected components.

    No approximate state matching: key = two integration-state receipts + actual
    u20 dtype/shape/bytes. H2 and all rows of a common boundary remain together.
    """
    boundaries = sorted({int(s["source_boundary_index"]) for s in identities})
    parent = {c: c for c in boundaries}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    grouped = defaultdict(list)
    for i, item in enumerate(identities):
        u = controls[i]
        key = (item["input_integration_sha256"], item["target_integration_sha256"],
               u.dtype.str, u.shape, u.tobytes())
        grouped[key].append(i)
    aliases = []
    for rows in grouped.values():
        cs = sorted({int(identities[i]["source_boundary_index"]) for i in rows})
        if len(cs) > 1:
            aliases.append(sorted(rows))
            for c in cs[1:]:
                a, b = find(cs[0]), find(c)
                parent[max(a, b)] = min(a, b)
    members = defaultdict(list)
    for c in boundaries:
        members[find(c)].append(c)
    components = sorted(members.values())
    lookup = {c: k for k, cs in enumerate(components) for c in cs}
    row_groups = np.array([lookup[int(s["source_boundary_index"])] for s in identities], dtype=np.int64)
    return sorted(aliases), components, row_groups


@dataclass(frozen=True)
class SavedPhoneTransitions:
    x58: np.ndarray
    u20: np.ndarray
    target58: np.ndarray
    physical_delta52: np.ndarray
    source_boundary_index: np.ndarray
    branch_id: np.ndarray
    H2_step: np.ndarray
    alias_component: np.ndarray
    cross_boundary_alias_rows: tuple
    boundary_components: tuple
    identities: tuple
    source_pins: dict

    def require_single_partition(self, labels):
        """Reject any train/test/validation split of this connected prefix.

        Arbitrary cohort assignments cannot bypass the alias component. Even an
        all-one label is only a fitting partition, not a generalization holdout.
        """
        require(len(labels) == len(self.x58), "One partition label per original row required")
        for group in np.unique(self.alias_component):
            selected = [labels[i] for i in np.flatnonzero(self.alias_component == group)]
            require(len(set(selected)) == 1, "Split leaks an alias-connected source prefix; no independent split available")
        require(len(set(labels)) == 1, "This fixed single-lineage dataset has no qualified multi-partition split")
        return {"partition": labels[0], "rows": len(labels), "independent_holdout_qualified": False}

    def branch_rows(self):
        """All original H2 pairs, without shuffling or dropping aliases."""
        return np.arange(len(self.x58), dtype=np.int64).reshape(-1, 2)


def load_saved_dataset(report_path, *, expected_report_sha256, overlap_audit_path, expected_overlap_sha256):
    m = metadata_preflight(report_path, expected_report_sha256=expected_report_sha256,
                           overlap_audit_path=overlap_audit_path, expected_overlap_sha256=expected_overlap_sha256)
    ids = read(m["paths"]["sample_identities322.json"])
    keys = ("input_state58", "control20", "next_state58", "physical_world_delta52",
            "source_boundary_index", "branch_id", "H2_step")
    with np.load(m["paths"]["transitions322_F64.npz"], allow_pickle=False) as archive:
        a = {k: archive[k].copy() for k in keys}
    for k, width in (("input_state58", 58), ("control20", 20), ("next_state58", 58), ("physical_world_delta52", 52)):
        require(a[k].shape == (322, width) and a[k].dtype == np.float64 and bool(np.isfinite(a[k]).all()), "Original F64 state/action/delta ABI required")
    require(len(ids) == 322, "All322 identities required")
    for k in ("source_boundary_index", "branch_id", "H2_step"):
        require(a[k].shape == (322,) and a[k].dtype == np.int64, "Original integer identity ABI required")
    for i, s in enumerate(ids):
        c, b, h = i // 46, i // 2 + 1, i % 2
        require(s["sample_id"] == i and s["source_boundary_index"] == a["source_boundary_index"][i] == c
                and s["source_boundary_cohort"] == f"SOURCE_BOUNDARY_{c:03d}"
                and s["branch_id"] == a["branch_id"][i] == b and s["H2_step"] == a["H2_step"][i] == h,
                "Original boundary/branch/H2 order changed")
        require(s["control_duration_s"] == .02 and s["control_held_substeps"] == 128
                and s["input_branch_relative_POST_tick"] == 128*h
                and s["target_branch_relative_POST_tick"] == 128*(h+1), "Integer H20ms timing changed")
        if h == 1:
            require(np.array_equal(a["input_state58"][i].view(np.uint8), a["next_state58"][i-1].view(np.uint8))
                    and ids[i-1]["target_integration_sha256"] == s["input_integration_sha256"], "H2 state/receipt link broken")
    aliases, components, groups = alias_components(ids, a["control20"])
    prior = m["overlap"]
    require(aliases == [v["sample_ids"] for v in prior["aliases"]]
            and components == [v["source_boundary_indices"] for v in prior["cohort_union_groups"]]
            and len(aliases) == 6 and components == [list(range(7))], "Original6 aliases/single connected prefix changed")
    for value in (*a.values(), groups):
        value.flags.writeable = False
    return SavedPhoneTransitions(a["input_state58"], a["control20"], a["next_state58"],
        a["physical_world_delta52"], a["source_boundary_index"], a["branch_id"], a["H2_step"], groups,
        tuple(tuple(v) for v in aliases), tuple(tuple(v) for v in components), tuple(ids),
        {"dataset_report": m["dataset_report_pin"], "overlap_receipt": m["overlap_receipt_pin"],
         **m["report"]["outputs"]})
