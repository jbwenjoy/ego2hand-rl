"""Generate/validate an UNRUN proposal; no simulator, Torch or training entry."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("D:/workspace/supermimic-chord-study")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def generate():
    assert not (ROOT / "proposal.json").exists(), "Refuse to replace a prior proposal"
    source = BASE / "outputs/a2_paired_v2_clip007_2000iter_seed0"
    config = read(source / "input_config.json")
    wrapper = ROOT / "src/penalty_study/penalty_only_objective.py"
    bindings = {
        "original_trainer": BASE / "scripts/train_controlled_pen.py",
        "geometry_module": BASE / "src/chord_study/contact_geometry.py",
        "A2_objective_module": BASE / "src/chord_study/contact_objective.py",
        "paired_v2_protocol": BASE / "configs/paired-contact-pilot-v2.json",
        "native_source_inventory": source / "source/inventory.json",
        "native_source_archive": source / "source/dex_source.zip",
        "A2_pilot_input_config": source / "input_config.json",
        "new_penalty_wrapper": wrapper,
        "proposal_generator": Path(__file__),
    }
    pins = {name: {"path": str(path), "sha256": sha(path)} for name, path in bindings.items()}
    configs = []
    for seed in (1,2,3):
        reference_arm = BASE / f"outputs/training_configs/replication_seeds123_v1/a2_paired_v2_2000iter_seed{seed}.json"
        baseline = read(reference_arm)
        proposal = {
            "schema": "supermimic_chord_penalty_only_PROPOSAL:1",
            "status": "unrun_not_accepted_by_current_trainer",
            "method": "P_ONLY",
            "seed": seed,
            "native_training_and_reference_template": {"path": str(reference_arm), "sha256": sha(reference_arm)},
            "training": baseline["training"], "reference": baseline["reference"],
            "contact_objective_frozen_inputs": baseline["contact_objective"],
            "only_objective_difference_from_A2": {"support_coefficient": 0., "A2_support_coefficient": 1.,
                                                 "penetration_coefficient": .1, "caller_dt": .02},
            "implementation_candidate": pins["new_penalty_wrapper"],
            "proposed_output": str(ROOT / f"outputs/penalty_only_paired_v2_2000iter_seed{seed}"),
            "finite_engineering_guard_seconds": 3600,
            "primary_budget": {"iterations":2000,"num_envs":128,"wm_updates":4000,"auxiliary_failure_updates":4000,
                               "actor_updates":1800,"critic_updates":7200,"real_transitions":648192},
            "secondary_OS_wall_seconds":1200,
            "comparators": {method: str(BASE / f"outputs/{method.lower()}_paired_v2_clip007_2000iter_seed{seed}") for method in ("A1L","A2")},
            "native_self_collision_supervision_weight": .1,
            "no_changes_to_real_collection_WM_native_reward_or_controller": True,
        }
        path = ROOT / f"configs/penalty_only_seed{seed}.proposal.json"
        write(path, proposal)
        configs.append({"seed":seed,"path":str(path),"sha256":sha(path)})
    frozen = ROOT / "source_baselines"
    frozen.mkdir()
    for name in ("original_trainer", "geometry_module", "A2_objective_module", "paired_v2_protocol"):
        shutil.copy2(pins[name]["path"], frozen / Path(pins[name]["path"]).name)
    design = {
        "schema":"supermimic_chord_penalty_followup_design:1","status":"design_only_no_training_entry_no_GPU_runs",
        "created_utc":datetime.now(timezone.utc).isoformat(),"source_bindings":pins,"proposal_configs":configs,
        "scientific_status":"Exploratory follow-up: rationale discussed after seed0; concrete proposal frozen after existing A1L/A2 n3 outcomes were known and before any P_ONLY run. Not part of the original preregistration.",
        "A1L_A2_n3_outcomes_known_at_proposal_freeze":True,
        "arms":{"A1L":{"support":0.,"penetration":0.},"P_ONLY":{"support":0.,"penetration":.1},"A2":{"support":1.,"penetration":.1}},
        "shared_bonus_multiplier_dt":.02,
        "identified_contrasts":{"P_ONLY_minus_A1L":"penalty increment without support attraction",
                                "A2_minus_P_ONLY":"support increment conditional on the same penalty"},
        "not_identified":"Without support-only(1,0), no complete2x2 interaction or unconditional support main-effect claim.",
        "historical_control_limit":"Minimal3 new P_ONLY runs use already completed A1L/A2 controls. All new-arm runs occur later, so method/order/cache history is not retrospectively randomized. Disclose this batch limitation; a fully contemporaneous randomized three-arm replication would require a separate approved9-run budget, not silently reuse these3 runs as that experiment.",
        "sampling":"Same1344 fixed surface points/64 training basis from A2. The P-only reward ignores support/basis values; full A2 forward is retained for a minimal exact shared-penalty adapter, not a compute-equivalence claim.",
        "execution_plan":{"training_seeds":[1,2,3],"GPU":"single GPU, strictly serial","order":"P_ONLY seed1,seed2,seed3; comparators already completed and immutable","fresh_from_scratch":True,
                          "no_auto_retry_or_quality_stopping":True,"not_bitwise_same_onpolicy_data":True,"primary_equal_sample_update_budget":True,
                          "secondary":"Latest complete checkpoint within1200s of actual Python OS creation; exact source/phase/cache and CPUhealth preserved", "no_best_checkpoint_selection":True},
        "evaluation":"Use the exact original frozen3s reference, task gates, physical substeps, contact512 diagnostic, seed sets and no-reset/no-assist contract for all three arms. No new thresholds after viewing results.",
        "future_entry_work":["Create a separate new method P_ONLY entry; current trainer permits A1/A1L/A2 only, so proposal configs are intentionally non-executable.",
                             "Use hash-locked original native runner and geometry/objective; wrap the original objective only during policy imagination. Do not relabel P_ONLY as A2.",
                             "Record wrapper hash, full source snapshot and formula; preserve None/native path and original WM/collect rewards.",
                             "Extend a separate follow-up audit/queue schema; never weaken or edit existing frozen A1L/A2 audit results."],
        "future_validation_before_GPU":["CPU value+state-gradient exact equality to -.1*original_A2_terms[penetration_prior], including contact/unknown/no-contact phases, deep penetration, threshold/ties, and phase independence.",
                                        "Same FP32 geometry under outer AMP, factoryRNG preserved, frozen targets no gradients, dt applied exactly once externally, native collect/WM rewards unchanged.",
                                        "Only after approval: bounded128env/H8 actual10iter smoke, complete optimizer counts/finite tensors, fixed physics/ref equality, actual cost/VRAM; then3 formal2000 runs. No method hyperparameter tuning to obtain success."],
        "cost":"Report actual compute separately. P_ONLY may have cheaper backward than A2 although full A2 forward is retained; equal updates do not equate compute. Do not dummy-pad computation to manufacture equality.",
        "launch_authorized":False,"torch_imported_by_generator":"torch" in sys.modules,
    }
    write(ROOT / "proposal.json", design)


def check():
    proposal = read(ROOT / "proposal.json")
    for pin in proposal["source_bindings"].values():
        assert sha(pin["path"]) == pin["sha256"]
    for record in proposal["proposal_configs"]:
        assert sha(record["path"]) == record["sha256"]
        cfg = read(record["path"])
        template = cfg["native_training_and_reference_template"]
        assert sha(template["path"]) == template["sha256"]
        original = read(template["path"])
        assert cfg["training"] == original["training"] and cfg["reference"] == original["reference"]
        assert cfg["contact_objective_frozen_inputs"] == original["contact_objective"]
    assert not proposal["launch_authorized"] and "torch" not in sys.modules
    return {"status":"passed","scope":"Unrun design/config/source-hash metadata only; no Torch/simulator/training imports",
            "proposal_sha256":sha(ROOT / "proposal.json"),"wrapper_sha256":sha(ROOT / "src/penalty_study/penalty_only_objective.py"),
            "proposal_count":3,"native_training_reference_and_A2_inputs_equal":True}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check-only",action="store_true");args=parser.parse_args()
    if not args.check_only:generate()
    result=check();print(json.dumps(result,indent=2))
    if not args.check_only:write(ROOT / "design_metadata_checks.json",result)
