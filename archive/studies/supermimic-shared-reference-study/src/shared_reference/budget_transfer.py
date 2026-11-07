"""Draft resource arithmetic and exact identity guards, not checkpoint loading."""
from .reader import _thaw
def shared_budget(n):
    if type(n) is not int or n not in (1,5,8,10): raise ValueError("Only declaredDRAFT N=1/5/8/10")
    return {"real_transitions":648192*n,"WM_optimizer_steps":4000*n,"failure_head_optimizer_steps":4000*n,
            "actor_optimizer_steps":1800*n,"critic_optimizer_steps":7200*n,"teacher_transitions":0,
            "shared_iterations":2000*n,"shared_warmup_steps":64*n,"shared_wm_only_iters":200*n,
            "status":"DRAFT_NOT_EXECUTED"}


def assert_matched_resource_counters(arm_counters,*,n):
    expected=shared_budget(n)
    if set(arm_counters)!={"A1L","P_ONLY","A2"}: raise ValueError("All3objective arms required")
    for arm,c in arm_counters.items():
        for key in ("real_transitions","WM_optimizer_steps","failure_head_optimizer_steps","actor_optimizer_steps","critic_optimizer_steps","teacher_transitions"):
            if c.get(key)!=expected[key]: raise ValueError("Unequalresource budget: "+arm+"/"+key)
    return {"counter_contract_match":True,"training_executed_or_verified":False,"scope":"CPU countercontract only; realstage ledgers/prefixtrace/teacher/pretraining/adaptation/wallcost still required"}


def assert_exact_transfer_identity(registry,track_name,descriptor):
    track=registry.track(track_name)
    expected={"registry_identity_sha256":registry.payload["registry_identity_sha256"],
              "native_reference_signature":track["native_reference_signature_CPU"],
              "model_sha256":registry.payload["proposed_controlled_physics"]["model"]["sha256"],
              "physics_context_sha256":registry.payload["proposed_controlled_physics"]["context"]["sha256"],
              "controller_sha256":registry.payload["proposed_controlled_physics"]["controller_pure"]["sha256"],"control_hz":50,
              "layout":_thaw(registry.payload["declared_layout"]),"data_exposure_source_ids":list(track["source_order"])}
    for key,value in expected.items():
        if descriptor.get(key)!=value: raise ValueError("Transferidentity mismatch, no silentcatalog/layout migration: "+key)
    if descriptor.get("format_version") in (2,3,103): raise ValueError("Oldcheckpoint format cannot silentlybecome a sharedtransfer contract")
    return {"exact_identity_match":True,"native_training_loader_bound":False,"resume_supported":False,"execution_authorized":False,
            "scope":"Metadata identityguard, not a checkpoint/optimizer/replay/environment resume implementation"}
