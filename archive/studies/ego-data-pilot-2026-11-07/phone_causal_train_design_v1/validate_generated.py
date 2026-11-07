"""Bounded pure metadata/schema validation. No native/model/NN/AD calls."""
from pathlib import Path
import hashlib,json
import numpy as np
import causal_contract as C

H=Path(__file__).resolve().parent;G=H/"generated_v1"

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def rejected(fn):
    try:fn()
    except ValueError:return True
    raise AssertionError("Invalid geometry/input accepted")

def main():
    contexts=json.loads((G/"contexts72.json").read_text());branches=json.loads((G/"all_baselines_then_sides_plan.json").read_text())
    report=json.loads((G/"report.json").read_text());topology=json.loads((G/"geometry_topology_contract.json").read_text())
    assert len(contexts)==72 and len(branches)==2952 and [b["branch_id"] for b in branches]==list(range(2952))
    assert all(b["kind"]=="BASELINE" for b in branches[:72]) and all(b["kind"]=="FINITE_SIDE" for b in branches[72:])
    planned_steps=sum(contexts[b["context_id"]]["n_substeps"] for b in branches if b["kind"]=="BASELINE" or b["available"])
    assert len({c["original_family_id"] for c in contexts})==12
    for b in branches[72:]:
        if b["available"]:assert abs(b["actual_delta_u_F64_rad"])>=1e-6
        if b["model_FP32_endpoint_collapsed"]:assert not b["new_action_response_check_eligible"]
    with np.load(G/"causal_masks_and_descriptors.npz",allow_pickle=False) as z:
        assert z["strict_object_action_zero72x20"].shape==(72,20) and not z["strict_object_action_zero72x20"].any()
    with np.load(G/"source_before_action_restore_arrays.npz",allow_pickle=False) as z:
        assert z["source_integration_state_F64"].shape==z["source_after_action_integration_state_F64"].shape==(72,300)
        assert z["source_integration_state_F64"].dtype==np.float64
    # A current geometry input with only one known pair; unknown features remain NaN.
    x=np.zeros((1,58));x[:,43:52]=np.eye(3).reshape(9)
    known=np.zeros((1,22),dtype=bool);known[0,0]=True
    p=np.full((1,22,3),np.nan);p[0,0]=[.03,0,0]
    r=np.full((1,22,9),np.nan);r[0,0]=np.eye(3).ravel()
    gap=np.full((1,22),np.nan);gap[0,0]=.03
    normal=np.full((1,22,3),np.nan);normal[0,0]=[1,0,0]
    batch=dict(state58=x,absolute_u20=np.zeros((1,20)),native_h_s=np.array([.0003125]),n_substeps=np.array([64],dtype=np.int64),
        pair_relative_translation_BODY_m=p,pair_relative_rotation_BODY9=r,pair_signed_gap_m=gap,pair_gap_lower_bound_m=gap.copy(),
        pair_geometry_known=known,normal_on_phone_BODY3=normal,normal_unique_known=known.copy(),
        pair_names=topology["actual_explicit_phone_pair_names"],model_sha256=C.MODEL_SHA256,source_state_schema="phone_COM_world_state58:1")
    C.validate_current_geometry_inputs(batch)
    reflection=r.copy();reflection[0,0]=np.diag([-1.,1,1]).ravel()
    rejected(lambda:C.validate_current_geometry_inputs({**batch,"pair_relative_rotation_BODY9":reflection}))
    rejected(lambda:C.validate_current_geometry_inputs({**batch,"whole_H_path_certificate":True}))
    rejected(lambda:C.validate_current_geometry_inputs({**batch,"native_h_s":np.array([-.0003125])}))
    rejected(lambda:C.validate_current_geometry_inputs({**batch,"n_substeps":np.array([64.])}))
    rejected(lambda:C.validate_current_geometry_inputs({**batch,"model_sha256":"0"*64}))
    result=dict(schema="phone_causal_metadata_design_validation:1",status="PASS_METADATA_PLAN_AND_FAIL_CLOSED_INPUT_SCHEMA_NOT_PHYSICAL_VALIDATION",
        source=pin(__file__),contract=pin(H/"causal_contract.py"),design_report=pin(G/"report.json"),
        planned_branch_records=2952,planned_executable_branches=sum(b["kind"]=="BASELINE" or b["available"] for b in branches),
        planned_executable_steps=planned_steps,unavailable_records_preserved=report["unavailable_sides"],
        checks=dict(order_all_baselines_then_sides=True,all72full300vector_sources=True,all20axis_records_preserved=True,
            no_strict_zero_certificates=True,unknown_geometry_NaN_accepted=True,reflection_rejected=True,
            future_target_oracle_rejected=True,invalid_clock_count_model_rejected=True),
        actual_counts=dict(derived_metadata_NPZ_reads=2,new_robot_FK_native_NN_AD_training=0))
    with (H/"validation_v1.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(H/"validation_v1.json"),planned_steps=planned_steps,actual_physics_NN=0)))

if __name__=="__main__":main()
