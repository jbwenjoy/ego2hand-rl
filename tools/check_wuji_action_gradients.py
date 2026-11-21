"""Test frozen world-model action derivatives against paired native physics interventions."""

import argparse
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.artifacts import file_pin
from supermimic_research.learning.flash_imagination import Dynamics, Scales, predictions


def block(schema,name):
    r,=[r for r in schema["blocks"] if r["name"]==name]
    return slice(r["start"],r["end"])


def spatial_quat_derivative(q,dq):
    norm=np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("Undefined predicted orientation derivative")
    q=q/norm
    dq=(dq-q*(q@dq))/norm
    return np.rad2deg(2*(q[0]*dq[1:]-dq[0]*q[1:]+np.cross(q[1:],dq[1:])))


def rotation_difference(plus,minus):
    return np.rad2deg((Rotation.from_quat(plus,scalar_first=True)*
                       Rotation.from_quat(minus,scalar_first=True).inv()).as_rotvec())


def vector_result(observed_difference,zero_difference,predicted_slope,amplitude,min_difference):
    actual=observed_difference/amplitude
    noise=float(np.linalg.norm(zero_difference))
    resolved=np.linalg.norm(observed_difference)>max(min_difference,3*noise)
    pred_norm=float(np.linalg.norm(predicted_slope))
    norm=float(np.linalg.norm(actual))
    cosine=(float(np.clip(predicted_slope@actual/(norm*pred_norm),-1,1)) if resolved and pred_norm>1e-10 else None)
    return dict(native_slope=actual.tolist(),model_slope=predicted_slope.tolist(),native_difference=float(np.linalg.norm(observed_difference)),
                zero_repeat_difference=noise,minimum_signal_difference=min_difference,scorable=bool(resolved),
                cosine=cosine,magnitude_ratio=pred_norm/norm if resolved else None)


def evaluate_stage(fixture,sidecar,schema,terms_names):
    saved=serialization.msgpack_restore(sidecar.read_bytes())
    st=saved["state"]
    model=Dynamics(len(st["scale"]["output_center"]))
    scales=Scales(**{k:jnp.asarray(v) for k,v in st["scale"].items()})
    params=jax.device_put(st["model_params"])
    with np.load(fixture,allow_pickle=False) as a:data=dict(a)
    initial=jnp.concatenate((jnp.asarray(data["obs0"]).reshape(1,-1),jnp.asarray(data["ctx0"]).reshape(1,-1)),axis=-1)
    obs_dim=data["obs0"].size
    actions=data["actions"]
    start=np.zeros(actions.shape[-1],np.float32)
    zero_indices=np.flatnonzero(np.asarray(data["pulse"])==0)
    if len(zero_indices)!=2 or not np.all(actions[zero_indices]==0) or not np.all(actions[:,1:]==0):
        raise ValueError("Expected two zero repeats and first-control-only pulses")
    z0,z1=zero_indices
    pos=block(schema,"object_position_e_m")
    quat=block(schema,"object_orientation_e_wxyz")
    wp=block(schema,"wrist_position_e_m")
    cws=terms_names.index("contact_wrench_support")
    def trajectory(first_action):
        x=initial
        outputs=[]
        for step in range(actions.shape[1]):
            action=first_action[None,:] if step==0 else jnp.zeros((1,len(start)),dtype=x.dtype)
            x,terms,_=predictions(model,params,scales,x,action)
            if saved["options"]["geometry_tail"]:
                sl=slice(obs_dim-saved["options"]["geometry_tail"],obs_dim)
                x=x.at[:,sl].set(initial[:,sl])
            outputs.append(jnp.concatenate((x[0,wp],x[0,pos],x[0,quat],terms[0,cws:cws+1],
                                           (terms@jnp.asarray(data["weights"]))*float(data["dt"]))))
        return jnp.stack(outputs)
    jvp=jax.jit(lambda v:jax.jvp(trajectory,(jnp.asarray(start),),(v,)))
    predicted_pairs=jax.jit(jax.vmap(trajectory))(jnp.asarray(actions[:,0,:]))
    predicted_pairs=np.asarray(predicted_pairs)
    rows=[]
    axes=np.unique(np.asarray(data["axis"])[np.asarray(data["pulse"])!=0])
    for axis in axes:
        pair=np.flatnonzero((data["axis"]==axis)&(data["pulse"]!=0))
        if len(pair)!=2:raise ValueError("Unpaired axis")
        minus,plus=pair[np.argsort(data["pulse"][pair])]
        amp=float(data["pulse"][plus]-data["pulse"][minus])
        tangent=np.zeros_like(start)
        tangent[int(axis)]=1.
        np.testing.assert_allclose((actions[plus,0]-actions[minus,0])/amp,tangent,rtol=1e-6,atol=1e-6)
        prediction,gradient=map(np.asarray,jvp(tangent))
        for step in range(actions.shape[1]):
            valid=bool(np.asarray(data["valid"])[[z0,z1,minus,plus],:step+1].all())
            row=dict(axis=int(axis),horizon=step+1,valid=valid,normalized_action_span=amp)
            rows.append(row)
            if not valid:continue
            nxt=data["next_obs"][:,step]
            reward=data["reward"][:,step].reshape(-1,1)
            support=data["objective_terms"][:,step,cws:cws+1]
            qp=nxt[:,quat]
            g=gradient[step]
            pairs=[("wrist_mm",nxt[:,wp]*1000,1000*g[:3],.05),
                   ("object_mm",nxt[:,pos]*1000,1000*g[3:6],.05),
                   ("support_term",support,g[10:11],.001),
                   ("reward",reward,g[11:12],.0001)]
            for key,truth,model_slope,threshold in pairs:
                row[key]=vector_result(truth[plus]-truth[minus],truth[z1]-truth[z0],model_slope,amp,threshold)
            row["object_deg"]=vector_result(rotation_difference(qp[plus],qp[minus]),rotation_difference(qp[z1],qp[z0]),
                                            spatial_quat_derivative(prediction[step,6:10],g[6:10]),amp,.05)
            # Verify that any mismatch with real physics is not a mismatch between our analytic
            # derivative and the learned model's own, finite-amplitude response.
            model_span=predicted_pairs[plus,step]-predicted_pairs[minus,step]
            row["model_finite_span_vs_linear_span_mm"]=float(np.linalg.norm(1000*(model_span[3:6]-g[3:6]*amp)))
            model_angle_span=rotation_difference(predicted_pairs[plus,step,6:10],predicted_pairs[minus,step,6:10])
            row["model_finite_span_vs_linear_span_deg"]=float(np.linalg.norm(
                model_angle_span-spatial_quat_derivative(prediction[step,6:10],g[6:10])*amp))
            measured_world=data["next_object"][[minus,plus],step,:3]
            np.testing.assert_allclose(np.linalg.norm(measured_world[1]-measured_world[0]),
                                       np.linalg.norm(nxt[plus,pos]-nxt[minus,pos]),atol=2e-6,rtol=1e-4)
            contact=data["substep_contact"][[minus,plus],:step+1,:,0]
            row["native_peak_hand_load_N"]=float(np.max(contact))
    return dict(fixture=file_pin(fixture),checkpoint=file_pin(sidecar),rows=rows,
                training_steps=int(saved["native_environment_steps"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture",action="append",type=Path,required=True)
    p.add_argument("--schema-report",type=Path,required=True)
    p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.schema_report.read_bytes())
    out={}
    for s in json.loads(a.ledger.read_bytes()):
        if s["stream"]!="shared_wm":continue
        ck=Path(s["checkpoint"])
        sidecar=ck.with_name(ck.name.replace("policy_","world_model_").replace(".safetensors",".msgpack"))
        out[str(s["round"])]=[evaluate_stage(f,sidecar,report["policy_schema"]["observation"],report["objective_names"]) for f in a.fixture]
    a.out.write_text(json.dumps(dict(model_results=out,tool=file_pin(__file__),
        scope="CPU physical finite differences vs JAX directional derivative at zero of the saved model rollout. "
              "No derivative agreement assigned to unresolved physical responses or windows crossing terminal state."),indent=2,allow_nan=False))
