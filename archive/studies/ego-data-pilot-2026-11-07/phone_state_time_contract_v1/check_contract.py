"""Four closed-form checks. No physics, FK, native model or NN dependencies."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import state_contract as c

HERE = Path(__file__).resolve().parent


def pin(path):
    p = Path(path).resolve()
    b = p.read_bytes()
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def close(a, b, atol=2e-14):
    np.testing.assert_allclose(a, b, rtol=0, atol=atol)


def fixture_offset_and_cross_sign():
    # 90deg about world Z: body X becomes world Y.
    r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    q = np.arange(20, dtype=np.float32) / np.float32(10)
    qd = -q
    s = c.to_com_world_state(q, qd, np.array([1., 2., 3.]), r,
                            np.array([.5, -.5, 1.]), np.array([0., 0., 3.]),
                            np.array([.02, 0., 0.]))
    close(s[40:43], [1., 2.02, 3.])
    close(s[52:55], [.44, -.5, 1.])  # +Z omega x +Y lever = -X.
    close(s[55:58], [0., 0., 3.])
    raw = c.from_com_world_state(s, np.array([.02, 0., 0.]))
    close(raw["p_root_world"], [1., 2., 3.])
    close(raw["v_root_world"], [.5, -.5, 1.])
    assert np.array_equal(s[:20], q.astype(np.float64))
    assert np.array_equal(s[20:40], qd.astype(np.float64))


def fixture_rotation_batch_roundtrip():
    r = np.array([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],
                  [[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]], np.eye(3)])
    q = np.arange(60, dtype=np.float64).reshape(3, 20) / 100
    qd = q / 3
    p = np.array([[1.,2.,3.],[-2.,1.,.4],[.1,.2,.3]])
    v = np.array([[.2,.3,.4],[.1,.6,.5],[-.3,.2,.1]])
    w = np.array([[1.,2.,3.],[-2.,1.,4.],[3.,1.,-2.]])
    ipos = np.array([[.02,.01,-.03],[-.01,.04,.02],[.03,-.02,.01]])
    s = c.to_com_world_state(q, qd, p, r.reshape(3,9), v, w, ipos)
    close(s[:,55:58], [[-2.,1.,3.],[-2.,-4.,1.],[3.,1.,-2.]])
    close(s[:,40:43] - p, [[-.01,.02,-.03],[-.01,-.02,.04],[.03,-.02,.01]])
    raw = c.from_com_world_state(s, ipos)
    for name, expected in (("q20",q),("qd20",qd),("p_root_world",p),
                           ("R_body_world",r),("v_root_world",v),("omega_body",w)):
        close(raw[name], expected)
    a = .37
    rounded_r = np.array([[np.cos(a),-np.sin(a),0.],
                          [np.sin(a),np.cos(a),0.],[0.,0.,1.]],dtype=np.float32)
    rounded_s = c.to_com_world_state(q,qd,p,np.broadcast_to(rounded_r,(3,3,3)),v,w,ipos)
    close(c.from_com_world_state(rounded_s,ipos)["omega_body"],w)
    try:
        c.to_com_world_state(q, qd, p, np.broadcast_to(np.diag([1.,1.,-1.]),(3,3,3)), v,w,ipos)
    except ValueError:
        pass
    else:
        raise AssertionError("reflection must be rejected")


def fixture_analytic_COM_derivative():
    # Analytic constant-omega rotation and translating root; no integration.
    omega = 2.3
    t0, delta = .37, 1e-5
    ipos = np.array([.03,-.02,.01])
    vroot = np.array([.2,-.1,.04])
    def values(t):
        a = omega*t
        r = np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
        return c.to_com_world_state(np.zeros(20),np.zeros(20),
            np.array([.1,.2,.3])+t*vroot,r,vroot,np.array([0.,0.,omega]),ipos)
    before, mid, after = values(t0-delta),values(t0),values(t0+delta)
    derivative = (after[40:43] - before[40:43])/(2*delta)
    close(derivative, mid[52:55], atol=2e-10)
    static_rotation = c.to_com_world_state(np.zeros(20),np.zeros(20),np.zeros(3),
        np.eye(3),vroot,np.zeros(3),ipos)
    close(static_rotation[52:55], vroot)


def fixture_integer_tick_time():
    for h, n, last_pre in ((.0003125,64,.0196875),(.00015625,128,.01984375)):
        p = c.TickPlan(h)
        assert p.substeps == n
        ticks = np.arange(n+1,dtype=np.int64)
        close(p.nominal_time_s(ticks)[[0,-2,-1]], [0.,last_pre,.02], atol=4e-18)
        masks = p.masks(ticks)
        assert masks["solver_PRE"].sum()==n and masks["integrated_POST"].sum()==n
        assert masks["terminal_POST"].sum()==1 and masks["initial"].sum()==1
        # An intentionally drifting saved clock cannot alter the supplied ticks.
        saved = np.cumsum(np.full(n,np.float32(h)),dtype=np.float32)
        saved = np.concatenate((np.zeros(1,dtype=np.float32),saved))
        desc = p.native_clock_descriptors(saved,ticks)
        assert np.array_equal(desc["tick"],ticks)
        assert np.any(desc["native_minus_nominal_s"] != 0)
    for h in (.005,.0003):
        try:
            c.TickPlan(h)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported timestep must be rejected")
    try:
        c.TickPlan(.0003125).masks(np.array([0.,64.]))
    except ValueError:
        pass
    else:
        raise AssertionError("float tick labels must be rejected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("proof output must be fresh")
    freeze = json.loads((HERE/"SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    for key, old in freeze["source_pins"].items():
        assert pin(old["path"]) == old, key
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    fixtures = [fixture_offset_and_cross_sign,fixture_rotation_batch_roundtrip,
                fixture_analytic_COM_derivative,fixture_integer_tick_time]
    checks = {}
    try:
        for f in fixtures:
            f()
            checks[f.__name__] = True
        assert not any(n in sys.modules for n in ("torch","mujoco","mujoco_warp","warp"))
        result = {"schema":"phone_state_time_contract_CPU_proof:1", "status":"PASS_PURE_NUMPY_ANALYTIC_4",
            "passed":len(checks),"total":len(fixtures),"checks":checks,"source_freeze":pin(HERE/"SOURCE_FREEZE.json"),
            "source_pins":freeze["source_pins"], "time_plans":[c.TickPlan(h).declaration() for h in (.0003125,.00015625)],
            "actual_model_or_FK_or_Env_or_native_forward_or_step_or_NN_or_GPU_calls":0,
            "native_saved_array_reads":0,"elapsed_s":time.perf_counter()-started}
    except Exception as exc:
        (args.output/"failure.json").write_text(json.dumps({"checks":checks,"exception":repr(exc)},indent=2),encoding="utf-8")
        raise
    (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"passed":result["passed"],"total":result["total"],"report":str((args.output/"report.json").resolve())}))


if __name__ == "__main__":
    main()
