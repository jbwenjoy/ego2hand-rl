"""Inspect an actual fresh native plant before, during and after host physics export."""
from pathlib import Path
import json,sys,copy
import numpy as np
import torch,mujoco
import collect_guarded_finger_response as b
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.runtime.pen_cpu_model_export import export_pen_cpu_model


def capture(m,body):
    adr=int(m.body_bvhadr[body]);n=int(m.body_bvhnum[body]);ids=np.flatnonzero(m.geom_bodyid==body)
    return {key:getattr(m,key)[body].tolist() for key in ('body_ipos','body_iquat','body_inertia')}|dict(
        geom_ids=ids.tolist(),geom_names=[m.geom(int(i)).name for i in ids],geom_pos=m.geom_pos[ids].tolist(),
        geom_quat=m.geom_quat[ids].tolist(),geom_size=m.geom_size[ids].tolist(),geom_aabb=m.geom_aabb[ids].tolist(),
        bvh_address=adr,bvh_aabb=m.bvh_aabb[adr:adr+n].tolist(),bvh_nodeid=m.bvh_nodeid[adr:adr+n].tolist())


def main(out):
    out.mkdir(exist_ok=False)
    with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as col:
        manifest=b.factory();identity=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
        with b.CostProfile(identity,manifest) as profile:
            rt=profile.create_runtime(b.OBJECTIVE,1)
            try:
                host=rt.env.sim.mj_model;wp=rt.env.sim.wp_model;body=rt.features.body;host_before=capture(host,body)
                # save_physics operates on its own copy. Retain both serialized stages.
                hostfile=out/'live_host.mjb';mujoco.mj_saveModel(host,str(hostfile))
                dest=out/'cpu_bundle';bundle=export_pen_cpu_model(rt,dest);export=dest/'device_parameters'
                repaired=capture(mujoco.MjModel.from_binary_path(bundle['model']['path']),body)
                exported=capture(mujoco.MjModel.from_binary_path(str(export/'model.mjb')),body)
                assert capture(host,body)==host_before
                arrays={k:getattr(wp,k).numpy()[0,body].tolist() for k in ('body_ipos','body_iquat','body_inertia')}
                for k,v in arrays.items():np.testing.assert_array_equal(v,exported[k])
            finally:rt.close()
    write_json(out/'report.json',dict(scope='Fresh W1 after runtime construction; captured live host model, device arrays, and unmodified save_physics export. No native integration.',factory=manifest['generated'],host=file_pin(hostfile),export=file_pin(export/'model.mjb'),wp=arrays,host_before=host_before,exported=exported,repaired=repaired,cpu_bundle=file_pin(dest/'report.json')))
    print('host iquat',host_before['body_iquat'],'wp',arrays['body_iquat'],'export',exported['body_iquat'],'bvh unchanged',exported['bvh_aabb']==host_before['bvh_aabb'])
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
