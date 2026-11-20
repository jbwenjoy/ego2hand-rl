"""Match native terminal penetration to a rigid geom pair at the saved pre-step pose."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


def localize(report_file, out):
    if out.exists():
        raise FileExistsError(out)
    source = file_pin(report_file)
    report = json.loads(verify_file_pin(source).read_bytes())
    rows = []
    for arm, arm_result in report['arms'].items():
        for case in arm_result['cases']:
            original = verify_file_pin(case['case'])
            audit = json.loads(original.read_bytes())
            export = json.loads((original.parent/'cpu_export/report.json').read_bytes())
            model = mujoco.MjModel.from_binary_path(str(verify_file_pin(export['model'])))
            data = mujoco.MjData(model)
            free, = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
            body = model.jnt_bodyid[free]
            with np.load(verify_file_pin(audit['trace'])) as trace:
                failed = (trace['pre_penetration_m'] > .002).nonzero()[0]
                if len(failed) == 0:
                    rows.append(dict(arm=arm, rep=audit['rep'], crossing=None))
                    continue
                tick = int(failed[0])
                data.qpos[:] = trace['pre_qpos'][tick]
                data.mocap_pos[:], data.mocap_quat[:] = trace['post_mocap_pos'][tick], trace['post_mocap_quat'][tick]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                mujoco.mj_collision(model, data)
                pairs = {}
                for con in data.contact:
                    ids = np.asarray(con.geom)
                    if (model.geom_bodyid[ids] == body).sum() != 1 or con.dist >= 0:
                        continue
                    pair = tuple(model.geom(int(i)).name for i in ids)
                    pairs[pair] = max(pairs.get(pair, 0.), -float(con.dist))
                ranked = sorted(pairs.items(), key=lambda pair: -pair[1])
                gpu = float(trace['pre_penetration_m'][tick])
                rows.append(dict(arm=arm, rep=audit['rep'], crossing=tick+1,
                    phase=tick//8+1, native_mm=1000*gpu, cpu_pairs_mm=[
                        dict(geoms=g, depth_mm=1000*d) for g,d in ranked[:3]],
                    disagreement_mm=1000*abs(ranked[0][1]-gpu) if ranked else None))
    write_json(out, dict(source=source, method='The native trace stores depth only. CPU geom IDs are inferred at the first crossing on the matched native pre-step qpos, using the refitted exported model.', rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    localize(args.report, args.out)
