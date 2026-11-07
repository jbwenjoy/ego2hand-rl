"""Read back frozen native manifolds and retain checked separating directions."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.wrench_separation import separating_witness
from supermimic_research.mechanics.contact_modes import contact_wrench_basis


def main(args):
    import mujoco
    parent,reference,out=map(lambda p:p.resolve(),[args.parent,args.reference,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((parent/'report.json').read_text());bind=json.loads((reference/'binding.json').read_text())
    for pin in report['sources']:verify_file_pin(pin)
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    allowed=[model.geom(f'right_finger{i}_{s}').id for i in [1,2,3] for s in ['link3_col','link4_col','tip_sensor_col']]
    records=[];count=0
    for row in report['results']:
        state=np.load(verify_file_pin(row['candidate']));saved=np.load(verify_file_pin(row['basis']))
        data.qpos[:]=state['qpos27'];data.qvel[:]=0;data.ctrl[:]=state['ctrl20'];mujoco.mj_forward(model,data)
        rebuilt=contact_wrench_basis(model,data,bind['mug_part_geom_ids'],bind['mug_body_id'],allowed);count+=rebuilt['native_EFC_unit_queries']
        W=rebuilt['native_wrench'];L=rebuilt['native_loads'];error=float(np.abs(W-saved['wrench6xN']).max());lerror=float(np.abs(L-saved['joint_load20xN']).max())
        if error>1e-10 or lerror>1e-10:raise ValueError('Saved/native manifold mismatch')
        separation=separating_witness(W,saved['target6'])
        if not separation['verified']:raise ValueError('Retain missing separation instead of calling solver status proof')
        R=data.xmat[bind['mug_body_id']].reshape(3,3);y=np.array(separation['witness_dual_to_F_N_T_Nm'])
        records.append(dict(target_depth_m=row['target_depth_m'],basis=row['basis'],native_EFC_columns=W.shape[1],
            native_wrench_error=error,native_hand_load_error=lerror,force_only_feasible=row['static']['force_only']['success'],
            cone6D_feasible=row['static']['cone6D']['success'],separation=separation,
            dimensional_dual_BODY_force=R.T@y[:3],dimensional_dual_BODY_moment=R.T@y[3:]))
    out.mkdir(parents=True);result=dict(schema='mug_source_patch_support_separation:1',sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/wrench_separation.py'),file_pin(ROOT/'src/supermimic_research/mechanics/contact_modes.py')],
        parent=file_pin(parent/'report.json'),binding=file_pin(reference/'binding.json'),records=records,native_EFC_columns=count,
        verified_separations=sum(r['separation']['verified'] for r in records),integration_steps=0,neural_updates=0,
        scope='Native frozen ray support separation with explicit COM and moment scaling. Only a checked numerical local cone witness, not global infeasibility, native compliance force attainability, measured source gravity/contact force or learning-gradient evidence.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')
    print(json.dumps(dict(report=file_pin(out/'report.json'),verified=result['verified_separations'],EFC_columns=count,
        force_only=[r['force_only_feasible'] for r in records],directions=[r['separation']['witness_scaled6'] for r in records])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['parent','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
