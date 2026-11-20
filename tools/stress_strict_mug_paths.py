"""Check pose and substep geometry jointly on frozen optimized mug trajectories under altered priors."""
import argparse
from pathlib import Path
import tempfile
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from audit_mug_refined_paths import geometry,h3


def run(out,stage):
    root=Path(__file__).resolve().parents[1]/'runs'
    clips=(1884,1888);sources={c:h3.file_pin(root/f'mug_strict_{stage}_{c}_v1/report.json') for c in clips}
    inputs=h3.physics.read_json(h3.file_pin(root/'mug_annotated_windows_original_v1/intent.json'))['inputs']
    asset=h3.verify_file_pin(inputs['model']);bind=h3.physics.read_json(inputs['binding']);old=h3.physics.load_arrays(inputs['reference'])
    # mass/inertia scale, sliding friction scale, contact time-constant scale,
    # initial mug translation along its own x axis (m).
    variants={'nominal':(1.,1.,1.,0.),'mass105':(1.05,1.,1.,0.),'mass095':(.95,1.,1.,0.),
        'mu095':(1.,.95,1.,0.),'mu105':(1.,1.05,1.,0.),'heavy_low':(1.15,.8,1.,0.),
        'time075':(1.,1.,.75,0.),'time125':(1.,1.,1.25,0.),
        'start_x_plus_mm':(1.,1.,1.,.001),'start_x_minus_mm':(1.,1.,1.,-.001)}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(sources=sources,code=[h3.file_pin(__file__),h3.file_pin(Path(__file__).with_name('audit_mug_refined_paths.py'))],
        conditions=variants,scope='Fixed final strict commands, zero initial velocity. Multiply inertia by the mass factor. Only hand/mug explicit pairs receive the designated friction and solref time scaling. Initial pose shifts are in the initial mug BODY axes and the target stays unchanged. Re-audit actual geometry, without parameter-specific retuning.'))
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        for tag,(mass,mu,compliance,offset) in variants.items():
            m=mujoco.MjModel.from_binary_path(str(asset))
            body=bind['mug_body_id'];hand=np.isin(np.arange(m.ngeom),[m.geom(n).id for n in bind['hand_collision_names']])
            mug=m.geom_bodyid==body;a,b=m.pair_geom1,m.pair_geom2;pairs=(hand[a]&mug[b])|(mug[a]&hand[b])
            assert pairs.sum()==len(bind['hand_collision_names'])*len(bind['mug_part_geom_ids'])
            m.pair_friction[pairs,:2]*=mu
            assert (m.pair_solref[pairs,0]>0).all()
            m.pair_solref[pairs,0]*=compliance
            data=mujoco.MjData(m)
            if mass!=1:
                m.body_mass[body]*=mass;m.body_inertia[body]*=mass;mujoco.mj_setConst(m,data)
            model_path=Path(tmp)/(tag+'.mjb');mujoco.mj_saveModel(m,str(model_path))
            for clip,folder in zip(clips,('mug_annotated_windows_original_v1','mug_unilateral_interval_v1')):
                item,=h3.physics.read_json(h3.file_pin(root/folder/'report.json'))['rows']
                ref=h3.physics.load_arrays(item['reference']);T=ref['body']
                initial=old['initial_qpos27'].copy();qa=bind['mug_free_qpos']
                initial[qa:qa+3]+=Rotation.from_quat(initial[qa+np.array([4,5,6,3])]).apply([offset,0,0])
                z={**old,'initial_qpos27':initial,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
                   'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
                prior=h3.physics.read_json(sources[clip]);original=h3.physics.load_arrays(prior['trace']);cmd=original['ctrl']
                detail,tr=h3.physics.replay(model_path,bind,z,old['feedforward_PD_target20'][0],cmd,1.,1.)
                stats=geometry(model_path,bind,z,old['feedforward_PD_target20'][0],cmd,tr['qpos'])
                limits=np.array([2.,.2,.001]);invalid=np.flatnonzero((stats[:,:3]>limits).any(1))
                passed=min(round(detail['valid_s']/.02),int(invalid[0]) if len(invalid) else len(cmd))
                if tag=='nominal':
                    np.testing.assert_array_equal(original['qpos'],tr['qpos'])
                    assert passed==prior['strict_controls']
                trace=out/f'{clip}_{tag}.npz';np.savez_compressed(trace,**tr,geometry=stats)
                results.append(dict(clip=clip,variant=tag,strict_s=.02*passed,full_strict=passed==len(cmd),
                    strict_peak=stats[:passed].max(0).tolist() if passed else None,task=detail,trace=h3.file_pin(trace)))
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--stage',choices=['continue','extend'],default='extend')
    a=p.parse_args();run(a.out.resolve(),a.stage)
