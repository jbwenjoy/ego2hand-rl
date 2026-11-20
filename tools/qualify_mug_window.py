"""Compare a compiled native observer with the existing per-substep Python audit."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from test_mug_witness_mpc import context,h3,Rotation
from supermimic_research.mechanics.mug_window_native import MugWindow


class NativeStrictWindow:
    def __init__(self,P,library):
        self.P=P
        self.kernel=MugWindow(library,P.m,P.binding)
    def step(self,d,i,cmd):
        peaks=self.kernel.step(d,cmd);p,R,_,_=self.P.chart.motion(self.P.state(d))
        return np.r_[1000*np.linalg.norm(p-self.P.pos[i+1]),np.rad2deg((R*self.P.rot[i+1].inv()).magnitude()),peaks]


def run(build,out):
    root=Path(__file__).resolve().parents[1]/'runs';pkg=h3.file_pin(build/'report.json')
    bundle=h3.physics.read_json(pkg);lib=h3.verify_file_pin(bundle['helper'])
    inputs=h3.physics.read_json(h3.file_pin(root/'mug_annotated_windows_original_v1/intent.json'))['inputs']
    old=h3.physics.load_arrays(inputs['reference']);binding=h3.physics.read_json(inputs['binding']);asset=h3.verify_file_pin(inputs['model'])
    m=mujoco.MjModel.from_binary_path(str(asset));out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(build=pkg,code=[h3.file_pin(__file__),h3.file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/mug_window_native.py')],
        model=inputs['model'],scope='Fixed commands on nominal successful and geometry-violating paths, including a table-contact diagnostic. All six metrics and runtime endpoint/warmstart must agree exactly at every control. Interleave repeated fresh-data timings with one unmeasured warmup.'))
    rows=[]
    for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
        info,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows']
        ref=h3.physics.load_arrays(info['reference']);T=ref['body']
        P,python=context(m,binding,T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],Rotation.from_matrix(T[:,:3,:3]))
        native=NativeStrictWindow(P,lib)
        p=h3.physics.read_json(h3.file_pin(root/f'mug_strict_extend_{clip}_v1/report.json'))['trace']
        for name,cmds in [('strict',h3.physics.load_arrays(p)['ctrl']),
                          ('ungated',np.load(root/f'mug_path_refinement_v1/{clip}_8451/best.npz')['commands']),
                          ('table',np.tile(old['feedforward_PD_target20'][0],(20,1)))]:
            def make():
                d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=old['feedforward_PD_target20'][0]
                mujoco.mj_forward(m,d)
                if name=='table':
                    g=m.geom('mug_task_table').id;qa=P.chart.qa
                    d.qpos[qa:qa+3]=d.geom_xpos[g]+[0,0,m.geom_size[g,2]+.07]
                    mujoco.mj_forward(m,d)
                return d
            d=make();e=make();metrics=[]
            for i,cmd in enumerate(cmds):
                a=python.step(d,i,cmd);b=native.step(e,i,cmd);np.testing.assert_array_equal(a,b)
                for k in ('qpos','qvel','qacc_warmstart','ctrl'):
                    np.testing.assert_array_equal(getattr(d,k),getattr(e,k))
                metrics.append(a)
            timings={'python':[],'native':[]}
            if name=='strict':
                for n in range(4):
                    order=[('python',python),('native',native)]
                    for key,checker in (order if n%2 else order[::-1]):
                        d=make();t=perf_counter()
                        for i,cmd in enumerate(cmds):checker.step(d,i,cmd)
                        timings[key].append(perf_counter()-t)
            rows.append(dict(clip=clip,case=name,controls=len(cmds),substeps=64*len(cmds),exact=True,
                maximum=np.max(metrics,axis=0).tolist(),times=timings))
    if max(r['maximum'][2] for r in rows)<=.1:raise ValueError('Table case did not exercise normal-force query')
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();run(a.build.resolve(),a.out.resolve())
