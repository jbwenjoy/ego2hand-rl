"""Time full candidate-depth scoring from a native host state on one CPU thread."""
import argparse
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import torch

from build_pen_contact_counterfactual_data import boundaries, depth, geometry, prepare, read
from fit_pen_action_depth import grouped_indices, read_npz
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_action_depth import ActionDepthHead, inputs
from supermimic_research.strict_json import write_json


def main(run, out):
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    fit = read(file_pin(run/'intent.json'))
    frozen = read(file_pin(run/'frozen.json'))
    source = read(fit['dataset_source'])
    dataset = read_npz(fit['fit'])
    groups = grouped_indices(dataset, dataset['split']=='val')
    refit = read(read(source['intent'])['refit'])
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(refit['model'])))
    data, geom = mujoco.MjData(model), geometry(model)
    ck = torch.load(verify_file_pin(frozen['models']['state_action']),map_location='cpu',weights_only=True)
    net = ActionDepthHead(True)
    net.load_state_dict(ck['state_dict'])
    net.eval().requires_grad_(False)
    write_json(out/'intent.json',dict(fit=fit['fit'],head=frozen['models']['state_action'],model=refit['model'],
               code=file_pin(__file__),torch_threads=1,groups=len(groups),candidates_per_source=groups.shape[1],
               scope='CPU resident model; preloaded states and proposed PDs. Include MjData reset, setting measured state, current geometry queries, tensor conversion and network forward for the learned scorer; each physical score integrates eight native substeps. Report warmed single-process wall times.'))
    def evaluate(method, group_ids):
        pred = []
        for idx in group_ids:
            row = idx[0]
            q,v,w,mount = (dataset[k][row] for k in ('qpos27','qvel26','warm26','mount7'))
            t = dataset['time_s'][row]
            pd = dataset['pd20'][idx]
            if method=='head':
                prepare(model,data,q,v,w,mount,t)
                b = boundaries(model,data,geom)
                state = native_state55(q,v,np.arange(7,27),np.arange(6,26),0,0)
                features = [torch.as_tensor(a,dtype=torch.float32).expand(len(pd),-1) for a in (state,w,b)]
                with torch.inference_mode():
                    x=inputs(*features,torch.as_tensor(pd,dtype=torch.float32))
                    pred.append(net.predict_mm(x).numpy())
            else:
                block = np.zeros((len(pd),2))
                for i,command in enumerate(pd):
                    prepare(model,data,q,v,w,mount,t)
                    data.ctrl[:] = command
                    for _ in range(8):
                        mujoco.mj_step(model,data)
                        np.maximum(block[i],depth(data,*geom[:2]),out=block[i])
                pred.append(block)
        return np.stack(pred)
    timings={'head':[],'native':[]}
    for method in timings:
        evaluate(method,groups[:2])
    for rep in range(5):
        for method in (('native','head') if rep%2 else ('head','native')):
            t=perf_counter()
            p=evaluate(method,groups)
            timings[method].append(perf_counter()-t)
            if method=='native':
                np.testing.assert_array_equal(p,dataset['cpu_mm'][groups])
            else:
                checked=p.copy()
    # Check the feature reconstruction used inside every timed head invocation.
    array = lambda k: torch.as_tensor(dataset[k][groups],dtype=torch.float32)
    with torch.inference_mode():
        expected=net.predict_mm(inputs(*[array(k) for k in ('state55','warm26','boundary_mm','pd20')])).numpy()
    np.testing.assert_allclose(checked,expected,rtol=1e-5,atol=2e-6)
    med = {m:float(np.median(ts)) for m,ts in timings.items()}
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),wall_s=timings,median_s=med,
        ms_per_source={m:t*1000/len(groups) for m,t in med.items()},
        speed_ratio=med['native']/med['head'],native_labels_exact=True,
        head_max_batch_error_mm=float(np.abs(checked-expected).max()),
        validation_head_depth_mae_mm=abs(checked-dataset['cpu_mm'][groups]).mean(axis=(0,1)).tolist()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    main(args.run.resolve(),args.out.resolve())
