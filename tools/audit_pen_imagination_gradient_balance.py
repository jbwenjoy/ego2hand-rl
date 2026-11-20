"""Measure how much the capacity term changes the shared starting Actor gradient."""
from pathlib import Path
import json,sys,tomllib
from types import SimpleNamespace
import numpy as np
import torch
from pilot_pen_paired_imagination import ROOT,READ,LoadForecast,load_capacity_head,model_pair,imagined_objectives,verify_file_pin,file_pin,write_json


def main(run,out):
    out.mkdir(exist_ok=False)
    intent=READ(file_pin(run/'intent.json'));pins=intent['pins']
    load_state=lambda key:torch.load(verify_file_pin(pins[key]),map_location='cpu',weights_only=False)
    parent=load_state('actor');actor,_,_=model_pair('cpu',SimpleNamespace(**parent['config']));actor.load_state_dict(parent['actor']);actor.double().eval().requires_grad_(False)
    layer=actor.net[-1];layer.requires_grad_(True)
    w=load_state('load');plant=LoadForecast(torch.zeros(2,len(w['center'])),torch.zeros(2,len(w['out_center'])));plant.load_state_dict(w);plant.double().eval().requires_grad_(False)
    cap=load_capacity_head(load_state('capacity'),dtype=torch.float64,device='cpu').eval().requires_grad_(False)
    with np.load(verify_file_pin(pins['starts'])) as f:
        t={k:torch.from_numpy(f[k]).double() for k in ('state55','previous_raw','previous_ema','ref_joints','ref_pose_wxyz','ref_links','mount_wxyz','encoder_bias','default_joint_pos','default_joint_vel','lower','upper','capacity_start','capacity_eligible')}
        mass=float(f['mass_kg'][0]);inertia=torch.tensor(f['inertia_BODY'][0]);gravity=torch.tensor(f['gravity_WORLD'][0]);ph=f['phase'];roles=f['role']
    cfg=tomllib.loads((ROOT/'configs/pen_full_pose_native_v1.toml').read_text())['reward']
    torch.set_num_threads(2)
    def subgradient(term):
        gw,gb=torch.autograd.grad(term,(layer.weight,layer.bias),retain_graph=True)
        return torch.cat([gw[4:12].flatten(),gb[4:12]])
    def inspect(indices):
        data={k:v[indices] for k,v in t.items()};terms=imagined_objectives(actor,plant,cap,data,cfg,mass,inertia,gravity)
        base=subgradient(terms['base']);extra=subgradient(-.2*terms['capacity_error'])
        n=float(base.norm());m=float(extra.norm())
        return dict(n_rows=len(indices),eligible_slots=int(data['capacity_eligible'].sum()),base_norm=n,added_norm=m,
                    extra_base_ratio=m/n if n else None,cos_base_extra=float(base@extra)/(m*n) if n*m else None,
                    cos_A1_A2=float((base+extra)@base/((base+extra).norm()*base.norm())))
    pooled=inspect(np.arange(len(ph)))
    locals_=[dict(role=str(role),phase=int(p),**inspect([i])) for i,(role,p) in enumerate(zip(roles,ph))]
    valid=[r for r in locals_ if r['added_norm']>0]
    write_json(out/'report.json',dict(pilot=file_pin(run/'intent.json'),scope='Start-weight loss gradients restricted to the eight fitted output rows; pooled derivatives are computed before Adam normalization and trust projection.',pooled=pooled,locals=locals_))
    print(pooled,'nonzero rows',len(valid),'negative cosine',sum(x['cos_base_extra']<0 for x in valid),'median ratio',np.median([x['extra_base_ratio'] for x in valid]))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
