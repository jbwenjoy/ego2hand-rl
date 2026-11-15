"""Common read-only finite-action response summaries for loss ablations."""
import numpy as np
import torch

from .pen_branch_evaluation import physical_pair_derivative
from .pen_native_response import vector_comparison


def pooled_error(rows):
    if not rows:return dict(pairs=0)
    denominator=sum(row['target_L2']**2 for row in rows)
    return dict(pairs=len(rows),pooled_relative_L2_error=float(np.sqrt(
        sum((row['relative_L2_error']*row['target_L2'])**2 for row in rows)/max(denominator,1e-24))),
        median_pair_relative_error=float(np.median([row['relative_L2_error'] for row in rows])),
        max_pair_relative_error=max(row['relative_L2_error'] for row in rows),zero_response_comparator=1.)


def response_diagnostics(model,data,pairs,directions,scale,device):
    with torch.no_grad():pred=model.prediction(torch.tensor(data['x'],dtype=torch.float32,device=device)).cpu().numpy()
    rows=[]
    for pair in pairs:
        if not pair['eligible']:continue
        ip,im=pair['plus_index'],pair['minus_index'];epsilon=pair['epsilon']
        ad=physical_pair_derivative(model,data['x'][ip],data['x'][im],data['y'][ip],data['y'][im],directions[pair['direction']],epsilon,scale,device)
        native=(data['y'][ip]-data['y'][im])/(2*epsilon);secant=(pred[ip]-pred[im])/(2*epsilon)
        row={k:pair[k] for k in ['anchor','dt','direction','epsilon']};row['AD']=ad
        row['secant']={name:vector_comparison(secant[sl]/scale[sl],native[sl]/scale[sl]) for name,sl in [('object12',slice(40,52)),('wrench6',slice(78,84))]}
        rows.append(row)
    summaries={}
    for anchor in ['ALL',0,2,4,6]:
        chosen=[row for row in rows if anchor=='ALL' or row['anchor']==anchor]
        summaries[str(anchor)]={method:{block:pooled_error([row[method][block] for row in chosen]) for block in ['object12','wrench6']} for method in ['AD','secant']}
    return dict(rows=rows,summary=summaries)
