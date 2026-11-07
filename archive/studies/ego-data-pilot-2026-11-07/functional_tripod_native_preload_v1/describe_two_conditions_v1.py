"""Describe saved measured preload routes; no model/physics/network calls."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
out=H/'saved_two_conditions_v1';out.mkdir(exist_ok=False);rows=[];parents=[]
reports=[H/'completed_run1/report.json',H/'completed_source1mm_run1/report.json']
for condition,path in zip(['exact','1mm algorithmic'],reports):
    r=read(path);parents.append(pin(path));assert len(r['candidates'])==1;c=r['candidates'][0]
    probes=[(v['route'],v['probe']) for v in c['routes'] if v['available']]
    probes.append(('bounded native-wrench search',c['final_repeat_probes'][0]));assert len(probes)==4
    for route,rec in probes:
        p=read(checked(rec));F=np.array(p['pad_only_wrench_world6'][:3]);M=np.array(p['pad_only_wrench_world6'][3:]);target=np.array(p['target_wrench_world6'])
        rows.append(dict(condition=condition,route=route,probe=rec,pad_Fn3_N=p['pad_normal_N3'],pad_force_world_N=F.tolist(),pad_moment_COM_world_Nm=M.tolist(),
            force_error_norm_N=float(np.linalg.norm(F-target[:3])),moment_error_norm_Nm=float(np.linalg.norm(M-target[3:])),
            target_upward_force_N=float(target[2]),other_hand_Fn_N=p['other_hand']['Fn'],table_Fn_N=p['table']['Fn'],
            measured_loaded_entry=p['measurements']['native_loaded_entry'],holding_qualified=False))
labels=['ctrl=q','normal 1N/pad','normal 3N/pad','bounded search'];fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
for k,condition in enumerate(['exact','1mm algorithmic']):
    these=[v for v in rows if v['condition']==condition];x=np.arange(4)+(k-.5)*.32
    vals=[[v['pad_force_world_N'][2] for v in these],[v['force_error_norm_N'] for v in these],[v['moment_error_norm_Nm'] for v in these]]
    for ax,y in zip(axes,vals):ax.bar(x,y,width=.3,label=condition)
for ax in axes:ax.set_xticks(np.arange(4),labels,rotation=18);ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8)
axes[0].axhline(1.962,color='black',linestyle=':',label='mg');axes[0].set_ylabel('designated pads upward force [N]');axes[0].legend(fontsize=8)
axes[1].set_ylabel('pad-only force error norm [N]');axes[2].set_ylabel('pad-only COM moment error norm [Nm]')
fig.suptitle('Fresh native measurements at two fixed compressed poses\nThree-pad loading is present; target wrench and dynamic holding are not qualified')
fig.savefig(out/'native_wrench_routes.png',dpi=150);plt.close(fig)
with (out/'measured_route_rows.json').open('x',encoding='utf8') as f:json.dump(rows,f,indent=2,allow_nan=False)
with (out/'report.json').open('x',encoding='utf8') as f:json.dump(dict(schema='two_condition_native_preload_saved_description:1',source=pin(__file__),producer_reports=parents,
    rows=8,native_NN_AD_training_calls=0,outputs={'rows':pin(out/'measured_route_rows.json'),'figure':pin(out/'native_wrench_routes.png')},
    scope='Four declared/optimized commands at each of two different fixed poses, measured by saved fresh native forward. No dynamic tracking, heldout control benefit, real-force calibration or force/moment mixed-unit norm.'),f,indent=2,allow_nan=False)
print(json.dumps({'report':pin(out/'report.json')}))
