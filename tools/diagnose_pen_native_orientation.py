"""Compare actual full SO3 and directed shaft errors; no twist decomposition."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    if a.output.exists():raise ValueError('Fresh orientation diagnostic')
    rows=[]
    for path in a.reports:
        run=json.loads(path.read_text())
        for c in run['cases']:
            z=np.load(verify_file_pin(c['trace']));t=np.arange(1,len(z['post_qpos'])+1)*c['dt']
            target=Slerp(z['time_s'],Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]]))(t)
            actual=Rotation.from_quat(z['post_quaternion_wxyz'][:,[1,2,3,0]])
            shaft=np.arccos(np.clip(np.sum(target.as_matrix()[:,:,0]*actual.as_matrix()[:,:,0],axis=1),-1,1))
            full=(actual*target.inv()).magnitude();qa=c['model_layout']['object_qposadr']
            initial=Rotation.from_quat(z['initial_qpos'][qa+3:qa+7][[1,2,3,0]])
            ref0=Rotation.from_quat(z['quaternion_wxyz'][0][[1,2,3,0]])
            rms=float(np.sqrt(np.mean(full**2)))
            checks=dict(original_full_orientation_RMS=abs(rms-c['summary']['metrics']['angle_RMS_rad'])<1e-12,
                directed_shaft_is_lower_bound=bool(np.all(shaft<=full+1e-7)),original_full_orientation_max=abs(float(full.max())-c['summary']['metrics']['angle_max_rad'])<1e-12)
            rows.append(dict(run=file_pin(path),mode=c['name'],dt=c['dt'],trace=c['trace'],checks=checks,
                initial_full_deg=float(np.rad2deg((initial*ref0.inv()).magnitude())),shaft_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(shaft**2)))),full_RMS_deg=float(np.rad2deg(rms))))
    passed=sum(sum(c['checks'].values()) for c in rows);total=sum(len(c['checks']) for c in rows)
    a.output.mkdir();write(a.output/'report.json',dict(schema='pen_native_orientation_error:2',source=file_pin(__file__),cases=rows,passed=passed,total=total,
        native_integrations=0,neural_updates=0,scope='Actual recorded full SO3 and directed shaft errors vs original Slerp reference. Shaft is not an axial twist decomposition; no new task/policy qualification.'))
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[passed,total])))
    if passed!=total:raise ValueError('Saved orientation interpretation differs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
