"""Fixed original/projection, two-dt native pen diagnostic, no training."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.projected_pen_dynamics import evaluate_projected_pen


def main(args):
    if args.output.exists():raise ValueError('Fresh dynamics output')
    projection=json.loads(args.projection.read_text());cfgpath=verify_file_pin(projection['config']);cfg=tomllib.loads(cfgpath.read_text());case=projection['case']
    for pin in projection['sources']:verify_file_pin(pin)
    source_report=json.loads(verify_file_pin(projection['source']).read_text());base_cfg=tomllib.loads(verify_file_pin(source_report['config']).read_text())
    clip_path=(cfgpath.parent/cfg['source_clip']).resolve();clip=np.load(clip_path);qref=clip['reference_qpos'].astype(np.float64);proposal=np.load(verify_file_pin(projection['proposal']));source=np.load(verify_file_pin(case['trace']))
    args.output.mkdir();sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/projected_pen_dynamics.py'),file_pin(ROOT/'src/supermimic_research/mechanics/controlled_snapshot.py'),
        file_pin(ROOT/'src/supermimic_research/mechanics/compound_contacts.py'),file_pin(ROOT/'src/supermimic_research/mechanics/compound_holding.py'),file_pin(ROOT/'src/supermimic_research/runtime/controlled_evaluation.py')]
    intent=dict(schema='controlled_pen_projection_native_dynamics:1',projection=file_pin(args.projection),clip=file_pin(clip_path),sources=sources,neural_updates=0,
        configurations=['ORIGINAL_ZERO_RESIDUAL','PROJECTED_CONSTANT_RESIDUAL'],dt=[.005,.0025],source_reference_frames=369,planned_duration_s=7.36)
    (args.output/'intent.json').write_text(json.dumps(intent,indent=2)+'\n',encoding='utf8');cases=[]
    for name in intent['configurations']:
        for dt in intent['dt']:
            delta=np.zeros(20) if name=='ORIGINAL_ZERO_RESIDUAL' else proposal['delta20'];qpos=source['initial_qpos'] if name=='ORIGINAL_ZERO_RESIDUAL' else proposal['qpos'];folder=args.output/f'{name}_{dt}'
            result=evaluate_projected_pen(case,qref,delta,qpos,dt,base_cfg,folder);result.update(name=name,trace=file_pin(folder/'trace.npz'));cases.append(result)
            (folder/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            print(json.dumps(dict(name=name,dt=dt,duration=result['summary']['metrics']['actual_duration_s'],qualified=result['summary']['qualified'],termination=result['termination'])),flush=True)
    for pin in sources:verify_file_pin(pin)
    result=dict(**intent,cases=cases,total_native_steps=sum(c['native_steps'] for c in cases),
        scope='One predefined bounded geometry correction followed by declared constant residual baseline, native diagnostic at two dt. All original clocks/velocities retained, failures kept; not successful learning/ego generalization.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--projection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
