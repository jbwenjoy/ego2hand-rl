"""Run isolated workers and evaluations, recording partial and final comparisons."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
MATRIX=[(256,32),(512,32),(1024,32),(1024,64),(1024,128),(2048,32),(4096,32)]


def write(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)


def run(args,log):
    with log.open('x',encoding='utf-8') as f:
        return subprocess.run([sys.executable,'-X','utf8','-u',*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode


def main(out):
    out=out.resolve();results=[];read=lambda p:json.loads(p.read_bytes())
    for w,n in MATRIX:
        label=f'W{w}_U{n}';folder=out/label
        write(out/'status.json',dict(state='training',case=label))
        rc=run(['tools/experiment_pen_batch_updates.py',w,n,folder,out/'parent.json'],out/(label+'.log'))
        record=dict(case=label,training_exit=rc,evaluations=[])
        if rc==0:
            stats=read(folder/'training_report.json');record['training']=stats
            for actor in stats['actors']:
                actor_path=Path(actor['path']);target=folder/('eval_'+actor_path.stem)
                write(out/'status.json',dict(state='evaluating',case=label,actor=actor_path.stem))
                rc_eval=run(['tools/evaluate_pen_scaled.py',actor_path,target,3],folder/(target.name+'.log'))
                ev=dict(actor=actor_path.stem,exit_code=rc_eval)
                if rc_eval==0:
                    summary=read(target/'report.json')
                    ev.update(successes=summary['successes'],total=summary['total'],cases=[])
                    for row in summary['cases']:
                        m=read(Path(row['report']['path']))['summary']['metrics']
                        ev['cases'].append(dict(valid_s=row['valid_s'],stop=row['stop'],angle_rms_deg=m['angle_RMS_rad']*180/3.141592653589793))
                record['evaluations'].append(ev)
        results.append(record);write(out/'comparison.json',results)
    write(out/'status.json',dict(state='finished',completed_training=sum(r['training_exit']==0 for r in results),total=len(results)))


if __name__=='__main__':main(Path(sys.argv[1]))
