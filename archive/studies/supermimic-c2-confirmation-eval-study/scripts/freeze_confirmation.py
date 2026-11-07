"""Finalize only the new evaluation plans; no execution, training or GPU IO."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def main():
    path=ROOT/'configs/c2-confirmation-evaluation-v1.json';p=read(path)
    additions={'queue_cpu_checker':ROOT/'scripts/check_confirmation_queue_cpu.py',
      'qualification_runner':ROOT/'scripts/run_format103_qualification.py',
      'checkpoint103_header_checker':ROOT/'scripts/check_checkpoint103_cpu.py',
      'confirmation_analysis':ROOT/'scripts/analyze_confirmation.py',
      'confirmation_analysis_protocol':ROOT/'analysis_protocol_v1.json'}
    for key,target in additions.items():
        if target.is_file():p['source_pins'][key]=pin(target)
    for key,v in p['source_pins'].items():
        current=sha(v['path'])
        if Path(v['path']).is_relative_to(ROOT):v['sha256']=current
        else:assert current==v['sha256'],'Previously frozen external source changed: '+key
    write(path,p)
    qualification=copy.deepcopy(p);qualificationpath=ROOT/'configs/format103-qualification-v1.json'
    qualification.update(status='CPU_prepared_engineering_qualification_GPU_not_authorized',
      output_root=str(ROOT/'outputs/format103_qualification_v1'),qualification_job_id=p['jobs'][0]['job_id'],
      scope='Only the fixed first schedule job is invoked once for format103 engineering qualification; never pooled/reused in the27 confirmation jobs')
    for j in qualification['jobs']:
        out=Path(qualification['output_root'])/j['job_id'];j['output_dir']=str(out);j['wrapper_metadata_dir']=str(out)+'_wrapper'
        args=j['original_evaluator_args'];args[args.index('--output')+1]=str(out)
        command=j['command'];command[command.index('--plan')+1]=str(qualificationpath);command[command.index('--output')+1]=str(out)
    write(qualificationpath,qualification)
    write(ROOT/'outputs/preparation_v1/frozen_manifest.json',{'scope':'CPU prepared, GPU authorization external',
      'main_plan':pin(path),'qualification_plan':pin(qualificationpath),
      'sources':{k:v for k,v in p['source_pins'].items() if k in {'evaluator','worker','queue','pure_projection','lifecycle','checkpoint103_contract',*additions}},
      'audited_model_index':p['source_pins']['audited_model_index'],'frozen_schedule':p['source_pins']['frozen_schedule'],
      'GPU_started':False})
    print(json.dumps({'main_plan':pin(path),'qualification_plan':pin(qualificationpath),'source_pins':len(p['source_pins'])}))

if __name__=='__main__':main()
