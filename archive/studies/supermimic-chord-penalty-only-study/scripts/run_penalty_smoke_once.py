"""One bounded GPU smoke under the shared physical-GPU lock; no retry."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    spec=importlib.util.spec_from_file_location('penalty_queue_helpers',ROOT/'scripts/run_matched_training_queue.py')
    queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
    plan=queue.read(ROOT/'configs/penalty-only-queue-v1.json');queue.validate(plan)
    proof=queue.read(ROOT/'outputs/penalty_cpu_factory_recheck_v2/checks.json')
    assert proof['passed'] and proof['tests_run']==2 and proof['source_sha256']['wrapper']==queue.digest(ROOT/'src/penalty_study/penalty_only_objective.py')
    config=ROOT/'configs/training/penalty_only_10iter_smoke_seed1.json'
    cfg=queue.read(config);run=ROOT/cfg['output_dir'];assert not run.exists()
    logs=ROOT/'outputs/penalty_only_10iter_smoke_launch_v1';logs.mkdir(exist_ok=False)
    record={'status':'starting','config_sha256':queue.digest(config),'source_entry_sha256':plan['trainer_sha256'],
            'followup_protocol_sha256':cfg['followup_protocol_sha256'],'shared_lock':plan['shared_physical_GPU_lock'],
            'inherited_CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'production':'native compile+TF32 unchanged, eager P_ONLY prior',
            'scope':'One10iter engineering run, no long training and no task-quality selection'}
    started=time.perf_counter();process=None
    try:
        with queue.gpu_queue_lock():
            queue.reject_external_GPU_work()
            with (logs/'stdout.log').open('w',encoding='utf-8') as out,(logs/'stderr.log').open('w',encoding='utf-8') as err,(logs/'gpu_telemetry.csv').open('w',newline='',encoding='utf-8') as telemetry:
                writer=csv.writer(telemetry);writer.writerow(['sample_utc','device_timestamp','used_MiB','free_MiB','utilization_pct'])
                record['Popen_prelaunch_utc']=queue.utc()
                process=subprocess.Popen([plan['python_executable'],'-u',plan['trainer'],'--config',str(config)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                record.update(status='running',launcher_pid=process.pid,Popen_postlaunch_utc=queue.utc())
                queue.write(logs/'launch.json',record);print(json.dumps(record),flush=True)
                while process.poll() is None:
                    assert time.perf_counter()-started<900,'Finite smoke process watchdog exceeded'
                    try:
                        runtime=queue.read(run/'runtime.json')
                        if 'actual_process_OS_observation' not in record:
                            record['actual_pid']=runtime['pid'];record['runtime_started_utc']=runtime['started_utc']
                            record['actual_process_OS_observation']=queue.observe_actual_process(runtime['pid'],process.pid,plan['trainer'],str(config))
                            queue.write(logs/'launch.json',record)
                    except (OSError,ValueError):pass
                    sample=subprocess.run(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                    writer.writerow([queue.utc(),*next(csv.reader([sample.stdout.strip()]))]);telemetry.flush();time.sleep(.2)
            record['training_process_exit_code']=process.returncode
            record['GPU_reserved_wall_seconds_to_process_exit']=time.perf_counter()-started
            assert process.returncode==0,f'Smoke training exited{process.returncode}'
            job={'output':str(run)}
            queue.audit(plan,job,10,logs/'checkpoint10_cpu_audit')
            record.update(status='passed',finished_utc=queue.utc(),reserved_wall_seconds_including_CPU_audit=time.perf_counter()-started)
            queue.write(logs/'launch.json',record);print(json.dumps(record,indent=2),flush=True)
    except BaseException as error:
        if process is not None:queue.stop_owned_child(process)
        record.update(status='failed',error=repr(error),finished_utc=queue.utc(),reserved_wall_seconds=time.perf_counter()-started)
        queue.write(logs/'launch.json',record);raise


if __name__=='__main__':main()
