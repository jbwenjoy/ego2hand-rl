"""Compile a version-checked Windows mug-window helper in a fresh run directory."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import mujoco
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def main(out,vcvars):
    source=Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/native/mug_window.c'
    root=Path(mujoco.__file__).resolve().parent
    header,=(root/'include').rglob('mujoco.h')
    out.mkdir(exist_ok=False)
    setup=dict(os.environ,MUG_VCVARS=str(vcvars.resolve(strict=True)))
    raw=subprocess.check_output('cmd.exe /d /c call "%MUG_VCVARS%" >nul && set',
                                executable=os.environ['COMSPEC'],env=setup,text=True,errors='replace')
    env=dict(os.environ)
    for line in raw.splitlines():
        key,sep,value=line.partition('=')
        if sep and key:env[key.upper()]=value
    compiler=shutil.which('cl.exe',path=env['PATH'])
    if not compiler:raise RuntimeError('MSVC compiler unavailable in configured environment')
    result=subprocess.run([compiler,'/nologo','/O2','/std:c11','/LD',f'/I{header.parent.parent}',str(source),
                           '/link','/OUT:mug_window.dll'],cwd=out,env=env,capture_output=True,text=True,errors='replace')
    (out/'build.log').write_text(result.stdout+result.stderr,encoding='utf8')
    result.check_returncode()
    write_json(out/'report.json',dict(source=file_pin(source),builder=file_pin(__file__),header=file_pin(header),
        runtime=file_pin(root/'mujoco.dll'),compiler=file_pin(compiler),helper=file_pin(out/'mug_window.dll'),
        mujoco_version=mujoco.__version__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--vcvars',type=Path,required=True)
    a=p.parse_args();main(a.out.resolve(),a.vcvars)
