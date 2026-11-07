"""Bounded Windows metadata rename retry; no payload or simulation callbacks."""
from __future__ import annotations
import math
import os
import time
import json
from pathlib import Path
import tempfile


def replace_with_bounded_retry(source, target, *, max_seconds=.75,
                               replace_fn=None, clock=None, sleep=None):
    """Rename the same prewritten temporary file; retry only permission denial.

    At most32replace calls, within a monotonic retry budget<=1s. A blocked OS
    syscall is outside Python's timing guarantee. No reserialization, file
    rewrite, physics/control callback, flag read, or process retry occurs here.
    Other errors propagate immediately; exhaustion propagates the last denial.
    """
    if type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 0<max_seconds<=1:
        raise ValueError('Metadata retry budget must be finite and in(0,1]seconds')
    replace_fn=os.replace if replace_fn is None else replace_fn
    clock=time.monotonic if clock is None else clock
    sleep=time.sleep if sleep is None else sleep
    started=clock();deadline=started+max_seconds;attempts=0;last_error=None
    while attempts<32:
        if attempts and clock()>=deadline:
            raise last_error
        attempts+=1
        try:
            replace_fn(source,target)
            return {'attempts':attempts,'elapsed_seconds':clock()-started,
                    'retry_budget_seconds':max_seconds}
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (None,5,32):
                raise
            last_error=error
            remaining=deadline-clock()
            if remaining<=0 or attempts>=32:
                raise
            sleep(min(.005*(2**min(attempts-1,4)),.05,remaining))
    raise last_error


def write_json(path, value, *, max_seconds=.75):
    """Compatible lifecycle.write replacement: one JSON payload and one temp.

    Same indent/ASCII/newline/allow_nan=False convention as the reviewed writer.
    Failed commits retain the temporary evidence. The caller's value is never
    updated and the same already written temporary is used for all retries.
    """
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(payload)
    replace_with_bounded_retry(temporary,path,max_seconds=max_seconds)
