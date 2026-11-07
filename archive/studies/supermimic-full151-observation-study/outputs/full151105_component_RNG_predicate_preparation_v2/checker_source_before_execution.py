"""One alias/serialization counterexample; no real component/checkpoint reads."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'scripts/audit_WM_component_checkpoint_cpu_v2.py'
OUT=ROOT/'outputs/full151105_component_RNG_predicate_preparation_v2'


def main():
    OUT.mkdir(exist_ok=False);s=importlib.util.spec_from_file_location('component_rng_v2',SRC);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    pins={'auditor':a.pin(SRC),'checker':a.pin(__file__)}
    for k,p in [('auditor',SRC),('checker',Path(__file__))]:ast.parse(p.read_bytes());(OUT/(k+'_source_before_execution.py')).write_bytes(p.read_bytes())
    a.write(OUT/'source_freeze.json',{'source_pins':pins,'actual_component_reads':0,'reason':'Only replace cross-serialization pickle digest equality with exact saved raw fields; old source/reports retained.'})
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch_CPU':torch.arange(32,dtype=torch.uint8),'CUDA_initialized':False}
    equal_copy=copy.deepcopy(rng);raw=a.raw_rng_comparison(rng,equal_copy,torch)
    host=dict(rng);host['torch_CPU']=bytes(range(32))
    alias=pickle.dumps((host,host),protocol=5);distinct=pickle.dumps((host,copy.deepcopy(host)),protocol=5)
    changed=copy.deepcopy(rng);changed['numpy'][1][0]^=np.uint32(1)
    check=raw['all_raw_values_equal'] and alias!=distinct and not a.raw_rng_comparison(rng,changed,torch)['all_raw_values_equal'] and not torch.cuda.is_initialized()
    np.savez(OUT/'synthetic_alias_pickle_bytes.npz',aliased=np.frombuffer(alias,np.uint8).copy(),equal_values_distinct_graph=np.frombuffer(distinct,np.uint8).copy())
    report={'schema':'full151105_component_raw_RNG_predicate_CPU_proof:1','source_pins':pins,'passed':int(check),'total':1,
      'checks':{'equal_raw_states_different_pickle_alias_graphs_pass_but_MT_onebit_change_rejects':bool(check)},
      'raw_comparison':raw,'actual_component_reads':0,'models_FW_AD_Env_physics_GPU':0,
      'scope':'Illustrates pickle object-graph memo sensitivity, not a unique diagnosis of the real in-memory versus torch.save/load representation change.'}
    a.write(OUT/'report.json',report);ready={'source_pins':pins,'cpu_report':a.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':1}}
    a.write(OUT/'READY.json',ready);print(json.dumps({'READY':a.pin(OUT/'READY.json'),**ready},indent=2));return 0 if check else 2


if __name__=='__main__':raise SystemExit(main())
