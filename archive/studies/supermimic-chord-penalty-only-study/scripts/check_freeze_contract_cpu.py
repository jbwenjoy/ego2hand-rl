"""Exercise real run-source preparation and the auditor's file contract on CPU."""
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    entry=load('penalty_entry_contract',ROOT/'scripts/train_controlled_pen.py')
    config=json.loads((ROOT/'configs/training/penalty_only_10iter_smoke_seed1.json').read_text())
    out=ROOT/'outputs/entry_freeze_contract_cpu_v1';out.mkdir(parents=True,exist_ok=False)
    source=entry.freeze_source(Path(config['dex_root']),out)
    recipe,clips,provenance=entry.prepare_reference(config['reference'],out)
    entry.freeze_paired_protocol(config,out,provenance,source)
    sys.path.insert(0,str(ROOT/'src'))
    from penalty_study.followup import freeze
    details=freeze(config,ROOT,out,source)
    frozen=entry.freeze_contact_inputs(config['contact_objective'],out)
    frozen['penalty_wrapper_sha256']=details['wrapper_sha256'];frozen['followup_protocol_sha256']=details['protocol_sha256']
    entry.write_json(out/'source/contact_objective/frozen_inputs.json',frozen)
    model=entry.build_penalty_only_objective(frozen,'cpu')
    import torch
    assert all(buffer.device.type=='cpu' for buffer in model.buffers())
    auditor=load('penalty_audit_contract',ROOT/'scripts/audit_training_checkpoint.py')
    checks=auditor.followup_contract_checks(out,config,source,model.metadata)
    assert len(checks)==8 and all(checks.values())
    current=out/'source/train_controlled_pen.py';original=out/'source/original_train_controlled_pen.py'
    assert entry.digest(current)==entry.digest(ROOT/'scripts/train_controlled_pen.py')==source['script_sha256']
    assert entry.digest(original)=='823711c1a7c5876942506986741f20171b3c09a292bf3656bda17c1b5a3fd65a'
    current_bytes=current.read_bytes()
    try:
        current.write_bytes(original.read_bytes())
        broken=auditor.followup_contract_checks(out,config,source,model.metadata)
        assert broken['P_ONLY_original_trainer8237_preserved'] is False
    finally:current.write_bytes(current_bytes)
    missing=original.with_suffix('.missing_fixture')
    original.rename(missing)
    try:
        try:auditor.followup_contract_checks(out,config,source,model.metadata)
        except FileNotFoundError:missing_rejected=True
        else:missing_rejected=False
        assert missing_rejected
    finally:missing.rename(original)
    assert all(auditor.followup_contract_checks(out,config,source,model.metadata).values())
    report={'status':'passed','scope':'Actual source/reference/bundle freeze and auditor file-contract only; no simulator, checkpoint, optimizer or training was executed',
            'checks':checks,'entry_snapshot_preserved':True,'original8237_saved_separately':True,
            'overwrite_entry_fixture_rejected':True,'missing_original_filename_fixture_rejected':True,
            'entry_sha256':entry.digest(current),'original_sha256':entry.digest(original),'helper_sha256':entry.digest(ROOT/'src/penalty_study/followup.py'),
            'wrapper_sha256':details['wrapper_sha256'],'followup_protocol_sha256':details['protocol_sha256'],
            'reference_derived_sha256':provenance['derived_sha256'],'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
            'CUDA_initialized_flag':torch.cuda.is_initialized(),'CUDA_device_count':torch.cuda.device_count(),'CUDA_available':torch.cuda.is_available(),
            'all_model_buffers_CPU':True}
    assert report['CUDA_device_count']==0 and report['CUDA_available'] is False
    entry.write_json(out/'checks.json',report);entry.write_json(out/'constructed_metadata.json',model.metadata)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
