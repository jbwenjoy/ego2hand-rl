"""Standard-library validation of the independent P_ONLY follow-up contract."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate(config, workspace):
    workspace=Path(workspace)
    path=Path(config['followup_protocol_file'])
    if not path.is_absolute():path=workspace/path
    assert sha(path)==config['followup_protocol_sha256'],'Follow-up protocol changed'
    protocol=read(path)
    assert protocol['schema']=='supermimic_chord_penalty_only_followup:1'
    assert protocol['method']=='P_ONLY' and config['method']=='P_ONLY'
    assert protocol['objective']=={'support_coefficient':0.,'penetration_coefficient':.1,'caller_dt_s':.02,'formula':'-.1*penetration_prior'}
    assert protocol['n3_outcomes_known_before_followup'] is True
    assert config['seed'] in protocol['seeds']==[1,2,3]
    paired=workspace/config['paired_protocol_file']
    assert sha(paired)==protocol['inherited_paired_protocol_sha256']
    for name,pin in protocol['source_bindings'].items():
        assert sha(pin['path'])==pin['sha256'],f'Follow-up source changed: {name}'
    for relative,expected in protocol['vendored_chord_study_sha256'].items():
        assert sha(workspace/'src'/relative)==expected,f'Vendored source changed: {relative}'
    native=read(protocol['native_inventory']['path'])
    assert sha(protocol['native_inventory']['path'])==protocol['native_inventory']['sha256']
    assert all(sha(Path(native['dex_root'])/relative)==expected for relative,expected in native['files'].items())
    template=protocol['A2_templates'][str(config['seed'])]
    assert sha(template['path'])==template['sha256']
    base=read(template['path'])
    assert config['reference']==base['reference'] and config['contact_objective']==base['contact_objective']
    assert config['compile_fallback'] is False
    if config['run_role']=='primary_matched_arm':
        assert config['training']==base['training'] and config['max_attempt_seconds']==3600
    else:
        assert config['run_role']=='engineering_smoke'
        assert config['training']==protocol['engineering_smoke_training'] and config['max_attempt_seconds']==600
    return path,protocol


def freeze(config, workspace, output, inventory):
    path,protocol=validate(config,workspace)
    output=Path(output)
    shutil.copy2(path,output/'source/followup_protocol.json')
    for name in ('penalty_wrapper','original_trainer','entry_diff','allowed_changes'):
        pin=protocol['source_bindings'][name]
        filename='original_train_controlled_pen.py' if name=='original_trainer' else Path(pin['path']).name
        shutil.copy2(pin['path'],output/'source'/filename)
    # All wrapper/helper code is frozen, in addition to the unchanged12 base modules.
    import zipfile
    with zipfile.ZipFile(output/'source/penalty_study_source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((Path(workspace)/'src/penalty_study').glob('*.py')):
            archive.write(file,'penalty_study/'+file.name)
    details={'protocol_sha256':sha(path),'wrapper_sha256':protocol['source_bindings']['penalty_wrapper']['sha256'],
             'original_trainer_sha256':protocol['source_bindings']['original_trainer']['sha256'],
             'method':'P_ONLY','support_coefficient':0.,'penetration_coefficient':.1,'caller_dt_s':.02}
    inventory['penalty_only_followup']=details
    (output/'source/inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    return details
