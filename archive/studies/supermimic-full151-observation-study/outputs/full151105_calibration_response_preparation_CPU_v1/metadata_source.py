"""New full151 identity and the previously qualified atomic JSON writer."""
from pathlib import Path
import hashlib,importlib.util
from .io import jsonable

FORMAT_VERSION=105
MODE='rough_observation_reference_full151_v1'
VARIANT='legacy_H123_full151_SCP8_knownrough_v1'
DESCRIPTOR_SCHEMA='full151_knownrough_controller_input_descriptor:1'
BUNDLE_SCHEMA='full151_knownrough_input_bundle:1'
WRITER=Path('D:/workspace/supermimic-rough-prefix-learning-study/vendor/runtime_io.py')
WRITER_SHA='159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1'
if hashlib.sha256(WRITER.read_bytes()).hexdigest()!=WRITER_SHA:raise ValueError('Qualified atomic metadata writer changed')
spec=importlib.util.spec_from_file_location('_full151_qualified_atomic_json',WRITER)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,value):module.write_json(path,jsonable(value))
