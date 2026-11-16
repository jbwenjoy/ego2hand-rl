"""Lossless snapshot transport;payload SHA/size verified before deserialization."""
import gzip
import hashlib
import io
from pathlib import Path
import torch
from ..artifacts import file_pin,verify_file_pin

SCHEMA='byte_exact_owned_snapshot_gzip_transport:1'


def pack_bytes(path,payload):
    path=Path(path)
    if path.exists() or type(payload) is not bytes:raise ValueError('Fresh transport destination and explicit serialized bytes required')
    compressed=gzip.compress(payload,compresslevel=1,mtime=0);path.write_bytes(compressed)
    return dict(schema=SCHEMA,compression='GZIP_LEVEL1_MTIME0',compressed=file_pin(path),uncompressed_bytes=len(payload),uncompressed_sha256=hashlib.sha256(payload).hexdigest())


def unpack_bytes(record):
    if record.get('schema')!=SCHEMA or record.get('compression')!='GZIP_LEVEL1_MTIME0' or type(record.get('uncompressed_bytes')) is not int or record['uncompressed_bytes']<0:
        raise ValueError('Explicit exact-byte snapshot transport required')
    path=verify_file_pin(record['compressed']);expected=record['uncompressed_bytes']
    with gzip.open(path,'rb') as stream:
        payload=stream.read(expected+1)
        if len(payload)!=expected or stream.read(1):raise ValueError('Snapshot uncompressed size differs')
    if hashlib.sha256(payload).hexdigest()!=record['uncompressed_sha256']:raise ValueError('Snapshot uncompressed hash differs')
    return payload


def save_owned(path,snapshot):
    if not isinstance(snapshot,dict) or snapshot.get('schema')!='controlled_pen_owned_state:4' or type(snapshot.get('worlds')) is not int or snapshot['worlds']!=1 or snapshot['metadata'].get('fixed_frame_features')!='LIVE':
        raise ValueError('Owned native4 LIVE W1 snapshot required;no schema conversion')
    buffer=io.BytesIO();torch.save(snapshot,buffer);return pack_bytes(path,buffer.getvalue())


def load_owned(record):
    # There is no torch.load/native write until BOTH complete hashes pass.
    payload=unpack_bytes(record);snapshot=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    if not isinstance(snapshot,dict) or snapshot.get('schema')!='controlled_pen_owned_state:4' or type(snapshot.get('worlds')) is not int or snapshot['worlds']!=1 or snapshot['metadata'].get('fixed_frame_features')!='LIVE':raise ValueError('Transported snapshot identity differs')
    return snapshot
