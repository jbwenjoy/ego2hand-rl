"""Compare JSON audit representations; never coerce typed snapshot state."""
import json

from ..strict_json import scalar


def audit_json_equal(saved_json, typed_metadata):
    expected=json.loads(json.dumps(typed_metadata,allow_nan=False,default=scalar))
    return saved_json==expected
