"""Explicit metadata-only initialization of a separately identified backend."""
import copy

from ..learning.world_model import raw_tree_equal
from .shared_snapshot_transport import tree_digest, validate

FIELD = 'termination_backend'


def import_historical_state(state, target_metadata, contract):
    validate(state)
    if FIELD in state['metadata'] or FIELD not in target_metadata:
        raise ValueError('Historical state and explicit target metadata required')
    if target_metadata[FIELD] != contract:
        raise ValueError('Target termination identity differs')
    expected = copy.deepcopy(state['metadata']); expected[FIELD] = copy.deepcopy(contract)
    if expected != target_metadata:
        raise ValueError('Only the declared termination metadata addition is permitted')
    old_digest = tree_digest(state)
    result = dict(state, metadata=expected)
    unchanged = {k: raw_tree_equal(v, result[k]) for k, v in state.items() if k != 'metadata'}
    rollback = dict(result, metadata=state['metadata'])
    if not all(unchanged.values()) or not raw_tree_equal(rollback, state) or tree_digest(rollback) != old_digest:
        raise ValueError('Historical numerical/provenance tree changed')
    audit = dict(schema='pen_solver_rule_metadata_only_import:1',
                 original_tree_sha256=old_digest, imported_tree_sha256=tree_digest(result),
                 original_metadata=copy.deepcopy(state['metadata']), imported_metadata=copy.deepcopy(expected),
                 added_metadata_field=FIELD, unchanged_top_level_fields=unchanged,
                 rollback_exact=True, native_python_rng_unchanged=True,
                 scope='Explicit candidate initialization, not equivalent future dynamics or old-backend qualification.')
    return result, audit


def verify_imported_capture(original, captured, contract):
    imported, audit = import_historical_state(original, captured['metadata'], contract)
    # Capture has new source/engine receipts; they are verified separately.
    fields = ['native', 'python', 'rng', 'config', 'worlds', 'host_model_digest',
              'reference_signature', 'cache_metadata', 'metric_metadata', 'immutable',
              'immutable_arrays', 'immutable_scalars', 'exclusions', 'schema']
    checks = {k: raw_tree_equal(imported[k], captured[k]) for k in fields}
    if not all(checks.values()):
        raise ValueError('Imported capture differs from complete historical state: ' + repr(checks))
    return dict(import_audit=audit, readback_fields=checks)
