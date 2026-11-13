"""Exact resident model-array snapshots with explicit scalar compatibility."""
import dataclasses
import enum
import hashlib
import numpy as np


def model_leaves(model):
    """Traverse dataclasses and tuples; retain every array and scalar leaf."""
    import warp as wp
    arrays, scalars = {}, {}

    def visit(value, path):
        if isinstance(value, wp.array):
            arrays[path] = value
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name), f'{path}.{field.name}' if path else field.name)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f'{path}.{index}')
        elif isinstance(value, enum.Enum):
            scalars[path] = value.value
        elif value is None or isinstance(value, (bool, int, float, str)):
            scalars[path] = value
        else:
            raise ValueError(f'Unsupported model snapshot leaf {path}: {type(value)}')

    visit(model, '')
    return arrays, scalars


def capture_model(model):
    arrays, scalars = model_leaves(model)
    values = {k: v.numpy().copy() for k, v in arrays.items()}
    fingerprints = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in values.items()}
    return values, scalars, fingerprints


def restore_model_arrays(model, values, scalars):
    """Copy only into an independent model after checking its full layout.

    Scalar or callback changes require rebuilding a compatible model; they
    are never silently patched around a captured graph.
    """
    arrays, current = model_leaves(model)
    if current != scalars or set(arrays) != set(values):
        raise ValueError('Exact model scalar and array-key compatibility required')
    # Validate every field before any device mutation.
    for key, target in arrays.items():
        template = target.numpy()
        if template.shape != values[key].shape or template.dtype != values[key].dtype:
            raise ValueError(f'GPU model snapshot layout mismatch at {key}')
        if not np.isfinite(values[key]).all():
            raise ValueError(f'Nonfinite model snapshot at {key}')
    for key, target in arrays.items():
        if target.size:
            target.assign(values[key])
