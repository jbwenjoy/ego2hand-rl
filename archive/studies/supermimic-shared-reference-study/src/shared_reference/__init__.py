"""Observation-only shared-reference data boundary (G1), not a physical reader."""
from .reader import (
    CatalogEntry,
    ObservationTrajectory,
    VerifiedCatalog,
    geometry_arrays,
    load_observation,
    open_catalog,
    read_three_field_export,
    validate_observation_arrays,
)

__all__ = ["CatalogEntry", "ObservationTrajectory", "VerifiedCatalog", "open_catalog",
           "load_observation", "geometry_arrays", "read_three_field_export", "validate_observation_arrays"]
