"""Dataset preparation Subsystem.

The Python API: ``build_prepared(config, progress=NULL_PROGRESS)``,
``open_prepared(prepared_root, dataset_id)`` and
``compute_features(history, spec)``.
"""

from .build import BuildFailure, BuildRef, build_prepared
from .config import PreparedConfig, load_prepared_config
from .dataset import PreparedDataset, open_prepared
from .scaler import Scaler, ScalerError, load_scaler
from .spec import SPECS, FeatureSpec, compute_features

__all__ = [
    "SPECS",
    "BuildFailure",
    "BuildRef",
    "FeatureSpec",
    "PreparedConfig",
    "PreparedDataset",
    "Scaler",
    "ScalerError",
    "build_prepared",
    "compute_features",
    "load_prepared_config",
    "load_scaler",
    "open_prepared",
]
