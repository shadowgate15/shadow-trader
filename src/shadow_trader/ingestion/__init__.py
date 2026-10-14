"""Market data ingestion Subsystem.

The Python API (spec section 9): ``build_dataset(config, bar_size)``,
``open_dataset(data_root, dataset_id)``, ``LiveIngester(config, adapter).run()``
and ``BarBuilder`` (``push``, ``flush``; ``advance`` closes slots on a clock).
"""

from .backfill import build_dataset
from .bars import BarBuilder
from .dataset import Dataset, DatasetRef, open_dataset
from .live import LiveIngester

__all__ = [
    "BarBuilder",
    "Dataset",
    "DatasetRef",
    "LiveIngester",
    "build_dataset",
    "open_dataset",
]
