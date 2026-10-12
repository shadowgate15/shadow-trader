"""Dataset storage: the write-once raw cache.

Raw cache layout: ``<data_root>/raw/<source>/<instrument>/<period>.<ext>``
with a ``<file>.meta.json`` sidecar. Files are byte-exact and never
overwritten or pruned; a corrected re-download is a new file
(``<period>.r2.<ext>``, ``<period>.r3.<ext>``, ...) whose sidecar names the
file it supersedes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


class RawCache:
    def __init__(self, data_root: Path) -> None:
        self._root = Path(data_root) / "raw"

    def path_for(
        self, source: str, instrument: str, period: str, ext: str, revision: int = 1
    ) -> Path:
        suffix = "" if revision == 1 else f".r{revision}"
        return self._root / source / instrument / f"{period}{suffix}.{ext}"

    def has(self, source: str, instrument: str, period: str, ext: str) -> bool:
        return self.path_for(source, instrument, period, ext).exists()

    def store(
        self,
        source: str,
        instrument: str,
        period: str,
        ext: str,
        data: bytes,
        *,
        request: Mapping[str, Any],
        fetched_at: datetime,
        http: Mapping[str, Any],
        supersede: bool = False,
    ) -> Path:
        """Write ``data`` and its sidecar; never replace an existing file.

        Raises ``FileExistsError`` if the period is cached and ``supersede``
        is false. With ``supersede`` the bytes go to the next free revision
        and the sidecar records the file it supersedes.
        """
        revision = 1
        path = self.path_for(source, instrument, period, ext)
        superseded: str | None = None
        if path.exists():
            if not supersede:
                raise FileExistsError(f"{path} is already cached")
            while path.exists():
                superseded = path.name
                revision += 1
                path = self.path_for(source, instrument, period, ext, revision)
        path.parent.mkdir(parents=True, exist_ok=True)

        sidecar = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "request": dict(request),
            "fetched_at": fetched_at.isoformat(),
            "http": dict(http),
            "supersedes": superseded,
        }
        # Sidecar first, data last: a data file without a sidecar never exists.
        meta_path = path.with_name(path.name + ".meta.json")
        _write_new(meta_path, json.dumps(sidecar, indent=2, sort_keys=True).encode())
        _write_new(path, data)
        return path


def _write_new(path: Path, data: bytes) -> None:
    """Create ``path`` with ``data`` atomically; fail if it already exists."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.link(tmp, path)  # raises FileExistsError rather than replacing
    finally:
        os.unlink(tmp)
