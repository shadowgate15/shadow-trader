import hashlib
import json
from datetime import UTC, datetime

import pytest

from shadow_trader.ingestion.dataset import RawCache

FETCHED_AT = datetime(2026, 10, 12, 9, 30, tzinfo=UTC)
REQUEST = {"date": "2024", "datemonth": "202403", "fxpair": "EURUSD"}
HTTP = {"status": 200, "content_length": 5}


def store(cache, data=b"PK-ok", **kwargs):
    return cache.store(
        "histdata",
        "EURUSD",
        "202403",
        "zip",
        data,
        request=REQUEST,
        fetched_at=FETCHED_AT,
        http=HTTP,
        **kwargs,
    )


def sidecar_of(path):
    return json.loads(path.with_name(path.name + ".meta.json").read_text())


def test_stores_bytes_exactly_at_the_spec_layout(tmp_path):
    cache = RawCache(tmp_path)

    path = store(cache, b"\x00\x01exact\xff")

    assert path == tmp_path / "raw" / "histdata" / "EURUSD" / "202403.zip"
    assert path.read_bytes() == b"\x00\x01exact\xff"
    assert cache.has("histdata", "EURUSD", "202403", "zip")


def test_sidecar_records_hash_request_fetch_time_and_http(tmp_path):
    data = b"some zip bytes"
    path = store(RawCache(tmp_path), data)

    sidecar = sidecar_of(path)

    assert sidecar["sha256"] == hashlib.sha256(data).hexdigest()
    assert sidecar["request"] == REQUEST
    assert sidecar["fetched_at"] == "2026-10-12T09:30:00+00:00"
    assert sidecar["http"] == HTTP
    assert sidecar["size"] == len(data)
    assert sidecar["supersedes"] is None


def test_a_cached_file_is_never_overwritten(tmp_path):
    cache = RawCache(tmp_path)
    path = store(cache, b"first")

    with pytest.raises(FileExistsError):
        store(cache, b"second")

    assert path.read_bytes() == b"first"


def test_corrected_redownload_is_a_new_file_noting_supersession(tmp_path):
    cache = RawCache(tmp_path)
    first = store(cache, b"first")

    second = store(cache, b"second", supersede=True)

    assert second != first
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    assert sidecar_of(second)["supersedes"] == "202403.zip"
    third = store(cache, b"third", supersede=True)
    assert third not in (first, second)
    assert sidecar_of(third)["supersedes"] == second.name
