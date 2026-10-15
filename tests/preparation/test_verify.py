import json

import polars as pl
import pyarrow.parquet as pq

from shadow_trader import main

from .synthetic import write_ingestion, write_prepared_toml


def run(argv):
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def built(tmp_path, capsys):
    parent = write_ingestion(tmp_path)
    config = write_prepared_toml(tmp_path, parent.dataset_id)
    assert run(["prepare", "build", "--config", str(config), "-q"]) == 0
    first, *rest = capsys.readouterr().out.splitlines()
    ids = {line.split()[0]: line.split()[1] for line in rest}
    return config, first.split()[-1], ids, tmp_path / "prepared"


def verify(config, build_id):
    return run(["prepare", "verify", build_id, "--config", str(config)])


def snapshot(root):
    return {
        str(p): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def failed_lines(out):
    return [line for line in out.splitlines() if line.startswith("FAIL")]


def test_intact_build_passes_every_check_and_changes_nothing(tmp_path, capsys):
    config, build_id, _, _ = built(tmp_path, capsys)
    before = snapshot(tmp_path)

    assert verify(config, build_id) == 0

    out = capsys.readouterr().out
    assert out.count("PASS") >= 5 and not failed_lines(out)
    assert snapshot(tmp_path) == before


def test_corrupted_partition_fails_its_hash_check(tmp_path, capsys):
    config, build_id, ids, prepared = built(tmp_path, capsys)
    part = next((prepared / "datasets" / ids["train"]).rglob("*.parquet"))
    table = pq.read_table(part)
    frame = pl.DataFrame(table).with_columns(pl.col("close") + 1.0)
    pq.write_table(frame.to_arrow().cast(table.schema), part)

    assert verify(config, build_id) == 1

    assert any("train" in line for line in failed_lines(capsys.readouterr().out))


def test_broken_bound_fails_boundary_check(tmp_path, capsys):
    config, build_id, ids, prepared = built(tmp_path, capsys)
    path = prepared / "datasets" / ids["train"] / "manifest.json"
    manifest = json.loads(path.read_text())
    snapped = manifest["split"]["snapped_bounds"]
    snapped["validation_start"] = snapped["train_start"]
    path.write_text(json.dumps(manifest))

    assert verify(config, build_id) == 1

    assert any("boundary" in line for line in failed_lines(capsys.readouterr().out))


def test_removed_scaler_artifact_fails_scaler_check(tmp_path, capsys):
    config, build_id, _, prepared = built(tmp_path, capsys)
    for artifact in (prepared / "scalers").glob("*.json"):
        artifact.unlink()

    assert verify(config, build_id) == 1

    assert any("scaler" in line for line in failed_lines(capsys.readouterr().out))


def test_unknown_build_fails_and_bad_config_is_exit_2(tmp_path, capsys):
    config, _, _, _ = built(tmp_path, capsys)

    assert verify(config, "deadbeef") == 1
    assert run(["prepare", "verify", "x", "--config", str(tmp_path / "no.toml")]) == 2
