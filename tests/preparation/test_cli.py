import os
import subprocess
import sys

from shadow_trader import main

from .synthetic import write_ingestion, write_prepared_toml


def run(argv):
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def test_build_exits_0_and_prints_the_dataset_id(tmp_path, capsys):
    parent = write_ingestion(tmp_path)
    path = write_prepared_toml(tmp_path, parent.dataset_id)

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("built prepared dataset ")
    dataset_id = out.split()[-1]
    assert (tmp_path / "prepared/datasets" / dataset_id / "manifest.json").is_file()


def test_missing_parent_dataset_exits_1(tmp_path, capsys):
    path = write_prepared_toml(tmp_path, "deadbeef")

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 1

    assert "deadbeef" in capsys.readouterr().err


def test_bar_size_mismatch_with_the_spec_exits_1(tmp_path, capsys):
    parent = write_ingestion(tmp_path, bar_size=15)
    path = write_prepared_toml(tmp_path, parent.dataset_id)

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 1

    assert "bar size mismatch" in capsys.readouterr().err
    assert not (tmp_path / "prepared").exists()


def test_several_instruments_are_a_build_failure(tmp_path, capsys):
    parent = write_ingestion(tmp_path, instruments=("EURUSD", "GBPUSD"))
    path = write_prepared_toml(tmp_path, parent.dataset_id)

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 1

    assert "exactly one" in capsys.readouterr().err


def test_bad_config_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.toml"
    path.write_text("[nope]\n")

    assert run(["prepare", "build", "--config", str(path)]) == 2
    assert run(["prepare", "build", "--config", str(tmp_path / "absent.toml")]) == 2
    assert "config error" in capsys.readouterr().err


def test_flags_override_only_the_config_path(tmp_path):
    path = write_prepared_toml(tmp_path, "abc")

    assert run(["prepare", "build", "--config", str(path), "--spec-version", "2"]) == 2


def build_in_subprocess(config_path, threads):
    env = {**os.environ, "POLARS_MAX_THREADS": str(threads)}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from shadow_trader import main; "
            "sys.exit(main(['prepare', 'build', '--config', sys.argv[1], '-q']))",
            str(config_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()[-1]


def test_dataset_hash_is_identical_at_one_and_many_threads(tmp_path):
    parent = write_ingestion(tmp_path)
    one_root, many_root = tmp_path / "one", tmp_path / "many"
    ids = []
    for root, threads in ((one_root, 1), (many_root, 4)):
        root.mkdir()
        path = write_prepared_toml(root, parent.dataset_id)
        path.write_text(
            path.read_text().replace(
                str(root / "ingestion"), str(tmp_path / "ingestion")
            )
        )
        ids.append(build_in_subprocess(path, threads))

    assert ids[0] == ids[1]
