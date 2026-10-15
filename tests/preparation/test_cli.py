import os
import subprocess
import sys

from shadow_trader import main

from .synthetic import split_toml, write_ingestion, write_prepared_toml


def run(argv):
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def test_build_exits_0_and_prints_the_build_and_dataset_ids(tmp_path, capsys):
    parent = write_ingestion(tmp_path)
    path = write_prepared_toml(tmp_path, parent.dataset_id)

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 0

    first, *rest = capsys.readouterr().out.splitlines()
    assert first.startswith("built prepared build ")
    assert (tmp_path / "prepared/builds" / f"{first.split()[-1]}.json").is_file()
    roles = [
        line for line in rest if line.split()[0] in ("train", "validation", "test")
    ]
    assert [line.split()[0] for line in roles] == ["train", "validation", "test"]
    for line in roles:
        dataset_id = line.split()[1]
        manifest = tmp_path / "prepared/datasets" / dataset_id / "manifest.json"
        assert manifest.is_file()


def test_failed_boundary_check_exits_1_naming_it_and_writes_nothing(tmp_path, capsys):
    parent = write_ingestion(tmp_path)
    path = write_prepared_toml(
        tmp_path, parent.dataset_id, split=split_toml(episode_bars=5000)
    )

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 1

    assert "embargo_exceeds_dataset" in capsys.readouterr().err
    assert not (tmp_path / "prepared").exists()


def test_missing_episode_bars_is_a_config_error(tmp_path, capsys):
    parent = write_ingestion(tmp_path)
    path = write_prepared_toml(
        tmp_path, parent.dataset_id, split=split_toml(episode_bars=None)
    )

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 2

    assert "embargo" in capsys.readouterr().err


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


def test_several_instruments_build_on_the_shared_grid(tmp_path):
    parent = write_ingestion(tmp_path, instruments=("EURUSD", "GBPUSD"))
    path = write_prepared_toml(tmp_path, parent.dataset_id)

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 0


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
    # the build id and the three role / Dataset id pairs; the report path differs
    lines = [x for x in out.stdout.splitlines() if not x.startswith("report ")]
    return "\n".join(lines).split()


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


def test_build_prints_one_summary_screen(tmp_path, capsys):
    parent = write_ingestion(tmp_path)
    path = write_prepared_toml(tmp_path, parent.dataset_id)

    assert run(["prepare", "build", "--config", str(path)]) == 0

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    build_id = lines[0].split()[-1]
    by_key = {line.split()[0]: line.split() for line in lines[1:]}
    for role in ("train", "validation", "test"):
        _, dataset_id, _, rows, _, usable = by_key[role]
        assert (tmp_path / "prepared/datasets" / dataset_id).is_dir()
        assert int(rows) > 0 and usable.endswith("%")
    scaler_id = by_key["scaler"][1]
    assert (tmp_path / "prepared/scalers" / f"{scaler_id}.json").is_file()
    assert "boundary_check pass" in lines
    assert any(line.startswith("warnings ") for line in lines)
    report = by_key["report"][1]
    assert report == str(tmp_path / "prepared/builds" / f"{build_id}.report.json")
    assert "\x1b" not in captured.out  # result lines are plain text


def test_build_failure_names_the_code_and_the_failed_report(tmp_path, capsys):
    parent = write_ingestion(tmp_path)
    path = write_prepared_toml(
        tmp_path, parent.dataset_id, split=split_toml(episode_bars=5000)
    )

    assert run(["prepare", "build", "--config", str(path), "-q"]) == 1

    err = capsys.readouterr().err
    assert "embargo_exceeds_dataset" in err
    assert str(tmp_path / "prepared.failed") in err
