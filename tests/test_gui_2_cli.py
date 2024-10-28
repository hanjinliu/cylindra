from pathlib import Path

import pytest

from cylindra import _config

from ._const import PROJECT_DIR_13PF, PROJECT_DIR_14PF, TEST_DIR


def test_help(run_cli):
    # test help
    for cmd in [
        "average",
        "config",
        "find",
        "new",
        "open",
        "plugin",
        "preview",
        "run",
        "workflow",
    ]:
        run_cli("cylindra", cmd, "--help")


def test_start_gui(run_cli):
    run_cli("cylindra")


def test_preview(run_cli):
    run_cli("cylindra", "preview", PROJECT_DIR_14PF / "project.json")
    run_cli("cylindra", "preview", PROJECT_DIR_14PF / "project.json", "--gui")
    run_cli("cylindra", "preview", PROJECT_DIR_14PF / "test_tar.tar")
    run_cli("cylindra", "preview", PROJECT_DIR_14PF / "test_zip.zip")
    run_cli("cylindra", "preview", PROJECT_DIR_14PF / "test_tar.tar::project.json")
    run_cli("cylindra", "preview", PROJECT_DIR_14PF / "test_tar.tar::Mole-0.csv")
    with pytest.raises(ValueError):
        run_cli("cylindra", "preview", PROJECT_DIR_14PF / "script.py")
    with pytest.raises(FileNotFoundError):
        run_cli("cylindra", "preview", PROJECT_DIR_14PF.with_name("NOT_EXISTS"))


def test_new_and_open(run_cli, tmpdir):
    run_cli(
        "cylindra", "new",
        Path(tmpdir) / "test-project",
        "--image", TEST_DIR / "14pf_MT.tif",
        "--multiscales", "1", "2",
        "--missing_wedge", "-60", "50",
        "--molecules", PROJECT_DIR_14PF / "Mole-*",
    )  # fmt: skip
    run_cli("cylindra", "open", Path(tmpdir) / "test-project")


def test_config(run_cli, tmpdir):
    run_cli("cylindra", "config", "--list")
    run_cli("cylindra", "config", PROJECT_DIR_13PF)
    run_cli("cylindra", "config", PROJECT_DIR_14PF / "script.py", "--remove")
    run_cli("cylindra", "config", PROJECT_DIR_14PF, "--remove")

    with _config.patch_config_dir(tmpdir):
        from cylindra.components.spline._config import SplineConfig

        d = Path(tmpdir) / "temp"
        d.mkdir()
        config_path = d / "temp-config.json"
        SplineConfig().to_file(config_path)
        run_cli("cylindra", "config", config_path, "--import")


def test_average(run_cli, tmpdir):
    tmpdir = Path(tmpdir)
    run_cli(
        "cylindra", "average",
        TEST_DIR / "test_project_*",
        "--molecules", "Mole-*",
        "--size", "10.0",
        "--output", tmpdir / "test.tif",
        "--filter", "col('nth') % 2 == 0",
        "--split", "--seed", "123",
    )  # fmt: skip


def test_run(run_cli, tmpdir):
    run_cli("cylindra", "run", PROJECT_DIR_14PF, "--headless")
    run_cli(
        "cylindra",
        "run",
        PROJECT_DIR_14PF,
        "--headless",
        "-o",
        Path(tmpdir) / "test.tar",
    )


def test_find(run_cli):
    run_cli("cylindra", "find", "**/*.zip")
    run_cli("cylindra", "find", "**/*.zip", "--called", "register_path")
    run_cli("cylindra", "find", "**/*.zip", "--called", "ui.register_path", "--abs")
    run_cli(
        "cylindra",
        "find",
        "**/*.zip",
        "--props\"col('npf')==13\"\"col('start')==3\"",
    )
    run_cli(
        "cylindra",
        "find",
        "**/*.zip",
        "--date-before",
        "251014",
        "--date-after",
        "150101",
    )


def test_workflow(run_cli, tmpdir):
    code = "import numpy as np\ndef main(ui):\n    print(ui.default_config)\n"

    with _config.patch_workflow_path(tmpdir):
        Path(tmpdir).joinpath("test-cli.py").write_text(code)
        run_cli("cylindra", "workflow")
        run_cli("cylindra", "workflow", "test-cli.py")
    run_cli("cylindra", "workflow", "--list")


def test_plugin(run_cli, tmpdir):
    run_cli("cylindra", "plugin", "list")
    run_cli("cylindra", "plugin", "new", tmpdir)
