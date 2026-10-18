"""The pure layers never import gymnasium."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import shadow_trader.environment as environment

PACKAGE = Path(environment.__file__).parent
PURE_MODULES = [
    "__init__",
    "config",
    "engine",
    "ledger",
    "env_config",
    "reward",
    "market_data",
    "report",
]


def imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_pure_modules_do_not_import_gymnasium() -> None:
    for name in PURE_MODULES:
        assert "gymnasium" not in imported_roots(PACKAGE / f"{name}.py"), name


def test_importing_the_package_does_not_load_gymnasium() -> None:
    code = "import sys, shadow_trader.environment;sys.exit('gymnasium' in sys.modules)"
    assert subprocess.run([sys.executable, "-I", "-c", code]).returncode == 0


def test_env_exports_load_lazily() -> None:
    code = (
        "import sys, shadow_trader.environment as e;"
        "assert 'gymnasium' not in sys.modules;"
        "e.make_env; e.TradingEnv;"
        "assert 'gymnasium' in sys.modules"
    )
    assert subprocess.run([sys.executable, "-I", "-c", code]).returncode == 0
