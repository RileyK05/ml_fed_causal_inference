"""The four packages stay isolated: projects import fedcore, never each other;
fedcore imports no project."""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGES = {"fedcore": "core", "fedci": "dml", "fedenc": "encoder", "fedcde": "cde"}


def _imports(pkg: str) -> set[str]:
    found = set()
    for path in (REPO / PACKAGES[pkg] / pkg).rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        found |= set(re.findall(r"^\s*(?:from|import)\s+(fed\w+)", text, re.MULTILINE))
    return found - {pkg}


@pytest.mark.parametrize("pkg", list(PACKAGES))
def test_only_core_is_shared(pkg):
    allowed = set() if pkg == "fedcore" else {"fedcore"}
    assert _imports(pkg) <= allowed, f"{pkg} imports {_imports(pkg) - allowed}"
