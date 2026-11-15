"""Keeps the repo-feature deck (docs/repo-feature/build_deck.py) in sync
with the code, as CLAUDE.md requires: every public ``graphrag`` name and
every example script must appear in the deck, and every class-like name or
``.py`` file the deck mentions must still exist.

When one of these fails after a code change, update ``SLIDES`` in
``build_deck.py``; don't loosen the test.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pptx import Presentation

import graphrag

ROOT = Path(__file__).resolve().parents[2]
DECK_SCRIPT = ROOT / "docs" / "repo-feature" / "build_deck.py"
CODE_DIRS = ("src", "examples", "tests", "docs")

# CamelCase words in the deck that are names of things, not code.
NOT_CODE = {"GraphRAG", "OpenAI"}


def _load_deck_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_deck", DECK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _public_names() -> list[tuple[str, str]]:
    """(package, name) for every public name in a ``graphrag`` module:
    its ``__all__``, or for a plain module without one (``settings``), the
    classes it defines.
    """
    names = []
    for info in pkgutil.iter_modules(graphrag.__path__):
        module = importlib.import_module(f"graphrag.{info.name}")
        exported = getattr(module, "__all__", None)
        if exported is None:
            exported = [
                name
                for name, obj in vars(module).items()
                if isinstance(obj, type)
                and obj.__module__ == module.__name__
                and not name.startswith("_")
            ]
        names.extend((info.name, name) for name in exported)
    return names


def _python_files() -> list[Path]:
    return [path for directory in CODE_DIRS for path in (ROOT / directory).rglob("*.py")]


def _defined_names() -> set[str]:
    """Every class, function and assigned name in the repo's own code."""
    names: set[str] = set()
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


@pytest.fixture(scope="module")
def deck() -> ModuleType:
    return _load_deck_module()


@pytest.fixture(scope="module")
def deck_path(deck, tmp_path_factory) -> Path:
    return deck.build_deck(tmp_path_factory.mktemp("deck") / "repo-feature.pptx")


@pytest.fixture(scope="module")
def deck_text(deck_path) -> str:
    """All text on the built slides: text boxes, shapes and table cells."""
    parts = []
    for slide in Presentation(str(deck_path)).slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if shape.has_table:
                parts.extend(cell.text for row in shape.table.rows for cell in row.cells)
    return "\n".join(parts)


def test_deck_builds_one_slide_per_spec(deck, deck_path):
    assert len(Presentation(str(deck_path)).slides) == len(deck.SLIDES)


@pytest.mark.parametrize(
    "package, name", _public_names(), ids=[f"{p}.{n}" for p, n in _public_names()]
)
def test_every_public_name_is_in_the_deck(deck_text, package, name):
    assert re.search(rf"(?<!\w){re.escape(name)}(?!\w)", deck_text), (
        f"graphrag.{package}.{name} isn't mentioned in the repo-feature deck; "
        "add it to SLIDES in docs/repo-feature/build_deck.py"
    )


@pytest.mark.parametrize("script", sorted((ROOT / "examples").glob("*.py")), ids=lambda p: p.name)
def test_every_example_script_is_in_the_deck(deck_text, script):
    assert script.name in deck_text, (
        f"examples/{script.name} isn't mentioned in the repo-feature deck; "
        "add it to SLIDES in docs/repo-feature/build_deck.py"
    )


def test_deck_names_no_missing_code(deck_text):
    """CamelCase words (an uppercase letter after the first character, and
    some lowercase) are taken to be class names and must exist in the code.
    """
    words = set(re.findall(r"\b[A-Z]\w*\b", deck_text))
    camel = {w for w in words if any(c.islower() for c in w) and any(c.isupper() for c in w[1:])}
    missing = sorted(camel - _defined_names() - NOT_CODE)
    assert not missing, f"The repo-feature deck names code that doesn't exist: {missing}"


def test_deck_names_no_missing_files(deck_text):
    existing = {path.name for path in _python_files()}
    missing = sorted(set(re.findall(r"\b\w+\.py\b", deck_text)) - existing)
    assert not missing, f"The repo-feature deck names .py files that don't exist: {missing}"
