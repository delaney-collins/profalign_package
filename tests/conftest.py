"""
conftest.py — shared fixtures and markers for profalign tests.

Markers:
    @pytest.mark.integration — requires HH-suite (hhalign) installed
    @pytest.mark.slow        — takes >30 seconds

Usage:
    pytest                          # unit tests only
    pytest -m integration           # integration tests only
    pytest -m "not integration"     # skip integration tests (default CI)
"""

import sys
import shutil
import pytest
from pathlib import Path

# Centralized sys.path setup — all tests import from lib/ via this
_lib_dir = str(Path(__file__).resolve().parent.parent / "lib")
_project_dir = str(Path(__file__).resolve().parent.parent)
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires HH-suite installed")
    config.addinivalue_line("markers", "slow: takes >30 seconds")


@pytest.fixture
def msa_dir():
    """Path to example MSA directory. Skips test if not found."""
    path = Path(__file__).resolve().parent.parent / "examples" / "msa"
    if not path.exists():
        pytest.skip(f"Example MSA directory not found: {path}")
    return path


@pytest.fixture
def has_hhalign():
    """Skip test if hhalign is not installed."""
    if shutil.which("hhalign") is None:
        pytest.skip("hhalign not found in PATH")
    return True
