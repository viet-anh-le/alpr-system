"""Root pytest configuration — registers custom markers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# api/main.py uses absolute 'core.*' imports (no 'api.' prefix) so 'api/'
# must be on sys.path when running from the project root.
_API_DIR = str(Path(__file__).resolve().parent / "api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)
_LPR_DIR = str(Path(__file__).resolve().parent / "LPRNet")
if _LPR_DIR not in sys.path:
    sys.path.insert(0, _LPR_DIR)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: fast unit tests with no external dependencies")
    config.addinivalue_line("markers", "integration: tests that require external services")
    config.addinivalue_line("markers", "e2e: end-to-end tests")
