"""Shared fixtures / markers for the numerical stability track (CIP-0011)."""

import pytest

# Ensure the marker is always registered even if pyproject.toml is skipped.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "numerical: adversarial / extreme-input numerical stability (CIP-0011)",
    )
