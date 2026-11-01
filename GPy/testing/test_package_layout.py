# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""Installed layout must not include dead Python 2 trees (#1251)."""

from pathlib import Path

import GPy


def test_installed_package_has_no_python2_trees():
    root = Path(GPy.__file__).resolve().parent
    assert not (root / "old_tests").exists()
    assert not (root / "kern" / "src" / "todo").exists()
    assert not (root / "testing" / "mpi_test__.py").exists()
