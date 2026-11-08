# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""Installed layout must not include dead trees (#1251, #930 / #1213)."""

from pathlib import Path

import GPy


def test_installed_package_has_no_python2_trees():
    root = Path(GPy.__file__).resolve().parent
    assert not (root / "old_tests").exists()
    assert not (root / "kern" / "src" / "todo").exists()
    assert not (root / "testing" / "mpi_test__.py").exists()


def test_installed_package_has_no_symbolic_ad_stack():
    """#930: sympy symbolic AD helpers were unused and not a declared dependency."""
    root = Path(GPy.__file__).resolve().parent
    assert not (root / "util" / "functions.py").exists()
    assert not (root / "util" / "ln_diff_erfs.py").exists()
    assert not (root / "util" / "symbolic.py").exists()
    assert not (root / "core" / "symbolic.py").exists()
    assert not (root / "kern" / "src" / "symbolic.py").exists()
