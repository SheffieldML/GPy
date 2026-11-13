# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""GPy must not silence DeprecationWarning for the whole process (CIP-0010)."""

import importlib
import sys
import warnings


def _purge_gpy_modules():
    for name in list(sys.modules):
        if name == "GPy" or name.startswith("GPy."):
            del sys.modules[name]


def test_import_gpy_does_not_reinstall_deprecationwarning_ignore():
    """After warnings are enabled, importing GPy must not blanket-ignore them.

    CPython already ignores ``DeprecationWarning`` by default. The bug was that
    ``GPy/__init__.py`` called ``filterwarnings("ignore", ...)`` on every
    import, which re-silenced deprecations for users who had turned them on
    (``python -Wd``, ``PYTHONWARNINGS=default``, or an explicit filter) and
    for the rest of the process. CIP-0010 / REQ-0003 / Discussion #1297.
    """
    warnings.resetwarnings()
    warnings.simplefilter("default")
    assert not any(
        f[0] == "ignore" and f[2] is DeprecationWarning for f in warnings.filters
    ), "precondition: no DeprecationWarning ignore before import"

    _purge_gpy_modules()
    importlib.import_module("GPy")

    gpy_style_blanket = [
        f
        for f in warnings.filters
        if f[0] == "ignore"
        and f[1] is None
        and f[3] is None
        and f[2] is DeprecationWarning
    ]
    assert not gpy_style_blanket, (
        f"import GPy re-installed process-wide DeprecationWarning ignore: {gpy_style_blanket}"
    )
