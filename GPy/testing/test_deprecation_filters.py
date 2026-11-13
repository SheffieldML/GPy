# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""GPy must not silence DeprecationWarning for the whole process (CIP-0010)."""

import subprocess
import sys
import textwrap


def test_import_gpy_does_not_reinstall_deprecationwarning_ignore():
    """After warnings are enabled, importing GPy must not blanket-ignore them.

    CPython already ignores ``DeprecationWarning`` by default. The bug was that
    ``GPy/__init__.py`` called ``filterwarnings("ignore", ...)`` on every
    import, which re-silenced deprecations for users who had turned them on
    (``python -Wd``, ``PYTHONWARNINGS=default``, or an explicit filter) and
    for the rest of the process. CIP-0010 / REQ-0003 / Discussion #1297.

    Run in a subprocess so we do not purge/reimport ``GPy`` mid-suite (that
    dual-loads classes and breaks later tests).
    """
    code = textwrap.dedent(
        """        import warnings
        warnings.resetwarnings()
        warnings.simplefilter("default")
        assert not any(
            f[0] == "ignore" and f[2] is DeprecationWarning for f in warnings.filters
        ), "precondition: no DeprecationWarning ignore before import"
        import GPy  # noqa: F401
        gpy_style_blanket = [
            f
            for f in warnings.filters
            if f[0] == "ignore"
            and f[1] is None
            and f[3] is None
            and f[2] is DeprecationWarning
        ]
        assert not gpy_style_blanket, gpy_style_blanket
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
