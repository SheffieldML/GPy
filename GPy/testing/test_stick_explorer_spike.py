import importlib
import warnings

import numpy as np


def _deprecated_visualize_warnings(caught):
    return [
        item
        for item in caught
        if issubclass(item.category, DeprecationWarning)
        and "vidden" in str(item.message)
        and "discussions/1252" in str(item.message)
    ]


def test_visualize_import_warns_and_points_at_vidden():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        mod = importlib.import_module("GPy.plotting.matplot_dep.visualize")
        importlib.reload(mod)
    assert _deprecated_visualize_warnings(caught)
    message = str(_deprecated_visualize_warnings(caught)[0].message)
    assert "macOS" in message or "macos" in message.lower()
    assert "core inference library" in message


def test_visualize_constructor_warns_and_points_at_discussion():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        mod = importlib.import_module("GPy.plotting.matplot_dep.visualize")
        importlib.reload(mod)
        mod.data_show(np.zeros((1, 3)))
    assert _deprecated_visualize_warnings(caught)


def test_stick_explorer_spike_warns_and_points_at_vidden():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        try:
            importlib.import_module("GPy.examples.stick_explorer_spike")
        except ImportError as exc:
            assert "vidden" in str(exc)
            return
    assert any(
        issubclass(item.category, DeprecationWarning) and "vidden" in str(item.message)
        for item in caught
    )
