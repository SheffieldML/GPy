import importlib
import warnings


def test_visualize_import_warns_and_points_at_vidden():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        mod = importlib.import_module("GPy.plotting.matplot_dep.visualize")
        importlib.reload(mod)
    assert any(
        issubclass(item.category, DeprecationWarning) and "vidden" in str(item.message)
        for item in caught
    )


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
