"""Deprecated. The stick explorer moved to vidden.

Install https://github.com/lawrennd/vidden and run::

    from vidden.explore import stick_explorer
    stick_explorer()

or from that repository::

    panel serve examples/stick.py --show
"""

from __future__ import annotations

import warnings

_MSG = (
    "GPy.examples.stick_explorer_spike has moved to the vidden package. "
    "Install vidden (https://github.com/lawrennd/vidden) and import "
    "vidden.explore, or run `panel serve examples/stick.py --show` there."
)

warnings.warn(_MSG, DeprecationWarning, stacklevel=2)

try:
    from vidden.explore import stick_explorer
except ImportError as exc:
    raise ImportError(_MSG) from exc

__all__ = ["stick_explorer"]
