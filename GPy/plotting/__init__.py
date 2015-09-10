# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

try:
    from . import matplot_dep
except (ImportError, NameError):
    # Matplotlib not available
    import warnings
    warnings.warn(ImportWarning("Matplotlib not available, install newest version of Matplotlib for plotting"))
    #sys.modules['matplotlib'] =
    #sys.modules[__name__+'.matplot_dep'] = ImportWarning("Matplotlib not available, install newest version of Matplotlib for plotting")