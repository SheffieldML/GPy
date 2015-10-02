# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

try:
    from ..util.config import config
    lib = config.get('plotting', 'library')
    if lib == 'matplotlib':
        import matplotlib
        from . import matplot_dep as plotting_library
except (ImportError, NameError):
    import warnings
    warnings.warn(ImportWarning("{} not available, install newest version of {} for plotting").format(lib, lib))
    config.set('plotting', 'library', 'none')

if config.get('plotting', 'library') is not 'none':
    # Inject the plots into classes here:

    # Already converted to new style:
    from . import gpy_plot
    
    from ..core import GP
    GP.plot_data = gpy_plot.data_plots.plot_data

    # Still to convert to new style:
    GP.plot = plotting_library.models_plots.plot_fit
    GP.plot_f = plotting_library.models_plots.plot_fit_f
    GP.plot_density = plotting_library.models_plots.plot_density
    
    GP.plot_errorbars_trainset = plotting_library.models_plots.plot_errorbars_trainset
    GP.plot_magnification = plotting_library.dim_reduction_plots.plot_magnification
        
