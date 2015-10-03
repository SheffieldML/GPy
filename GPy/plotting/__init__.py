# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

try:
    #===========================================================================
    # Load in your plotting library here and 
    # save it under the name plotting_library!
    # This is hooking the library in 
    # for the usage in GPy:
    from ..util.config import config
    lib = config.get('plotting', 'library')
    if lib == 'matplotlib':
        import matplotlib
        from .matplot_dep import plot_definitions
        plotting_library = plot_definitions.MatplotlibPlots()

    #===========================================================================
except (ImportError, NameError):
    raise
    import warnings
    warnings.warn(ImportWarning("{} not available, install newest version of {} for plotting".format(lib, lib)))
    config.set('plotting', 'library', 'none')

if config.get('plotting', 'library') is not 'none':
    # Inject the plots into classes here:

    # Already converted to new style:
    from . import gpy_plot
    
    from ..core import GP
    GP.plot_data = gpy_plot.data_plots.plot_data
    GP.plot_errorbars_trainset = gpy_plot.data_plots.plot_errorbars_trainset
    GP.plot_mean = gpy_plot.gp_plots.plot_mean
    GP.plot_confidence = gpy_plot.gp_plots.plot_confidence
    GP.plot_density = gpy_plot.gp_plots.plot_density
    GP.plot_samples = gpy_plot.gp_plots.plot_samples
    GP.plot = gpy_plot.gp_plots.plot
    GP.plot_magnificaion = gpy_plot.latent_plots.plot_magnification
    
    from ..core import SparseGP
    SparseGP.plot_inducing = gpy_plot.data_plots.plot_inducing
    
    from ..core import GPLVM
    GPLVM.plot_latent = gpy_plot.latent_plots.plot_latent
    
    from ..kern import Kern
    Kern.plot_covariance = gpy_plot.kern_plots.plot_kern
    
    # Variational plot!
    
    from . import matplot_dep
    # Still to convert to new style:
    GP.plot = matplot_dep.models_plots.plot_fit
    GP.plot_f = matplot_dep.models_plots.plot_fit_f
    
    GP.plot_magnification = matplot_dep.dim_reduction_plots.plot_magnification
        
