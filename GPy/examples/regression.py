# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Gaussian Processes regression examples
"""
import pylab as pb
import numpy as np
import GPy


def toy_rbf_1d():
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    data = GPy.util.datasets.toy_rbf_1d()

    # create simple GP model
    m = GPy.models.GP_regression(data['X'],data['Y'])

    # optimize
    m.ensure_default_constraints()
    m.optimize()
    # plot
    m.plot()
    print(m)
    return m

def rogers_girolami_olympics():
    """Run a standard Gaussian process regression on the Rogers and Girolami olympics data."""
    data = GPy.util.datasets.rogers_girolami_olympics()

    # create simple GP model
    m = GPy.models.GP_regression(data['X'],data['Y'])

    # optimize
    m.ensure_default_constraints()
    m.optimize()

    # plot
    m.plot(plot_limits = (1850, 2050))
    print(m)
    return m

def della_gatta_TRP63_gene_expression(number=942):
    """Run a standard Gaussian process regression on the della Gatta et al TRP63 Gene Expression data set for a given gene number."""


def toy_rbf_1d_50():
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    data = GPy.util.datasets.toy_rbf_1d_50()

    # create simple GP model
    m = GPy.models.GP_regression(data['X'],data['Y'])

    # optimize
    m.ensure_default_constraints()
    m.optimize()

    # plot
    m.plot()
    print(m)
    return m

def silhouette():
    """Predict the pose of a figure given a silhouette. This is a task from Agarwal and Triggs 2004 ICML paper."""
    data = GPy.util.datasets.silhouette()

    # create simple GP model
    m = GPy.models.GP_regression(data['X'],data['Y'])

    # optimize
    m.ensure_default_constraints()
    m.optimize()

    print(m)
    return m


def multiple_optima(gene_number=937,resolution=80, model_restarts=10, seed=10000):
    """Show an example of a multimodal error surface for Gaussian process regression. Gene 939 has bimodal behaviour where the noisey mode is higher."""

    # Contour over a range of length scales and signal/noise ratios.
    length_scales = np.linspace(0.1, 60., resolution)
    log_SNRs = np.linspace(-3., 4., resolution)

    data = GPy.util.datasets.della_gatta_TRP63_gene_expression(gene_number)
    # Sub sample the data to ensure multiple optima
    #data['Y'] = data['Y'][0::2, :]
    #data['X'] = data['X'][0::2, :]

    # Remove the mean (no bias kernel to ensure signal/noise is in RBF/white)
    data['Y'] = data['Y'] - np.mean(data['Y'])

    lls = GPy.examples.regression.contour_data(data, length_scales, log_SNRs, GPy.kern.rbf)
    pb.contour(length_scales, log_SNRs, np.exp(lls), 20)
    ax = pb.gca()
    pb.xlabel('length scale')
    pb.ylabel('log_10 SNR')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Now run a few optimizations
    models = []
    optim_point_x = np.empty(2)
    optim_point_y = np.empty(2)
    np.random.seed(seed=seed)
    for i in range(0, model_restarts):
        kern = GPy.kern.rbf(1, variance=np.random.exponential(1.), lengthscale=np.random.exponential(50.)) + GPy.kern.white(1,variance=np.random.exponential(1.))
        
        m = GPy.models.GP_regression(data['X'],data['Y'], kernel=kern)
        optim_point_x[0] = m.get('rbf_lengthscale')
        optim_point_y[0] = np.log10(m.get('rbf_variance')) - np.log10(m.get('white_variance'));
        
        # optimize
        m.ensure_default_constraints()
        m.optimize(xtol=1e-6,ftol=1e-6)

        optim_point_x[1] = m.get('rbf_lengthscale')
        optim_point_y[1] = np.log10(m.get('rbf_variance')) - np.log10(m.get('white_variance'));
        
        pb.arrow(optim_point_x[0], optim_point_y[0], optim_point_x[1]-optim_point_x[0], optim_point_y[1]-optim_point_y[0], label=str(i), head_length=1, head_width=0.5, fc='k', ec='k')
        models.append(m)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return (models, lls)

def contour_data(data, length_scales, log_SNRs, signal_kernel_call=GPy.kern.rbf):
    """Evaluate the GP objective function for a given data set for a range of signal to noise ratios and a range of lengthscales.

    :data_set: A data set from the utils.datasets director.
    :length_scales: a list of length scales to explore for the contour plot.
    :log_SNRs: a list of base 10 logarithm signal to noise ratios to explore for the contour plot.
    :signal_kernel: a kernel to use for the 'signal' portion of the data."""

    lls = []
    total_var = np.var(data['Y'])
    for log_SNR in log_SNRs:
        SNR = 10**log_SNR
        length_scale_lls = []
        for length_scale in length_scales:
            noise_var = 1.
            signal_var = SNR
            noise_var = noise_var/(noise_var + signal_var)*total_var
            signal_var = signal_var/(noise_var + signal_var)*total_var

            signal_kernel = signal_kernel_call(1, variance=signal_var, lengthscale=length_scale)
            noise_kernel = GPy.kern.white(1, variance=noise_var)
            kernel = signal_kernel + noise_kernel
            K = kernel.K(data['X'])
            total_var = (np.dot(np.dot(data['Y'].T,GPy.util.linalg.pdinv(K)[0]), data['Y'])/data['Y'].shape[0])[0,0]
            noise_var *= total_var
            signal_var *= total_var
            
            kernel = signal_kernel_call(1, variance=signal_var, lengthscale=length_scale) + GPy.kern.white(1, variance=noise_var)

            model = GPy.models.GP_regression(data['X'], data['Y'], kernel=kernel)
            model.constrain_positive('')
            length_scale_lls.append(model.log_likelihood())
        lls.append(length_scale_lls)
    return np.array(lls)
