# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
Gaussian Processes regression examples
"""
try:
    from matplotlib import pyplot as pb
except:
    pass
import numpy as np
import GPy

def olympic_marathon_men(optimize=True, plot=True):
    """Run a standard Gaussian process regression on the Olympic marathon data."""
    try:import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.olympic_marathon_men()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'], data['Y'])

    # set the lengthscale to be something sensible (defaults to 1)
    m.kern.lengthscale = 10.

    if optimize:
        m.optimize('bfgs', max_iters=200)
    if plot:
        m.plot(plot_limits=(1850, 2050))

    return m

def coregionalization_toy(optimize=True, plot=True):
    """
    A simple demonstration of coregionalization on two sinusoidal functions.
    """
    #build a design matrix with a column of integers indicating the output
    X1 = np.random.rand(50, 1) * 8
    X2 = np.random.rand(30, 1) * 5

    #build a suitable set of observed variables
    Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
    Y2 = np.sin(X2) + np.random.randn(*X2.shape) * 0.05 + 2.

    m = GPy.models.GPCoregionalizedRegression(X_list=[X1,X2], Y_list=[Y1,Y2])

    if optimize:
        m.optimize('bfgs', max_iters=100)

    if plot:
        slices = GPy.util.multioutput.get_slices([X1,X2])
        m.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})
        m.plot(fixed_inputs=[(1,1)],which_data_rows=slices[1],Y_metadata={'output_index':1},ax=pb.gca())
    return m

def coregionalization_sparse(optimize=True, plot=True):
    """
    A simple demonstration of coregionalization on two sinusoidal functions using sparse approximations.
    """
    #build a design matrix with a column of integers indicating the output
    X1 = np.random.rand(50, 1) * 8
    X2 = np.random.rand(30, 1) * 5

    #build a suitable set of observed variables
    Y1 = np.sin(X1) + np.random.randn(*X1.shape) * 0.05
    Y2 = np.sin(X2) + np.random.randn(*X2.shape) * 0.05 + 2.

    m = GPy.models.SparseGPCoregionalizedRegression(X_list=[X1,X2], Y_list=[Y1,Y2])

    if optimize:
        m.optimize('bfgs', max_iters=100)

    if plot:
        slices = GPy.util.multioutput.get_slices([X1,X2])
        m.plot(fixed_inputs=[(1,0)],which_data_rows=slices[0],Y_metadata={'output_index':0})
        m.plot(fixed_inputs=[(1,1)],which_data_rows=slices[1],Y_metadata={'output_index':1},ax=pb.gca())
        pb.ylim(-3,)

    return m

def epomeo_gpx(max_iters=200, optimize=True, plot=True):
    """
    Perform Gaussian process regression on the latitude and longitude data
    from the Mount Epomeo runs. Requires gpxpy to be installed on your system
    to load in the data.
    """
    try:import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.epomeo_gpx()
    num_data_list = []
    for Xpart in data['X']:
        num_data_list.append(Xpart.shape[0])

    num_data_array = np.array(num_data_list)
    num_data = num_data_array.sum()
    Y = np.zeros((num_data, 2))
    t = np.zeros((num_data, 2))
    start = 0
    for Xpart, index in zip(data['X'], range(len(data['X']))):
        end = start+Xpart.shape[0]
        t[start:end, :] = np.hstack((Xpart[:, 0:1],
                                    index*np.ones((Xpart.shape[0], 1))))
        Y[start:end, :] = Xpart[:, 1:3]

    num_inducing = 200
    Z = np.hstack((np.linspace(t[:,0].min(), t[:, 0].max(), num_inducing)[:, None],
                   np.random.randint(0, 4, num_inducing)[:, None]))

    k1 = GPy.kern.RBF(1)
    k2 = GPy.kern.Coregionalize(output_dim=5, rank=5)
    k = k1**k2

    m = GPy.models.SparseGPRegression(t, Y, kernel=k, Z=Z, normalize_Y=True)
    m.constrain_fixed('.*variance', 1.)
    m.inducing_inputs.constrain_fixed()
    m.Gaussian_noise.variance.constrain_bounded(1e-3, 1e-1)
    m.optimize(max_iters=max_iters,messages=True)

    return m

def multiple_optima(gene_number=937, resolution=80, model_restarts=10, seed=10000, max_iters=300, optimize=True, plot=True):
    """
    Show an example of a multimodal error surface for Gaussian process
    regression. Gene 939 has bimodal behaviour where the noisy mode is
    higher.
    """

    # Contour over a range of length scales and signal/noise ratios.
    length_scales = np.linspace(0.1, 60., resolution)
    log_SNRs = np.linspace(-3., 4., resolution)

    try:import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.della_gatta_TRP63_gene_expression(data_set='della_gatta',gene_number=gene_number)
    # data['Y'] = data['Y'][0::2, :]
    # data['X'] = data['X'][0::2, :]

    data['Y'] = data['Y'] - np.mean(data['Y'])

    lls = GPy.examples.regression._contour_data(data, length_scales, log_SNRs, GPy.kern.RBF)
    if plot:
        pb.contour(length_scales, log_SNRs, np.exp(lls), 20, cmap=pb.cm.jet)
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
        # kern = GPy.kern.RBF(1, variance=np.random.exponential(1.), lengthscale=np.random.exponential(50.))
        kern = GPy.kern.RBF(1, variance=np.random.uniform(1e-3, 1), lengthscale=np.random.uniform(5, 50))

        m = GPy.models.GPRegression(data['X'], data['Y'], kernel=kern)
        m.likelihood.variance = np.random.uniform(1e-3, 1)
        optim_point_x[0] = m.rbf.lengthscale
        optim_point_y[0] = np.log10(m.rbf.variance) - np.log10(m.likelihood.variance);

        # optimize
        if optimize:
            m.optimize('scg', xtol=1e-6, ftol=1e-6, max_iters=max_iters)

        optim_point_x[1] = m.rbf.lengthscale
        optim_point_y[1] = np.log10(m.rbf.variance) - np.log10(m.likelihood.variance);

        if plot:
            pb.arrow(optim_point_x[0], optim_point_y[0], optim_point_x[1] - optim_point_x[0], optim_point_y[1] - optim_point_y[0], label=str(i), head_length=1, head_width=0.5, fc='k', ec='k')
        models.append(m)

    if plot:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return m # (models, lls)

def _contour_data(data, length_scales, log_SNRs, kernel_call=GPy.kern.RBF):
    """
    Evaluate the GP objective function for a given data set for a range of
    signal to noise ratios and a range of lengthscales.

    :data_set: A data set from the utils.datasets director.
    :length_scales: a list of length scales to explore for the contour plot.
    :log_SNRs: a list of base 10 logarithm signal to noise ratios to explore for the contour plot.
    :kernel: a kernel to use for the 'signal' portion of the data.
    """

    lls = []
    total_var = np.var(data['Y'])
    kernel = kernel_call(1, variance=1., lengthscale=1.)
    model = GPy.models.GPRegression(data['X'], data['Y'], kernel=kernel)
    for log_SNR in log_SNRs:
        SNR = 10.**log_SNR
        noise_var = total_var / (1. + SNR)
        signal_var = total_var - noise_var
        model.kern['.*variance'] = signal_var
        model.likelihood.variance = noise_var
        length_scale_lls = []

        for length_scale in length_scales:
            model['.*lengthscale'] = length_scale
            length_scale_lls.append(model.log_likelihood())

        lls.append(length_scale_lls)

    return np.array(lls)


def olympic_100m_men(optimize=True, plot=True):
    """Run a standard Gaussian process regression on the Rogers and Girolami olympics data."""
    try:import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.olympic_100m_men()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'], data['Y'])

    # set the lengthscale to be something sensible (defaults to 1)
    m.rbf.lengthscale = 10

    if optimize:
        m.optimize('bfgs', max_iters=200)

    if plot:
        m.plot(plot_limits=(1850, 2050))
    return m

def toy_rbf_1d(optimize=True, plot=True):
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    try:import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.toy_rbf_1d()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'], data['Y'])

    if optimize:
        m.optimize('bfgs')
    if plot:
        m.plot()

    return m

def toy_rbf_1d_50(optimize=True, plot=True):
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    try:import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.toy_rbf_1d_50()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'], data['Y'])

    if optimize:
        m.optimize('bfgs')
    if plot:
        m.plot()

    return m

def toy_poisson_rbf_1d_laplace(optimize=True, plot=True):
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    optimizer='scg'
    x_len = 100
    X = np.linspace(0, 10, x_len)[:, None]
    f_true = np.random.multivariate_normal(np.zeros(x_len), GPy.kern.RBF(1).K(X))
    Y = np.array([np.random.poisson(np.exp(f)) for f in f_true])[:,None]

    kern = GPy.kern.RBF(1)
    poisson_lik = GPy.likelihoods.Poisson()
    laplace_inf = GPy.inference.latent_function_inference.Laplace()

    # create simple GP Model
    m = GPy.core.GP(X, Y, kernel=kern, likelihood=poisson_lik, inference_method=laplace_inf)

    if optimize:
        m.optimize(optimizer)
    if plot:
        m.plot()
        # plot the real underlying rate function
        pb.plot(X, np.exp(f_true), '--k', linewidth=2)

    return m

def toy_ARD(max_iters=1000, kernel_type='linear', num_samples=300, D=4, optimize=True, plot=True):
    # Create an artificial dataset where the values in the targets (Y)
    # only depend in dimensions 1 and 3 of the inputs (X). Run ARD to
    # see if this dependency can be recovered
    X1 = np.sin(np.sort(np.random.rand(num_samples, 1) * 10, 0))
    X2 = np.cos(np.sort(np.random.rand(num_samples, 1) * 10, 0))
    X3 = np.exp(np.sort(np.random.rand(num_samples, 1), 0))
    X4 = np.log(np.sort(np.random.rand(num_samples, 1), 0))
    X = np.hstack((X1, X2, X3, X4))

    Y1 = np.asarray(2 * X[:, 0] + 3).reshape(-1, 1)
    Y2 = np.asarray(4 * (X[:, 2] - 1.5 * X[:, 0])).reshape(-1, 1)
    Y = np.hstack((Y1, Y2))

    Y = np.dot(Y, np.random.rand(2, D));
    Y = Y + 0.2 * np.random.randn(Y.shape[0], Y.shape[1])
    Y -= Y.mean()
    Y /= Y.std()

    if kernel_type == 'linear':
        kernel = GPy.kern.Linear(X.shape[1], ARD=1)
    elif kernel_type == 'rbf_inv':
        kernel = GPy.kern.RBF_inv(X.shape[1], ARD=1)
    else:
        kernel = GPy.kern.RBF(X.shape[1], ARD=1)
    kernel += GPy.kern.White(X.shape[1]) + GPy.kern.Bias(X.shape[1])
    m = GPy.models.GPRegression(X, Y, kernel)
    # len_prior = GPy.priors.inverse_gamma(1,18) # 1, 25
    # m.set_prior('.*lengthscale',len_prior)

    if optimize:
        m.optimize(optimizer='scg', max_iters=max_iters)

    if plot:
        m.kern.plot_ARD()

    return m

def toy_ARD_sparse(max_iters=1000, kernel_type='linear', num_samples=300, D=4, optimize=True, plot=True):
    # Create an artificial dataset where the values in the targets (Y)
    # only depend in dimensions 1 and 3 of the inputs (X). Run ARD to
    # see if this dependency can be recovered
    X1 = np.sin(np.sort(np.random.rand(num_samples, 1) * 10, 0))
    X2 = np.cos(np.sort(np.random.rand(num_samples, 1) * 10, 0))
    X3 = np.exp(np.sort(np.random.rand(num_samples, 1), 0))
    X4 = np.log(np.sort(np.random.rand(num_samples, 1), 0))
    X = np.hstack((X1, X2, X3, X4))

    Y1 = np.asarray(2 * X[:, 0] + 3)[:, None]
    Y2 = np.asarray(4 * (X[:, 2] - 1.5 * X[:, 0]))[:, None]
    Y = np.hstack((Y1, Y2))

    Y = np.dot(Y, np.random.rand(2, D));
    Y = Y + 0.2 * np.random.randn(Y.shape[0], Y.shape[1])
    Y -= Y.mean()
    Y /= Y.std()

    if kernel_type == 'linear':
        kernel = GPy.kern.Linear(X.shape[1], ARD=1)
    elif kernel_type == 'rbf_inv':
        kernel = GPy.kern.RBF_inv(X.shape[1], ARD=1)
    else:
        kernel = GPy.kern.RBF(X.shape[1], ARD=1)
    #kernel += GPy.kern.Bias(X.shape[1])
    X_variance = np.ones(X.shape) * 0.5
    m = GPy.models.SparseGPRegression(X, Y, kernel, X_variance=X_variance)
    # len_prior = GPy.priors.inverse_gamma(1,18) # 1, 25
    # m.set_prior('.*lengthscale',len_prior)

    if optimize:
        m.optimize(optimizer='scg', max_iters=max_iters)

    if plot:
        m.kern.plot_ARD()

    return m

def robot_wireless(max_iters=100, kernel=None, optimize=True, plot=True):
    """Predict the location of a robot given wirelss signal strength readings."""
    try:import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.robot_wireless()

    # create simple GP Model
    m = GPy.models.GPRegression(data['Y'], data['X'], kernel=kernel)

    # optimize
    if optimize:
        m.optimize(max_iters=max_iters)

    Xpredict = m.predict(data['Ytest'])[0]
    if plot:
        pb.plot(data['Xtest'][:, 0], data['Xtest'][:, 1], 'r-')
        pb.plot(Xpredict[:, 0], Xpredict[:, 1], 'b-')
        pb.axis('equal')
        pb.title('WiFi Localization with Gaussian Processes')
        pb.legend(('True Location', 'Predicted Location'))

    sse = ((data['Xtest'] - Xpredict)**2).sum()

    print(('Sum of squares error on test data: ' + str(sse)))
    return m

def silhouette(max_iters=100, optimize=True, plot=True):
    """Predict the pose of a figure given a silhouette. This is a task from Agarwal and Triggs 2004 ICML paper."""
    try:import pods
    except ImportError:
        print('pods unavailable, see https://github.com/sods/ods for example datasets')
        return
    data = pods.datasets.silhouette()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'], data['Y'])

    # optimize
    if optimize:
        m.optimize(messages=True, max_iters=max_iters)

    print(m)
    return m

def sparse_GP_regression_1D(num_samples=400, num_inducing=5, max_iters=100, optimize=True, plot=True, checkgrad=False):
    """Run a 1D example of a sparse GP regression."""
    # sample inputs and outputs
    X = np.random.uniform(-3., 3., (num_samples, 1))
    Y = np.sin(X) + np.random.randn(num_samples, 1) * 0.05
    # construct kernel
    rbf = GPy.kern.RBF(1)
    # create simple GP Model
    m = GPy.models.SparseGPRegression(X, Y, kernel=rbf, num_inducing=num_inducing)

    if checkgrad:
        m.checkgrad()

    if optimize:
        m.optimize('tnc', max_iters=max_iters)

    if plot:
        m.plot()

    return m

def sparse_GP_regression_2D(num_samples=400, num_inducing=50, max_iters=100, optimize=True, plot=True, nan=False):
    """Run a 2D example of a sparse GP regression."""
    np.random.seed(1234)
    X = np.random.uniform(-3., 3., (num_samples, 2))
    Y = np.sin(X[:, 0:1]) * np.sin(X[:, 1:2]) + np.random.randn(num_samples, 1) * 0.05
    if nan:
        inan = np.random.binomial(1,.2,size=Y.shape)
        Y[inan] = np.nan

    # construct kernel
    rbf = GPy.kern.RBF(2)

    # create simple GP Model
    m = GPy.models.SparseGPRegression(X, Y, kernel=rbf, num_inducing=num_inducing)

    # contrain all parameters to be positive (but not inducing inputs)
    m['.*len'] = 2.

    m.checkgrad()

    # optimize
    if optimize:
        m.optimize('tnc', messages=1, max_iters=max_iters)

    # plot
    if plot:
        m.plot()

    print(m)
    return m

def uncertain_inputs_sparse_regression(max_iters=200, optimize=True, plot=True):
    """Run a 1D example of a sparse GP regression with uncertain inputs."""
    fig, axes = pb.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # sample inputs and outputs
    S = np.ones((20, 1))
    X = np.random.uniform(-3., 3., (20, 1))
    Y = np.sin(X) + np.random.randn(20, 1) * 0.05
    # likelihood = GPy.likelihoods.Gaussian(Y)
    Z = np.random.uniform(-3., 3., (7, 1))

    k = GPy.kern.RBF(1)
    # create simple GP Model - no input uncertainty on this one
    m = GPy.models.SparseGPRegression(X, Y, kernel=k, Z=Z)

    if optimize:
        m.optimize('scg', messages=1, max_iters=max_iters)

    if plot:
        m.plot(ax=axes[0])
        axes[0].set_title('no input uncertainty')
    print(m)

    # the same Model with uncertainty
    m = GPy.models.SparseGPRegression(X, Y, kernel=GPy.kern.RBF(1), Z=Z, X_variance=S)
    if optimize:
        m.optimize('scg', messages=1, max_iters=max_iters)
    if plot:
        m.plot(ax=axes[1])
        axes[1].set_title('with input uncertainty')
        fig.canvas.draw()

    print(m)
    return m

def simple_mean_function(max_iters=100, optimize=True, plot=True):
    """
    The simplest possible mean function. No parameters, just a simple Sinusoid.
    """
    #create  simple mean function
    mf = GPy.core.Mapping(1,1)
    mf.f = np.sin
    mf.update_gradients = lambda a,b: None

    X = np.linspace(0,10,50).reshape(-1,1)
    Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape)

    k =GPy.kern.RBF(1)
    lik = GPy.likelihoods.Gaussian()
    m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
    if optimize:
        m.optimize(max_iters=max_iters)
    if plot:
        m.plot(plot_limits=(-10,15))
    return m

def parametric_mean_function(max_iters=100, optimize=True, plot=True):
    """
    A linear mean function with parameters that we'll learn alongside the kernel
    """
    #create  simple mean function
    mf = GPy.core.Mapping(1,1)
    mf.f = np.sin

    X = np.linspace(0,10,50).reshape(-1,1)
    Y = np.sin(X) + 0.5*np.cos(3*X) + 0.1*np.random.randn(*X.shape) + 3*X

    mf = GPy.mappings.Linear(1,1)

    k =GPy.kern.RBF(1)
    lik = GPy.likelihoods.Gaussian()
    m = GPy.core.GP(X, Y, kernel=k, likelihood=lik, mean_function=mf)
    if optimize:
        m.optimize(max_iters=max_iters)
    if plot:
        m.plot()
    return m


def warped_gp_cubic_sine(max_iters=100):
    """
    A test replicating the cubic sine regression problem from
    Snelson's paper.
    """
    X = (2 * np.pi) * np.random.random(151) - np.pi
    Y = np.sin(X) + np.random.normal(0,0.2,151)
    Y = np.array([np.power(abs(y),float(1)/3) * (1,-1)[y<0] for y in Y])
    X = X[:, None]
    Y = Y[:, None]

    warp_k = GPy.kern.RBF(1)
    warp_f = GPy.util.warping_functions.TanhFunction(n_terms=2)
    warp_m = GPy.models.WarpedGP(X, Y, kernel=warp_k, warping_function=warp_f)
    warp_m['.*\.d'].constrain_fixed(1.0)
    m = GPy.models.GPRegression(X, Y)
    m.optimize_restarts(parallel=False, robust=True, num_restarts=5, max_iters=max_iters)
    warp_m.optimize_restarts(parallel=False, robust=True, num_restarts=5, max_iters=max_iters)
    #m.optimize(max_iters=max_iters)
    #warp_m.optimize(max_iters=max_iters)

    print(warp_m)
    print(warp_m['.*warp.*'])

    warp_m.predict_in_warped_space = False
    warp_m.plot(title="Warped GP - Latent space")
    warp_m.predict_in_warped_space = True
    warp_m.plot(title="Warped GP - Warped space")
    m.plot(title="Standard GP")
    warp_m.plot_warping()
    pb.show()
