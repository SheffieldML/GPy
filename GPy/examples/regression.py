# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
Gaussian Processes regression examples
"""
import pylab as pb
import numpy as np
import GPy


def toy_rbf_1d(optimizer='tnc', max_nb_eval_optim=100):
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    data = GPy.util.datasets.toy_rbf_1d()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'],data['Y'])

    # optimize
    m.ensure_default_constraints()
    m.optimize(optimizer, max_f_eval=max_nb_eval_optim)
    # plot
    m.plot()
    print(m)
    return m

def rogers_girolami_olympics(optim_iters=100):
    """Run a standard Gaussian process regression on the Rogers and Girolami olympics data."""
    data = GPy.util.datasets.rogers_girolami_olympics()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'],data['Y'])

    #set the lengthscale to be something sensible (defaults to 1)
    m['rbf_lengthscale'] = 10

    # optimize
    m.ensure_default_constraints()
    m.optimize(max_f_eval=optim_iters)

    # plot
    m.plot(plot_limits = (1850, 2050))
    print(m)
    return m

def toy_rbf_1d_50(optim_iters=100):
    """Run a simple demonstration of a standard Gaussian process fitting it to data sampled from an RBF covariance."""
    data = GPy.util.datasets.toy_rbf_1d_50()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'],data['Y'])

    # optimize
    m.ensure_default_constraints()
    m.optimize(max_f_eval=optim_iters)

    # plot
    m.plot()
    print(m)
    return m

def silhouette(optim_iters=100):
    """Predict the pose of a figure given a silhouette. This is a task from Agarwal and Triggs 2004 ICML paper."""
    data = GPy.util.datasets.silhouette()

    # create simple GP Model
    m = GPy.models.GPRegression(data['X'],data['Y'])

    # optimize
    m.ensure_default_constraints()
    m.optimize(messages=True,max_f_eval=optim_iters)

    print(m)
    return m

def coregionalisation_toy2(optim_iters=100):
    """
    A simple demonstration of coregionalisation on two sinusoidal functions.
    """
    X1 = np.random.rand(50,1)*8
    X2 = np.random.rand(30,1)*5
    index = np.vstack((np.zeros_like(X1),np.ones_like(X2)))
    X = np.hstack((np.vstack((X1,X2)),index))
    Y1 = np.sin(X1) + np.random.randn(*X1.shape)*0.05
    Y2 = np.sin(X2) + np.random.randn(*X2.shape)*0.05 + 2.
    Y = np.vstack((Y1,Y2))

    k1 = GPy.kern.rbf(1) + GPy.kern.bias(1)
    k2 = GPy.kern.Coregionalise(2,1)
    k = k1.prod(k2,tensor=True)
    m = GPy.models.GPRegression(X,Y,kernel=k)
    m.constrain_fixed('.*rbf_var',1.)
    #m.constrain_positive('.*kappa')
    m.ensure_default_constraints()
    m.optimize('sim',messages=1,max_f_eval=optim_iters)

    pb.figure()
    Xtest1 = np.hstack((np.linspace(0,9,100)[:,None],np.zeros((100,1))))
    Xtest2 = np.hstack((np.linspace(0,9,100)[:,None],np.ones((100,1))))
    mean, var,low,up = m.predict(Xtest1)
    GPy.util.plot.gpplot(Xtest1[:,0],mean,low,up)
    mean, var,low,up = m.predict(Xtest2)
    GPy.util.plot.gpplot(Xtest2[:,0],mean,low,up)
    pb.plot(X1[:,0],Y1[:,0],'rx',mew=2)
    pb.plot(X2[:,0],Y2[:,0],'gx',mew=2)
    return m

def coregionalisation_toy(optim_iters=100):
    """
    A simple demonstration of coregionalisation on two sinusoidal functions.
    """
    X1 = np.random.rand(50,1)*8
    X2 = np.random.rand(30,1)*5
    index = np.vstack((np.zeros_like(X1),np.ones_like(X2)))
    X = np.hstack((np.vstack((X1,X2)),index))
    Y1 = np.sin(X1) + np.random.randn(*X1.shape)*0.05
    Y2 = -np.sin(X2) + np.random.randn(*X2.shape)*0.05
    Y = np.vstack((Y1,Y2))

    k1 = GPy.kern.rbf(1)
    k2 = GPy.kern.Coregionalise(2,2)
    k = k1.prod(k2,tensor=True)
    m = GPy.models.GPRegression(X,Y,kernel=k)
    m.constrain_fixed('.*rbf_var',1.)
    #m.constrain_positive('kappa')
    m.ensure_default_constraints()
    m.optimize(max_f_eval=optim_iters)

    pb.figure()
    Xtest1 = np.hstack((np.linspace(0,9,100)[:,None],np.zeros((100,1))))
    Xtest2 = np.hstack((np.linspace(0,9,100)[:,None],np.ones((100,1))))
    mean, var,low,up = m.predict(Xtest1)
    GPy.util.plot.gpplot(Xtest1[:,0],mean,low,up)
    mean, var,low,up = m.predict(Xtest2)
    GPy.util.plot.gpplot(Xtest2[:,0],mean,low,up)
    pb.plot(X1[:,0],Y1[:,0],'rx',mew=2)
    pb.plot(X2[:,0],Y2[:,0],'gx',mew=2)
    return m


def coregionalisation_sparse(optim_iters=100):
    """
    A simple demonstration of coregionalisation on two sinusoidal functions using sparse approximations.
    """
    X1 = np.random.rand(500,1)*8
    X2 = np.random.rand(300,1)*5
    index = np.vstack((np.zeros_like(X1),np.ones_like(X2)))
    X = np.hstack((np.vstack((X1,X2)),index))
    Y1 = np.sin(X1) + np.random.randn(*X1.shape)*0.05
    Y2 = -np.sin(X2) + np.random.randn(*X2.shape)*0.05
    Y = np.vstack((Y1,Y2))

    num_inducing = 40
    Z = np.hstack((np.random.rand(num_inducing,1)*8,np.random.randint(0,2,num_inducing)[:,None]))

    k1 = GPy.kern.rbf(1)
    k2 = GPy.kern.Coregionalise(2,2)
    k = k1.prod(k2,tensor=True) + GPy.kern.white(2,0.001)

    m = GPy.models.SparseGPRegression(X,Y,kernel=k,Z=Z)
    m.constrain_fixed('.*rbf_var',1.)
    m.constrain_fixed('iip')
    m.constrain_bounded('noise_variance',1e-3,1e-1)
    m.ensure_default_constraints()
    m.optimize_restarts(5, robust=True, messages=1, max_f_eval=optim_iters)

    #plotting:
    pb.figure()
    Xtest1 = np.hstack((np.linspace(0,9,100)[:,None],np.zeros((100,1))))
    Xtest2 = np.hstack((np.linspace(0,9,100)[:,None],np.ones((100,1))))
    mean, var,low,up = m.predict(Xtest1)
    GPy.util.plot.gpplot(Xtest1[:,0],mean,low,up)
    mean, var,low,up = m.predict(Xtest2)
    GPy.util.plot.gpplot(Xtest2[:,0],mean,low,up)
    pb.plot(X1[:,0],Y1[:,0],'rx',mew=2)
    pb.plot(X2[:,0],Y2[:,0],'gx',mew=2)
    y = pb.ylim()[0]
    pb.plot(Z[:,0][Z[:,1]==0],np.zeros(np.sum(Z[:,1]==0))+y,'r|',mew=2)
    pb.plot(Z[:,0][Z[:,1]==1],np.zeros(np.sum(Z[:,1]==1))+y,'g|',mew=2)
    return m


def multiple_optima(gene_number=937,resolution=80, model_restarts=10, seed=10000, optim_iters=300):
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

    lls = GPy.examples.regression._contour_data(data, length_scales, log_SNRs, GPy.kern.rbf)
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
        #kern = GPy.kern.rbf(1, variance=np.random.exponential(1.), lengthscale=np.random.exponential(50.))
        kern = GPy.kern.rbf(1, variance=np.random.uniform(1e-3,1), lengthscale=np.random.uniform(5,50))

        m = GPy.models.GPRegression(data['X'],data['Y'], kernel=kern)
        m['noise_variance'] = np.random.uniform(1e-3,1)
        optim_point_x[0] = m['rbf_lengthscale']
        optim_point_y[0] = np.log10(m['rbf_variance']) - np.log10(m['noise_variance']);

        # optimize
        m.ensure_default_constraints()
        m.optimize('scg', xtol=1e-6, ftol=1e-6, max_f_eval=optim_iters)

        optim_point_x[1] = m['rbf_lengthscale']
        optim_point_y[1] = np.log10(m['rbf_variance']) - np.log10(m['noise_variance']);

        pb.arrow(optim_point_x[0], optim_point_y[0], optim_point_x[1]-optim_point_x[0], optim_point_y[1]-optim_point_y[0], label=str(i), head_length=1, head_width=0.5, fc='k', ec='k')
        models.append(m)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return m #(models, lls)

def _contour_data(data, length_scales, log_SNRs, kernel_call=GPy.kern.rbf):
    """Evaluate the GP objective function for a given data set for a range of signal to noise ratios and a range of lengthscales.

    :data_set: A data set from the utils.datasets director.
    :length_scales: a list of length scales to explore for the contour plot.
    :log_SNRs: a list of base 10 logarithm signal to noise ratios to explore for the contour plot.
    :kernel: a kernel to use for the 'signal' portion of the data."""

    lls = []
    total_var = np.var(data['Y'])
    kernel = kernel_call(1, variance=1., lengthscale=1.)
    Model = GPy.models.GPRegression(data['X'], data['Y'], kernel=kernel)
    for log_SNR in log_SNRs:
        SNR = 10.**log_SNR
        noise_var = total_var/(1.+SNR)
        signal_var = total_var - noise_var
        Model.kern['.*variance'] = signal_var
        Model['noise_variance'] = noise_var
        length_scale_lls = []

        for length_scale in length_scales:
            Model['.*lengthscale'] = length_scale
            length_scale_lls.append(Model.log_likelihood())

        lls.append(length_scale_lls)

    return np.array(lls)

def sparse_GP_regression_1D(N = 400, num_inducing = 5, optim_iters=100):
    """Run a 1D example of a sparse GP regression."""
    # sample inputs and outputs
    X = np.random.uniform(-3.,3.,(N,1))
    Y = np.sin(X)+np.random.randn(N,1)*0.05
    # construct kernel
    rbf =  GPy.kern.rbf(1)
    noise = GPy.kern.white(1)
    kernel = rbf + noise
    # create simple GP Model
    m = GPy.models.SparseGPRegression(X, Y, kernel, num_inducing=num_inducing)

    m.ensure_default_constraints()

    m.checkgrad(verbose=1)
    m.optimize('tnc', messages = 1, max_f_eval=optim_iters)
    m.plot()
    return m

def sparse_GP_regression_2D(N = 400, num_inducing = 50, optim_iters=100):
    """Run a 2D example of a sparse GP regression."""
    X = np.random.uniform(-3.,3.,(N,2))
    Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(N,1)*0.05

    # construct kernel
    rbf =  GPy.kern.rbf(2)
    noise = GPy.kern.white(2)
    kernel = rbf + noise

    # create simple GP Model
    m = GPy.models.SparseGPRegression(X,Y,kernel, num_inducing = num_inducing)

    # contrain all parameters to be positive (but not inducing inputs)
    m.ensure_default_constraints()
    m.set('.*len',2.)

    m.checkgrad()

    # optimize and plot
    m.optimize('tnc', messages = 1, max_f_eval=optim_iters)
    m.plot()
    print(m)
    return m

def uncertain_inputs_sparse_regression(optim_iters=100):
    """Run a 1D example of a sparse GP regression with uncertain inputs."""
    fig, axes = pb.subplots(1,2,figsize=(12,5))

    # sample inputs and outputs
    S = np.ones((20,1))
    X = np.random.uniform(-3.,3.,(20,1))
    Y = np.sin(X)+np.random.randn(20,1)*0.05
    #likelihood = GPy.likelihoods.Gaussian(Y)
    Z = np.random.uniform(-3.,3.,(7,1))

    k = GPy.kern.rbf(1) + GPy.kern.white(1)

    # create simple GP Model - no input uncertainty on this one
    m = GPy.models.SparseGPRegression(X, Y, kernel=k, Z=Z)
    m.ensure_default_constraints()
    m.optimize('scg', messages=1, max_f_eval=optim_iters)
    m.plot(ax=axes[0])
    axes[0].set_title('no input uncertainty')


    #the same Model with uncertainty
    m = GPy.models.SparseGPRegression(X, Y, kernel=k, Z=Z, X_variance=S)
    m.ensure_default_constraints()
    m.optimize('scg', messages=1, max_f_eval=optim_iters)
    m.plot(ax=axes[1])
    axes[1].set_title('with input uncertainty')
    print(m)

    fig.canvas.draw()

    return m
