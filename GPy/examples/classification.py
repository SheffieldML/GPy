# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Gaussian Processes classification examples
"""
MPL_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    MPL_AVAILABLE = False

import GPy

default_seed = 10000


def oil(num_inducing=50, max_iters=100, kernel=None, optimize=True, plot=True):
    """
    Run a Gaussian process classification on the three phase oil data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    """
    try:
        import pods
    except ImportError:
        raise ImportWarning(
            "Need pods for example datasets. See https://github.com/sods/ods, or pip install pods."
        )
    data = pods.datasets.oil()
    X = data["X"]
    Xtest = data["Xtest"]
    Y = data["Y"][:, 0:1]
    Ytest = data["Ytest"][:, 0:1]
    Y[Y.flatten() == -1] = 0
    Ytest[Ytest.flatten() == -1] = 0

    # Create GP model
    m = GPy.models.SparseGPClassification(
        X, Y, kernel=kernel, num_inducing=num_inducing
    )
    m.Ytest = Ytest

    # Contrain all parameters to be positive
    # m.tie_params('.*len')
    m[".*len"] = 10.0

    # Optimize
    if optimize:
        m.optimize(messages=1)
    print(m)

    # Test
    probs = m.predict(Xtest)[0]
    GPy.util.classification.conf_matrix(probs, Ytest)
    return m


def toy_linear_1d_classification(seed=default_seed, optimize=True, plot=True):
    """
    Simple 1D classification example using EP approximation

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """
    try:
        import pods
    except ImportError:
        raise ImportWarning(
            "Need pods for example datasets. See https://github.com/sods/ods, or pip install pods."
        )
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data["Y"][:, 0:1]
    Y[Y.flatten() == -1] = 0

    # Model definition
    m = GPy.models.GPClassification(data["X"], Y)

    # Optimize
    if optimize:
        # m.update_likelihood_approximation()
        # Parameters optimization:
        m.optimize()
        # m.update_likelihood_approximation()
        # m.pseudo_EM()

    # Plot
    if MPL_AVAILABLE and plot:
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m


def toy_linear_1d_classification_laplace(seed=default_seed, optimize=True, plot=True):
    """
    Simple 1D classification example using Laplace approximation

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:
        import pods
    except ImportError:
        print("pods unavailable, see https://github.com/sods/ods for example datasets")
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data["Y"][:, 0:1]
    Y[Y.flatten() == -1] = 0

    likelihood = GPy.likelihoods.Bernoulli()
    laplace_inf = GPy.inference.latent_function_inference.Laplace()
    kernel = GPy.kern.RBF(1)

    # Model definition
    m = GPy.core.GP(
        data["X"], Y, kernel=kernel, likelihood=likelihood, inference_method=laplace_inf
    )

    # Optimize
    if optimize:
        m.optimize("scg", messages=True)

    return m

    # Plot
    if MPL_AVAILABLE and plot:
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m


def sparse_toy_linear_1d_classification(
    num_inducing=10, seed=default_seed, optimize=True, plot=True
):
    """
    Sparse 1D classification example

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:
        import pods
    except ImportError:
        print("pods unavailable, see https://github.com/sods/ods for example datasets")
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data["Y"][:, 0:1]
    Y[Y.flatten() == -1] = 0

    # Model definition
    m = GPy.models.SparseGPClassification(data["X"], Y, num_inducing=num_inducing)
    m[".*len"] = 4.0

    # Optimize
    if optimize:
        m.optimize()

    # Plot
    if MPL_AVAILABLE and plot:
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m


def sparse_toy_linear_1d_classification_uncertain_input(
    num_inducing=10, seed=default_seed, optimize=True, plot=True
):
    """
    Sparse 1D classification example

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:
        import pods
    except ImportError:
        print("pods unavailable, see https://github.com/sods/ods for example datasets")
    import numpy as np

    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data["Y"][:, 0:1]
    Y[Y.flatten() == -1] = 0
    X = data["X"]
    X_var = np.random.uniform(0.3, 0.5, X.shape)

    # Model definition
    m = GPy.models.SparseGPClassificationUncertainInput(
        X, X_var, Y, num_inducing=num_inducing
    )
    m[".*len"] = 4.0

    # Optimize
    if optimize:
        m.optimize()

    # Plot
    if MPL_AVAILABLE and plot:
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m


def toy_heaviside(seed=default_seed, max_iters=100, optimize=True, plot=True):
    """
    Simple 1D classification example using a heavy side gp transformation

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:
        import pods
    except ImportError:
        print("pods unavailable, see https://github.com/sods/ods for example datasets")
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data["Y"][:, 0:1]
    Y[Y.flatten() == -1] = 0

    # Model definition
    kernel = GPy.kern.RBF(1)
    likelihood = GPy.likelihoods.Bernoulli(
        gp_link=GPy.likelihoods.link_functions.Heaviside()
    )
    ep = GPy.inference.latent_function_inference.expectation_propagation.EP()
    m = GPy.core.GP(
        X=data["X"],
        Y=Y,
        kernel=kernel,
        likelihood=likelihood,
        inference_method=ep,
        name="gp_classification_heaviside",
    )
    # m = GPy.models.GPClassification(data['X'], likelihood=likelihood)

    # Optimize
    if optimize:
        # Parameters optimization:
        for _ in range(5):
            m.optimize(max_iters=int(max_iters / 5))
        print(m)

    # Plot
    if MPL_AVAILABLE and plot:
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m


def crescent_data(
    model_type="Full",
    num_inducing=10,
    seed=default_seed,
    kernel=None,
    optimize=True,
    plot=True,
):
    """
    Run a Gaussian process classification on the crescent data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param inducing: number of inducing variables (only used for 'FITC' or 'DTC').
    :type inducing: int
    :param seed: seed value for data generation.
    :type seed: int
    :param kernel: kernel to use in the model
    :type kernel: a GPy kernel
    """
    try:
        import pods
    except ImportError:
        print("pods unavailable, see https://github.com/sods/ods for example datasets")
    data = pods.datasets.crescent_data(seed=seed)
    Y = data["Y"]
    Y[Y.flatten() == -1] = 0

    if model_type == "Full":
        m = GPy.models.GPClassification(data["X"], Y, kernel=kernel)

    elif model_type == "DTC":
        m = GPy.models.SparseGPClassification(
            data["X"], Y, kernel=kernel, num_inducing=num_inducing
        )
        m[".*len"] = 10.0

    elif model_type == "FITC":
        m = GPy.models.FITCClassification(
            data["X"], Y, kernel=kernel, num_inducing=num_inducing
        )
        m[".*len"] = 3.0
    if optimize:
        m.optimize(messages=1)

    if MPL_AVAILABLE and plot:
        m.plot()

    print(m)
    return m
