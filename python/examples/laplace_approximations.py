import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from coxGP.python.likelihoods.Laplace import Laplace
from coxGP.python.likelihoods.likelihood_function import student_t


def student_t_approx():
    """
    Example of regressing with a student t likelihood
    """
    #Start a function, any function
    X = np.sort(np.random.uniform(0, 15, 70))[:, None]
    Y = np.sin(X)

    #Add student t random noise to datapoints
    deg_free = 1
    noise = t.rvs(deg_free, loc=1.8, scale=1, size=Y.shape)
    Y += noise

    # Kernel object
    print X.shape
    kernel = GPy.kern.rbf(X.shape[1])

    #A GP should completely break down due to the points as they get a lot of weight
    # create simple GP model
    m = GPy.models.GP_regression(X, Y, kernel=kernel)

    # optimize
    m.ensure_default_constraints()
    m.optimize()
    # plot
    #m.plot()
    print m

    #with a student t distribution, since it has heavy tails it should work well
    likelihood_function = student_t(deg_free, sigma=1)
    lap = Laplace(Y, likelihood_function)
    cov = kernel.K(X)
    lap.fit_full(cov)


def noisy_laplace_approx():
    """
    Example of regressing with a student t likelihood
    """
    #Start a function, any function
    X = np.sort(np.random.uniform(0, 15, 70))[:, None]
    Y = np.sin(X)

    #Add some extreme value noise to some of the datapoints
    percent_corrupted = 0.05
    corrupted_datums = int(np.round(Y.shape[0] * percent_corrupted))
    indices = np.arange(Y.shape[0])
    np.random.shuffle(indices)
    corrupted_indices = indices[:corrupted_datums]
    print corrupted_indices
    noise = np.random.uniform(-10, 10, (len(corrupted_indices), 1))
    Y[corrupted_indices] += noise

    #A GP should completely break down due to the points as they get a lot of weight
    # create simple GP model
    m = GPy.models.GP_regression(X, Y)

    # optimize
    m.ensure_default_constraints()
    m.optimize()
    # plot
    m.plot()
    print m

    #with a student t distribution, since it has heavy tails it should work well
