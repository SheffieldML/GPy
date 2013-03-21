import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from coxGP.python.likelihoods.Laplace import Laplace
from coxGP.python.likelihoods.likelihood_function import student_t


def student_t_approx():
    """
    Example of regressing with a student t likelihood
    """
    #Start a function, any function
    X = np.sort(np.random.uniform(0, 15, 100))[:, None]
    Y = np.sin(X)

    #Add student t random noise to datapoints
    deg_free = 100000.5
    real_var = 4
    t_rv = t(deg_free, loc=0, scale=real_var)
    noise = t_rv.rvs(size=Y.shape)
    Y += noise

    #Add some extreme value noise to some of the datapoints
    #percent_corrupted = 0.15
    #corrupted_datums = int(np.round(Y.shape[0] * percent_corrupted))
    #indices = np.arange(Y.shape[0])
    #np.random.shuffle(indices)
    #corrupted_indices = indices[:corrupted_datums]
    #print corrupted_indices
    #noise = t_rv.rvs(size=(len(corrupted_indices), 1))
    #Y[corrupted_indices] += noise

    # Kernel object
    print X.shape
    kernel = GPy.kern.rbf(X.shape[1])

    #A GP should completely break down due to the points as they get a lot of weight
    # create simple GP model
    #m = GPy.models.GP_regression(X, Y, kernel=kernel)

    ## optimize
    #m.ensure_default_constraints()
    #m.optimize()
    ## plot
    ##m.plot()
    #print m

    #with a student t distribution, since it has heavy tails it should work well
    likelihood_function = student_t(deg_free, sigma=real_var)
    lap = Laplace(Y, likelihood_function)
    cov = kernel.K(X)
    lap.fit_full(cov)

    test_range = np.arange(0, 10, 0.1)
    plt.plot(test_range, t_rv.pdf(test_range))
    for i in xrange(X.shape[0]):
        mode = lap.f_hat[i]
        covariance = lap.hess_hat_i[i,i]
        scaling = np.exp(lap.ln_z_hat)
        normalised_approx = norm(loc=mode, scale=covariance)
        print "Normal with mode %f, and variance %f" % (mode, covariance)
        plt.plot(test_range, scaling*normalised_approx.pdf(test_range))
    plt.show()
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

    # Likelihood object
    t_distribution = student_t(deg_free, sigma=real_var)
    stu_t_likelihood = Laplace(Y, t_distribution)
    kernel = GPy.kern.rbf(X.shape[1]) + GPy.kern.bias(X.shape[1])

    m = GPy.models.GP(X, stu_t_likelihood, kernel)
    m.ensure_default_constraints()

    m.update_likelihood_approximation()
    print "NEW MODEL"
    print(m)

    # optimize
    #m.optimize()
    #print(m)

    # plot
    m.plot()
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

    m.optimize()
    print(m)

    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
    return m


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
