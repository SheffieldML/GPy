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
    real_var = 0.1
    #Start a function, any function
    X = np.linspace(0.0, 10.0, 30)[:, None]
    Y = np.sin(X) + np.random.randn(*X.shape)*real_var
    Yc = Y.copy()

    #Y = Y/Y.max()

    #Yc[10] += 100
    Yc[25] += 10
    Yc[23] += 10
    Yc[24] += 10
    #Yc = Yc/Yc.max()

    #Add student t random noise to datapoints
    deg_free = 20 #100000.5
    real_sd = np.sqrt(real_var)
    #t_rv = t(deg_free, loc=0, scale=real_var)
    #noise = t_rvrvs(size=Y.shape)
    #Y += noise

    #Add some extreme value noise to some of the datapoints
    #percent_corrupted = 0.15
    #corrupted_datums = int(np.round(Y.shape[0] * percent_corrupted))
    #indices = np.arange(Y.shape[0])
    #np.random.shuffle(indices)
    #corrupted_indices = indices[:corrupted_datums]
    #print corrupted_indices
    #noise = t_rv.rvs(size=(len(corrupted_indices), 1))
    #Y[corrupted_indices] += noise
    plt.figure(1)
    plt.suptitle('Gaussian likelihood')
    # Kernel object
    kernel1 = GPy.kern.rbf(X.shape[1])
    kernel2 = kernel1.copy()
    kernel3 = kernel1.copy()
    kernel4 = kernel1.copy()

    print "Clean Gaussian"
    #A GP should completely break down due to the points as they get a lot of weight
    # create simple GP model
    m = GPy.models.GP_regression(X, Y, kernel=kernel1)
    ## optimize
    m.ensure_default_constraints()
    #m.unconstrain('noise')
    #m.constrain_fixed('noise', 0.1)
    m.optimize()
    # plot
    plt.subplot(211)
    m.plot()
    print m

    ##Corrupt
    print "Corrupt Gaussian"
    m = GPy.models.GP_regression(X, Yc, kernel=kernel2)
    m.ensure_default_constraints()
    #m.unconstrain('noise')
    #m.constrain_fixed('noise', 0.1)
    m.optimize()
    plt.subplot(212)
    m.plot()
    print m

    ##with a student t distribution, since it has heavy tails it should work well
    ##likelihood_function = student_t(deg_free, sigma=real_var)
    ##lap = Laplace(Y, likelihood_function)
    ##cov = kernel.K(X)
    ##lap.fit_full(cov)

    ##test_range = np.arange(0, 10, 0.1)
    ##plt.plot(test_range, t_rv.pdf(test_range))
    ##for i in xrange(X.shape[0]):
        ##mode = lap.f_hat[i]
        ##covariance = lap.hess_hat_i[i,i]
        ##scaling = np.exp(lap.ln_z_hat)
        ##normalised_approx = norm(loc=mode, scale=covariance)
        ##print "Normal with mode %f, and variance %f" % (mode, covariance)
        ##plt.plot(test_range, scaling*normalised_approx.pdf(test_range))
    ##plt.show()

    plt.figure(2)
    plt.suptitle('Student-t likelihood')
    edited_real_sd = real_sd

    # Likelihood object
    t_distribution = student_t(deg_free, sigma=edited_real_sd)
    stu_t_likelihood = Laplace(Yc, t_distribution)

    print "Clean student t"
    m = GPy.models.GP(X, stu_t_likelihood, kernel3)
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    # optimize
    m.optimize()
    print(m)
    # plot
    plt.subplot(211)
    m.plot_f()
    plt.ylim(-2.5,2.5)
    #import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

    print "Corrupt student t"
    t_distribution = student_t(deg_free, sigma=edited_real_sd)
    corrupt_stu_t_likelihood = Laplace(Yc, t_distribution)
    m = GPy.models.GP(X, corrupt_stu_t_likelihood, kernel4)
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(212)
    m.plot()
    plt.ylim(-2.5,2.5)
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
