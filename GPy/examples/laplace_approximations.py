import GPy
import numpy as np
import matplotlib.pyplot as plt

def timing():
    real_var = 0.1
    times = 1
    deg_free = 10
    real_sd = np.sqrt(real_var)
    the_is = np.zeros(times)
    X = np.linspace(0.0, 10.0, 300)[:, None]

    for a in xrange(times):
        Y = np.sin(X) + np.random.randn(*X.shape)*real_var
        Yc = Y.copy()

        Yc[10] += 100
        Yc[25] += 10
        Yc[23] += 10
        Yc[24] += 10
        Yc[250] += 10
        #Yc[4] += 10000

        edited_real_sd = real_sd
        kernel1 = GPy.kern.rbf(X.shape[1])

        t_distribution = GPy.likelihoods.likelihood_functions.student_t(deg_free, sigma=edited_real_sd)
        corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, rasm=True)
        m = GPy.models.GP(X, corrupt_stu_t_likelihood, kernel1)
        m.ensure_default_constraints()
        m.update_likelihood_approximation()
        m.optimize()
        the_is[a] = m.likelihood.i

    print the_is
    print np.mean(the_is)

def debug_student_t_noise_approx():
    real_var = 0.2
    #Start a function, any function
    X = np.linspace(0.0, 10.0, 30)[:, None]
    Y = np.sin(X) + np.random.randn(*X.shape)*real_var

    X_full = np.linspace(0.0, 10.0, 500)[:, None]
    Y_full = np.sin(X_full)

    #Y = Y/Y.max()

    #Add student t random noise to datapoints
    deg_free = 10000
    real_sd = np.sqrt(real_var)
    print "Real noise: ", real_sd

    initial_var_guess = 0.01
    #t_rv = t(deg_free, loc=0, scale=real_var)
    #noise = t_rvrvs(size=Y.shape)
    #Y += noise

    plt.figure(1)
    plt.suptitle('Gaussian likelihood')
    # Kernel object
    kernel1 = GPy.kern.rbf(X.shape[1])
    kernel2 = kernel1.copy()
    kernel3 = kernel1.copy()
    kernel4 = kernel1.copy()
    kernel5 = kernel1.copy()
    kernel6 = kernel1.copy()

    print "Clean Gaussian"
    #A GP should completely break down due to the points as they get a lot of weight
    # create simple GP model
    m = GPy.models.GP_regression(X, Y, kernel=kernel1)
    # optimize
    m.ensure_default_constraints()
    m.optimize()
    # plot
    plt.subplot(131)
    m.plot()
    plt.plot(X_full, Y_full)
    print m

    plt.suptitle('Student-t likelihood')
    edited_real_sd = initial_var_guess #real_sd

    print "Clean student t, rasm"
    t_distribution = GPy.likelihoods.likelihood_functions.student_t(deg_free, sigma=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, rasm=True)
    m = GPy.models.GP(X, stu_t_likelihood, kernel6)
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(132)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.ylim(-2.5, 2.5)

    print "Clean student t, ncg"
    t_distribution = GPy.likelihoods.likelihood_functions.student_t(deg_free, sigma=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y, t_distribution, rasm=False)
    m = GPy.models.GP(X, stu_t_likelihood, kernel3)
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(133)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.ylim(-2.5, 2.5)

    plt.show()

def student_t_approx():
    """
    Example of regressing with a student t likelihood
    """
    real_var = 0.2
    #Start a function, any function
    X = np.linspace(0.0, 10.0, 30)[:, None]
    Y = np.sin(X) + np.random.randn(*X.shape)*real_var
    Yc = Y.copy()

    X_full = np.linspace(0.0, 10.0, 500)[:, None]
    Y_full = np.sin(X_full)

    #Y = Y/Y.max()

    Yc[10] += 100
    Yc[25] += 10
    Yc[23] += 10
    Yc[24] += 10
    #Yc = Yc/Yc.max()

    #Add student t random noise to datapoints
    deg_free = 1000000000000
    real_sd = np.sqrt(real_var)
    print "Real noise: ", real_sd

    initial_var_guess = 0.01
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
    kernel5 = kernel1.copy()
    kernel6 = kernel1.copy()

    print "Clean Gaussian"
    #A GP should completely break down due to the points as they get a lot of weight
    # create simple GP model
    m = GPy.models.GP_regression(X, Y, kernel=kernel1)
    # optimize
    m.ensure_default_constraints()
    m.optimize()
    # plot
    plt.subplot(211)
    m.plot()
    plt.plot(X_full, Y_full)
    print m

    #Corrupt
    print "Corrupt Gaussian"
    m = GPy.models.GP_regression(X, Yc, kernel=kernel2)
    m.ensure_default_constraints()
    m.optimize()
    plt.subplot(212)
    m.plot()
    plt.plot(X_full, Y_full)
    print m

    plt.figure(2)
    plt.suptitle('Student-t likelihood')
    edited_real_sd = initial_var_guess #real_sd

    print "Clean student t, rasm"
    t_distribution = GPy.likelihoods.likelihood_functions.student_t(deg_free, sigma=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, rasm=True)
    m = GPy.models.GP(X, stu_t_likelihood, kernel6)
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(222)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.ylim(-2.5, 2.5)

    print "Corrupt student t, rasm"
    t_distribution = GPy.likelihoods.likelihood_functions.student_t(deg_free, sigma=edited_real_sd)
    corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, rasm=True)
    m = GPy.models.GP(X, corrupt_stu_t_likelihood, kernel4)
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(224)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.ylim(-2.5, 2.5)

    print "Clean student t, ncg"
    t_distribution = GPy.likelihoods.likelihood_functions.student_t(deg_free, sigma=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y, t_distribution, rasm=False)
    m = GPy.models.GP(X, stu_t_likelihood, kernel3)
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(221)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.ylim(-2.5, 2.5)

    print "Corrupt student t, ncg"
    t_distribution = GPy.likelihoods.likelihood_functions.student_t(deg_free, sigma=edited_real_sd)
    corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, rasm=False)
    m = GPy.models.GP(X, corrupt_stu_t_likelihood, kernel5)
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(223)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.ylim(-2.5, 2.5)


    ###with a student t distribution, since it has heavy tails it should work well
    ###likelihood_function = student_t(deg_free, sigma=real_var)
    ###lap = Laplace(Y, likelihood_function)
    ###cov = kernel.K(X)
    ###lap.fit_full(cov)

    ###test_range = np.arange(0, 10, 0.1)
    ###plt.plot(test_range, t_rv.pdf(test_range))
    ###for i in xrange(X.shape[0]):
        ###mode = lap.f_hat[i]
        ###covariance = lap.hess_hat_i[i,i]
        ###scaling = np.exp(lap.ln_z_hat)
        ###normalised_approx = norm(loc=mode, scale=covariance)
        ###print "Normal with mode %f, and variance %f" % (mode, covariance)
        ###plt.plot(test_range, scaling*normalised_approx.pdf(test_range))
    ###plt.show()

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
