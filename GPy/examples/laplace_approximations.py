import GPy
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

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

        t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=edited_real_sd)
        corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, opt='rasm')
        m = GPy.models.GP(X, corrupt_stu_t_likelihood, kernel1)
        m.ensure_default_constraints()
        m.update_likelihood_approximation()
        m.optimize()
        the_is[a] = m.likelihood.i

    print the_is
    print np.mean(the_is)

def v_fail_test():
    #plt.close('all')
    real_var = 0.1
    X = np.linspace(0.0, 10.0, 50)[:, None]
    Y = np.sin(X) + np.random.randn(*X.shape)*real_var
    Y = Y/Y.max()

    #Add student t random noise to datapoints
    deg_free = 10
    real_sd = np.sqrt(real_var)
    print "Real noise std: ", real_sd

    kernel1 = GPy.kern.white(X.shape[1]) #+ GPy.kern.white(X.shape[1])

    edited_real_sd = 0.3#real_sd
    edited_real_sd = real_sd

    print "Clean student t, rasm"
    t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')
    m = GPy.models.GP(X, stu_t_likelihood, kernel1)
    m.constrain_positive('')
    vs = 25
    noises = 30
    checkgrads = np.zeros((vs, noises))
    vs_noises = np.zeros((vs, noises))
    for v_ind, v in enumerate(np.linspace(1, 100, vs)):
        m.likelihood.likelihood_function.v = v
        print v
        for noise_ind, noise in enumerate(np.linspace(0.0001, 100, noises)):
            m['t_noise'] = noise
            m.update_likelihood_approximation()
            checkgrads[v_ind, noise_ind] = m.checkgrad()
            vs_noises[v_ind, noise_ind] = (float(v)/(float(v) - 2))*(noise**2)

    plt.figure()
    plt.title('Checkgrads')
    plt.imshow(checkgrads, interpolation='nearest')
    plt.xlabel('noise')
    plt.ylabel('v')

    #plt.figure()
    #plt.title('variance change')
    #plt.imshow(vs_noises, interpolation='nearest')
    #plt.xlabel('noise')
    #plt.ylabel('v')
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
    print(m)

def student_t_obj_plane():
    plt.close('all')
    X = np.linspace(0, 1, 50)[:, None]
    real_std = 0.002
    noise = np.random.randn(*X.shape)*real_std
    Y = np.sin(X*2*np.pi) + noise
    deg_free = 1000

    kernelgp = GPy.kern.rbf(X.shape[1]) # + GPy.kern.white(X.shape[1])
    mgp = GPy.models.GP_regression(X, Y, kernel=kernelgp)
    mgp.ensure_default_constraints()
    mgp['noise'] = real_std**2
    print "Gaussian"
    print mgp

    kernelst = kernelgp.copy()
    t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=(real_std**2))
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')
    m = GPy.models.GP(X, stu_t_likelihood, kernelst)
    m.ensure_default_constraints()
    m.constrain_fixed('t_no', real_std**2)
    vs = 10
    ls = 10
    objs_t = np.zeros((vs, ls))
    objs_g = np.zeros((vs, ls))
    rbf_vs = np.linspace(1e-6, 8, vs)
    rbf_ls = np.linspace(1e-2, 8, ls)
    for v_id, rbf_v in enumerate(rbf_vs):
        for l_id, rbf_l in enumerate(rbf_ls):
            m['rbf_v'] = rbf_v
            m['rbf_l'] = rbf_l
            mgp['rbf_v'] = rbf_v
            mgp['rbf_l'] = rbf_l
            objs_t[v_id, l_id] = m.log_likelihood()
            objs_g[v_id, l_id] = mgp.log_likelihood()
    plt.figure()
    plt.subplot(211)
    plt.title('Student t')
    plt.imshow(objs_t, interpolation='none')
    plt.xlabel('variance')
    plt.ylabel('lengthscale')
    plt.subplot(212)
    plt.title('Gaussian')
    plt.imshow(objs_g, interpolation='none')
    plt.xlabel('variance')
    plt.ylabel('lengthscale')
    plt.show()
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
    return objs_t

def student_t_f_check():
    plt.close('all')
    X = np.linspace(0, 1, 50)[:, None]
    real_std = 0.2
    noise = np.random.randn(*X.shape)*real_std
    Y = np.sin(X*2*np.pi) + noise
    deg_free = 1000

    kernelgp = GPy.kern.rbf(X.shape[1]) # + GPy.kern.white(X.shape[1])
    mgp = GPy.models.GP_regression(X, Y, kernel=kernelgp)
    mgp.ensure_default_constraints()
    mgp.randomize()
    mgp.optimize()
    print "Gaussian"
    print mgp
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

    kernelst = kernelgp.copy()
    #kernelst += GPy.kern.bias(X.shape[1])
    t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=0.05)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')
    m = GPy.models.GP(X, stu_t_likelihood, kernelst)
    #m['rbf_v'] = mgp._get_params()[0]
    #m['rbf_l'] = mgp._get_params()[1] + 1
    m.ensure_default_constraints()
    #m.constrain_fixed('rbf_v', mgp._get_params()[0])
    #m.constrain_fixed('rbf_l', mgp._get_params()[1])
    #m.constrain_bounded('t_no', 2*real_std**2, 1e3)
    #m.constrain_positive('bias')
    m.constrain_positive('t_no')
    m.randomize()
    m['t_no'] = 0.3
    m.likelihood.X = X
    #print m
    plt.figure()
    plt.subplot(211)
    m.plot()
    print "OPTIMIZED ONCE"
    plt.subplot(212)
    m.optimize()
    m.plot()
    print "final optimised student t"
    print m
    print "real GP"
    print mgp
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
    return m

def student_t_fix_optimise_check():
    plt.close('all')
    real_var = 0.1
    real_std = np.sqrt(real_var)
    X = np.random.rand(200)[:, None]
    noise = np.random.randn(*X.shape)*real_std
    Y = np.sin(X*2*np.pi) + noise
    X_full = X
    Y_full = np.sin(X_full)
    Y = Y/Y.max()
    Y_full = Y_full/Y_full.max()
    deg_free = 1000

    #GP
    kernelgp = GPy.kern.rbf(X.shape[1]) # + GPy.kern.white(X.shape[1])
    mgp = GPy.models.GP_regression(X, Y, kernel=kernelgp)
    mgp.ensure_default_constraints()
    mgp.randomize()
    mgp.optimize()

    kernelst = kernelgp.copy()
    real_stu_t_std2 = (real_std**2)*((deg_free - 2)/float(deg_free))

    t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=real_stu_t_std2)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')

    plt.figure(1)
    plt.suptitle('Student likelihood')
    m = GPy.models.GP(X, stu_t_likelihood, kernelst)
    m.constrain_fixed('rbf_var', mgp._get_params()[0])
    m.constrain_fixed('rbf_len', mgp._get_params()[1])
    m.constrain_positive('t_noise')
    #m.ensure_default_constraints()

    m.update_likelihood_approximation()
    print "T std2 {} converted from original data, LL: {}".format(real_stu_t_std2, m.log_likelihood())
    plt.subplot(231)
    m.plot()
    plt.title('Student t original data noise')

    #Fix student t noise variance to same a GP
    gp_noise = mgp._get_params()[2]
    m['t_noise_std2'] = gp_noise
    m.update_likelihood_approximation()
    print "T std2 {} same as GP noise, LL: {}".format(gp_noise, m.log_likelihood())
    plt.subplot(232)
    m.plot()
    plt.title('Student t GP noise')

    #Fix student t noise to variance converted from the GP
    real_stu_t_std2gp = (gp_noise)*((deg_free - 2)/float(deg_free))
    m['t_noise_std2'] = real_stu_t_std2gp
    m.update_likelihood_approximation()
    print "T std2 {} converted to student t noise from GP noise, LL: {}".format(m.likelihood.likelihood_function.sigma2, m.log_likelihood())
    plt.subplot(233)
    m.plot()
    plt.title('Student t GP noise converted')

    m.constrain_positive('t_noise_std2')
    m.randomize()
    m.update_likelihood_approximation()
    plt.subplot(234)
    m.plot()
    plt.title('Student t fixed rbf')
    m.optimize()
    print "T std2 {} var {} after optimising, LL: {}".format(m.likelihood.likelihood_function.sigma2, m.likelihood.likelihood_function.variance, m.log_likelihood())
    plt.subplot(235)
    m.plot()
    plt.title('Student t fixed rbf optimised')

    plt.figure(2)
    mrbf = m.copy()
    mrbf.unconstrain('')
    mrbf.constrain_fixed('t_noise', m.likelihood.likelihood_function.sigma2)
    gp_var = mgp._get_params()[0]
    gp_len = mgp._get_params()[1]
    mrbf.constrain_fixed('rbf_var', gp_var)
    mrbf.constrain_positive('rbf_len')
    mrbf.randomize()
    print "Before optimize"
    print mrbf
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
    mrbf.checkgrad(verbose=1)
    plt.subplot(121)
    mrbf.plot()
    plt.title('Student t fixed noise')
    mrbf.optimize()
    print "After optimize"
    print mrbf
    plt.subplot(122)
    mrbf.plot()
    plt.title('Student t fixed noise optimized')
    print mrbf

    plt.figure(3)
    print "GP noise {} after optimising, LL: {}".format(gp_noise, mgp.log_likelihood())
    plt.suptitle('Gaussian likelihood optimised')
    mgp.plot()
    print "Real std: {}".format(real_std)
    print "Real variance {}".format(real_std**2)

    #import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

    print "Len should be: {}".format(gp_len)
    return mrbf

def debug_student_t_noise_approx():
    plot = False
    real_var = 0.1
    #Start a function, any function
    #X = np.linspace(0.0, 10.0, 50)[:, None]
    X = np.random.rand(100)[:, None]
    #X = np.random.rand(100)[:, None]
    #X = np.array([0.5, 1])[:, None]
    Y = np.sin(X*2*np.pi) + np.random.randn(*X.shape)*real_var + 1
    #Y = X + np.random.randn(*X.shape)*real_var
    #ty = np.array([1., 9.97733584, 4.17841363])[:, None]
    #Y = ty

    X_full = X
    Y_full = np.sin(X_full) + 1

    Y = Y/Y.max()

    #Add student t random noise to datapoints
    deg_free = 100

    real_sd = np.sqrt(real_var)
    print "Real noise std: ", real_sd

    initial_var_guess = 0.3
    #t_rv = t(deg_free, loc=0, scale=real_var)
    #noise = t_rvrvs(size=Y.shape)
    #Y += noise

    plt.close('all')
    # Kernel object
    kernel1 = GPy.kern.rbf(X.shape[1]) #+ GPy.kern.white(X.shape[1])
    #kernel1 = GPy.kern.linear(X.shape[1]) + GPy.kern.white(X.shape[1])
    kernel2 = kernel1.copy()
    kernel3 = kernel1.copy()
    kernel4 = kernel1.copy()
    kernel5 = kernel1.copy()
    kernel6 = kernel1.copy()

    print "Clean Gaussian"
    #A GP should completely break down due to the points as they get a lot of weight
    # create simple GP model
    #m = GPy.models.GP_regression(X, Y, kernel=kernel1)
    ## optimize
    #m.ensure_default_constraints()
    #m.optimize()
    ## plot
    #if plot:
        #plt.figure(1)
        #plt.suptitle('Gaussian likelihood')
        #plt.subplot(131)
        #m.plot()
        #plt.plot(X_full, Y_full)
    #print m

    real_stu_t_std = np.sqrt(real_var*((deg_free - 2)/float(deg_free)))
    edited_real_sd = real_stu_t_std**2 #initial_var_guess #real_sd
    #edited_real_sd = real_sd

    print "Clean student t, rasm"
    t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')

    m = GPy.models.GP(X, stu_t_likelihood, kernel6)
    #m['rbf_len'] = 1.5
    #m.constrain_fixed('rbf_v', 1.0898)
    #m.constrain_fixed('rbf_l', 0.2651)
    #m.constrain_fixed('t_noise_std2', edited_real_sd)
    #m.constrain_positive('rbf')
    m.constrain_positive('t_noise_std2')
    #m.constrain_positive('')
    #m.constrain_bounded('t_noi', 0.001, 10)
    #m.constrain_fixed('t_noi', real_stu_t_std)
    #m.constrain_fixed('white', 0.01)
    #m.constrain_fixed('t_no', 0.01)
    #m['rbf_var'] = 0.20446332
    #m['rbf_leng'] = 0.85776241
    #m['t_noise'] = 0.667083294421005
    m.ensure_default_constraints()
    m.update_likelihood_approximation()
    #m.optimize(messages=True)
    print(m)
    #return m
    #m.optimize('lbfgsb', messages=True, callback=m._update_params_callback)
    if plot:
        plt.suptitle('Student-t likelihood')
        plt.subplot(132)
        m.plot()
        plt.plot(X_full, Y_full)
        plt.ylim(-2.5, 2.5)
    print "Real noise std: ", real_sd
    print "or Real noise std: ", real_stu_t_std
    return m

    #print "Clean student t, ncg"
    #t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=edited_real_sd)
    #stu_t_likelihood = GPy.likelihoods.Laplace(Y, t_distribution, opt='ncg')
    #m = GPy.models.GP(X, stu_t_likelihood, kernel3)
    #m.ensure_default_constraints()
    #m.update_likelihood_approximation()
    #m.optimize()
    #print(m)
    #if plot:
        #plt.subplot(133)
        #m.plot()
        #plt.plot(X_full, Y_full)
        #plt.ylim(-2.5, 2.5)

    #plt.show()

def student_t_approx():
    """
    Example of regressing with a student t likelihood
    """
    real_std = 0.1
    #Start a function, any function
    X = np.linspace(0.0, 10.0, 50)[:, None]
    Y = np.sin(X) + np.random.randn(*X.shape)*real_std
    Yc = Y.copy()

    X_full = np.linspace(0.0, 10.0, 500)[:, None]
    Y_full = np.sin(X_full)

    Y = Y/Y.max()

    Yc[10] += 100
    Yc[25] += 10
    Yc[23] += 10
    Yc[26] += 1000
    Yc[24] += 10
    #Yc = Yc/Yc.max()

    #Add student t random noise to datapoints
    deg_free = 8
    print "Real noise: ", real_std

    initial_var_guess = 0.1
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
    plt.title('Gaussian clean')
    print m

    #Corrupt
    print "Corrupt Gaussian"
    m = GPy.models.GP_regression(X, Yc, kernel=kernel2)
    m.ensure_default_constraints()
    #m.optimize()
    plt.subplot(212)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.title('Gaussian corrupt')
    print m

    plt.figure(2)
    plt.suptitle('Student-t likelihood')
    edited_real_sd = real_std #initial_var_guess

    print "Clean student t, rasm"
    t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')
    m = GPy.models.GP(X, stu_t_likelihood, kernel6)
    m.ensure_default_constraints()
    m.constrain_positive('t_noise')
    m.randomize()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(222)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.ylim(-2.5, 2.5)
    plt.title('Student-t rasm clean')

    print "Corrupt student t, rasm"
    t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=edited_real_sd)
    corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, opt='rasm')
    m = GPy.models.GP(X, corrupt_stu_t_likelihood, kernel4)
    m.ensure_default_constraints()
    m.constrain_positive('t_noise')
    m.randomize()
    m.update_likelihood_approximation()
    m.optimize()
    print(m)
    plt.subplot(224)
    m.plot()
    plt.plot(X_full, Y_full)
    plt.ylim(-2.5, 2.5)
    plt.title('Student-t rasm corrupt')

    return m

    #print "Clean student t, ncg"
    #t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=edited_real_sd)
    #stu_t_likelihood = GPy.likelihoods.Laplace(Y, t_distribution, opt='ncg')
    #m = GPy.models.GP(X, stu_t_likelihood, kernel3)
    #m.ensure_default_constraints()
    #m.update_likelihood_approximation()
    #m.optimize()
    #print(m)
    #plt.subplot(221)
    #m.plot()
    #plt.plot(X_full, Y_full)
    #plt.ylim(-2.5, 2.5)
    #plt.title('Student-t ncg clean')

    #print "Corrupt student t, ncg"
    #t_distribution = GPy.likelihoods.likelihood_functions.Student_t(deg_free, sigma2=edited_real_sd)
    #corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, opt='ncg')
    #m = GPy.models.GP(X, corrupt_stu_t_likelihood, kernel5)
    #m.ensure_default_constraints()
    #m.update_likelihood_approximation()
    #m.optimize()
    #print(m)
    #plt.subplot(223)
    #m.plot()
    #plt.plot(X_full, Y_full)
    #plt.ylim(-2.5, 2.5)
    #plt.title('Student-t ncg corrupt')


    ###with a student t distribution, since it has heavy tails it should work well
    ###likelihood_function = student_t(deg_free, sigma2=real_var)
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

def gaussian_f_check():
    plt.close('all')
    X = np.linspace(0, 1, 50)[:, None]
    real_std = 0.2
    noise = np.random.randn(*X.shape)*real_std
    Y = np.sin(X*2*np.pi) + noise

    kernelgp = GPy.kern.rbf(X.shape[1]) # + GPy.kern.white(X.shape[1])
    mgp = GPy.models.GP_regression(X, Y, kernel=kernelgp)
    mgp.ensure_default_constraints()
    mgp.randomize()
    mgp.optimize()
    print "Gaussian"
    print mgp
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

    kernelg = kernelgp.copy()
    #kernelst += GPy.kern.bias(X.shape[1])
    N, D = X.shape
    g_distribution = GPy.likelihoods.likelihood_functions.Gaussian(variance=0.1, N=N, D=D)
    g_likelihood = GPy.likelihoods.Laplace(Y.copy(), g_distribution, opt='rasm')
    m = GPy.models.GP(X, g_likelihood, kernelg)
    #m['rbf_v'] = mgp._get_params()[0]
    #m['rbf_l'] = mgp._get_params()[1] + 1
    m.ensure_default_constraints()
    #m.constrain_fixed('rbf_v', mgp._get_params()[0])
    #m.constrain_fixed('rbf_l', mgp._get_params()[1])
    #m.constrain_bounded('t_no', 2*real_std**2, 1e3)
    #m.constrain_positive('bias')
    m.constrain_positive('noise_var')
    m.randomize()
    m['noise_variance'] = 0.1
    m.likelihood.X = X
    plt.figure()
    plt.subplot(211)
    m.plot()
    plt.subplot(212)
    m.optimize()
    m.plot()
    print "final optimised gaussian"
    print m
    print "real GP"
    print mgp
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT
