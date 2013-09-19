import GPy
import numpy as np
import matplotlib.pyplot as plt
from GPy.util import datasets
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

        t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=edited_real_sd)
        corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, opt='rasm')
        m = GPy.models.GPRegression(X, Yc.copy(), kernel1, likelihood=corrupt_stu_t_likelihood)
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
    t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')
    m = GPy.models.GPRegression(X, Y.copy(), kernel1, likelihood=stu_t_likelihood)
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
    mgp = GPy.models.GPRegression(X, Y, kernel=kernelgp)
    mgp.ensure_default_constraints()
    mgp['noise'] = real_std**2
    print "Gaussian"
    print mgp

    kernelst = kernelgp.copy()
    t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=(real_std**2))
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')
    m = GPy.models.GPRegression(X, Y, kernelst, likelihood=stu_t_likelihood)
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
    mgp = GPy.models.GPRegression(X, Y, kernel=kernelgp)
    mgp.ensure_default_constraints()
    mgp.randomize()
    mgp.optimize()
    print "Gaussian"
    print mgp
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

    kernelst = kernelgp.copy()
    #kernelst += GPy.kern.bias(X.shape[1])
    t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=0.05)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')
    m = GPy.models.GPRegression(X, Y.copy(), kernelst, likelihood=stu_t_likelihood)
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
    mgp = GPy.models.GPRegression(X, Y.copy(), kernel=kernelgp)
    mgp.ensure_default_constraints()
    mgp.randomize()
    mgp.optimize()

    kernelst = kernelgp.copy()
    real_stu_t_std2 = (real_std**2)*((deg_free - 2)/float(deg_free))

    t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=real_stu_t_std2)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')

    plt.figure(1)
    plt.suptitle('Student likelihood')
    m = GPy.models.GPRegression(X, Y.copy(), kernelst, likelihood=stu_t_likelihood)
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
    #m = GPy.models.GPRegression(X, Y, kernel=kernel1)
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
    t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')

    m = GPy.models.GPRegression(X, Y, kernel6, likelihood=stu_t_likelihood)
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
    #t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=edited_real_sd)
    #stu_t_likelihood = GPy.likelihoods.Laplace(Y, t_distribution, opt='ncg')
    #m = GPy.models.GPRegression(X, stu_t_likelihood, kernel3)
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
    X = np.linspace(0.0, np.pi*2, 100)[:, None]
    Y = np.sin(X) + np.random.randn(*X.shape)*real_std
    Yc = Y.copy()

    X_full = np.linspace(0.0, np.pi*2, 500)[:, None]
    Y_full = np.sin(X_full)

    Y = Y/Y.max()

    Yc[75:80] += 1

    #Yc[10] += 100
    #Yc[25] += 10
    #Yc[23] += 10
    #Yc[26] += 1000
    #Yc[24] += 10
    #Yc = Yc/Yc.max()

    #Add student t random noise to datapoints
    deg_free = 5
    print "Real noise: ", real_std

    initial_var_guess = 0.5
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
    m = GPy.models.GPRegression(X, Y, kernel=kernel1)
    # optimize
    m.ensure_default_constraints()
    m.randomize()
    m.optimize()
    # plot
    ax = plt.subplot(211)
    m.plot(ax=ax)
    plt.plot(X_full, Y_full)
    plt.ylim(-1.5, 1.5)
    plt.title('Gaussian clean')
    print m

    #Corrupt
    print "Corrupt Gaussian"
    m = GPy.models.GPRegression(X, Yc, kernel=kernel2)
    m.ensure_default_constraints()
    m.randomize()
    m.optimize()
    ax = plt.subplot(212)
    m.plot(ax=ax)
    plt.plot(X_full, Y_full)
    plt.ylim(-1.5, 1.5)
    plt.title('Gaussian corrupt')
    print m

    plt.figure(2)
    plt.suptitle('Student-t likelihood')
    edited_real_sd = initial_var_guess

    print "Clean student t, rasm"
    t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution, opt='rasm')
    m = GPy.models.GPRegression(X, Y.copy(), kernel6, likelihood=stu_t_likelihood)
    m.ensure_default_constraints()
    m.constrain_positive('t_noise')
    m.randomize()
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    #m.update_likelihood_approximation()
    m.optimize()
    print(m)
    ax = plt.subplot(211)
    m.plot(ax=ax)
    plt.plot(X_full, Y_full)
    plt.ylim(-1.5, 1.5)
    plt.title('Student-t rasm clean')

    print "Corrupt student t, rasm"
    t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=edited_real_sd)
    corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, opt='rasm')
    m = GPy.models.GPRegression(X, Yc.copy(), kernel4, likelihood=corrupt_stu_t_likelihood)
    m.ensure_default_constraints()
    m.constrain_positive('t_noise')
    m.randomize()
    #m.update_likelihood_approximation()
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    m.optimize()
    print(m)
    ax = plt.subplot(212)
    m.plot(ax=ax)
    plt.plot(X_full, Y_full)
    plt.ylim(-1.5, 1.5)
    plt.title('Student-t rasm corrupt')

    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    return m

    #print "Clean student t, ncg"
    #t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=edited_real_sd)
    #stu_t_likelihood = GPy.likelihoods.Laplace(Y, t_distribution, opt='ncg')
    #m = GPy.models.GPRegression(X, Y, kernel3, likelihood=stu_t_likelihood)
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
    #t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=edited_real_sd)
    #corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution, opt='ncg')
    #m = GPy.models.GPRegression(X, Y, kernel5, likelihood=corrupt_stu_t_likelihood)
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
    m = GPy.models.GPRegression(X, Y)

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
    mgp = GPy.models.GPRegression(X, Y, kernel=kernelgp)
    mgp.ensure_default_constraints()
    mgp.randomize()
    mgp.optimize()
    print "Gaussian"
    print mgp

    kernelg = kernelgp.copy()
    #kernelst += GPy.kern.bias(X.shape[1])
    N, D = X.shape
    g_distribution = GPy.likelihoods.functions.Gaussian(variance=0.1, N=N, D=D)
    g_likelihood = GPy.likelihoods.Laplace(Y.copy(), g_distribution, opt='rasm')
    m = GPy.models.GPRegression(X, Y, kernelg, likelihood=g_likelihood)
    m.likelihood.X = X
    #m['rbf_v'] = mgp._get_params()[0]
    #m['rbf_l'] = mgp._get_params()[1] + 1
    m.ensure_default_constraints()
    #m.constrain_fixed('rbf_v', mgp._get_params()[0])
    #m.constrain_fixed('rbf_l', mgp._get_params()[1])
    #m.constrain_bounded('t_no', 2*real_std**2, 1e3)
    #m.constrain_positive('bias')
    m.constrain_positive('noise_var')
    #m['noise_variance'] = 0.1
    #m.likelihood.X = X
    m.randomize()
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    plt.figure()
    ax = plt.subplot(211)
    m.plot(ax=ax)

    m.optimize()
    ax = plt.subplot(212)
    m.plot(ax=ax)

    print "final optimised gaussian"
    print m
    print "real GP"
    print mgp
    import ipdb; ipdb.set_trace() ### XXX BREAKPOINT

def boston_example():
    import sklearn
    from sklearn.cross_validation import KFold
    data = datasets.boston_housing()
    X = data['X'].copy()
    Y = data['Y'].copy()
    Y = Y-Y.mean()
    Y = Y/Y.std()
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    num_folds = 10
    kf = KFold(len(Y), n_folds=num_folds, indices=True)
    score_folds = np.zeros((4, num_folds))
    def rmse(Y, Ystar):
        return np.sqrt(np.mean((Y-Ystar)**2))
    #for train, test in kf:
    for n, (train, test) in enumerate(kf):
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        print "Fold {}".format(n)

        noise = np.exp(-2)

        #Gaussian GP
        print "Gauss GP"
        kernelgp = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1], variance=0.01)
        mgp = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelgp)
        mgp.ensure_default_constraints()
        mgp['noise'] = noise
        mgp.constrain_fixed('white', 0.01)
        print mgp
        mgp.optimize(messages=1)
        Y_test_pred = mgp.predict(X_test)
        score_folds[0, n] = rmse(Y_test, Y_test_pred[0])
        print mgp
        print score_folds
        #plt.figure()
        #plt.scatter(X_test[:, 0], Y_test_pred[0])
        #plt.scatter(X_test[:, 0], Y_test, c='r', marker='x')
        #plt.title('GP gauss')

        print "Gaussian Laplace GP"
        sigma2_start = 1
        kernelstu = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1], variance=0.1)
        N, D = Y_train.shape
        g_distribution = GPy.likelihoods.functions.Gaussian(variance=noise, N=N, D=D)
        g_likelihood = GPy.likelihoods.Laplace(Y_train.copy(), g_distribution, opt='rasm')
        mg = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelstu, likelihood=g_likelihood)
        mg.ensure_default_constraints()
        mg.constrain_positive('noise_variance')
        mg.constrain_fixed('white', 0.01)
        mg['noise'] = noise
        print mg
        try:
            mg.optimize(messages=1)
        except Exception:
            print "Blew up"
        Y_test_pred = mg.predict(X_test)
        score_folds[1, n] = rmse(Y_test, Y_test_pred[0])
        print score_folds
        print mg
        #plt.figure()
        #plt.scatter(X_test[:, 0], Y_test_pred[0])
        #plt.scatter(X_test[:, 0], Y_test, c='r', marker='x')
        #plt.title('Lap gauss')

        #Student t likelihood
        deg_free = 5
        print "Student-T GP {}df".format(deg_free)
        kernelstu = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1], variance=0.1)
        t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=noise)
        stu_t_likelihood = GPy.likelihoods.Laplace(Y_train.copy(), t_distribution, opt='rasm')
        mstu_t = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelstu, likelihood=stu_t_likelihood)
        mstu_t.ensure_default_constraints()
        mstu_t.constrain_fixed('white', 0.01)
        #mstu_t.constrain_positive('t_noise')
        mstu_t.constrain_bounded('t_noise', 0.001, 1000)
        mstu_t['t_noise'] = noise
        print mstu_t
        try:
            mstu_t.optimize(messages=1)
        except Exception:
            print "Blew up"
        Y_test_pred = mstu_t.predict(X_test)
        score_folds[2, n] = rmse(Y_test, Y_test_pred[0])
        print score_folds
        print mstu_t
        #plt.figure()
        #plt.scatter(X_test[:, 0], Y_test_pred[0])
        #plt.scatter(X_test[:, 0], Y_test, c='r', marker='x')
        #plt.title('Stu t {}df'.format(deg_free))

        deg_free = 3
        print "Student-T GP {}df".format(deg_free)
        kernelstu = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1], variance=0.1)
        t_distribution = GPy.likelihoods.functions.StudentT(deg_free, sigma2=noise)
        stu_t_likelihood = GPy.likelihoods.Laplace(Y_train.copy(), t_distribution, opt='rasm')
        mstu_t = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelstu, likelihood=stu_t_likelihood)
        mstu_t.ensure_default_constraints()
        mstu_t.constrain_fixed('white', 0.01)
        #mstu_t.constrain_positive('t_noise')
        mstu_t.constrain_bounded('t_noise', 0.001, 1000)
        mstu_t['t_noise'] = noise
        print mstu_t
        try:
            mstu_t.optimize(messages=1)
        except Exception:
            print "Blew up"
        mstu_t.optimize(messages=1)
        Y_test_pred = mstu_t.predict(X_test)
        score_folds[3, n] = rmse(Y_test, Y_test_pred[0])
        print score_folds
        print mstu_t
        #plt.figure()
        #plt.scatter(X_test[:, 0], Y_test_pred[0])
        #plt.scatter(X_test[:, 0], Y_test, c='r', marker='x')
        #plt.title('Stu t {}df'.format(deg_free))


def plot_f_approx(model):
    plt.figure()
    model.plot(ax=plt.gca())
    plt.plot(model.X, model.likelihood.f_hat, c='g')
