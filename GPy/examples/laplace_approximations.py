import GPy
import numpy as np
import matplotlib.pyplot as plt
from GPy.util import datasets
np.random.seed(1)

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

    #Slightly noisy data
    Yc[75:80] += 1

    #Very noisy data
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

    plt.figure(1)
    plt.suptitle('Gaussian likelihood')
    # Kernel object
    kernel1 = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1])
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
    m.constrain_fixed('white', 1e-4)
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
    m.constrain_fixed('white', 1e-4)
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
    t_distribution = GPy.likelihoods.noise_model_constructors.student_t(deg_free=deg_free, sigma2=edited_real_sd)
    stu_t_likelihood = GPy.likelihoods.Laplace(Y.copy(), t_distribution)
    m = GPy.models.GPRegression(X, Y.copy(), kernel6, likelihood=stu_t_likelihood)
    m.ensure_default_constraints()
    m.constrain_positive('t_noise')
    m.constrain_fixed('white', 1e-4)
    m.randomize()
    #m.update_likelihood_approximation()
    m.optimize()
    print(m)
    ax = plt.subplot(211)
    m.plot(ax=ax)
    plt.plot(X_full, Y_full)
    plt.ylim(-1.5, 1.5)
    plt.title('Student-t rasm clean')

    print "Corrupt student t, rasm"
    t_distribution = GPy.likelihoods.noise_model_constructors.student_t(deg_free=deg_free, sigma2=edited_real_sd)
    corrupt_stu_t_likelihood = GPy.likelihoods.Laplace(Yc.copy(), t_distribution)
    m = GPy.models.GPRegression(X, Yc.copy(), kernel4, likelihood=corrupt_stu_t_likelihood)
    m.ensure_default_constraints()
    m.constrain_positive('t_noise')
    m.constrain_fixed('white', 1e-4)
    m.randomize()
    for a in range(1):
        m.randomize()
        m_start = m.copy()
        print m
        m.optimize('scg', messages=1)
    print(m)
    ax = plt.subplot(212)
    m.plot(ax=ax)
    plt.plot(X_full, Y_full)
    plt.ylim(-1.5, 1.5)
    plt.title('Student-t rasm corrupt')

    return m

    #with a student t distribution, since it has heavy tails it should work well
    #likelihood_function = student_t(deg_free=deg_free, sigma2=real_var)
    #lap = Laplace(Y, likelihood_function)
    #cov = kernel.K(X)
    #lap.fit_full(cov)

    #test_range = np.arange(0, 10, 0.1)
    #plt.plot(test_range, t_rv.pdf(test_range))
    #for i in xrange(X.shape[0]):
        #mode = lap.f_hat[i]
        #covariance = lap.hess_hat_i[i,i]
        #scaling = np.exp(lap.ln_z_hat)
        #normalised_approx = norm(loc=mode, scale=covariance)
        #print "Normal with mode %f, and variance %f" % (mode, covariance)
        #plt.plot(test_range, scaling*normalised_approx.pdf(test_range))
    #plt.show()

    return m

def boston_example():
    import sklearn
    from sklearn.cross_validation import KFold
    optimizer='bfgs'
    messages=0
    data = datasets.boston_housing()
    degrees_freedoms = [3, 5, 8, 10]
    X = data['X'].copy()
    Y = data['Y'].copy()
    X = X-X.mean(axis=0)
    X = X/X.std(axis=0)
    Y = Y-Y.mean()
    Y = Y/Y.std()
    num_folds = 10
    kf = KFold(len(Y), n_folds=num_folds, indices=True)
    num_models = len(degrees_freedoms) + 3 #3 for baseline, gaussian, gaussian laplace approx
    score_folds = np.zeros((num_models, num_folds))
    pred_density = score_folds.copy()

    def rmse(Y, Ystar):
        return np.sqrt(np.mean((Y-Ystar)**2))

    for n, (train, test) in enumerate(kf):
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        print "Fold {}".format(n)

        noise = 1e-1 #np.exp(-2)
        rbf_len = 0.5
        data_axis_plot = 4
        plot = False
        kernelstu = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1]) + GPy.kern.bias(X.shape[1])
        kernelgp = GPy.kern.rbf(X.shape[1]) + GPy.kern.white(X.shape[1]) + GPy.kern.bias(X.shape[1])

        #Baseline
        score_folds[0, n] = rmse(Y_test, np.mean(Y_train))

        #Gaussian GP
        print "Gauss GP"
        mgp = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelgp.copy())
        mgp.ensure_default_constraints()
        mgp.constrain_fixed('white', 1e-5)
        mgp['rbf_len'] = rbf_len
        mgp['noise'] = noise
        print mgp
        mgp.optimize(optimizer=optimizer, messages=messages)
        Y_test_pred = mgp.predict(X_test)
        score_folds[1, n] = rmse(Y_test, Y_test_pred[0])
        pred_density[1, n] = np.mean(mgp.log_predictive_density(X_test, Y_test))
        print mgp
        print pred_density
        if plot:
            plt.figure()
            plt.scatter(X_test[:, data_axis_plot], Y_test_pred[0])
            plt.scatter(X_test[:, data_axis_plot], Y_test, c='r', marker='x')
            plt.title('GP gauss')

        print "Gaussian Laplace GP"
        N, D = Y_train.shape
        g_distribution = GPy.likelihoods.noise_model_constructors.gaussian(variance=noise, N=N, D=D)
        g_likelihood = GPy.likelihoods.Laplace(Y_train.copy(), g_distribution)
        mg = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelstu.copy(), likelihood=g_likelihood)
        mg.ensure_default_constraints()
        mg.constrain_positive('noise_variance')
        mg.constrain_fixed('white', 1e-5)
        mg['rbf_len'] = rbf_len
        mg['noise'] = noise
        print mg
        try:
            mg.optimize(optimizer=optimizer, messages=messages)
        except Exception:
            print "Blew up"
        Y_test_pred = mg.predict(X_test)
        score_folds[2, n] = rmse(Y_test, Y_test_pred[0])
        pred_density[2, n] = np.mean(mg.log_predictive_density(X_test, Y_test))
        print pred_density
        print mg
        if plot:
            plt.figure()
            plt.scatter(X_test[:, data_axis_plot], Y_test_pred[0])
            plt.scatter(X_test[:, data_axis_plot], Y_test, c='r', marker='x')
            plt.title('Lap gauss')

        for stu_num, df in enumerate(degrees_freedoms):
            #Student T
            print "Student-T GP {}df".format(df)
            t_distribution = GPy.likelihoods.noise_model_constructors.student_t(deg_free=df, sigma2=noise)
            stu_t_likelihood = GPy.likelihoods.Laplace(Y_train.copy(), t_distribution)
            mstu_t = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelstu.copy(), likelihood=stu_t_likelihood)
            mstu_t.ensure_default_constraints()
            mstu_t.constrain_fixed('white', 1e-5)
            mstu_t.constrain_bounded('t_noise', 0.0001, 1000)
            mstu_t['rbf_len'] = rbf_len
            mstu_t['t_noise'] = noise
            print mstu_t
            try:
                mstu_t.optimize(optimizer=optimizer, messages=messages)
            except Exception:
                print "Blew up"
            Y_test_pred = mstu_t.predict(X_test)
            score_folds[3+stu_num, n] = rmse(Y_test, Y_test_pred[0])
            pred_density[3+stu_num, n] = np.mean(mstu_t.log_predictive_density(X_test, Y_test))
            print pred_density
            print mstu_t
            if plot:
                plt.figure()
                plt.scatter(X_test[:, data_axis_plot], Y_test_pred[0])
                plt.scatter(X_test[:, data_axis_plot], Y_test, c='r', marker='x')
                plt.title('Stu t {}df'.format(df))

    print "Average scores: {}".format(np.mean(score_folds, 1))
    print "Average pred density: {}".format(np.mean(pred_density, 1))

    #Plotting
    stu_t_legends = ['Student T, df={}'.format(df) for df in degrees_freedoms]
    legends = ['Baseline', 'Gaussian', 'Laplace Approx Gaussian'] + stu_t_legends

    #Plot boxplots for RMSE density
    fig = plt.figure()
    ax=fig.add_subplot(111)
    plt.title('RMSE')
    bp = ax.boxplot(score_folds.T, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    xtickNames = plt.setp(ax, xticklabels=legends)
    plt.setp(xtickNames, rotation=45, fontsize=8)
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Distribution')
    #Make grid and put it below boxes
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
    ax.set_axisbelow(True)

    #Plot boxplots for predictive density
    fig = plt.figure()
    ax=fig.add_subplot(111)
    plt.title('Predictive density')
    bp = ax.boxplot(pred_density[1:,:].T, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    xtickNames = plt.setp(ax, xticklabels=legends[1:])
    plt.setp(xtickNames, rotation=45, fontsize=8)
    ax.set_ylabel('Mean Log probability P(Y*|Y)')
    ax.set_xlabel('Distribution')
    #Make grid and put it below boxes
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
    ax.set_axisbelow(True)
    return mstu_t

def precipitation_example():
    import sklearn
    from sklearn.cross_validation import KFold
    data = datasets.boston_housing()
    X = data['X'].copy()
    Y = data['Y'].copy()
    X = X-X.mean(axis=0)
    X = X/X.std(axis=0)
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
