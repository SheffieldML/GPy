# Copyright (c) 2014, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import numpy as np
from GPy.util import datasets
try:
    import matplotlib.pyplot as plt
except:
    pass

def student_t_approx(optimize=True, plot=True):
    """
    Example of regressing with a student t likelihood using Laplace
    """
    real_std = 0.1
    #Start a function, any function
    X = np.linspace(0.0, np.pi*2, 100)[:, None]
    Y = np.sin(X) + np.random.randn(*X.shape)*real_std
    Y = Y/Y.max()
    Yc = Y.copy()

    X_full = np.linspace(0.0, np.pi*2, 500)[:, None]
    Y_full = np.sin(X_full)
    Y_full = Y_full/Y_full.max()

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
    deg_free = 1
    print("Real noise: ", real_std)
    initial_var_guess = 0.5
    edited_real_sd = initial_var_guess

    # Kernel object
    kernel1 = GPy.kern.RBF(X.shape[1]) + GPy.kern.White(X.shape[1])
    kernel2 = GPy.kern.RBF(X.shape[1]) + GPy.kern.White(X.shape[1])
    kernel3 = GPy.kern.RBF(X.shape[1]) + GPy.kern.White(X.shape[1])
    kernel4 = GPy.kern.RBF(X.shape[1]) + GPy.kern.White(X.shape[1])

    #Gaussian GP model on clean data
    m1 = GPy.models.GPRegression(X, Y.copy(), kernel=kernel1)
    # optimize
    m1['.*white'].constrain_fixed(1e-5)
    m1.randomize()

    #Gaussian GP model on corrupt data
    m2 = GPy.models.GPRegression(X, Yc.copy(), kernel=kernel2)
    m2['.*white'].constrain_fixed(1e-5)
    m2.randomize()

    #Student t GP model on clean data
    t_distribution = GPy.likelihoods.StudentT(deg_free=deg_free, sigma2=edited_real_sd)
    laplace_inf = GPy.inference.latent_function_inference.Laplace()
    m3 = GPy.core.GP(X, Y.copy(), kernel3, likelihood=t_distribution, inference_method=laplace_inf)
    m3['.*t_scale2'].constrain_bounded(1e-6, 10.)
    m3['.*white'].constrain_fixed(1e-5)
    m3.randomize()

    #Student t GP model on corrupt data
    t_distribution = GPy.likelihoods.StudentT(deg_free=deg_free, sigma2=edited_real_sd)
    laplace_inf = GPy.inference.latent_function_inference.Laplace()
    m4 = GPy.core.GP(X, Yc.copy(), kernel4, likelihood=t_distribution, inference_method=laplace_inf)
    m4['.*t_scale2'].constrain_bounded(1e-6, 10.)
    m4['.*white'].constrain_fixed(1e-5)
    m4.randomize()
    print(m4)
    debug=True
    if debug:
        m4.optimize(messages=1)
        from matplotlib import pyplot as pb
        pb.plot(m4.X, m4.inference_method.f_hat)
        pb.plot(m4.X, m4.Y, 'rx')
        m4.plot()
        print(m4)
        return m4

    if optimize:
        optimizer='scg'
        print("Clean Gaussian")
        m1.optimize(optimizer, messages=1)
        print("Corrupt Gaussian")
        m2.optimize(optimizer, messages=1)
        print("Clean student t")
        m3.optimize(optimizer, messages=1)
        print("Corrupt student t")
        m4.optimize(optimizer, messages=1)

    if plot:
        plt.figure(1)
        plt.suptitle('Gaussian likelihood')
        ax = plt.subplot(211)
        m1.plot(ax=ax)
        plt.plot(X_full, Y_full)
        plt.ylim(-1.5, 1.5)
        plt.title('Gaussian clean')

        ax = plt.subplot(212)
        m2.plot(ax=ax)
        plt.plot(X_full, Y_full)
        plt.ylim(-1.5, 1.5)
        plt.title('Gaussian corrupt')

        plt.figure(2)
        plt.suptitle('Student-t likelihood')
        ax = plt.subplot(211)
        m3.plot(ax=ax)
        plt.plot(X_full, Y_full)
        plt.ylim(-1.5, 1.5)
        plt.title('Student-t rasm clean')

        ax = plt.subplot(212)
        m4.plot(ax=ax)
        plt.plot(X_full, Y_full)
        plt.ylim(-1.5, 1.5)
        plt.title('Student-t rasm corrupt')

    return m1, m2, m3, m4

def boston_example(optimize=True, plot=True):
    raise NotImplementedError("Needs updating")
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
        print("Fold {}".format(n))

        noise = 1e-1 #np.exp(-2)
        rbf_len = 0.5
        data_axis_plot = 4
        kernelstu = GPy.kern.RBF(X.shape[1]) + GPy.kern.white(X.shape[1]) + GPy.kern.bias(X.shape[1])
        kernelgp = GPy.kern.RBF(X.shape[1]) + GPy.kern.white(X.shape[1]) + GPy.kern.bias(X.shape[1])

        #Baseline
        score_folds[0, n] = rmse(Y_test, np.mean(Y_train))

        #Gaussian GP
        print("Gauss GP")
        mgp = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelgp.copy())
        mgp.constrain_fixed('.*white', 1e-5)
        mgp['.*len'] = rbf_len
        mgp['.*noise'] = noise
        print(mgp)
        if optimize:
            mgp.optimize(optimizer=optimizer, messages=messages)
        Y_test_pred = mgp.predict(X_test)
        score_folds[1, n] = rmse(Y_test, Y_test_pred[0])
        pred_density[1, n] = np.mean(mgp.log_predictive_density(X_test, Y_test))
        print(mgp)
        print(pred_density)

        print("Gaussian Laplace GP")
        N, D = Y_train.shape
        g_distribution = GPy.likelihoods.noise_model_constructors.gaussian(variance=noise, N=N, D=D)
        g_likelihood = GPy.likelihoods.Laplace(Y_train.copy(), g_distribution)
        mg = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelstu.copy(), likelihood=g_likelihood)
        mg.constrain_positive('noise_variance')
        mg.constrain_fixed('.*white', 1e-5)
        mg['rbf_len'] = rbf_len
        mg['noise'] = noise
        print(mg)
        if optimize:
            mg.optimize(optimizer=optimizer, messages=messages)
        Y_test_pred = mg.predict(X_test)
        score_folds[2, n] = rmse(Y_test, Y_test_pred[0])
        pred_density[2, n] = np.mean(mg.log_predictive_density(X_test, Y_test))
        print(pred_density)
        print(mg)

        for stu_num, df in enumerate(degrees_freedoms):
            #Student T
            print("Student-T GP {}df".format(df))
            t_distribution = GPy.likelihoods.noise_model_constructors.student_t(deg_free=df, sigma2=noise)
            stu_t_likelihood = GPy.likelihoods.Laplace(Y_train.copy(), t_distribution)
            mstu_t = GPy.models.GPRegression(X_train.copy(), Y_train.copy(), kernel=kernelstu.copy(), likelihood=stu_t_likelihood)
            mstu_t.constrain_fixed('.*white', 1e-5)
            mstu_t.constrain_bounded('.*t_scale2', 0.0001, 1000)
            mstu_t['rbf_len'] = rbf_len
            mstu_t['.*t_scale2'] = noise
            print(mstu_t)
            if optimize:
                mstu_t.optimize(optimizer=optimizer, messages=messages)
            Y_test_pred = mstu_t.predict(X_test)
            score_folds[3+stu_num, n] = rmse(Y_test, Y_test_pred[0])
            pred_density[3+stu_num, n] = np.mean(mstu_t.log_predictive_density(X_test, Y_test))
            print(pred_density)
            print(mstu_t)

    if plot:
        plt.figure()
        plt.scatter(X_test[:, data_axis_plot], Y_test_pred[0])
        plt.scatter(X_test[:, data_axis_plot], Y_test, c='r', marker='x')
        plt.title('GP gauss')

        plt.figure()
        plt.scatter(X_test[:, data_axis_plot], Y_test_pred[0])
        plt.scatter(X_test[:, data_axis_plot], Y_test, c='r', marker='x')
        plt.title('Lap gauss')

        plt.figure()
        plt.scatter(X_test[:, data_axis_plot], Y_test_pred[0])
        plt.scatter(X_test[:, data_axis_plot], Y_test, c='r', marker='x')
        plt.title('Stu t {}df'.format(df))

    print("Average scores: {}".format(np.mean(score_folds, 1)))
    print("Average pred density: {}".format(np.mean(pred_density, 1)))

    if plot:
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

#def precipitation_example():
    #import sklearn
    #from sklearn.cross_validation import KFold
    #data = datasets.boston_housing()
    #X = data['X'].copy()
    #Y = data['Y'].copy()
    #X = X-X.mean(axis=0)
    #X = X/X.std(axis=0)
    #Y = Y-Y.mean()
    #Y = Y/Y.std()
    #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    #num_folds = 10
    #kf = KFold(len(Y), n_folds=num_folds, indices=True)
    #score_folds = np.zeros((4, num_folds))
    #def rmse(Y, Ystar):
        #return np.sqrt(np.mean((Y-Ystar)**2))
    ##for train, test in kf:
    #for n, (train, test) in enumerate(kf):
        #X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        #print "Fold {}".format(n)

