import numpy as np
import GPy

def load_housing_data():
    X = np.loadtxt('housing.data')
    X, Y = X[:,:-1], X[:,-1:]

    #scale the X data
    xmax, xmin = X.max(0), X.min(0)
    X = (X-xmin)/(xmax-xmin)

    #loy the response
    Y = np.log(Y)
    return X, Y

def fit_full_GP():
    X, Y = load_housing_data()
    k = GPy.kern.RBF(X.shape[1], ARD=True) + GPy.kern.Linear(X.shape[1])
    m = GPy.models.GPRegression(X, Y, kernel=k)
    m.optimize('bfgs', max_iters=400, gtol=0)
    return m

def fit_svgp_st():
    np.random.seed(0)
    X, Y = load_housing_data()

    Z = X[np.random.permutation(X.shape[0])[:100]]
    k = GPy.kern.RBF(X.shape[1], ARD=True) + GPy.kern.Linear(X.shape[1]) + GPy.kern.White(1,0.01)

    lik = GPy.likelihoods.StudentT(deg_free=3.)
    m = GPy.core.SVGP(X, Y, Z=Z, kernel=k, likelihood=lik)
    [m.optimize('scg', max_iters=40, gtol=0, messages=1, xtol=0, ftol=0) for i in range(10)]
    m.optimize('bfgs', max_iters=4000, gtol=0, messages=1, xtol=0, ftol=0)
    return m






if __name__=="__main__":
    import timeit


