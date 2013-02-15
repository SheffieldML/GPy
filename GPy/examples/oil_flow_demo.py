# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import cPickle as pickle
import numpy as np
import pylab as pb
import GPy
import pylab as plt
np.random.seed(3)

def plot_oil(X, theta, labels, label):
    plt.figure()
    X = X[:,np.argsort(theta)[:2]]
    flow_type = (X[labels[:,0]==1])
    plt.plot(flow_type[:,0], flow_type[:,1], 'rx')
    flow_type = (X[labels[:,1]==1])
    plt.plot(flow_type[:,0], flow_type[:,1], 'gx')
    flow_type = (X[labels[:,2]==1])
    plt.plot(flow_type[:,0], flow_type[:,1], 'bx')
    plt.title(label)

data = pickle.load(open('../../../GPy_assembla/datasets/oil_flow_3classes.pickle', 'r'))

Y = data['DataTrn']
N, D = Y.shape
selected = np.random.permutation(N)[:350]
labels = data['DataTrnLbls'][selected]
Y = Y[selected]
N, D = Y.shape
Y -= Y.mean(axis=0)
# Y /= Y.std(axis=0)

Q = 5
k = GPy.kern.linear(Q, ARD = True) + GPy.kern.white(Q)
m = GPy.models.Bayesian_GPLVM(Y, Q, kernel = k, M = 20)
m.constrain_positive('(rbf|bias|S|linear|white|noise)')

# m.unconstrain('noise')
# m.constrain_fixed('noise_precision', 50.0)
# m.unconstrain('white')
# m.constrain_bounded('white', 1e-6, 10.0)
# plot_oil(m.X, np.array([1,1]), labels, 'PCA initialization')
m.optimize(messages = True)
# m.optimize('tnc', messages = True)
# plot_oil(m.X, m.kern.parts[0].lengthscale, labels, 'B-GPLVM')
# # pb.figure()
# m.plot()
# pb.title('PCA initialisation')
# pb.figure()
# m.optimize(messages = 1)
# m.plot()
# pb.title('After optimisation')
# m = GPy.models.GPLVM(Y, Q)
# m.constrain_positive('(white|rbf|bias|noise)')
# m.optimize()
# plot_oil(m.X, np.array([1,1]), labels, 'GPLVM')
