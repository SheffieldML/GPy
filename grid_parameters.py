import numpy as np
import pylab as pb
pb.ion()
import sys
import GPy

pb.close('all')

N = 200
M = 15
resolution=5

X = np.linspace(0,12,N)[:,None]
Z = np.linspace(0,12,M)[:,None] # inducing points (fixed for now)
Y = np.sin(X) + np.random.randn(*X.shape)/np.sqrt(50.)
#k = GPy.kern.rbf(1)
k = GPy.kern.Matern32(1) + GPy.kern.white(1)

models = [GPy.models.sparse_GP_regression(X,Y,Z=Z,kernel=k)
    ,GPy.models.sparse_GP_regression(X,Y,Z=Z,kernel=k)
    ,GPy.models.sparse_GP_regression(X,Y,Z=Z,kernel=k)
    ,GPy.models.sparse_GP_regression(X,Y,Z=Z,kernel=k)]
models[0].scale_factor = 1.
models[1].scale_factor = 10.
models[2].scale_factor = 100.
models[3].scale_factor = 1000.
    #GPy.models.sgp_debugB(X,Y,Z=Z,kernel=k),
    #GPy.models.sgp_debugC(X,Y,Z=Z,kernel=k)]#,
    #GPy.models.sgp_debugE(X,Y,Z=Z,kernel=k)]

[m.constrain_fixed('white',0.1) for m in models]

#xx,yy = np.mgrid[1.5:4:0+resolution*1j,-2:2:0+resolution*1j]
xx,yy = np.mgrid[3:16:0+resolution*1j,-2:1:0+resolution*1j]

lls = []
cgs = []
grads = []
count = 0
for l,v in zip(xx.flatten(),yy.flatten()):
    count += 1
    print count, 'of', resolution**2
    sys.stdout.flush()

    [m.set('lengthscale',l) for m in models]
    [m.set('_variance',10.**v) for m in models]
    lls.append([m.log_likelihood() for m in models])
    grads.append([m.log_likelihood_gradients() for m in models])
    cgs.append([m.checkgrad(verbose=0,return_ratio=True) for m in models])

lls = np.array(zip(*lls)).reshape(-1,resolution,resolution)
cgs = np.array(zip(*cgs)).reshape(-1,resolution,resolution)

for ll,cg in zip(lls,cgs):
    pb.figure()
    pb.contourf(xx,yy,ll,100,cmap=pb.cm.gray)
    pb.colorbar()
    try:
        pb.contour(xx,yy,np.exp(ll),colors='k')
    except:
        pass
    pb.scatter(xx.flatten(),yy.flatten(),20,np.log(np.abs(cg.flatten())),cmap=pb.cm.jet,linewidth=0)
    pb.colorbar()

