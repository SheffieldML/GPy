import numpy as np
import pylab as pb
pb.ion()
import sys
import GPy

pb.close('all')

N = 1000
M = 10
resolution=5

X = np.linspace(0,12,N)[:,None]
Z = np.linspace(0,12,M)[:,None] # inducing points (fixed for now)
Y = np.sin(X) + np.random.randn(*X.shape)/np.sqrt(50.)
k = GPy.kern.rbf(1)


models = [GPy.models.sparse_GP_regression(X,Y,Z=Z,kernel=k),
    GPy.models.sgp_debugB(X,Y,Z=Z,kernel=k),
    GPy.models.sgp_debugC(X,Y,Z=Z,kernel=k),
    GPy.models.sgp_debugE(X,Y,Z=Z,kernel=k)]
#[m.constrain_fixed('iip') for m in models]
#m.constrain_fixed('white',1e-6)
#[m.constrain_fixed('precision',50) for m in models]
#[m.ensure_default_constraints() for m in models]


xx,yy = np.mgrid[1.5:4:0+resolution*1j,-2:2:0+resolution*1j]

lls = []
cgs = []
for l,v in zip(xx.flatten(),yy.flatten()):
    [m.set('lengthscale',l) for m in models]
    [m.set('rbf_variance',10.**v) for m in models]
    lls.append(models[0].log_likelihood())
    cgs.append([m.checkgrad(verbose=0,return_ratio=True) for m in models])

lls = np.array(lls).reshape(resolution,resolution)
cgs = np.array(zip(*cgs),dtype=np.float64).reshape(-1,resolution,resolution)

for cg in cgs:
    pb.figure()
    pb.contourf(xx,yy,lls,cmap=pb.cm.jet)
    pb.colorbar()
    pb.scatter(xx.flatten(),yy.flatten(),20,np.log(np.abs(cg.flatten())),cmap=pb.cm.gray,linewidth=0)
    pb.colorbar()

