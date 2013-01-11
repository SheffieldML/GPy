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


m = GPy.models.sparse_GP_regression(X,Y,Z=Z,kernel=k)
m.constrain_fixed('iip')
#m.constrain_fixed('white',1e-6)
m.constrain_fixed('precision',50)
m.ensure_default_constraints()


xx,yy = np.mgrid[1.5:4:0+resolution*1j,-2:2:0+resolution*1j]

lls = []
cgs = []
for l,v in zip(xx.flatten(),yy.flatten()):
    m.set('lengthscale',l)
    m.set('rbf_variance',10.**v)
    lls.append(m.log_likelihood())
    cgs.append(m.checkgrad())
    #m.plot()

lls = np.array(lls).reshape(resolution,resolution)
cgs = np.array(cgs,dtype=np.float64).reshape(resolution,resolution)

pb.contourf(xx,yy,lls,np.linspace(-500,560,100),linewidths=2,cmap=pb.cm.jet)
pb.colorbar()
pb.scatter(xx.flatten(),yy.flatten(),10,cgs.flatten(),linewidth=0,cmap=pb.cm.gray)
pb.figure()
#pb.imshow(lls,origin='upper',cmap=pb.cm.jet,extent=[xx[0,0],xx[-1,0],yy[0].min(),yy[0].max()],vmin=-500)
pb.scatter(xx.flatten(),yy.flatten(),10,lls.flatten(),linewidth=0,cmap=pb.cm.jet)
pb.colorbar()
pb.figure()
#pb.imshow(cgs,origin='upper',cmap=pb.cm.jet,extent=[xx[0,0],xx[-1,0],yy[0].min(),yy[0].max()])
pb.scatter(xx.flatten(),yy.flatten(),10,cgs.flatten(),linewidth=0,cmap=pb.cm.jet)
pb.colorbar()

