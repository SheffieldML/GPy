# The detailed explanations of the commands used in this file can be found in the tutorial section

import pylab as pb
import numpy as np
import GPy
pb.ion()

ker1 = GPy.kern.rbf(1)  # Equivalent to ker1 = GPy.kern.rbf(D=1, variance=1., lengthscale=1.)
ker2 = GPy.kern.rbf(D=1, variance = .75, lengthscale=2.)
ker3 = GPy.kern.rbf(1, .5, .5)

print ker2
ker1.plot()
ker2.plot()
ker3.plot()

k1 = GPy.kern.rbf(1,1.,2.)
k2 = GPy.kern.Matern32(1, 0.5, 0.2)

# Product of kernels
k_prod = k1.prod(k2)
k_prodorth = k1.prod_orthogonal(k2)

# Sum of kernels
k_add = k1.add(k2)
k_addorth = k1.add_orthogonal(k2)    

pb.figure(figsize=(8,8))
pb.subplot(2,2,1)
k_prod.plot()
pb.title('prod')
pb.subplot(2,2,2)
k_prodorth.plot()
pb.title('prod_orthogonal')
pb.subplot(2,2,3)
k_add.plot()
pb.title('add')
pb.subplot(2,2,4)
k_addorth.plot()
pb.title('add_orthogonal')
pb.subplots_adjust(wspace=0.3, hspace=0.3)

k1 = GPy.kern.rbf(1,1.,2)
k2 = GPy.kern.periodic_Matern52(1,variance=1e3, lengthscale=1, period = 1.5, lower=-5., upper = 5)

k = k1 * k2  # equivalent to k = k1.prod(k2)
print k

# Simulate sample paths
X = np.linspace(-5,5,501)[:,None]
Y = np.random.multivariate_normal(np.zeros(501),k.K(X),1)

# plot
pb.figure(figsize=(10,4))
pb.subplot(1,2,1)
k.plot()
pb.subplot(1,2,2)
pb.plot(X,Y.T)
pb.ylabel("Sample path")
pb.subplots_adjust(wspace=0.3)

k = (k1+k2)*(k1+k2)
print k.parts[0].name, '\n', k.parts[1].name, '\n', k.parts[2].name, '\n', k.parts[3].name

k1 = GPy.kern.rbf(1)
k2 = GPy.kern.Matern32(1)
k3 = GPy.kern.white(1)

k = k1 + k2 + k3
print k

k.constrain_positive('var')
k.constrain_fixed(np.array([1]),1.75)
k.tie_param('len')
k.unconstrain('white')
k.constrain_bounded('white',lower=1e-5,upper=.5)
print k

k_cst = GPy.kern.bias(1,variance=1.)
k_mat = GPy.kern.Matern52(1,variance=1., lengthscale=3)
Kanova = (k_cst + k_mat).prod_orthogonal(k_cst + k_mat)
print Kanova

# sample inputs and outputs
X = np.random.uniform(-3.,3.,(40,2))
Y = 0.5*X[:,:1] + 0.5*X[:,1:] + 2*np.sin(X[:,:1]) * np.sin(X[:,1:])

# Create GP regression model
m = GPy.models.GP_regression(X,Y,Kanova)
pb.figure(figsize=(5,5))
m.plot()

pb.figure(figsize=(20,3))
pb.subplots_adjust(wspace=0.5)
pb.subplot(1,5,1)
m.plot()
pb.subplot(1,5,2)
pb.ylabel("=   ",rotation='horizontal',fontsize='30')
pb.subplot(1,5,3)
m.plot(which_functions=[False,True,False,False])
pb.ylabel("cst          +",rotation='horizontal',fontsize='30')
pb.subplot(1,5,4)
m.plot(which_functions=[False,False,True,False])
pb.ylabel("+   ",rotation='horizontal',fontsize='30')
pb.subplot(1,5,5)
pb.ylabel("+   ",rotation='horizontal',fontsize='30')
m.plot(which_functions=[False,False,False,True])

import pylab as pb
import numpy as np
import GPy
pb.ion()

ker1 = GPy.kern.rbf(D=1)  # Equivalent to ker1 = GPy.kern.rbf(D=1, variance=1., lengthscale=1.)
ker2 = GPy.kern.rbf(D=1, variance = .75, lengthscale=3.)
ker3 = GPy.kern.rbf(1, .5, .25)

ker1.plot()
ker2.plot()
ker3.plot()
#pb.savefig("Figures/tuto_kern_overview_basicdef.png")

kernels = [GPy.kern.rbf(1), GPy.kern.exponential(1), GPy.kern.Matern32(1), GPy.kern.Matern52(1),  GPy.kern.Brownian(1), GPy.kern.bias(1), GPy.kern.linear(1), GPy.kern.spline(1), GPy.kern.periodic_exponential(1), GPy.kern.periodic_Matern32(1), GPy.kern.periodic_Matern52(1), GPy.kern.white(1)]
kernel_names = ["GPy.kern.rbf", "GPy.kern.exponential", "GPy.kern.Matern32", "GPy.kern.Matern52", "GPy.kern.Brownian", "GPy.kern.bias", "GPy.kern.linear", "GPy.kern.spline", "GPy.kern.periodic_exponential", "GPy.kern.periodic_Matern32", "GPy.kern.periodic_Matern52", "GPy.kern.white"]

pb.figure(figsize=(16,12))
pb.subplots_adjust(wspace=.5, hspace=.5)
for i, kern in enumerate(kernels):
   pb.subplot(3,4,i+1)
   kern.plot(x=7.5,plot_limits=[0.00001,15.])
   pb.title(kernel_names[i]+ '\n')

# actual plot for the noise
i = 11
X = np.linspace(0.,15.,201)
WN = 0*X
WN[100] = 1.
pb.subplot(3,4,i+1)
pb.plot(X,WN,'b')
