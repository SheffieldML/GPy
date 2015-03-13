import GPy
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 2000)[:, None]
Y = np.sin(X) + np.random.randn(*X.shape)*0.1

kernel = GPy.kern.Matern32(X.shape[1])
m = GPy.models.StateSpace(X,Y, kernel)

m.optimize()

print m

kernel1 = GPy.kern.Matern32(X.shape[1])
m1  = GPy.models.GPRegression(X,Y, kernel1)

m1.optimize()

print m1