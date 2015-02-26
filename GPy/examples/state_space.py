import GPy
import numpy as np
import matplotlib.pyplot as plt
from GPy.models.state_space import StateSpace

X = np.linspace(0, 10, 2000)[:, None]
Y = np.sin(X) + np.random.randn(*X.shape)*0.1

kernel = GPy.kern.Matern32(X.shape[1])
m = StateSpace(X,Y, kernel)
