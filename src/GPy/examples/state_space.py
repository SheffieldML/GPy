import GPy
import numpy as np
import matplotlib.pyplot as plt

import GPy.models.state_space_model as SS_model

def state_space_example():
    X = np.linspace(0, 10, 2000)[:, None]
    Y = np.sin(X) + np.random.randn(*X.shape)*0.1

    kernel1 = GPy.kern.Matern32(X.shape[1])
    m1  = GPy.models.GPRegression(X,Y, kernel1)

    print(m1)
    m1.optimize(optimizer='bfgs',messages=True)

    print(m1)

    kernel2 = GPy.kern.sde_Matern32(X.shape[1])
    #m2  = SS_model.StateSpace(X,Y, kernel2)
    m2 = GPy.models.StateSpace(X,Y, kernel2)
    print(m2)

    m2.optimize(optimizer='bfgs',messages=True)

    print(m2)

    return m1, m2

