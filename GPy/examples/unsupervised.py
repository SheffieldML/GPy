"""
Usupervised learning with Gaussian Processes.
"""
import pylab as pb
import numpy as np
import GPy


######################################
## Oil data subsampled to 100 points.
def oil_100():
    data = GPy.util.datasets.oil_100()

    # create simple GP model
    m = GPy.models.GPLVM(data['X'], 2)


    # optimize
    m.ensure_default_constraints()
    m.optimize()

    # plot
    print(m)
    return m

