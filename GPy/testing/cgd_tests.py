'''
Created on 26 Apr 2013

@author: maxz
'''
import unittest
import numpy
from GPy.inference.conjugate_gradient_descent import CGD
import pylab
import time
from scipy.optimize.optimize import rosen, rosen_der


class Test(unittest.TestCase):

    def testMinimizeSquare(self):
        f = lambda x: x ** 2 + 2 * x - 2

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testMinimizeSquare']
#     unittest.main()
    N = 2
    A = numpy.random.rand(N) * numpy.eye(N)
    b = numpy.random.rand(N)
#     f = lambda x: numpy.dot(x.T.dot(A), x) + numpy.dot(x.T, b)
#     df = lambda x: numpy.dot(A, x) - b

    f = rosen
    df = rosen_der
    x0 = numpy.random.randn(N) * .5

    opt = CGD()

    fig = pylab.figure("cgd optimize")
    if fig.axes:
        ax = fig.axes[0]
        ax.cla()
    else:
        ax = fig.add_subplot(111, projection='3d')

    interpolation = 40
    x, y = numpy.linspace(-1, 1, interpolation)[:, None], numpy.linspace(-1, 1, interpolation)[:, None]
    X, Y = numpy.meshgrid(x, y)
    fXY = numpy.array([f(numpy.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())]).reshape(interpolation, interpolation)

    ax.plot_wireframe(X, Y, fXY)
    xopts = [x0.copy()]
    optplts, = ax.plot3D([x0[0]], [x0[1]], zs=f(x0), marker='o', color='r')

    def callback(x, *a, **kw):
        xopts.append(x.copy())
        time.sleep(.3)
        optplts._verts3d = [numpy.array(xopts)[:, 0], numpy.array(xopts)[:, 1], [f(xs) for xs in xopts]]
        fig.canvas.draw()

    res = opt.fmin(f, df, x0, callback, messages=True, report_every=1)
