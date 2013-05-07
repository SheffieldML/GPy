'''
Created on 26 Apr 2013

@author: maxz
'''
import unittest
import numpy
from GPy.inference.conjugate_gradient_descent import CGD, RUNNING
import pylab
import time
from scipy.optimize.optimize import rosen, rosen_der


class Test(unittest.TestCase):

    def testMinimizeSquare(self):
        N = 100
        A = numpy.random.rand(N) * numpy.eye(N)
        b = numpy.random.rand(N) * 0
        f = lambda x: numpy.dot(x.T.dot(A), x) - numpy.dot(x.T, b)
        df = lambda x: numpy.dot(A, x) - b

        opt = CGD()

        restarts = 10
        for _ in range(restarts):
            try:
                x0 = numpy.random.randn(N) * 300
                res = opt.opt(f, df, x0, messages=0,
                               maxiter=1000, gtol=1e-10)
                assert numpy.allclose(res[0], 0, atol=1e-3)
                break
            except:
                # RESTART
                pass
        else:
            raise AssertionError("Test failed for {} restarts".format(restarts))

    def testRosen(self):
        N = 20
        f = rosen
        df = rosen_der

        opt = CGD()

        restarts = 10
        for _ in range(restarts):
            try:
                x0 = numpy.random.randn(N) * .5
                res = opt.opt(f, df, x0, messages=0,
                               maxiter=5e2, gtol=1e-2)
                assert numpy.allclose(res[0], 1, atol=.1)
                break
            except:
                # RESTART
                pass
        else:
            raise AssertionError("Test failed for {} restarts".format(restarts))

if __name__ == "__main__":
#     import sys;sys.argv = ['',
#                            'Test.testMinimizeSquare',
#                            'Test.testRosen',
#                            ]
#     unittest.main()

    N = 2
    A = numpy.random.rand(N) * numpy.eye(N)
    b = numpy.random.rand(N) * 0
#     f = lambda x: numpy.dot(x.T.dot(A), x) - numpy.dot(x.T, b)
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

    raw_input("enter to start optimize")
    res = [0]

    def callback(*r):
        xopts.append(r[0].copy())
#         time.sleep(.3)
        optplts._verts3d = [numpy.array(xopts)[:, 0], numpy.array(xopts)[:, 1], [f(xs) for xs in xopts]]
        fig.canvas.draw()
        if r[-1] != RUNNING:
            res[0] = r

    p, c = opt.opt_async(f, df, x0.copy(), callback, messages=True, maxiter=1000,
                   report_every=20, gtol=1e-12)


    pylab.ion()
    pylab.show()

    pass
