# Copyright (c) 2012-2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy

class GDUpdateRule():
    _gradnat = None
    _gradnatold = None
    def __init__(self, initgrad, initgradnat=None):
        self.grad = initgrad
        if initgradnat:
            self.gradnat = initgradnat
        else:
            self.gradnat = initgrad
        # self.grad, self.gradnat
    def _gamma(self):
        raise NotImplemented("""Implement gamma update rule here, 
        you can use self.grad and self.gradold for parameters, as well as
        self.gradnat and self.gradnatold for natural gradients.""")
    def __call__(self, grad, gradnat=None, si=None, *args, **kw):
        """
        Return gamma for given gradients and optional natural gradients
        """
        if not gradnat:
            gradnat = grad
        self.gradold = self.grad
        self.gradnatold = self.gradnat
        self.grad = grad
        self.gradnat = gradnat
        self.si = si
        return self._gamma(*args, **kw)

class FletcherReeves(GDUpdateRule):
    '''
    Fletcher Reeves update rule for gamma
    '''
    def _gamma(self, *a, **kw):
        tmp = numpy.dot(self.grad.T, self.gradnat)
        if tmp:
            return tmp / numpy.dot(self.gradold.T, self.gradnatold)
        return tmp

class PolakRibiere(GDUpdateRule):
    '''
    Fletcher Reeves update rule for gamma
    '''
    def _gamma(self, *a, **kw):
        tmp = numpy.dot((self.grad - self.gradold).T, self.gradnat)
        if tmp:
            return tmp / numpy.dot(self.gradold.T, self.gradnatold)
        return tmp
