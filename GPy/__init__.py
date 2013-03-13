# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import kern
import models
import inference
import util
import examples
from core import priors
import likelihoods
import testing
from numpy.testing import Tester
from nose.tools import nottest

@nottest
def tests():
    Tester(testing).test(verbose=10)
