# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
import inspect
import pkgutil
import os
import random


class ExamplesTests(unittest.TestCase):
    def _checkgrad(self, model):
        self.assertTrue(model.checkgrad())

    def _model_instance(self, model):
        self.assertTrue(isinstance(model, GPy.models))

"""
def model_instance_generator(model):
    def check_model_returned(self):
        self._model_instance(model)
    return check_model_returned

def checkgrads_generator(model):
    def model_checkgrads(self):
        self._checkgrad(model)
    return model_checkgrads
"""

def model_checkgrads(model):
    assert model.checkgrad()


def model_instance(model):
    assert isinstance(model, GPy.core.model)


def test_models():
    examples_path = os.path.dirname(GPy.examples.__file__)
    #Load modules
    for loader, module_name, is_pkg in pkgutil.iter_modules([examples_path]):
        #Load examples
        module_examples = loader.find_module(module_name).load_module(module_name)
        print "MODULE", module_examples
        print "Before"
        print inspect.getmembers(module_examples, predicate=inspect.isfunction)
        functions = [ func for func in inspect.getmembers(module_examples, predicate=inspect.isfunction) if func[0].startswith('_') is False ][::-1]
        print "After"
        print functions
        for example in functions:
            print "Testing example: ", example[0]
            #Generate model
            model = example[1]()
            print model

            #Create tests for instance check
            """
            test = model_instance_generator(model)
            test.__name__ = 'test_instance_%s' % example[0]
            setattr(ExamplesTests, test.__name__, test)

            #Create tests for checkgrads check
            test = checkgrads_generator(model)
            test.__name__ = 'test_checkgrads_%s' % example[0]
            setattr(ExamplesTests, test.__name__, test)
            """
            model_checkgrads.description = 'test_checkgrads_%s' % example[0]
            yield model_checkgrads, model
            model_instance.description = 'test_instance_%s' % example[0]
            yield model_instance, model

if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
