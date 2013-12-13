# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
import inspect
import pkgutil
import os
import random
from nose.tools import nottest
import sys
import itertools

class ExamplesTests(unittest.TestCase):
    def _checkgrad(self, Model):
        self.assertTrue(Model.checkgrad())

    def _model_instance(self, Model):
        self.assertTrue(isinstance(Model, GPy.models))

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
    model.randomize()
    #assert model.checkgrad()
    return model.checkgrad()

def model_instance(model):
    #assert isinstance(model, GPy.core.model)
    return isinstance(model, GPy.core.model.Model)

def flatten_nested(lst):
    result = []
    for element in lst:
        if hasattr(element, '__iter__'):
            result.extend(flatten_nested(element))
        else:
            result.append(element)
    return result

#@nottest
def test_models():
    optimize=False
    plot=True
    examples_path = os.path.dirname(GPy.examples.__file__)
    # Load modules
    failing_models = {}
    for loader, module_name, is_pkg in pkgutil.iter_modules([examples_path]):
        # Load examples
        module_examples = loader.find_module(module_name).load_module(module_name)
        print "MODULE", module_examples
        print "Before"
        print inspect.getmembers(module_examples, predicate=inspect.isfunction)
        functions = [ func for func in inspect.getmembers(module_examples, predicate=inspect.isfunction) if func[0].startswith('_') is False ][::-1]
        print "After"
        print functions
        for example in functions:
            #if example[0] in ['oil', 'silhouette', 'GPLVM_oil_100', 'brendan_faces']:
                #print "SKIPPING"
                #continue

            print "Testing example: ", example[0]
            # Generate model

            try:
                models = [ example[1](optimize=optimize, plot=plot) ]
                #If more than one model returned, flatten them
                models = flatten_nested(models)
            except Exception as e:
                failing_models[example[0]] = "Cannot make model: \n{e}".format(e=e)
            else:
                print models
                model_checkgrads.description = 'test_checkgrads_%s' % example[0]
                try:
                    for model in models:
                        if not model_checkgrads(model):
                            failing_models[model_checkgrads.description] = False
                except Exception as e:
                    failing_models[model_checkgrads.description] = e

                model_instance.description = 'test_instance_%s' % example[0]
                try:
                    for model in models:
                        if not model_instance(model):
                            failing_models[model_instance.description] = False
                except Exception as e:
                    failing_models[model_instance.description] = e

            #yield model_checkgrads, model
            #yield model_instance, model

        print "Finished checking module {m}".format(m=module_name)
        if len(failing_models.keys()) > 0:
            print "Failing models: "
            print failing_models

    if len(failing_models.keys()) > 0:
        print failing_models
        raise Exception(failing_models)


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    # unittest.main()
    test_models()
