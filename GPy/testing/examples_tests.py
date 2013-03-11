# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class ExamplesTests(unittest.TestCase):
    def test_check_model_returned(self):
        pass

    def test_model_checkgrads(self):
        pass

    def test_all_examples(self):
        examples_module = __import__("GPy").examples
        #Load models

        #Loop through models
        #for model in models:
            #self.assertTrue(m.checkgrad())


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
