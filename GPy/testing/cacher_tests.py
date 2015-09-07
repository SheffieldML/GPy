'''
Created on 4 Sep 2015

@author: maxz
'''
import unittest
from GPy.util.caching import Cacher
from pickle import PickleError


class Test(unittest.TestCase):
    def setUp(self):
        def op(x):
            return x
        self.cache = Cacher(op, 1)

    def test_pickling(self):
        self.assertRaises(PickleError, self.cache.__getstate__)
        self.assertRaises(PickleError, self.cache.__setstate__)

    def test_copy(self):
        tmp = self.cache.__deepcopy__()
        assert(tmp.operation is self.cache.operation)
        self.assertEqual(tmp.limit, self.cache.limit)

    def test_reset(self):
        self.cache.reset()
        self.assertDictEqual(self.cache.cached_input_ids, {}, )
        self.assertDictEqual(self.cache.cached_outputs, {}, )
        self.assertDictEqual(self.cache.inputs_changed, {}, )

    def test_name(self):
        assert(self.cache.__name__ == self.cache.operation.__name__)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()