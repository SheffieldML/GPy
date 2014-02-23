'''
Created on 12 Feb 2014

@author: maxz
'''
import unittest
import numpy as np
from GPy.core.parameterization.index_operations import ParameterIndexOperations,\
    ParameterIndexOperationsView

one, two, three = 'one', 'two', 'three'

class Test(unittest.TestCase):

    def setUp(self):
        self.param_index = ParameterIndexOperations()
        self.param_index.add(one, [3])
        self.param_index.add(two, [0,5])
        self.param_index.add(three, [2,4,7])

    def test_remove(self):
        self.param_index.remove(three, np.r_[3:10])
        self.assertListEqual(self.param_index[three].tolist(), [2])
        self.param_index.remove(one, [1])
        self.assertListEqual(self.param_index[one].tolist(), [3])        

    def test_shift_left(self):
        self.param_index.shift_left(1, 2)
        self.assertListEqual(self.param_index[three].tolist(), [2,5])
        self.assertListEqual(self.param_index[two].tolist(), [0,3])
        self.assertListEqual(self.param_index[one].tolist(), [1])        


    def test_index_view(self):
        #=======================================================================
        #          0    1    2    3    4    5    6    7    8    9
        #                        one
        #         two                      two
        #                   three     three          three
        # view:             [0    1    2    3    4    5    ]
        #=======================================================================
        view = ParameterIndexOperationsView(self.param_index, 2, 6)
        self.assertSetEqual(set(view.properties()), set([one, two, three]))
        for v,p in zip(view.properties_for(np.r_[:6]), self.param_index.properties_for(np.r_[2:2+6])):
            self.assertEqual(v, p)
        self.assertSetEqual(set(view[two]), set([3]))
        self.assertSetEqual(set(self.param_index[two]), set([0, 5]))
        view.add(two, np.array([0]))
        self.assertSetEqual(set(view[two]), set([0,3]))
        self.assertSetEqual(set(self.param_index[two]), set([0, 2, 5]))
        view.clear()
        for v,p in zip(view.properties_for(np.r_[:6]), self.param_index.properties_for(np.r_[2:2+6])):
            self.assertEqual(v, p)
            self.assertEqual(v, [])
        param_index = ParameterIndexOperations()
        param_index.add(one, [3])
        param_index.add(two, [0,5])
        param_index.add(three, [2,4,7])
        view2 = ParameterIndexOperationsView(param_index, 2, 6)
        view.update(view2)
        for [i,v],[i2,v2] in zip(sorted(param_index.items()), sorted(self.param_index.items())):
            self.assertEqual(i, i2)
            self.assertTrue(np.all(v == v2))
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_index_view']
    unittest.main()