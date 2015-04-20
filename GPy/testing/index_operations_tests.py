# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
from GPy.core.parameterization.index_operations import ParameterIndexOperations,\
    ParameterIndexOperationsView

one, two, three = 'one', 'two', 'three'

class Test(unittest.TestCase):

    def setUp(self):
        self.param_index = ParameterIndexOperations()
        self.param_index.add(one, [3,9])
        self.param_index.add(two, [0,5])
        self.param_index.add(three, [2,4,7,10])
        self.view = ParameterIndexOperationsView(self.param_index, 2, 6)

    def test_clear(self):
        self.param_index.clear()
        self.assertDictEqual(self.param_index._properties, {})

    def test_remove(self):
        removed = self.param_index.remove(three, np.r_[3:13])
        self.assertListEqual(removed.tolist(), [4,7,10])
        self.assertListEqual(self.param_index[three].tolist(), [2])
        removed = self.param_index.remove(one, [1])
        self.assertListEqual(removed.tolist(), [])
        self.assertListEqual(self.param_index[one].tolist(), [3,9])
        self.assertListEqual(self.param_index.remove('not in there', []).tolist(), [])
        removed = self.param_index.remove(one, [9])
        self.assertListEqual(removed.tolist(), [9])
        self.assertListEqual(self.param_index[one].tolist(), [3])
        self.assertListEqual(self.param_index.remove('not in there', [2,3,4]).tolist(), [])
        self.assertListEqual(self.view.remove('not in there', [2,3,4]).tolist(), [])

    def test_shift_left(self):
        self.view.shift_left(0, 2)
        self.assertListEqual(self.param_index[three].tolist(), [2,5,8])
        self.assertListEqual(self.param_index[two].tolist(), [0,3])
        self.assertListEqual(self.param_index[one].tolist(), [7])
        #=======================================================================
        #          0    1    2    3    4    5    6    7    8    9    10
        #                                            one
        #         two            two
        #                   three          three          three
        # view:             [0    1    2    3    4    5    ]
        #=======================================================================
        self.assertListEqual(self.view[three].tolist(), [0,3])
        self.assertListEqual(self.view[two].tolist(), [1])
        self.assertListEqual(self.view[one].tolist(), [5])
        self.param_index.shift_left(7, 1)
        #=======================================================================
        #          0    1    2    3    4    5    6    7    8    9    10
        #
        #         two            two
        #                   three          three     three
        # view:             [0    1    2    3    4    5    ]
        #=======================================================================
        self.assertListEqual(self.param_index[three].tolist(), [2,5,7])
        self.assertListEqual(self.param_index[two].tolist(), [0,3])
        self.assertListEqual(self.param_index[one].tolist(), [])
        self.assertListEqual(self.view[three].tolist(), [0,3,5])
        self.assertListEqual(self.view[two].tolist(), [1])
        self.assertListEqual(self.view[one].tolist(), [])

    def test_shift_right(self):
        self.view.shift_right(3, 2)
        self.assertListEqual(self.param_index[three].tolist(), [2,4,9,12])
        self.assertListEqual(self.param_index[two].tolist(), [0,7])
        self.assertListEqual(self.param_index[one].tolist(), [3,11])

    def test_index_view(self):
        #=======================================================================
        #          0    1    2    3    4    5    6    7    8    9    10
        #                        one                           one
        #         two                      two
        #                   three     three          three          three
        # view:             [0    1    2    3    4    5    ]
        #=======================================================================
        self.view = ParameterIndexOperationsView(self.param_index, 2, 6)
        self.assertSetEqual(set(self.view.properties()), set([one, two, three]))
        for v,p in zip(self.view.properties_for(np.r_[:6]), self.param_index.properties_for(np.r_[2:2+6])):
            self.assertEqual(v, p)
        self.assertSetEqual(set(self.view[two]), set([3]))
        self.assertSetEqual(set(self.param_index[two]), set([0, 5]))
        self.view.add(two, np.array([0]))
        self.assertSetEqual(set(self.view[two]), set([0,3]))
        self.assertSetEqual(set(self.param_index[two]), set([0, 2, 5]))
        self.view.clear()
        for v,p in zip(self.view.properties_for(np.r_[:6]), self.param_index.properties_for(np.r_[2:2+6])):
            self.assertEqual(v, p)
            self.assertEqual(v, [])
        param_index = ParameterIndexOperations()
        param_index.add(one, [3,9])
        param_index.add(two, [0,5])
        param_index.add(three, [2,4,7,10])
        view2 = ParameterIndexOperationsView(param_index, 2, 8)
        self.view.update(view2)
        for [i,v],[i2,v2] in zip(sorted(param_index.items()), sorted(self.param_index.items())):
            self.assertEqual(i, i2)
            np.testing.assert_equal(v, v2)

    def test_view_of_view(self):
        #=======================================================================
        #          0    1    2    3    4    5    6    7    8    9    10
        #                        one                           one
        #         two                      two
        #                   three     three          three          three
        # view:             [0    1    2    3    4    5    ]
        # view2:                      [0    1    2    3    4    5    ]
        #=======================================================================
        view2 = ParameterIndexOperationsView(self.view, 2, 6)
        view2.shift_right(0, 2)

    def test_indexview_remove(self):
        removed = self.view.remove(two, [3])
        self.assertListEqual(removed.tolist(), [3])
        removed = self.view.remove(three, np.r_[:5])
        self.assertListEqual(removed.tolist(), [0, 2])

    def test_misc(self):
        #py3 fix
        #for k,v in self.param_index.copy()._properties.iteritems():
        for k,v in self.param_index.copy()._properties.items():
            self.assertListEqual(self.param_index[k].tolist(), v.tolist())
        self.assertEqual(self.param_index.size, 8)
        self.assertEqual(self.view.size, 5)

    def test_print(self):
        print(self.param_index)
        print(self.view)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_index_view']
    unittest.main()
