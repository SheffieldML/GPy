# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import unittest
from GPy.core.parameterization.parameterized import Parameterized
from GPy.core.parameterization.param import Param
import numpy

# One trigger in init
_trigger_start = -1

class ParamTestParent(Parameterized):
    parent_changed_count = _trigger_start
    def parameters_changed(self):
        self.parent_changed_count += 1

class ParameterizedTest(Parameterized):
    # One trigger after initialization
    params_changed_count = _trigger_start
    def parameters_changed(self):
        self.params_changed_count += 1

class Test(unittest.TestCase):

    def setUp(self):
        self.parent = ParamTestParent('test parent')
        self.par = ParameterizedTest('test model')
        self.par2 = ParameterizedTest('test model 2')
        self.p = Param('test parameter', numpy.random.normal(1,2,(10,3)))

        self.par.link_parameter(self.p)
        self.par.link_parameter(Param('test1', numpy.random.normal(0,1,(1,))))
        self.par.link_parameter(Param('test2', numpy.random.normal(0,1,(1,))))

        self.par2.link_parameter(Param('par2 test1', numpy.random.normal(0,1,(1,))))
        self.par2.link_parameter(Param('par2 test2', numpy.random.normal(0,1,(1,))))

        self.parent.link_parameter(self.par)
        self.parent.link_parameter(self.par2)

        self._observer_triggered = None
        self._trigger_count = 0
        self._first = None
        self._second = None

    def _trigger(self, me, which):
        self._observer_triggered = which
        self._trigger_count += 1
        if self._first is not None:
            self._second = self._trigger
        else:
            self._first = self._trigger

    def _trigger_priority(self, me, which):
        if self._first is not None:
            self._second = self._trigger_priority
        else:
            self._first = self._trigger_priority

    def test_observable(self):
        self.par.add_observer(self, self._trigger, -1)
        self.assertEqual(self.par.params_changed_count, 0, 'no params changed yet')
        self.assertEqual(self.par.params_changed_count, self.parent.parent_changed_count, 'parent should be triggered as often as param')

        self.p[0,1] = 3 # trigger observers
        self.assertIs(self._observer_triggered, self.p, 'observer should have triggered')
        self.assertEqual(self._trigger_count, 1, 'observer should have triggered once')
        self.assertEqual(self.par.params_changed_count, 1, 'params changed once')
        self.assertEqual(self.par.params_changed_count, self.parent.parent_changed_count, 'parent should be triggered as often as param')

        self.par.remove_observer(self)
        self.p[0,1] = 4
        self.assertIs(self._observer_triggered, self.p, 'observer should not have triggered')
        self.assertEqual(self._trigger_count, 1, 'observer should have triggered once')
        self.assertEqual(self.par.params_changed_count, 2, 'params changed second')
        self.assertEqual(self.par.params_changed_count, self.parent.parent_changed_count, 'parent should be triggered as often as param')

        self.par.add_observer(self, self._trigger, -1)
        self.p[0,1] = 4
        self.assertIs(self._observer_triggered, self.p, 'observer should have triggered')
        self.assertEqual(self._trigger_count, 2, 'observer should have triggered once')
        self.assertEqual(self.par.params_changed_count, 3, 'params changed second')
        self.assertEqual(self.par.params_changed_count, self.parent.parent_changed_count, 'parent should be triggered as often as param')

        self.par.remove_observer(self, self._trigger)
        self.p[0,1] = 3
        self.assertIs(self._observer_triggered, self.p, 'observer should not have triggered')
        self.assertEqual(self._trigger_count, 2, 'observer should have triggered once')
        self.assertEqual(self.par.params_changed_count, 4, 'params changed second')
        self.assertEqual(self.par.params_changed_count, self.parent.parent_changed_count, 'parent should be triggered as often as param')

    def test_set_params(self):
        self.assertEqual(self.par.params_changed_count, 0, 'no params changed yet')
        self.par.param_array[:] = 1
        self.par._trigger_params_changed()
        self.assertEqual(self.par.params_changed_count, 1, 'now params changed')
        self.assertEqual(self.parent.parent_changed_count, self.par.params_changed_count)

        self.par.param_array[:] = 2
        self.par._trigger_params_changed()
        self.assertEqual(self.par.params_changed_count, 2, 'now params changed')
        self.assertEqual(self.parent.parent_changed_count, self.par.params_changed_count)


    def test_priority_notify(self):
        self.assertEqual(self.par.params_changed_count, 0)
        self.par.notify_observers(0, None)
        self.assertEqual(self.par.params_changed_count, 1)
        self.assertEqual(self.parent.parent_changed_count, self.par.params_changed_count)

        self.par.notify_observers(0, -numpy.inf)
        self.assertEqual(self.par.params_changed_count, 2)
        self.assertEqual(self.parent.parent_changed_count, 1)

    def test_priority(self):
        self.par.add_observer(self, self._trigger, -1)
        self.par.add_observer(self, self._trigger_priority, 0)
        self.par.notify_observers(0)
        self.assertEqual(self._first, self._trigger_priority, 'priority should be first')
        self.assertEqual(self._second, self._trigger, 'priority should be first')

        self.par.remove_observer(self)
        self._first = self._second = None

        self.par.add_observer(self, self._trigger, 1)
        self.par.add_observer(self, self._trigger_priority, 0)
        self.par.notify_observers(0)
        self.assertEqual(self._first, self._trigger, 'priority should be second')
        self.assertEqual(self._second, self._trigger_priority, 'priority should be second')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
