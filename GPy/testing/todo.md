As off now, I am once through all of the tests and basic migration is done.

Now, fix the below things and todos before starting to get the tests running using pytest


+ update test script names according to pytest conversion
+ check for TODOs
+ + there are many associated with "iscloseto" functions from np.testing. Will have to figure out how these
+ + some tests are not that clear to me tbh
+ check nomenclature of test files and test classes and test functions
+ chatgpt says that I should replace delta with the decimal but a delta of 1e-4 should be decimal=4. Not sure about this yet  but that is something I need to fix later on
--> this gives more content to it: https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertAlmostEqual
I need to write a custom function that behaves accordingly as in some cases, np.testing.assert_almost_equal won't be applicable, https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_almost_equal.html
or how about this: `np.testing.assert_allclose(pcopy.param_array, par.param_array, atol=1e-6)`