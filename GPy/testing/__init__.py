# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""

MaxZ

"""
import unittest
import sys

def deepTest(reason):
    if reason:
        return lambda x:x
    return unittest.skip("Not deep scanning, enable deepscan by adding 'deep' argument")
