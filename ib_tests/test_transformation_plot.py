"""
Tests whether or not the tansformation plot works as expected.
It does not work on the normal build.

Author:
    Ilias Bilionis

Date:
    3/8/2015
"""

import sys
import os
# Make sure we load the GP that is here
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import GPy
import matplotlib.pyplot as plt


if __name__ == '__main__':
    f = GPy.constraints.Logexp()
    f.plot()
    plt.show(block=True)
