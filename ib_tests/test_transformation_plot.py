import sys
import os
# Make sure we load the GP that is here
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import GPy
import matplotlib.pyplot as plt

f = GPy.constraints.Logexp()
f.plot()
plt.show(block=True)
