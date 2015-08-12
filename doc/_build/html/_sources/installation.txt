==============
 Installation
==============


Linux
============


Windows
======================
One easy way to get a Python distribution with the required packages is to use the Anaconda environment from Continuum Analytics.

* Download and install the free version of Anaconda according to your operating system  from `their website <https://store.continuum.io>`_.
* Open a (new) terminal window:

  * Navigate to Applications/Accessories/cmd, or
  * open *anaconda Command Prompt* from windows *start*

You should now be able to launch a Python interpreter by typing *ipython* in the terminal. In the ipython prompt, you can check your installation by importing the libraries we will need later:
::
    $ import numpy
    $ import pylab

To install the latest version of GPy, *git* is required. A *git* client on Windows can be found `here <http://git-scm.com/download/win>`_. It is recommened to install with the option "*Use Git from the Windows Command Prompt*". Then, GPy can be installed with the following command
::
    pip install git+https://github.com/SheffieldML/GPy.git@devel

MacOSX
===================================

