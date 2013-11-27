GPy
===

A Gaussian processes framework in Python.

* [Online documentation](https://gpy.readthedocs.org/en/latest/)
* [Unit tests (Travis-CI)](https://travis-ci.org/SheffieldML/GPy)


Continuous integration status: ![CI status](https://travis-ci.org/SheffieldML/GPy.png)

Getting started
===============
Installing with pip
-------------------
The simplest way to install GPy is using pip. ubuntu users can do:

    sudo apt-get install python-pip
    pip install gpy

If you'd like to install from source, or want to contribute to the project (e.g. by sending pull requests via github), read on.

Ubuntu
------
For the most part, the developers are using ubuntu. To install the required packages:

    sudo apt-get install python-numpy python-scipy python-matplotlib

clone this git repository and add it to your path:

    git clone git@github.com:SheffieldML/GPy.git ~/SheffieldML
    echo 'PYTHONPATH=$PYTHONPATH:~/SheffieldML' >> ~/.bashrc


Windows
-------
On windows, we recommend the ![anaconda python distribution](http://continuum.io/downloads). We've also had luck with ![enthought](http://www.enthought.com). git clone or unzip the source to a suitable directory, and add an approptiate PYTHONPATH environment variable. 

On windows 7 (and possibly earlier versions) there's a bug in scipy version 0.13 which tries to write very long filenames. Reverting to scipy 0.12 seems to do the trick:

    conda install scipy=0.12

OSX
---
Everything appears to work out-of-the box using ![enthought](http://www.enthought.com) on osx Mavericks. Download/clone GPy, and then add GPy to your PYTHONPATH

    git clone git@github.com:SheffieldML/GPy.git ~/SheffieldML
    echo 'PYTHONPATH=$PYTHONPATH:~/SheffieldML' >> ~/.profile


Compiling documentation:
========================

The documentation is stored in doc/ and is compiled with the Sphinx Python documentation generator, and is written in the reStructuredText format.

The Sphinx documentation is available here: http://sphinx-doc.org/latest/contents.html


Installing dependencies:
------------------------

To compile the documentation, first ensure that Sphinx is installed. On Debian-based systems, this can be achieved as follows:

    sudo apt-get install python-pip
    sudo pip install sphinx

A LaTeX distribution is also required to compile the equations. Note that the extra packages are necessary to install the unicode packages. To compile the equations to PNG format for use in HTML pages, the package *dvipng* must be installed. IPython is also required. On Debian-based systems, this can be achieved as follows:

    sudo apt-get install texlive texlive-latex-extra texlive-base texlive-recommended
    sudo apt-get install dvipng
    sudo apt-get install ipython


Compiling documentation:
------------------------

The documentation can be compiled as follows:

    cd doc
    make html

The HTML files are then stored in doc/_build/


Running unit tests:
===================

Ensure nose is installed via pip:

    pip install nose

Run nosetests from the root directory of the repository:

    nosetests -v

