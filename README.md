# GPy

The Gaussian processes framework in Python.

* GPy [homepage](http://sheffieldml.github.io/GPy/)
* Tutorial [notebooks](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb)
* User [mailing-list](https://lists.shef.ac.uk/sympa/subscribe/gpy-users)
* Developer [documentation](http://pythonhosted.org/GPy/)
* Travis-CI [unit-tests](https://travis-ci.org/SheffieldML/GPy)
* [![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)

[![develstat](https://travis-ci.org/SheffieldML/GPy.svg?branch=devel)](https://travis-ci.org/SheffieldML/GPy) [![covdevel](http://codecov.io/github/SheffieldML/GPy/coverage.svg?branch=devel)](http://codecov.io/github/SheffieldML/GPy?branch=devel) [![Research software impact](http://depsy.org/api/package/pypi/GPy/badge.svg)](http://depsy.org/package/python/GPy) [![Code Health](https://landscape.io/github/SheffieldML/GPy/devel/landscape.svg?style=flat)](https://landscape.io/github/SheffieldML/GPy/devel)

## Updated Structure

We have pulled the core parameterization out of GPy. It is a package called [paramz](https://github.com/sods/paramz) and is the pure gradient based model optimization.

If you installed GPy with pip, just upgrade the package using:

    $ pip install --upgrade GPy

If you have the developmental version of GPy (using the develop or -e option) just install the dependencies by running

    $ python setup.py develop

again, in the GPy installation folder.

A warning: This usually works, but sometimes `distutils/setuptools` opens a
whole can of worms here, specially when compiled extensions are involved.
If that is the case, it is best to clean the repo and reinstall.

## Supported Platforms:

[<img src="https://www.python.org/static/community_logos/python-logo-generic.svg" height=40px>](https://www.python.org/)
[<img src="https://upload.wikimedia.org/wikipedia/commons/5/5f/Windows_logo_-_2012.svg" height=40px>](http://www.microsoft.com/en-gb/windows)
[<img src="https://upload.wikimedia.org/wikipedia/commons/8/8e/OS_X-Logo.svg" height=40px>](http://www.apple.com/osx/)
[<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>](https://en.wikipedia.org/wiki/List_of_Linux_distributions)

Python 2.7, 3.4 and higher

## Citation

    @Misc{gpy2014,
      author =   {{GPy}},
      title =    {{GPy}: A Gaussian process framework in python},
      howpublished = {\url{http://github.com/SheffieldML/GPy}},
      year = {since 2012}
    }

### Pronounciation:

We like to pronounce it 'g-pie'.

## Getting started: installing with pip

We are now requiring the newest version (0.16) of
[scipy](http://www.scipy.org/) and thus, we strongly recommend using
the  [anaconda python distribution](http://continuum.io/downloads).
With anaconda you can install GPy by the following:

    conda update scipy
    pip install gpy

We've also had luck with [enthought](http://www.enthought.com). Install scipy 0.16 (or later)
 and then pip install GPy:

    pip install gpy

If you'd like to install from source, or want to contribute to the project (i.e. by sending pull requests via github), read on.

### Troubleshooting installation problems

If you're having trouble installing GPy via `pip install GPy` here is a probable solution:

    git clone https://github.com/SheffieldML/GPy.git
    cd GPy
    git checkout devel
    python setup.py build_ext --inplace
    nosetests GPy/testing

### Direct downloads

[![PyPI version](https://badge.fury.io/py/GPy.svg)](https://pypi.python.org/pypi/GPy) [![source](https://img.shields.io/badge/download-source-green.svg)](https://pypi.python.org/pypi/GPy)
[![Windows](https://img.shields.io/badge/download-windows-orange.svg)](https://pypi.python.org/pypi/GPy)
[![MacOSX](https://img.shields.io/badge/download-macosx-blue.svg)](https://pypi.python.org/pypi/GPy)

# Saving models in a consistent way across versions:

As pickle is inconsistent across python versions and heavily dependent on class structure, it behaves inconsistent across versions. 
Pickling as meant to serialize models within the same environment, and not to store models on disk to be used later on.

To save a model it is best to save the m.param_array of it to disk (using numpy’s np.save).
Additionally, you save the script, which creates the model. 
In this script you can create the model using initialize=False as a keyword argument and with the data loaded as normal. 
You then set the model parameters by setting m.param_array[:] = loaded_params as the previously saved parameters. 
Then you initialize the model by m.initialize_parameter(), which will make the model usable. 
Be aware that up to this point the model is in an inconsistent state and cannot be used to produce any results.

```python
# let X, Y be data loaded above
# Model creation:
m = GPy.models.GPRegression(X, Y)
m.optimize()
# 1: Saving a model:
np.save('model_save.npy', m.param_array)
# 2: loading a model
# Model creation, without initialization:
m = GPy.models(GPRegression(X,Y,initialize=False)
m[:] = np.load('model_save.npy')
m.initialize_parameter()
print m
```

## Running unit tests:

Ensure nose is installed via pip:

    pip install nose

Run nosetests from the root directory of the repository:

    nosetests -v GPy/testing

or from within IPython

    import GPy; GPy.tests()

or using setuptools

    python setup.py test

## Ubuntu hackers

> Note: Right now the Ubuntu package index does not include scipy 0.16.0, and thus, cannot
> be used for GPy. We hope this gets fixed soon.

For the most part, the developers are using ubuntu. To install the required packages:

    sudo apt-get install python-numpy python-scipy python-matplotlib

clone this git repository and add it to your path:

    git clone git@github.com:SheffieldML/GPy.git ~/SheffieldML
    echo 'PYTHONPATH=$PYTHONPATH:~/SheffieldML' >> ~/.bashrc


## Compiling documentation:

The documentation is stored in doc/ and is compiled with the Sphinx Python documentation generator, and is written in the reStructuredText format.

The Sphinx documentation is available here: http://sphinx-doc.org/latest/contents.html

**Installing dependencies:**

To compile the documentation, first ensure that Sphinx is installed. On Debian-based systems, this can be achieved as follows:

    sudo apt-get install python-pip
    sudo pip install sphinx

**Compiling documentation:**

The documentation can be compiled as follows:

    cd doc
    sphinx-apidoc -o source/ ../GPy/
    make html

The HTML files are then stored in doc/build/html

## Funding Acknowledgements

Current support for the GPy software is coming through the following projects.

* [EU FP7-HEALTH Project Ref 305626](http://radiant-project.eu) "RADIANT: Rapid Development and Distribution of Statistical Tools for High-Throughput Sequencing Data"

* [EU FP7-PEOPLE Project Ref 316861](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/mlpm/) "MLPM2012: Machine Learning for Personalized Medicine"

* MRC Special Training Fellowship "Bayesian models of expression in the transcriptome for clinical RNA-seq"

*  [EU FP7-ICT Project Ref 612139](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/wysiwyd/) "WYSIWYD: What You Say is What You Did"

Previous support for the GPy software came from the following projects:

- [BBSRC Project No BB/K011197/1](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/recombinant/) "Linking recombinant gene sequence to protein product manufacturability using CHO cell genomic resources"
- [EU FP7-KBBE Project Ref 289434](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/biopredyn/) "From Data to Models: New Bioinformatics Methods and Tools for Data-Driven Predictive Dynamic Modelling in Biotechnological Applications"
- [BBSRC Project No BB/H018123/2](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/iterative/) "An iterative pipeline of computational modelling and experimental design for uncovering gene regulatory networks in vertebrates"
- [Erasysbio](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/synergy/) "SYNERGY: Systems approach to gene regulation biology through nuclear receptors"
