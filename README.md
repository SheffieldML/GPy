# GPy

The Gaussian processes framework in Python.

* GPy [homepage](http://sheffieldml.github.io/GPy/)
* Tutorial [notebooks](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb)
* User [mailing-list](https://lists.shef.ac.uk/sympa/subscribe/gpy-users)
* Developer [documentation](http://gpy.readthedocs.io/) [documentation (devel branch)](https://gpy.readthedocs.io/en/devel/)
* Travis-CI [unit-tests](https://travis-ci.org/SheffieldML/GPy)
* [![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)
  [![Research software impact](http://depsy.org/api/package/pypi/GPy/badge.svg)](http://depsy.org/package/python/GPy)

## Status

| Branch | travis-ci.org | ci.appveyor.com | coveralls.io | codecov.io |
| --- | --- | --- | --- | --- |
| Default branch (`devel`) | [![travis-devel](https://travis-ci.org/SheffieldML/GPy.svg?branch=devel)](https://travis-ci.org/SheffieldML/GPy/branches) | [![appveyor-devel](https://ci.appveyor.com/api/projects/status/662o6tha09m2jix3/branch/devel?svg=true)](https://ci.appveyor.com/project/mzwiessele/gpy/branch/devel) | [![coveralls-devel](https://coveralls.io/repos/github/SheffieldML/GPy/badge.svg?branch=devel)](https://coveralls.io/github/SheffieldML/GPy?branch=devel) | [![codecov-devel](http://codecov.io/github/SheffieldML/GPy/coverage.svg?branch=devel)](http://codecov.io/github/SheffieldML/GPy?branch=devel) |
| Deployment branch (`deploy`) | [![travis-deploy](https://travis-ci.org/SheffieldML/GPy.svg?branch=deploy)](https://travis-ci.org/SheffieldML/GPy/branches) | [![appveyor-deploy](https://ci.appveyor.com/api/projects/status/662o6tha09m2jix3/branch/deploy?svg=true)](https://ci.appveyor.com/project/mzwiessele/gpy/branch/deploy) | [![coveralls-deploy](https://coveralls.io/repos/github/SheffieldML/GPy/badge.svg?branch=deploy)](https://coveralls.io/github/SheffieldML/GPy?branch=deploy) | [![codecov-deploy](http://codecov.io/github/SheffieldML/GPy/coverage.svg?branch=deploy)](http://codecov.io/github/SheffieldML/GPy?branch=deploy) |

## What's new:

From now on we keep track of changes in the CHANGELOG.md.
If you want your changes to show up there follow the [guidelines](#gl).
In particular tag your commits by the [gitchangelog](https://github.com/vaab/gitchangelog) commit message format.

## Contributing to GPy

We welcome any contributions to GPy, after all it is an open source project. We use the GitHub feature of pull requests for contributions.

For an in depth description of pull requests, please visit https://help.github.com/articles/using-pull-requests/ .

### Steps to a successful contribution:

 1. Fork GPy: https://help.github.com/articles/fork-a-repo/
 2. Make your changes to the source in your fork.
 3. Make sure the [guidelines](#gl) are met.
 4. Set up tests to test your code. We are using unittests in the testing subfolder of GPy. There is a good chance 
    that there is already a framework set up to test your new model in model_tests.py or kernel in kernel_tests.py. have a look at the source and you might be able to just add your model (or kernel or others) as an additional test in the appropriate file. There is more frameworks for testing the other bits and pieces, just head over to the testing folder and have a look.
 5. Create a pull request to the devel branch in GPy, see above.
 6. The tests will be running on your pull request. In the comments section we will be able to discuss the changes and help you with any problems. Let us know if there are any in the comments, so we can help.
 7. The pull request gets accepted and your awesome new feature will be in the next GPy release :)

For any further questions/suggestions head over to the issues section in GPy.

<a name=gl></a>
### Pull Request Guidelines

 - Check your code with PEP8 or pylint. Try to stick to 80 columns wide.
 - Separate commits per smallest concern.
 - Each functionality/bugfix commit should contain code, tests, and doc.
 - We are using gitchangelog to keep track of changes and log new features. So if you want your changes to show up in the changelog, make sure you follow the [gitchangelog](https://github.com/vaab/gitchangelog) commit message format.

## Support and questions to the community

Ask questions using the issues section.

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

Python 3.5 and higher

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

We are requiring a recent version (1.3.0 or later) of
[scipy](http://www.scipy.org/) and thus, we strongly recommend using
the  [anaconda python distribution](http://continuum.io/downloads).
With anaconda you can install GPy by the following:

    conda update scipy
    
Then potentially try,

    sudo apt-get update
    sudo apt-get install python3-dev
    sudo apt-get install build-essential   
    conda update anaconda
    
And finally,

    pip install gpy

We've also had luck with [enthought](http://www.enthought.com). Install scipy 1.3.0 (or later)
 and then pip install GPy:

    pip install gpy

If you'd like to install from source, or want to contribute to the project (i.e. by sending pull requests via github), read on.

### Troubleshooting installation problems

If you're having trouble installing GPy via `pip install GPy` here is a probable solution:

    git clone https://github.com/SheffieldML/GPy.git
    cd GPy
    git checkout devel
    python setup.py build_ext --inplace
    pytest .

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
m_load = GPy.models.GPRegression(X, Y, initialize=False)
m_load.update_model(False) # do not call the underlying expensive algebra on load
m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
m_load[:] = np.load('model_save.npy') # Load the parameters
m_load.update_model(True) # Call the algebra only once
print(m_load)
```
## For Admins and Developers:

### Running unit tests:

New way of running tests is using coverage:

Ensure nose and coverage is installed:

    pip install nose coverage

Run nosetests from root directory of repository:

    coverage run travis_tests.py

Create coverage report in htmlcov/

    coverage html

The coverage report is located in htmlcov/index.html

##### Legacy: using nosetests

Ensure nose is installed via pip:

    pip install nose

Run nosetests from the root directory of the repository:

    nosetests -v GPy/testing

or from within IPython

    import GPy; GPy.tests()

or using setuptools

    python setup.py test


### Compiling documentation:

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

alternatively:

```{shell}
cd doc
sphinx-build -b html -d build/doctrees -D graphviz_dot='<path to dot>' source build/html
```

The HTML files are then stored in doc/build/html

### Commit new patch to devel

If you want to merge a branch into devel make sure the following steps are met:

 - Create a local branch from the pull request and merge the current devel in.
 - Look through the changes on the pull request.
 - Check that tests are there and are checking code where applicable.
 - [optional] Make changes if necessary and commit and push to run tests.
 - [optional] Repeat the above until tests pass.
 - [optional] bump up the version of GPy using bumpversion. The configuration is done, so all you need is bumpversion [major|minor|patch].
 - Update the changelog using gitchangelog: `gitchangelog > CHANGELOG.md`
 - Commit the changes of the changelog as silent update: `git commit -m "chg: pkg: CHANGELOG update" CHANGELOG.md
 - Push the changes into devel.

A usual workflow should look like this:

    $ git fetch origin
    $ git checkout -b <pull-origin>-devel origin/<pull-origin>-devel
    $ git merge devel
    $ coverage run travis_tests.py

**Make changes for tests to cover corner cases (if statements, None arguments etc.)**
Then we are ready to make the last changes for the changelog and versioning:

    $ git commit -am "fix: Fixed tests for <pull-origin>"
    $ bumpversion patch # [optional]
    $ gitchangelog > CHANGELOG.md
    $ git commit -m "chg: pkg: CHANGELOG update" CHANGELOG.md

Now we can merge the pull request into devel:

    $ git checkout devel
    $ git merge --no-ff <pull-origin>-devel
    $ git push origin devel

This will update the devel branch of GPy.

### Deploying GPy

We have set up all deployment automatic.
Thus, all you need to do is create a pull request from devel to deploy.
Wait for the tests to finish (successfully!) and merge the pull request.
This will update the package on pypi for all platforms fully automatically.

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
