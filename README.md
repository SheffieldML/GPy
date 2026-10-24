# GPy

The Gaussian processes framework in Python.

* GPy [homepage](http://sheffieldml.github.io/GPy/)
* Tutorial [notebooks](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb)
* Multi-output / coregionalized regression notes: [`doc/source/tuto_coregionalized.rst`](doc/source/tuto_coregionalized.rst) (also on [Read the Docs](https://gpy.readthedocs.io/en/latest/tuto_coregionalized.html))
* Community Q&A: [GitHub Discussions](https://github.com/SheffieldML/GPy/discussions)
* Developer [documentation](https://gpy.readthedocs.io/en/latest/)
* [![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)
  [![Research software impact](http://depsy.org/api/package/pypi/GPy/badge.svg)](http://depsy.org/package/python/GPy)

## Status

| Branch | GitHub Actions | Codecov |
| --- | --- | --- |
| `main` | [![gha-main](https://github.com/SheffieldML/GPy/actions/workflows/test-and-deploy.yml/badge.svg?branch=main)](https://github.com/SheffieldML/GPy/actions/workflows/test-and-deploy.yml?query=branch%3Amain) | [![codecov-main](https://codecov.io/gh/SheffieldML/GPy/branch/main/graph/badge.svg)](https://codecov.io/gh/SheffieldML/GPy/branch/main) |

## What's new:

From now on we keep track of changes in the CHANGELOG.md.
If you want your changes to show up there follow the [guidelines](#gl).
In particular tag your commits by the [gitchangelog](https://github.com/vaab/gitchangelog) commit message format.

Packaging uses [Poetry](https://python-poetry.org/) (CIP-0003). **End users**
still install with `pip install gpy` from PyPI wheels. **Contributors** should
use Poetry for editable installs and a locked environment (`poetry.lock`).

## Contributing to GPy

We welcome any contributions to GPy, after all it is an open source project. We use the GitHub feature of pull requests for contributions.

For an in depth description of pull requests, please visit https://help.github.com/articles/using-pull-requests/ .

### Steps to a successful contribution:

 1. Fork GPy: https://help.github.com/articles/fork-a-repo/
 2. Make your changes to the source in your fork.
 3. Make sure the [guidelines](#gl) are met.
 4. Set up tests to test your code. We are using unittests in the testing subfolder of GPy. There is a good chance 
    that there is already a framework set up to test your new model in model_tests.py or kernel in kernel_tests.py. have a look at the source and you might be able to just add your model (or kernel or others) as an additional test in the appropriate file. There is more frameworks for testing the other bits and pieces, just head over to the testing folder and have a look.
 5. Create a pull request against **`main`**.
 6. The tests will be running on your pull request. In the comments section we will be able to discuss the changes and help you with any problems. Let us know if there are any in the comments, so we can help.
 7. The pull request gets accepted and your awesome new feature will be in the next GPy release :)

For any further questions/suggestions head over to the issues section in GPy.

<a name=gl></a>
### Pull Request Guidelines

 - Lint and format with [Ruff](https://docs.astral.sh/ruff/) (CIP-0005). CI runs `ruff check GPy` and `ruff format --check GPy` using the select set in `pyproject.toml`. Recommended local setup: `pip install pre-commit && pre-commit install` (hooks in `.pre-commit-config.yaml`). Or run `ruff check GPy` / `ruff format GPy` yourself.
 - Separate commits per smallest concern.
 - Each functionality/bugfix commit should contain code, tests, and doc.
 - We are using gitchangelog to keep track of changes and log new features. So if you want your changes to show up in the changelog, make sure you follow the [gitchangelog](https://github.com/vaab/gitchangelog) commit message format.

## Support and questions to the community

Ask questions using the issues section.

## Updated Structure

We have pulled the core parameterization out of GPy. It is a package called [paramz](https://github.com/sods/paramz) and is the pure gradient based model optimization.

If you installed GPy with pip, just upgrade the package using:

    $ pip install --upgrade gpy

For a developmental install from a clone, use Poetry (see below).

## Supported Platforms:

[<img src="https://www.python.org/static/community_logos/python-logo-generic.svg" height=40px>](https://www.python.org/)
[<img src="https://upload.wikimedia.org/wikipedia/commons/5/5f/Windows_logo_-_2012.svg" height=40px>](http://www.microsoft.com/en-gb/windows)
[<img src="https://upload.wikimedia.org/wikipedia/commons/8/8e/OS_X-Logo.svg" height=40px>](http://www.apple.com/osx/)
[<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>](https://en.wikipedia.org/wiki/List_of_Linux_distributions)

Python 3.11 and higher

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

### Contributor install (Poetry)

Requires [Poetry](https://python-poetry.org/docs/#installation) 1.8+ (2.x recommended).

```bash
git clone https://github.com/SheffieldML/GPy.git
cd GPy
poetry install --with dev
poetry run pytest GPy/testing
```

`poetry.lock` pins the contributor environment for reproducibility. PyPI /
`pip install gpy` installs do **not** use the lockfile.

You can also build and install a local wheel without Poetry at runtime:

```bash
poetry build
pip install dist/gpy-*.whl
```

### Plotting (optional)

Matplotlib and plotly are **optional**. Core modelling works with a plain
`pip install gpy`. For plots:

```bash
pip install 'gpy[plotting]'
```

If the configured library is missing, GPy imports successfully and continues
without plotting (see [#1209](https://github.com/SheffieldML/GPy/issues/1209)).

### Wheels and Apple Silicon / Linux aarch64

Release CI publishes:

- **manylinux** wheels for **x86_64** and **aarch64** (Cython extensions built in)
- **macOS** wheels from `macos-latest` (Apple Silicon)
- **Windows** wheels

If no wheel matches your platform, `pip` builds from the sdist (needs a C
compiler and Cython). Contributor installs via Poetry compile extensions locally
on any architecture CI already tests (`ubuntu`, `windows`, `macos` including
arm64).

### Troubleshooting installation problems

If you're having trouble installing GPy via `pip install gpy` here is a probable solution:

    git clone https://github.com/SheffieldML/GPy.git
    cd GPy
    git checkout main
    poetry install --with dev
    pytest .

### Direct downloads

[![PyPI version](https://badge.fury.io/py/GPy.svg)](https://pypi.python.org/pypi/GPy) [![source](https://img.shields.io/badge/download-source-green.svg)](https://pypi.python.org/pypi/GPy)
[![Windows](https://img.shields.io/badge/download-windows-orange.svg)](https://pypi.python.org/pypi/GPy)
[![MacOSX](https://img.shields.io/badge/download-macosx-blue.svg)](https://pypi.python.org/pypi/GPy)

# Saving models in a consistent way across versions:

As pickle is inconsistent across python versions and heavily dependent on class structure, it behaves inconsistent across versions.
Pickling as meant to serialize models within the same environment, and not to store models on disk to be used later on.

To save a model it is best to save the `m.param_array` of it to disk (using numpy’s `np.save`).
Additionally, you save the script which creates the model.
In this script you can create the model using `initialize=False` as a keyword argument and with the data loaded as normal.
Then call `m.initialize_parameter()` and load the saved values with **model slice assignment** `m[:] = loaded_params` (not `m.param_array[:] = …`).
Bare writes into `param_array` skip paramz observers, so lengthscales, predictions and the objective can stay at their defaults even though `m.param_array` prints the loaded numbers (see [#1217](https://github.com/SheffieldML/GPy/issues/1217)).
Be aware that up to the final `update_model(True)` the model is in an inconsistent state and cannot be used to produce any results.

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

With Poetry (recommended for contributors):

    poetry install --with dev
    poetry run pytest GPy/testing

Coverage (uses `.coveragerc`; also uploaded from CI on Ubuntu / Python 3.12):

    poetry run pytest GPy/testing --cov=GPy --cov-report=html --cov-report=term
    # open htmlcov/index.html

or using setuptools

    poetry run pytest GPy/testing


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

### Merging to the default branch

Open a pull request against **`main`**. CI (`Test Python Lib`) runs tests and
lint on the PR. The old `deploy` branch is stale and is not part of publishing.

Bump the Poetry version in `pyproject.toml` / `GPy/__version__.py` when cutting
a release, and add a CHANGELOG section.

### Deploying GPy (CIP-0008)

1. Merge the version bump to `main`.
2. On GitHub: **Releases → Draft a new release**.
3. Create tag `vX.Y.Z` on that commit (must match `poetry version -s`).
4. **Publish** the release (a draft does not upload).

That `release` event (`types: [published]`) builds wheels (Linux x86_64 and
aarch64, macOS, Windows; Python 3.11–3.14) and runs `twine upload` to PyPI.
To build wheels **without** uploading, run **Actions → Test Python Lib →
Run workflow** (`workflow_dispatch`).

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
