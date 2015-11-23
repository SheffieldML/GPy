# GPy

A Gaussian processes framework in Python.

* [GPy homepage](http://sheffieldml.github.io/GPy/)
* [Tutorial notebooks](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb)
* [User mailing list](https://lists.shef.ac.uk/sympa/subscribe/gpy-users)
* [Online documentation](https://gpy.readthedocs.org/en/latest/)
* [Unit tests (Travis-CI)](https://travis-ci.org/SheffieldML/GPy)
* [![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause) 

#### Continuous integration
|      | Travis-CI | Codecov | Readthedocs |
| ---: | :--: | :---: | :---: |
| **master:** | [![master](https://travis-ci.org/SheffieldML/GPy.svg?branch=master)](https://travis-ci.org/SheffieldML/GPy) | [![codecov.io](http://codecov.io/github/SheffieldML/GPy/coverage.svg?branch=master)](http://codecov.io/github/SheffieldML/GPy?branch=master) | [![mdocs](https://img.shields.io/badge/docs-master-blue.svg?style=flat)](http://gpy.readthedocs.org/en/master/) |
| **devel:**  | [![devel](https://travis-ci.org/SheffieldML/GPy.svg?branch=devel)](https://travis-ci.org/SheffieldML/GPy) | [![codecov.io](http://codecov.io/github/SheffieldML/GPy/coverage.svg?branch=devel)](http://codecov.io/github/SheffieldML/GPy?branch=devel) | [![ddocs](https://img.shields.io/badge/docs-devel-blue.svg?style=flat)](http://gpy.readthedocs.org/en/devel/) |

### Supported Platforms:

[<img src="https://www.python.org/static/community_logos/python-logo-generic.svg" height="40">](https://www.python.org/) [<img src="https://upload.wikimedia.org/wikipedia/commons/5/5f/Windows_logo_-_2012.svg" height="40">](http://www.microsoft.com/en-gb/windows) [<img src="https://upload.wikimedia.org/wikipedia/commons/8/8e/OS_X-Logo.svg" height="40">](http://www.apple.com/osx/) [<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height="40">](https://en.wikipedia.org/wiki/List_of_Linux_distributions)
----

Python 2.7, 3.3 and higher


### Citation

    @Misc{gpy2014,
      author =   {{The GPy authors}},
      title =    {{GPy}: A Gaussian process framework in python},
      howpublished = {\url{http://github.com/SheffieldML/GPy}},
      year = {2012--2015}
    }

### Pronounciation:

We like to pronounce it 'g-pie'.

### Getting started: installing with pip 

We are now requiring the newest version (0.16) of 
[scipy](http://www.scipy.org/) and thus, we strongly recommend using 
the  [anaconda python distribution](http://continuum.io/downloads).
With anaconda you can install GPy by the following:

    conda update scipy
    pip install gpy
    
We've also had luck with [enthought](http://www.enthought.com), 
although enthought currently (as of 8th Sep. 2015) does not support scipy 0.16.

If you'd like to install from source, or want to contribute to the project (e.g. by sending pull requests via github), read on.

### Troubleshooting installation problems

If you're having trouble installing GPy via `pip install GPy` here is a probable solution:

    git clone https://github.com/mikecroucher/GPy.git
    cd GPy
    git checkout devel
    python3 setup.py build_ext --inplace
    nosetests3 GPy/testing

### Direct downloads

[![PyPI version](https://badge.fury.io/py/GPy.svg)](https://pypi.python.org/pypi/GPy) [![source](https://img.shields.io/badge/download-source-green.svg)](https://github.com/SheffieldML/GPy/releases/latest)
[![Windows](https://img.shields.io/badge/download-windows-orange.svg)](https://github.com/SheffieldML/GPy/releases/latest)
[![MacOSX](https://img.shields.io/badge/download-macosx-blue.svg)](https://github.com/SheffieldML/GPy/releases/latest)

### Ubuntu hackers

> Note: Right now the Ubuntu package index does not include scipy 0.16.0, and thus, cannot
> be used for GPy. We hope this gets fixed soon.

For the most part, the developers are using ubuntu. To install the required packages:

    sudo apt-get install python-numpy python-scipy python-matplotlib

clone this git repository and add it to your path:

    git clone git@github.com:SheffieldML/GPy.git ~/SheffieldML
    echo 'PYTHONPATH=$PYTHONPATH:~/SheffieldML' >> ~/.bashrc


 
### OSX


We were working hard to make pre-built distributions ready. 
You can now install GPy via pip on MacOSX using 
[anaconda python distribution](http://continuum.io/downloads):

    conda update scipy
    pip install gpy

If this does not work, then you need to build GPy yourself, 
using the [development toolkits](https://developer.apple.com/xcode/). 
Download/clone GPy and run the build process:

    conda update scipy
    git clone git@github.com:SheffieldML/GPy.git ~/GPy
    cd ~/GPy
    python setup.py install

If you do not wish to build the C extensions (10 times speedup),
you can run the pure python installations, by just adding GPy
to your python path.

   echo 'PYTHONPATH=$PYTHONPATH:~/SheffieldML' >> ~/.profile



### Compiling documentation:


The documentation is stored in doc/ and is compiled with the Sphinx Python documentation generator, and is written in the reStructuredText format.

The Sphinx documentation is available here: http://sphinx-doc.org/latest/contents.html


##### Installing dependencies:


To compile the documentation, first ensure that Sphinx is installed. On Debian-based systems, this can be achieved as follows:

    sudo apt-get install python-pip
    sudo pip install sphinx

A LaTeX distribution is also required to compile the equations. Note that the extra packages are necessary to install the unicode packages. To compile the equations to PNG format for use in HTML pages, the package *dvipng* must be installed. IPython is also required. On Debian-based systems, this can be achieved as follows:

    sudo apt-get install texlive texlive-latex-extra texlive-base texlive-recommended
    sudo apt-get install dvipng
    sudo apt-get install ipython


#### Compiling documentation:


The documentation can be compiled as follows:

    cd doc
    make html

The HTML files are then stored in doc/_build/


## Running unit tests:


Ensure nose is installed via pip:

    pip install nose

Run nosetests from the root directory of the repository:

    nosetests -v GPy/testing

or from within IPython

    import GPy; GPy.tests()



## Funding Acknowledgements


Current support for the GPy software is coming through the following projects. 

* [EU FP7-HEALTH Project Ref 305626](http://radiant-project.eu) "RADIANT: Rapid Development and Distribution of Statistical Tools for High-Throughput Sequencing Data"

* [EU FP7-PEOPLE Project Ref 316861](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/mlpm/) "MLPM2012: Machine Learning for Personalized Medicine"

* MRC Special Training Fellowship "Bayesian models of expression in the transcriptome for clinical RNA-seq"

*  [EU FP7-ICT Project Ref 612139](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/wysiwyd/) "WYSIWYD: What You Say is What You Did"

Previous support for the GPy software came from the following projects:
* [BBSRC Project No BB/K011197/1](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/recombinant/) "Linking recombinant gene sequence to protein product manufacturability using CHO cell genomic resources"
* [EU FP7-KBBE Project Ref 289434](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/biopredyn/) "From Data to Models: New Bioinformatics Methods and Tools for Data-Driven Predictive Dynamic Modelling in Biotechnological Applications"
* [BBSRC Project No BB/H018123/2](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/iterative/) "An iterative pipeline of computational modelling and experimental design for uncovering gene regulatory networks in vertebrates"
* [Erasysbio](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/synergy/) "SYNERGY: Systems approach to gene regulation biology through nuclear receptors"
