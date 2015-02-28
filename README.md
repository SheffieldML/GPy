# GPy

A Gaussian processes framework in Python.

* [GPy homepage](http://sheffieldml.github.io/GPy/)
* [Tutorial notebooks](http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb)
* [User mailing list](https://lists.shef.ac.uk/sympa/subscribe/gpy-users)
* [Online documentation](https://gpy.readthedocs.org/en/latest/)
* [Unit tests (Travis-CI)](https://travis-ci.org/SheffieldML/GPy)

Continuous integration status: ![CI status](https://travis-ci.org/SheffieldML/GPy.png)

### Citation

    @Misc{gpy2014,
      author =   {The GPy authors},
      title =    {{GPy}: A Gaussian process framework in python},
      howpublished = {\url{http://github.com/SheffieldML/GPy}},
      year = {2012--2014}
    }

### Pronounciation

We like to pronounce it 'Gee-pie'.

### Getting started: installing with pip

The simplest way to install GPy is using pip. Ubuntu users can do:

    sudo apt-get install python-pip
    pip install gpy
    
On windows, we recommend the ![anaconda python distribution](http://continuum.io/downloads). We've also had luck with ![enthought](http://www.enthought.com).

On a fresh install of windows 8.1, we downloaded the Anaconda python distribution, started the anaconda command prompt and typed 

    pip install GPy

Everything seems to work: from here you can type `ipython` and then `import GPy; GPy.tests()`. Working as of 21/11/14

If you'd like to install from source, or want to contribute to the project (e.g. by sending pull requests via github), read on.

### Ubuntu hackers

For the most part, the developers are using ubuntu. To install the required packages:

    sudo apt-get install python-numpy python-scipy python-matplotlib

clone this git repository and add it to your path:

    git clone git@github.com:SheffieldML/GPy.git ~/SheffieldML
    echo 'PYTHONPATH=$PYTHONPATH:~/SheffieldML' >> ~/.bashrc


 
### OSX

Everything appears to work out-of-the box using ![enthought](http://www.enthought.com) on osx Mavericks. Download/clone GPy, and then add GPy to your PYTHONPATH

    git clone git@github.com:SheffieldML/GPy.git ~/SheffieldML
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

### Moving to Python 3
Work is underway to make GPy run on Python 3. We are not there yet! Changes performed so far have retained compatibility with Python 2.6 and above.

Work done so far:

* Used 2to3 to fix relative imports
* Used 2to3 to convert print from statement to function. Some advanced uses of print meant that this could not be done in a way that retained compatibility with old versions of Python. The oldest version of Python that is supported by this version is 2.6 due to the required future imports.
* Used 2to3 to convert exceptions to Python 3 friendly versions. There are a few oustanding string exceptions to take care of that 2to3 doesn't handle. Will need to do these manually
* Handled the different imports required for ConfigParser/configparser in Py2/Py3
* In utils/linalg.py:
    * Commented out the function cholupdate(L, x) since it doesn't appear to be used. Its definitely not in the tests.s
    * Put the import for scipy.weave in a try/except block so that it will gracefully fail in Py3
* Fixed a couple of urllib2 issues - had to be done mannual since 2to3 didn't help
		       



## Funding Acknowledgements


Current support for the GPy software is coming through the following projects. 

* [EU FP7-PEOPLE Project Ref 316861](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/mlpm/) "MLPM2012: Machine Learning for Personalized Medicine"

* [BBSRC Project No BB/K011197/1](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/recombinant/) "Linking recombinant gene sequence to protein product manufacturability using CHO cell genomic resources"

* MRC Special Training Fellowship "Bayesian models of expression in the transcriptome for clinical RNA-seq"

* [EU FP7-KBBE Project Ref 289434](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/biopredyn/) "From Data to Models: New Bioinformatics Methods and Tools for Data-Driven Predictive Dynamic Modelling in Biotechnological Applications"

*  [EU FP7-ICT Project Ref 612139](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/wysiwyd/) "WYSIWYD: What You Say is What You Did"

Previous support for the GPy software came from the following projects:

* [BBSRC Project No BB/H018123/2](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/iterative/) "An iterative pipeline of computational modelling and experimental design for uncovering gene regulatory networks in vertebrates"
* [Erasysbio](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/synergy/) "SYNERGY: Systems approach to gene regulation biology through nuclear receptors"
