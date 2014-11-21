# GPy 0.6 Changelog

GPy 0.6 brings a host of changes. At the core of GPy is a new framewrok for
handling parameteterised models, which gives much improved performace for many
case. The kernels have changed both in their internal structure and their
interface. We have added tutorials and improved docstrings (and thus sphinx
compiled documentation) everywhere. 

## kernel changes

The kernels now follow the pep8 guidelines and use CamelCase for class names.
this means that GPy.kern.linear(args) is now replaced with
GPy.kern.Linear(args) This does mean small changes to some users code.
Apologies for the inconvenience. 

The kernels can now (optionally) accept `active_dims`, an iterable which
describs which dimensions of the input the kernel should work on. This makes
constructing kernels which are products over different space (tensor product
kernels) much easier to implement.

The kernels include a base class called `Stationary`. Many kernels (Matern,
RBF, Exponential) inherrit from this, saving lots of code. Implementing new
Stationary kernels is straightforward. 

## Likelihood changes

The stucture of GPy has changed co that we can have a more 'plug-n-play',
modular codebase. There's now a base `Likelihood` class which skeletons the
implementation for many liklihoods, allowing flexible modelling. For example,
GPclassificatino with different approximations is now straightforward with the
implementation of the Bernouilli class. 


## parameterized changes

## Documentation changes

We have focussed efforts on documentation on providing IPython notebooks
containing examples. See
http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb


