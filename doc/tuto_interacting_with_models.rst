*************************************
Interacting with models
*************************************

The GPy model class has a set of features which are designed to make it simple to explore the parameter space of the model. By default, the scipy optimisers are used to fit GPy models (via model.optimize()), for which we provide mechanisms for 'free' optimisation: GPy can ensure that naturally positive parameters (such as variances) remain positive. But these mechanisms are much more powerful than simple reparameterisation, as we shall see. 

All of the examples included in GPy return an instance of a model class. We'll use GPy.examples.?? as an example::

    import pylab as pb
    pb.ion()
    import GPy
    m = GPy.examples.??

Examining the model using print
===============================
To see the current state of the model parameters, and the model's (marginal) likelihood just print the model::
    print m

?? output

Getting the model's likelihood and gradients
===========================================
foobar

Setting and fetching parameters by name
=======================================
foobar

Constraining and optimising the model
=====================================
A simple task in GPy is to ensure that the models' variances remain positive during optimisation. the models class has a function called constrain_positive(), which accepts a regex string as above. To constrain the models' variance to be positive::
    m.constrain_positive('variance')
    print m

Now we see that the variance of the model is constrained to be postive. GPy handles the effective change of gradients: see how m.objective_gradients has changed approriately


For convenience, we also provide a catch all function which ensures that anything which appears to require positivity is constrianed appropriately::
    m.ensure_default_constraints()


Fixing parameters
=================


Tying Parameters
================

Bounding parameters
===================


Further Reading
===============
All of the mechansiams for dealing with parameters are baked right into GPy.core.model, from which all of the classes in GPy.models inherrit. To learn how to construct your own model, you might want to read ??link?? creating_new_models. 

By deafult, GPy uses the tnc optimizer (from scipy.optimize.tnc). To use other optimisers, and to control the setting of those optimisers, as well as other funky features like automated restarts and diagnostics, you can read the optimization tutorial ??link??.



