# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import optimize
import sys, pdb
#import numdifftools as ndt
from parameterised import parameterised, truncate_pad
import priors
from ..util.linalg import jitchol
from ..inference import optimization

class model(parameterised):
    def __init__(self):
        parameterised.__init__(self)
        self.priors = [None for i in range(self.get_param().size)]
        self.optimization_runs = []
        self.sampling_runs = []
        self.set_param(self.get_param())
        self.preferred_optimizer = 'tnc'
    def get_param(self):
        raise NotImplementedError, "this needs to be implemented to utilise the model class"
    def set_param(self,x):
        raise NotImplementedError, "this needs to be implemented to utilise the model class"
    def log_likelihood(self):
        raise NotImplementedError, "this needs to be implemented to utilise the model class"
    def log_likelihood_gradients(self):
        raise NotImplementedError, "this needs to be implemented to utilise the model class"

    def set_prior(self,which,what):
        """
        Sets priors on the model parameters.

        Arguments
        ---------
        which -- string, regexp, or integer array
        what -- instance of a prior class

        Notes
        -----
        Asserts that the prior is suitable for the constraint. If the
        wrong constraint is in place, an error is raised.  If no
        constraint is in place, one is added (warning printed).

        For tied parameters, the prior will only be "counted" once, thus
        a prior object is only inserted on the first tied index
        """

        which = self.grep_param_names(which)

        #check tied situation
        tie_partial_matches = [tie for tie in self.tied_indices if (not set(tie).isdisjoint(set(which))) & (not set(tie)==set(which))]
        if len(tie_partial_matches):
            raise ValueError, "cannot place prior across partial ties"
        tie_matches = [tie for tie in self.tied_indices if set(which)==set(tie) ]
        if len(tie_matches)>1:
            raise ValueError, "cannot place prior across multiple ties"
        elif len(tie_matches)==1:
            which = which[:1]# just place a prior object on the first parameter


        #check constraints are okay
        if isinstance(what, (priors.gamma, priors.log_Gaussian)):
            assert not np.any(which[:,None]==self.constrained_negative_indices), "constraint and prior incompatible"
            assert not np.any(which[:,None]==self.constrained_bounded_indices), "constraint and prior incompatible"
            unconst = np.setdiff1d(which, self.constrained_positive_indices)
            if len(unconst):
                print "Warning: constraining parameters to be positive:"
                print '\n'.join([n for i,n in enumerate(self.get_param_names()) if i in unconst])
                print '\n'
                self.constrain_positive(unconst)
        elif isinstance(what,priors.Gaussian):
            assert not np.any(which[:,None]==self.all_constrained_indices()), "constraint and prior incompatible"
        else:
            raise ValueError, "prior not recognised"


        #store the prior in a local list
        for w in which:
            self.priors[w] = what

    def get(self,name):
        """
        get a model parameter by name
        """
        matches = self.grep_param_names(name)
        if len(matches):
            return self.get_param()[matches]
        else:
            raise AttributeError, "no parameter matches %s"%name

    def set(self,name,val):
        """
        Set a model parameter by name
        """
        matches = self.grep_param_names(name)
        if len(matches):
            x = self.get_param()
            x[matches] = val
            self.set_param(x)
        else:
            raise AttributeError, "no parameter matches %s"%name



    def log_prior(self):
        """evaluate the prior"""
        return np.sum([p.lnpdf(x) for p, x in zip(self.priors,self.get_param()) if p is not None])

    def log_prior_gradients(self):
        """evaluate the gradients of the priors"""
        x = self.get_param()
        ret = np.zeros(x.size)
        [np.put(ret,i,p.lnpdf_grad(xx)) for i,(p,xx) in enumerate(zip(self.priors,x)) if not p is None]
        return ret

    def extract_gradients(self):
        """
        Use self.log_likelihood_gradients and self.prior_gradients to get the gradients of the model.
        Adjust the gradient for constraints and ties, return.
        """
        g = self.log_likelihood_gradients() + self.log_prior_gradients()
        x = self.get_param()
        g[self.constrained_positive_indices] = g[self.constrained_positive_indices]*x[self.constrained_positive_indices]
        g[self.constrained_negative_indices] = g[self.constrained_negative_indices]*x[self.constrained_negative_indices]
        [np.put(g,i,g[i]*(x[i]-l)*(h-x[i])/(h-l)) for i,l,h in zip(self.constrained_bounded_indices, self.constrained_bounded_lowers, self.constrained_bounded_uppers)]
        [np.put(g,i,v) for i,v in [(t[0],np.sum(g[t])) for t in self.tied_indices]]
        if len(self.tied_indices) or len(self.constrained_fixed_indices):
            to_remove = np.hstack((self.constrained_fixed_indices+[t[1:] for t in self.tied_indices]))
            return np.delete(g,to_remove)
        else:
            return g

    def randomize(self):
        """
        Randomize the model.
        Make this draw from the prior if one exists, else draw from N(0,1)
        """
        #first take care of all parameters (from N(0,1))
        x = self.extract_param()
        x = np.random.randn(x.size)
        self.expand_param(x)
        #now draw from prior where possible
        x = self.get_param()
        [np.put(x,i,p.rvs(1)) for i,p in enumerate(self.priors) if not p is None]
        self.set_param(x)
        self.expand_param(self.extract_param())#makes sure all of the tied parameters get the same init (since there's only one prior object...)


    def optimize_restarts(self, Nrestarts=10, robust=False, verbose=True, **kwargs):
        """
        Perform random restarts of the model, and set the model to the best
        seen solution.

        If the robust flag is set, exceptions raised during optimizations will
        be handled silently.  If _all_ runs fail, the model is reset to the
        existing parameter values.

        Notes
        -----
        **kwargs are passed to the optimizer. They can be:
        :max_f_eval: maximum number of function evaluations
        :messages: whether to display during optimisation
        :verbose: whether to show informations about the current restart
        """

        initial_parameters = self.extract_param()
        for i in range(Nrestarts):
            try:
                self.randomize()
                self.optimize(**kwargs)
                if verbose:
                    print("Optimization restart {0}/{1}, f = {2}".format(i+1,
                                                                      Nrestarts,
                                                                      self.optimization_runs[-1].f_opt))

            except Exception as e:
                if robust:
                    print("Warning - optimization restart {0}/{1} failed".format(i+1, Nrestarts))
                else:
                    raise e
        if len(self.optimization_runs):
            i = np.argmin([o.f_opt for o in self.optimization_runs])
            self.expand_param(self.optimization_runs[i].x_opt)
        else:
            self.expand_param(initial_parameters)

    def ensure_default_constraints(self,warn=False):
        """
        Ensure that any variables which should clearly be positive have been constrained somehow.
        """
        positive_strings = ['variance','lengthscale', 'precision']
        for s in positive_strings:
            for i in self.grep_param_names(s):
                if not (i in self.all_constrained_indices()):
                    name = self.get_param_names()[i]
                    self.constrain_positive(name)
                    if warn:
                        print "Warning! constraining %s postive"%name


    def optimize(self, optimizer=None, start=None, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :max_f_eval: maximum number of function evaluations
        :messages: whether to display during optimisation
        :param optimzer: whice optimizer to use (defaults to self.preferred optimizer)
        :type optimzer: string TODO: valid strings?
        """
        if optimizer is None:
            optimizer = self.preferred_optimizer

        def f(x):
            self.expand_param(x)
            return -self.log_likelihood()-self.log_prior()
        def fp(x):
            self.expand_param(x)
            return -self.extract_gradients()
        def f_fp(x):
            self.expand_param(x)
            return -self.log_likelihood()-self.log_prior(),-self.extract_gradients()

        if start == None:
            start = self.extract_param()

        optimizer = optimization.get_optimizer(optimizer)
        opt = optimizer(start, model = self, **kwargs)
        opt.run(f_fp=f_fp, f=f, fp=fp)
        self.optimization_runs.append(opt)

        self.expand_param(opt.x_opt)

    def optimize_SGD(self, momentum = 0.1, learning_rate = 0.01, iterations = 20, **kwargs):
        # assert self.Y.shape[1] > 1, "SGD only works with D > 1"
        sgd = SGD.StochasticGD(self, iterations, learning_rate, momentum, **kwargs)
        sgd.run()
        self.optimization_runs.append(sgd)

    def Laplace_covariance(self):
        """return the covariance matric of a Laplace approximatino at the current (stationary) point"""
        #TODO add in the prior contributions for MAP estimation
        #TODO fix the hessian for tied, constrained and fixed components
        if hasattr(self, 'log_likelihood_hessian'):
            A = -self.log_likelihood_hessian()

        else:
            print "numerically calculating hessian. please be patient!"
            x = self.get_param()
            def f(x):
                self.set_param(x)
                return self.log_likelihood()
            h = ndt.Hessian(f)
            A = -h(x)
            self.set_param(x)
        # check for almost zero components on the diagonal which screw up the cholesky
        aa = np.nonzero((np.diag(A)<1e-6) & (np.diag(A)>0.))[0]
        A[aa,aa] = 0.
        return A

    def Laplace_evidence(self):
        """Returns an estiamte of the model evidence based on the Laplace approximation.
        Uses a numerical estimate of the hessian if none is available analytically"""
        A = self.Laplace_covariance()
        try:
            hld = np.sum(np.log(np.diag(jitchol(A)[0])))
        except:
            return np.nan
        return 0.5*self.get_param().size*np.log(2*np.pi) + self.log_likelihood() - hld

    def __str__(self):
        s = parameterised.__str__(self).split('\n')
        #add priors to the string
        strs = [str(p) if p is not None else '' for p in self.priors]
        width = np.array(max([len(p) for p in strs] + [5])) + 4

        s[0] = 'Marginal log-likelihood: {0:.3e}\n'.format(self.log_likelihood()) + s[0]
        s[0] += "|{h:^{col}}".format(h = 'Prior', col = width)
        s[1] += '-'*(width + 1)

        for p in range(2, len(strs)+2):
            s[p] += '|{prior:^{width}}'.format(prior = strs[p-2], width = width)

        return '\n'.join(s)


    def checkgrad(self, verbose=False, include_priors=False, step=1e-6, tolerance = 1e-3, *args):
        """
        Check the gradient of the model by comparing to a numerical estimate.
        If the overall gradient fails, invividual components are tested.
        """

        x = self.extract_param().copy()

        #choose a random direction to step in:
        dx = step*np.sign(np.random.uniform(-1,1,x.size))

        #evaulate around the point x
        self.expand_param(x+dx)
        f1,g1 = self.log_likelihood() + self.log_prior(), self.extract_gradients()
        self.expand_param(x-dx)
        f2,g2 = self.log_likelihood() + self.log_prior(), self.extract_gradients()
        self.expand_param(x)
        gradient = self.extract_gradients()

        numerical_gradient = (f1-f2)/(2*dx)
        ratio = (f1-f2)/(2*np.dot(dx,gradient))
        if verbose:
            print "Gradient ratio = ", ratio, '\n'
            sys.stdout.flush()

        if (np.abs(1.-ratio)<tolerance) and not np.isnan(ratio):
            if verbose:
                print 'Gradcheck passed'
        else:
            if verbose:
                print "Global check failed. Testing individual gradients\n"

                try:
                    names = self.extract_param_names()
                except NotImplementedError:
                    names = ['Variable %i'%i for i in range(len(x))]

                # Prepare for pretty-printing
                header = ['Name', 'Ratio', 'Difference', 'Analytical', 'Numerical']
                max_names = max([len(names[i]) for i in range(len(names))] + [len(header[0])])
                float_len = 10
                cols = [max_names]
                cols.extend([max(float_len, len(header[i])) for i in range(1, len(header))])
                cols = np.array(cols) + 5
                header_string = ["{h:^{col}}".format(h = header[i], col = cols[i]) for i in range(len(cols))]
                header_string = map(lambda x: '|'.join(x), [header_string])
                separator = '-'*len(header_string[0])
                print '\n'.join([header_string[0], separator])

            for i in range(len(x)):
                xx = x.copy()
                xx[i] += step
                self.expand_param(xx)
                f1,g1 = self.log_likelihood() + self.log_prior(), self.extract_gradients()[i]
                xx[i] -= 2.*step
                self.expand_param(xx)
                f2,g2 = self.log_likelihood() + self.log_prior(), self.extract_gradients()[i]
                self.expand_param(x)
                gradient = self.extract_gradients()[i]


                numerical_gradient = (f1-f2)/(2*step)
                ratio = (f1-f2)/(2*step*gradient)
                difference = np.abs((f1-f2)/2/step - gradient)

                if verbose:
                    if (np.abs(ratio-1)<tolerance):
                        formatted_name = "\033[92m {0} \033[0m".format(names[i])
                    else:
                        formatted_name = "\033[91m {0} \033[0m".format(names[i])
                    r = '%.6f' % float(ratio)
                    d = '%.6f' % float(difference)
                    g = '%.6f' % gradient
                    ng = '%.6f' % float(numerical_gradient)
                    grad_string = "{0:^{c0}}|{1:^{c1}}|{2:^{c2}}|{3:^{c3}}|{4:^{c4}}".format(formatted_name,r,d,g, ng, c0 = cols[0]+9, c1 = cols[1], c2 = cols[2], c3 = cols[3], c4 = cols[4])
                    print grad_string

            print ''
            
            return False
        return True
