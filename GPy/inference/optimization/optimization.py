# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import datetime as dt
from scipy import optimize
from warnings import warn

try:
    import rasmussens_minimize as rasm
    rasm_available = True
except ImportError:
    rasm_available = False
from .scg import SCG

class Optimizer():
    """
    Superclass for all the optimizers.

    :param x_init: initial set of parameters
    :param f_fp: function that returns the function AND the gradients at the same time
    :param f: function to optimize
    :param fp: gradients
    :param messages: print messages from the optimizer?
    :type messages: (True | False)
    :param max_f_eval: maximum number of function evaluations

    :rtype: optimizer object.

    """
    def __init__(self, x_init, messages=False, model=None, max_f_eval=1e4, max_iters=1e3,
                 ftol=None, gtol=None, xtol=None, bfgs_factor=None):
        self.opt_name = None
        self.x_init = x_init
        # Turning messages off and using internal structure for print outs:
        self.messages = False #messages
        self.f_opt = None
        self.x_opt = None
        self.funct_eval = None
        self.status = None
        self.max_f_eval = int(max_iters)
        self.max_iters = int(max_iters)
        self.bfgs_factor = bfgs_factor
        self.trace = None
        self.time = "Not available"
        self.xtol = xtol
        self.gtol = gtol
        self.ftol = ftol
        self.model = model

    def run(self, **kwargs):
        start = dt.datetime.now()
        self.opt(**kwargs)
        end = dt.datetime.now()
        self.time = str(end - start)

    def opt(self, f_fp=None, f=None, fp=None):
        raise NotImplementedError("this needs to be implemented to use the optimizer class")

    def plot(self):
        """
        See GPy.plotting.matplot_dep.inference_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import inference_plots
        inference_plots.plot_optimizer(self)


    def __str__(self):
        diagnostics = "Optimizer: \t\t\t\t %s\n" % self.opt_name
        diagnostics += "f(x_opt): \t\t\t\t %.3f\n" % self.f_opt
        diagnostics += "Number of function evaluations: \t %d\n" % self.funct_eval
        diagnostics += "Optimization status: \t\t\t %s\n" % self.status
        diagnostics += "Time elapsed: \t\t\t\t %s\n" % self.time
        return diagnostics

class opt_tnc(Optimizer):
    def __init__(self, *args, **kwargs):
        Optimizer.__init__(self, *args, **kwargs)
        self.opt_name = "TNC (Scipy implementation)"

    def opt(self, f_fp=None, f=None, fp=None):
        """
        Run the TNC optimizer

        """
        tnc_rcstrings = ['Local minimum', 'Converged', 'XConverged', 'Maximum number of f evaluations reached',
             'Line search failed', 'Function is constant']

        assert f_fp != None, "TNC requires f_fp"

        opt_dict = {}
        if self.xtol is not None:
            opt_dict['xtol'] = self.xtol
        if self.ftol is not None:
            opt_dict['ftol'] = self.ftol
        if self.gtol is not None:
            opt_dict['pgtol'] = self.gtol

        opt_result = optimize.fmin_tnc(f_fp, self.x_init, messages=self.messages,
                       maxfun=self.max_f_eval, **opt_dict)
        self.x_opt = opt_result[0]
        self.f_opt = f_fp(self.x_opt)[0]
        self.funct_eval = opt_result[1]
        self.status = tnc_rcstrings[opt_result[2]]

class opt_lbfgsb(Optimizer):
    def __init__(self, *args, **kwargs):
        Optimizer.__init__(self, *args, **kwargs)
        self.opt_name = "L-BFGS-B (Scipy implementation)"

    def opt(self, f_fp=None, f=None, fp=None):
        """
        Run the optimizer

        """
        rcstrings = ['Converged', 'Maximum number of f evaluations reached', 'Error']

        assert f_fp != None, "BFGS requires f_fp"

        if self.messages:
            iprint = 1
        else:
            iprint = -1

        opt_dict = {}
        if self.xtol is not None:
            print("WARNING: l-bfgs-b doesn't have an xtol arg, so I'm going to ignore it")
        if self.ftol is not None:
            print("WARNING: l-bfgs-b doesn't have an ftol arg, so I'm going to ignore it")
        if self.gtol is not None:
            opt_dict['pgtol'] = self.gtol
        if self.bfgs_factor is not None:
            opt_dict['factr'] = self.bfgs_factor

        opt_result = optimize.fmin_l_bfgs_b(f_fp, self.x_init, iprint=iprint,
                                            maxfun=self.max_iters, **opt_dict)
        self.x_opt = opt_result[0]
        self.f_opt = f_fp(self.x_opt)[0]
        self.funct_eval = opt_result[2]['funcalls']
        self.status = rcstrings[opt_result[2]['warnflag']]

        #a more helpful error message is available in opt_result in the Error case
        if opt_result[2]['warnflag']==2:
            self.status = 'Error' + str(opt_result[2]['task'])

class opt_simplex(Optimizer):
    def __init__(self, *args, **kwargs):
        Optimizer.__init__(self, *args, **kwargs)
        self.opt_name = "Nelder-Mead simplex routine (via Scipy)"

    def opt(self, f_fp=None, f=None, fp=None):
        """
        The simplex optimizer does not require gradients.
        """

        statuses = ['Converged', 'Maximum number of function evaluations made', 'Maximum number of iterations reached']

        opt_dict = {}
        if self.xtol is not None:
            opt_dict['xtol'] = self.xtol
        if self.ftol is not None:
            opt_dict['ftol'] = self.ftol
        if self.gtol is not None:
            print("WARNING: simplex doesn't have an gtol arg, so I'm going to ignore it")

        opt_result = optimize.fmin(f, self.x_init, (), disp=self.messages,
                   maxfun=self.max_f_eval, full_output=True, **opt_dict)

        self.x_opt = opt_result[0]
        self.f_opt = opt_result[1]
        self.funct_eval = opt_result[3]
        self.status = statuses[opt_result[4]]
        self.trace = None


class opt_rasm(Optimizer):
    def __init__(self, *args, **kwargs):
        Optimizer.__init__(self, *args, **kwargs)
        self.opt_name = "Rasmussen's Conjugate Gradient"

    def opt(self, f_fp=None, f=None, fp=None):
        """
        Run Rasmussen's Conjugate Gradient optimizer
        """

        assert f_fp != None, "Rasmussen's minimizer requires f_fp"
        statuses = ['Converged', 'Line search failed', 'Maximum number of f evaluations reached',
                'NaNs in optimization']

        opt_dict = {}
        if self.xtol is not None:
            print("WARNING: minimize doesn't have an xtol arg, so I'm going to ignore it")
        if self.ftol is not None:
            print("WARNING: minimize doesn't have an ftol arg, so I'm going to ignore it")
        if self.gtol is not None:
            print("WARNING: minimize doesn't have an gtol arg, so I'm going to ignore it")

        opt_result = rasm.minimize(self.x_init, f_fp, (), messages=self.messages,
                                   maxnumfuneval=self.max_f_eval)
        self.x_opt = opt_result[0]
        self.f_opt = opt_result[1][-1]
        self.funct_eval = opt_result[2]
        self.status = statuses[opt_result[3]]

        self.trace = opt_result[1]

class opt_SCG(Optimizer):
    def __init__(self, *args, **kwargs):
        if 'max_f_eval' in kwargs:
            warn("max_f_eval deprecated for SCG optimizer: use max_iters instead!\nIgnoring max_f_eval!", FutureWarning)
        Optimizer.__init__(self, *args, **kwargs)

        self.opt_name = "Scaled Conjugate Gradients"

    def opt(self, f_fp=None, f=None, fp=None):
        assert not f is None
        assert not fp is None

        opt_result = SCG(f, fp, self.x_init, display=self.messages,
                         maxiters=self.max_iters,
                         max_f_eval=self.max_f_eval,
                         xtol=self.xtol, ftol=self.ftol,
                         gtol=self.gtol)

        self.x_opt = opt_result[0]
        self.trace = opt_result[1]
        self.f_opt = self.trace[-1]
        self.funct_eval = opt_result[2]
        self.status = opt_result[3]

def get_optimizer(f_min):

    optimizers = {'fmin_tnc': opt_tnc,
          'simplex': opt_simplex,
          'lbfgsb': opt_lbfgsb,
          'scg': opt_SCG}

    if rasm_available:
        optimizers['rasmussen'] = opt_rasm

    for opt_name in optimizers.keys():
        if opt_name.lower().find(f_min.lower()) != -1:
            return optimizers[opt_name]

    raise KeyError('No optimizer was found matching the name: %s' % f_min)
