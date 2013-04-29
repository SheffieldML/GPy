# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import re
import copy
import cPickle
import os
from ..util.squashers import sigmoid
import warnings

def truncate_pad(string, width, align='m'):
    """
    A helper function to make aligned strings for parameterised.__str__
    """
    width = max(width, 4)
    if len(string) > width:
        return string[:width - 3] + '...'
    elif len(string) == width:
        return string
    elif len(string) < width:
        diff = width - len(string)
        if align == 'm':
            return ' ' * np.floor(diff / 2.) + string + ' ' * np.ceil(diff / 2.)
        elif align == 'l':
            return string + ' ' * diff
        elif align == 'r':
            return ' ' * diff + string
        else:
            raise ValueError

class parameterised(object):
    def __init__(self):
        """
        This is the base class for model and kernel. Mostly just handles tieing and constraining of parameters
        """
        self.tied_indices = []
        self.constrained_fixed_indices = []
        self.constrained_fixed_values = []
        self.constrained_positive_indices = np.empty(shape=(0,), dtype=np.int64)
        self.constrained_negative_indices = np.empty(shape=(0,), dtype=np.int64)
        self.constrained_bounded_indices = []
        self.constrained_bounded_uppers = []
        self.constrained_bounded_lowers = []

    def pickle(self, filename, protocol= -1):
        f = file(filename, 'w')
        cPickle.dump(self, f, protocol)
        f.close()

    def copy(self):
        """
        Returns a (deep) copy of the current model
        """

        return copy.deepcopy(self)

    @property
    def params(self):
        """
        Returns a **copy** of parameters in non transformed space
        
        :see_also: :py:func:`GPy.core.parameterised.params_transformed` 
        """
        return self._get_params()
    @params.setter
    def params(self, params):
        self._set_params(params)

    @property
    def params_transformed(self):
        """
        Returns a **copy** of parameters in transformed space
        
        :see_also: :py:func:`GPy.core.parameterised.params` 
        """
        return self._get_params_transformed()
    @params_transformed.setter
    def params_transformed(self, params):
        self._set_params_transformed(params)

    _get_set_deprecation = """get and set methods wont be available at next minor release
        in the next releases you will get and set with following syntax:
        Assume m is a model class:
        print m['var']          # > prints all parameters matching 'var'
        m['var'] = 2.           # > sets all parameters matching 'var' to 2.
        m['var'] = <array-like> # > sets parameters matching 'var' to <array-like>        
        """
    def get(self, name):
        warnings.warn(self._get_set_deprecation, FutureWarning, stacklevel=2)
        return self[name]

    def set(self, name, val):
        warnings.warn(self._get_set_deprecation, FutureWarning, stacklevel=2)
        self[name] = val

    def __getitem__(self, name, return_names=False):
        """
        Get a model parameter by name. The name is applied as a regular expression and all parameters that match that regular expression are returned.
        """
        matches = self.grep_param_names(name)
        if len(matches):
            if return_names:
                return self._get_params()[matches], np.asarray(self._get_param_names())[matches].tolist()
            else:
                return self._get_params()[matches]
        else:
            raise AttributeError, "no parameter matches %s" % name

    def __setitem__(self, name, val):
        """
        Set model parameter(s) by name. The name is provided as a regular expression. All parameters matching that regular expression are set to ghe given value.
        """
        matches = self.grep_param_names(name)
        if len(matches):
            val = np.array(val)
            assert (val.size == 1) or val.size == len(matches), "Shape mismatch: {}:({},)".format(val.size, len(matches))
            x = self.params
            x[matches] = val
            self.params = x
#             import ipdb;ipdb.set_trace()
#             self.params[matches] = val
        else:
            raise AttributeError, "no parameter matches %s" % name

    def tie_params(self, which):
        matches = self.grep_param_names(which)
        assert matches.size > 0, "need at least something to tie together"
        if len(self.tied_indices):
            assert not np.any(matches[:, None] == np.hstack(self.tied_indices)), "Some indices are already tied!"
        self.tied_indices.append(matches)
        # TODO only one of the priors will be evaluated. Give a warning message if the priors are not identical
        if hasattr(self, 'prior'):
            pass

        self._set_params_transformed(self._get_params_transformed())  # sets tied parameters to single value

    def untie_everything(self):
        """Unties all parameters by setting tied_indices to an empty list."""
        self.tied_indices = []

    def all_constrained_indices(self):
        """Return a np array of all the constrained indices"""
        ret = [np.hstack(i) for i in [self.constrained_bounded_indices, self.constrained_positive_indices, self.constrained_negative_indices, self.constrained_fixed_indices] if len(i)]
        if len(ret):
            return np.hstack(ret)
        else:
            return []
    def grep_param_names(self, expr):
        """
        Arguments
        ---------
        expr -- can be a regular expression object or a string to be turned into regular expression object.

        Returns
        -------
        the indices of self._get_param_names which match the regular expression.

        Notes
        -----
        Other objects are passed through - i.e. integers which were'nt meant for grepping
        """

        if type(expr) in [str, np.string_, np.str]:
            expr = re.compile(expr)
            return np.nonzero([expr.search(name) for name in self._get_param_names()])[0]
        elif type(expr) is re._pattern_type:
            return np.nonzero([expr.search(name) for name in self._get_param_names()])[0]
        else:
            return expr

    def Nparam_transformed(self):
            ties = 0
            for ar in self.tied_indices:
                ties += ar.size - 1
            return self.Nparam - len(self.constrained_fixed_indices) - ties

    def constrain_positive(self, which):
        """
        Set positive constraints.

        Arguments
        ---------
        which -- np.array(dtype=int), or regular expression object or string
        """
        matches = self.grep_param_names(which)
        assert not np.any(matches[:, None] == self.all_constrained_indices()), "Some indices are already constrained"
        self.constrained_positive_indices = np.hstack((self.constrained_positive_indices, matches))
        # check to ensure constraint is in place
        x = self._get_params()
        for i, xx in enumerate(x):
            if (xx < 0) & (i in matches):
                x[i] = -xx
        self._set_params(x)


    def unconstrain(self, which):
        """Unconstrain matching parameters.  does not untie parameters"""
        matches = self.grep_param_names(which)
        # positive/negative
        self.constrained_positive_indices = np.delete(self.constrained_positive_indices, np.nonzero(np.sum(self.constrained_positive_indices[:, None] == matches[None, :], 1))[0])
        self.constrained_negative_indices = np.delete(self.constrained_negative_indices, np.nonzero(np.sum(self.constrained_negative_indices[:, None] == matches[None, :], 1))[0])
        # bounded
        if len(self.constrained_bounded_indices):
            self.constrained_bounded_indices = [np.delete(a, np.nonzero(np.sum(a[:, None] == matches[None, :], 1))[0]) for a in self.constrained_bounded_indices]
            if np.hstack(self.constrained_bounded_indices).size:
                self.constrained_bounded_uppers, self.constrained_bounded_lowers, self.constrained_bounded_indices = zip(*[(u, l, i) for u, l, i in zip(self.constrained_bounded_uppers, self.constrained_bounded_lowers, self.constrained_bounded_indices) if i.size])
                self.constrained_bounded_uppers, self.constrained_bounded_lowers, self.constrained_bounded_indices = list(self.constrained_bounded_uppers), list(self.constrained_bounded_lowers), list(self.constrained_bounded_indices)
            else:
                self.constrained_bounded_uppers, self.constrained_bounded_lowers, self.constrained_bounded_indices = [], [], []
        # fixed:
        for i, indices in enumerate(self.constrained_fixed_indices):
            self.constrained_fixed_indices[i] = np.delete(indices, np.nonzero(np.sum(indices[:, None] == matches[None, :], 1))[0])
        # remove empty elements
        tmp = [(i, v) for i, v in zip(self.constrained_fixed_indices, self.constrained_fixed_values) if len(i)]
        if tmp:
            self.constrained_fixed_indices, self.constrained_fixed_values = zip(*tmp)
            self.constrained_fixed_indices, self.constrained_fixed_values = list(self.constrained_fixed_indices), list(self.constrained_fixed_values)
        else:
            self.constrained_fixed_indices, self.constrained_fixed_values = [], []



    def constrain_negative(self, which):
        """
        Set negative constraints.

        :param which: which variables to constrain
        :type which: regular expression string

        """
        matches = self.grep_param_names(which)
        assert not np.any(matches[:, None] == self.all_constrained_indices()), "Some indices are already constrained"
        self.constrained_negative_indices = np.hstack((self.constrained_negative_indices, matches))
        # check to ensure constraint is in place
        x = self._get_params()
        for i, xx in enumerate(x):
            if (xx > 0.) and (i in matches):
                x[i] = -xx
        self._set_params(x)



    def constrain_bounded(self, which, lower, upper):
        """Set bounded constraints.

        Arguments
        ---------
        which -- np.array(dtype=int), or regular expression object or string
        upper -- (float) the upper bound on the constraint
        lower -- (float) the lower bound on the constraint
        """
        matches = self.grep_param_names(which)
        assert not np.any(matches[:, None] == self.all_constrained_indices()), "Some indices are already constrained"
        assert lower < upper, "lower bound must be smaller than upper bound!"
        self.constrained_bounded_indices.append(matches)
        self.constrained_bounded_uppers.append(upper)
        self.constrained_bounded_lowers.append(lower)
        # check to ensure constraint is in place
        x = self._get_params()
        for i, xx in enumerate(x):
            if ((xx <= lower) | (xx >= upper)) & (i in matches):
                x[i] = sigmoid(xx) * (upper - lower) + lower
        self._set_params(x)


    def constrain_fixed(self, which, value=None):
        """
        Arguments
        ---------
        :param which: np.array(dtype=int), or regular expression object or string
        :param value: a float to fix the matched values to. If the value is not specified,
                 the parameter is fixed to the current value

        Notes
        -----
        Fixing a parameter which is tied to another, or constrained in some way will result in an error.
        To fix multiple parameters to the same value, simply pass a regular expression which matches both parameter names, or pass both of the indexes
        """
        matches = self.grep_param_names(which)
        assert not np.any(matches[:, None] == self.all_constrained_indices()), "Some indices are already constrained"
        self.constrained_fixed_indices.append(matches)
        if value != None:
            self.constrained_fixed_values.append(value)
        else:
            self.constrained_fixed_values.append(self._get_params()[self.constrained_fixed_indices[-1]])

        # self.constrained_fixed_values.append(value)
        self._set_params_transformed(self._get_params_transformed())

    def _get_params_transformed(self):
        """use self._get_params to get the 'true' parameters of the model, which are then tied, constrained and fixed"""
        x = self._get_params()
        x[self.constrained_positive_indices] = np.log(x[self.constrained_positive_indices])
        x[self.constrained_negative_indices] = np.log(-x[self.constrained_negative_indices])
        [np.put(x, i, np.log(np.clip(x[i] - l, 1e-10, np.inf) / np.clip(h - x[i], 1e-10, np.inf))) for i, l, h in zip(self.constrained_bounded_indices, self.constrained_bounded_lowers, self.constrained_bounded_uppers)]

        to_remove = self.constrained_fixed_indices + [t[1:] for t in self.tied_indices]
        if len(to_remove):
            return np.delete(x, np.hstack(to_remove))
        else:
            return x


    def _set_params_transformed(self, x):
        """ takes the vector x, which is then modified (by untying, reparameterising or inserting fixed values), and then call self._set_params"""

        # work out how many places are fixed, and where they are. tricky logic!
        Nfix_places = 0.
        if len(self.tied_indices):
            Nfix_places += np.hstack(self.tied_indices).size - len(self.tied_indices)
        if len(self.constrained_fixed_indices):
            Nfix_places += np.hstack(self.constrained_fixed_indices).size
        if Nfix_places:
            fix_places = np.hstack(self.constrained_fixed_indices + [t[1:] for t in self.tied_indices])
        else:
            fix_places = []

        free_places = np.setdiff1d(np.arange(Nfix_places + x.size, dtype=np.int), fix_places)

        # put the models values in the vector xx
        xx = np.zeros(Nfix_places + free_places.size, dtype=np.float64)

        xx[free_places] = x
        [np.put(xx, i, v) for i, v in zip(self.constrained_fixed_indices, self.constrained_fixed_values)]
        [np.put(xx, i, v) for i, v in [(t[1:], xx[t[0]]) for t in self.tied_indices] ]
        xx[self.constrained_positive_indices] = np.exp(xx[self.constrained_positive_indices])
        xx[self.constrained_negative_indices] = -np.exp(xx[self.constrained_negative_indices])
        [np.put(xx, i, low + sigmoid(xx[i]) * (high - low)) for i, low, high in zip(self.constrained_bounded_indices, self.constrained_bounded_lowers, self.constrained_bounded_uppers)]
        self._set_params(xx)

    def _get_param_names_transformed(self):
        """
        Returns the parameter names as propagated after constraining,
        tying or fixing, i.e. a list of the same length as _get_params_transformed()
        """
        n = self._get_param_names()

        # remove/concatenate the tied parameter names
        if len(self.tied_indices):
            for t in self.tied_indices:
                n[t[0]] = "<tie>".join([n[tt] for tt in t])
            remove = np.hstack([t[1:] for t in self.tied_indices])
        else:
            remove = np.empty(shape=(0,), dtype=np.int)

        # also remove the fixed params
        if len(self.constrained_fixed_indices):
            remove = np.hstack((remove, np.hstack(self.constrained_fixed_indices)))

        # add markers to show that some variables are constrained
        for i in self.constrained_positive_indices:
            n[i] = n[i] + '(+ve)'
        for i in self.constrained_negative_indices:
            n[i] = n[i] + '(-ve)'
        for i, l, h in zip(self.constrained_bounded_indices, self.constrained_bounded_lowers, self.constrained_bounded_uppers):
            for ii in i:
                n[ii] = n[ii] + '(bounded)'

        n = [nn for i, nn in enumerate(n) if not i in remove]
        return n

    def __str__(self, nw=30):
        """
        Return a string describing the parameter names and their ties and constraints
        """
        names = self._get_param_names()
        N = len(names)

        if not N:
            return "This object has no free parameters."
        header = ['Name', 'Value', 'Constraints', 'Ties']
        values = self._get_params()  # map(str,self._get_params())
        # sort out the constraints
        constraints = [''] * len(names)
        for i in self.constrained_positive_indices:
            constraints[i] = '(+ve)'
        for i in self.constrained_negative_indices:
            constraints[i] = '(-ve)'
        for i in self.constrained_fixed_indices:
            for ii in i:
                constraints[ii] = 'Fixed'
        for i, u, l in zip(self.constrained_bounded_indices, self.constrained_bounded_uppers, self.constrained_bounded_lowers):
            for ii in i:
                constraints[ii] = '(' + str(l) + ', ' + str(u) + ')'
        # sort out the ties
        ties = [''] * len(names)
        for i, tie in enumerate(self.tied_indices):
            for j in tie:
                ties[j] = '(' + str(i) + ')'

        values = ['%.4f' % float(v) for v in values]
        max_names = max([len(names[i]) for i in range(len(names))] + [len(header[0])])
        max_values = max([len(values[i]) for i in range(len(values))] + [len(header[1])])
        max_constraint = max([len(constraints[i]) for i in range(len(constraints))] + [len(header[2])])
        max_ties = max([len(ties[i]) for i in range(len(ties))] + [len(header[3])])
        cols = np.array([max_names, max_values, max_constraint, max_ties]) + 4
        columns = cols.sum()

        header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
        header_string = map(lambda x: '|'.join(x), [header_string])
        separator = '-' * len(header_string[0])
        param_string = ["{n:^{c0}}|{v:^{c1}}|{c:^{c2}}|{t:^{c3}}".format(n=names[i], v=values[i], c=constraints[i], t=ties[i], c0=cols[0], c1=cols[1], c2=cols[2], c3=cols[3]) for i in range(len(values))]


        return ('\n'.join([header_string[0], separator] + param_string)) + '\n'
