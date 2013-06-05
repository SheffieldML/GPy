# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import re
import copy
import cPickle
import warnings
import transformations

class Parameterised(object):
    def __init__(self):
        """
        This is the base class for model and kernel. Mostly just handles tieing and constraining of parameters
        """
        self.tied_indices = []
        self.fixed_indices = []
        self.fixed_values = []
        self.constrained_indices = []
        self.constraints = []

    def pickle(self, filename, protocol= -1):
        f = file(filename, 'w')
        cPickle.dump(self, f, protocol)
        f.close()

    def copy(self):
        """Returns a (deep) copy of the current model """
        return copy.deepcopy(self)

    @property
    def params(self):
        """
        Returns a **copy** of parameters in non transformed space

        :see_also: :py:func:`GPy.core.Parameterised.params_transformed`
        """
        return self._get_params()

    @params.setter
    def params(self, params):
        self._set_params(params)

    @property
    def params_transformed(self):
        """
        Returns a **copy** of parameters in transformed space

        :see_also: :py:func:`GPy.core.Parameterised.params`
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
    def get(self, regexp):
        warnings.warn(self._get_set_deprecation, FutureWarning, stacklevel=2)
        return self[regexp]

    def set(self, regexp, val):
        warnings.warn(self._get_set_deprecation, FutureWarning, stacklevel=2)
        self[regexp] = val

    def __getitem__(self, regexp, return_names=False):
        """
        Get a model parameter by name.  The name is applied as a regular
        expression and all parameters that match that regular expression are
        returned.
        """
        matches = self.grep_param_names(regexp)
        if len(matches):
            if return_names:
                return self._get_params()[matches], np.asarray(self._get_param_names())[matches].tolist()
            else:
                return self._get_params()[matches]
        else:
            raise AttributeError, "no parameter matches %s" % regexp

    def __setitem__(self, name, val):
        """
        Set model parameter(s) by name. The name is provided as a regular
        expression. All parameters matching that regular expression are set to
        the given value.
        """
        matches = self.grep_param_names(name)
        if len(matches):
            val = np.array(val)
            assert (val.size == 1) or val.size == len(matches), "Shape mismatch: {}:({},)".format(val.size, len(matches))
            x = self.params
            x[matches] = val
            self.params = x
        else:
            raise AttributeError, "no parameter matches %s" % name

    def tie_params(self, regexp):
        matches = self.grep_param_names(regexp)
        assert matches.size > 0, "need at least something to tie together"
        if len(self.tied_indices):
            assert not np.any(matches[:, None] == np.hstack(self.tied_indices)), "Some indices are already tied!"
        self.tied_indices.append(matches)
        # TODO only one of the priors will be evaluated. Give a warning message if the priors are not identical
        if hasattr(self, 'prior'):
            pass

        self._set_params_transformed(self._get_params_transformed()) # sets tied parameters to single value

    def untie_everything(self):
        """Unties all parameters by setting tied_indices to an empty list."""
        self.tied_indices = []

    def grep_param_names(self, regexp, transformed=False, search=False):
        """
        :param regexp: regular expression to select parameter names
        :type regexp: re | str | int
        :rtype: the indices of self._get_param_names which match the regular expression.

        Note:-
          Other objects are passed through - i.e. integers which weren't meant for grepping
        """

        if transformed:
            names = self._get_param_names_transformed()
        else:
            names = self._get_param_names()

        if type(regexp) in [str, np.string_, np.str]:
            regexp = re.compile(regexp)
        elif type(regexp) is re._pattern_type:
            pass
        else:
            return regexp
        if search:
            return np.nonzero([regexp.search(name) for name in names])[0]
        else:
            return np.nonzero([regexp.match(name) for name in names])[0]

    def num_params_transformed(self):
        removed = 0
        for tie in self.tied_indices:
            removed += tie.size - 1

        for fix in self.fixed_indices:
            removed += fix.size

        return len(self._get_params()) - removed

    def unconstrain(self, regexp):
        """Unconstrain matching parameters.  does not untie parameters"""
        matches = self.grep_param_names(regexp)

        # tranformed contraints:
        for match in matches:
            self.constrained_indices = [i[i <> match] for i in self.constrained_indices]

        # remove empty constraints
        tmp = zip(*[(i, t) for i, t in zip(self.constrained_indices, self.constraints) if len(i)])
        if tmp:
            self.constrained_indices, self.constraints = zip(*[(i, t) for i, t in zip(self.constrained_indices, self.constraints) if len(i)])
            self.constrained_indices, self.constraints = list(self.constrained_indices), list(self.constraints)

        # fixed:
        self.fixed_values = [np.delete(values, np.nonzero(np.sum(indices[:, None] == matches[None, :], 1))[0]) for indices, values in zip(self.fixed_indices, self.fixed_values)]
        self.fixed_indices = [np.delete(indices, np.nonzero(np.sum(indices[:, None] == matches[None, :], 1))[0]) for indices in self.fixed_indices]

        # remove empty elements
        tmp = [(i, v) for i, v in zip(self.fixed_indices, self.fixed_values) if len(i)]
        if tmp:
            self.fixed_indices, self.fixed_values = zip(*tmp)
            self.fixed_indices, self.fixed_values = list(self.fixed_indices), list(self.fixed_values)
        else:
            self.fixed_indices, self.fixed_values = [], []

    def constrain_negative(self, regexp):
        """ Set negative constraints. """
        self.constrain(regexp, transformations.negative_exponent())

    def constrain_positive(self, regexp):
        """ Set positive constraints. """
        self.constrain(regexp, transformations.logexp())

    def constrain_bounded(self, regexp, lower, upper):
        """ Set bounded constraints. """
        self.constrain(regexp, transformations.logistic(lower, upper))

    def all_constrained_indices(self):
        if len(self.constrained_indices) or len(self.fixed_indices):
            return np.hstack(self.constrained_indices + self.fixed_indices)
        else:
            return np.empty(shape=(0,))

    def constrain(self, regexp, transform):
        assert isinstance(transform, transformations.transformation)

        matches = self.grep_param_names(regexp)
        overlap = set(matches).intersection(set(self.all_constrained_indices()))
        if overlap:
            self.unconstrain(np.asarray(list(overlap)))
            print 'Warning: re-constraining these parameters'
            pn = self._get_param_names()
            for i in overlap:
                print pn[i]

        self.constrained_indices.append(matches)
        self.constraints.append(transform)
        x = self._get_params()
        x[matches] = transform.initialize(x[matches])
        self._set_params(x)

    def constrain_fixed(self, regexp, value=None):
        """
        Arguments
        ---------
        :param regexp: np.array(dtype=int), or regular expression object or string
        :param value: a float to fix the matched values to. If the value is not specified,
                 the parameter is fixed to the current value

        Notes
        -----
        Fixing a parameter which is tied to another, or constrained in some way will result in an error.
        To fix multiple parameters to the same value, simply pass a regular expression which matches both parameter names, or pass both of the indexes
        """
        matches = self.grep_param_names(regexp)
        overlap = set(matches).intersection(set(self.all_constrained_indices()))
        if overlap:
            self.unconstrain(np.asarray(list(overlap)))
            print 'Warning: re-constraining these parameters'
            pn = self._get_param_names()
            for i in overlap:
                print pn[i]

        self.fixed_indices.append(matches)
        if value != None:
            self.fixed_values.append(value)
        else:
            self.fixed_values.append(self._get_params()[self.fixed_indices[-1]])

        # self.fixed_values.append(value)
        self._set_params_transformed(self._get_params_transformed())

    def _get_params_transformed(self):
        """use self._get_params to get the 'true' parameters of the model, which are then tied, constrained and fixed"""
        x = self._get_params()
        [np.put(x, i, t.finv(x[i])) for i, t in zip(self.constrained_indices, self.constraints)]

        to_remove = self.fixed_indices + [t[1:] for t in self.tied_indices]
        if len(to_remove):
            return np.delete(x, np.hstack(to_remove))
        else:
            return x

    def _set_params_transformed(self, x):
        """ takes the vector x, which is then modified (by untying, reparameterising or inserting fixed values), and then call self._set_params"""
        self._set_params(self._untransform_params(x))

    def _untransform_params(self, x):
        """
        The transformation required for _set_params_transformed.

        This moves the vector x seen by the optimiser (unconstrained) to the
        valid parameter vector seen by the model

        Note:
          - This function is separate from _set_params_transformed for downstream flexibility
        """
        # work out how many places are fixed, and where they are. tricky logic!
        fix_places = self.fixed_indices + [t[1:] for t in self.tied_indices]
        if len(fix_places):
            fix_places = np.hstack(fix_places)
            Nfix_places = fix_places.size
        else:
            Nfix_places = 0

        free_places = np.setdiff1d(np.arange(Nfix_places + x.size, dtype=np.int), fix_places)

        # put the models values in the vector xx
        xx = np.zeros(Nfix_places + free_places.size, dtype=np.float64)

        xx[free_places] = x
        [np.put(xx, i, v) for i, v in zip(self.fixed_indices, self.fixed_values)]
        [np.put(xx, i, v) for i, v in [(t[1:], xx[t[0]]) for t in self.tied_indices] ]

        [np.put(xx, i, t.f(xx[i])) for i, t in zip(self.constrained_indices, self.constraints)]
        if hasattr(self, 'debug'):
            stop # @UndefinedVariable

        return xx

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
        if len(self.fixed_indices):
            remove = np.hstack((remove, np.hstack(self.fixed_indices)))

        # add markers to show that some variables are constrained
        for i, t in zip(self.constrained_indices, self.constraints):
            for ii in i:
                n[ii] = n[ii] + t.__str__()

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
        values = self._get_params() # map(str,self._get_params())
        # sort out the constraints
        constraints = [''] * len(names)
        for i, t in zip(self.constrained_indices, self.constraints):
            for ii in i:
                constraints[ii] = t.__str__()
        for i in self.fixed_indices:
            for ii in i:
                constraints[ii] = 'Fixed'
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
        # columns = cols.sum()

        header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
        header_string = map(lambda x: '|'.join(x), [header_string])
        separator = '-' * len(header_string[0])
        param_string = ["{n:^{c0}}|{v:^{c1}}|{c:^{c2}}|{t:^{c3}}".format(n=names[i], v=values[i], c=constraints[i], t=ties[i], c0=cols[0], c1=cols[1], c2=cols[2], c3=cols[3]) for i in range(len(values))]


        return ('\n'.join([header_string[0], separator] + param_string)) + '\n'
