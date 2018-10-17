# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import re
from ..core.parameterization import Parameterized
import numpy as np
import sympy as sym
from ..core.parameterization import Param
from sympy.utilities.lambdify import lambdastr, _imp_namespace, _get_namespace
from sympy.utilities.iterables import numbered_symbols
import scipy
import GPy


def getFromDict(dataDict, mapList):
    return reduce(lambda d, k: d[k], mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

class Symbolic_core():
    """
    Base model symbolic class.
    """

    def __init__(self, expressions, cacheable, derivatives=None, parameters=None, func_modules=[]):
        # Base class init, do some basic derivatives etc.

        # Func_modules sets up the right mapping for functions.
        func_modules += [{'gamma':scipy.special.gamma,
                          'gammaln':scipy.special.gammaln,
                          'erf':scipy.special.erf, 'erfc':scipy.special.erfc,
                          'erfcx':scipy.special.erfcx,
                          'polygamma':scipy.special.polygamma,
                          'normcdf':GPy.util.functions.normcdf,
                          'normcdfln':GPy.util.functions.normcdfln,
                          'logistic':GPy.util.functions.logistic,
                          'logisticln':GPy.util.functions.logisticln},
                         'numpy']

        self._set_expressions(expressions)
        self._set_variables(cacheable)
        self._set_derivatives(derivatives)
        self._set_parameters(parameters)
        # Convert the expressions to a list for common sub expression elimination
        # We should find the following type of expressions: 'function', 'derivative', 'second_derivative', 'third_derivative'. 
        self.update_expression_list()

        # Apply any global stabilisation operations to expressions.
        self.global_stabilize()

        # Helper functions to get data in and out of dictionaries.
        # this code from http://stackoverflow.com/questions/14692690/access-python-nested-dictionary-items-via-a-list-of-keys

        self.extract_sub_expressions()
        self._gen_code()
        self._set_namespace(func_modules)

    def _set_namespace(self, namespaces):
        """Set the name space for use when calling eval. This needs to contain all the relvant functions for mapping from symbolic python to the numerical python. It also contains variables, cached portions etc."""
        self.namespace = {}
        for m in namespaces[::-1]:
            buf = _get_namespace(m)
            self.namespace.update(buf)
        self.namespace.update(self.__dict__)

    def _set_expressions(self, expressions):
        """Extract expressions and variables from the user provided expressions."""
        self.expressions = {}
        for key, item in expressions.items():
            self.expressions[key] = {'function': item}

    def _set_variables(self, cacheable):
        """Pull the variable names out of the provided expressions and separate into cacheable expressions and normal parameters. Those that are only stored in the cache, the parameters are stored in this object."""
        # pull the parameters and inputs out of the symbolic pdf
        def extract_vars(expr):
            return [e for e in expr.atoms() if e.is_Symbol and e not in vars]
        self.cacheable = cacheable
        self.variables = {}
        vars = []
        for expression in self.expressions.values():
            vars += extract_vars(expression['function'])
        # inputs are assumed to be those things that are
        # cacheable. I.e. those things that aren't stored within the
        # object except as cached. For covariance functions this is X
        # and Z, for likelihoods F and for mapping functions X.
        self.cacheable_vars = [] # list of everything that's cacheable
        for var in cacheable:            
            self.variables[var] = [e for e in vars if e.name.split('_')[0]==var.lower()]
            self.cacheable_vars += self.variables[var]
        for var in cacheable:
            if not self.variables[var]:
                raise ValueError('Variable ' + var + ' was specified as cacheable but is not in expression. Expected to find symbols of the form ' + var.lower() + '_0 to represent ' + var)

        # things that aren't cacheable are assumed to be parameters.
        self.variables['theta'] = sorted([e for e in vars if not e in self.cacheable_vars],key=lambda e:e.name)

    def _set_derivatives(self, derivatives):
        # these are arguments for computing derivatives.
        def extract_derivative(function, derivative_arguments):
            return {theta.name : self.stabilize(sym.diff(function,theta)) for theta in derivative_arguments}
        derivative_arguments = []
        if derivatives is not None:
            for derivative in derivatives:
                derivative_arguments += self.variables[derivative]

            # Do symbolic work to compute derivatives.        
            for key, func in self.expressions.items():
                # if func['function'].is_Matrix:
                #     rows = func['function'].shape[0]
                #     cols = func['function'].shape[1]
                #     self.expressions[key]['derivative'] = sym.zeros(rows, cols)
                #     for i in range(rows):
                #         for j in range(cols):
                #             self.expressions[key]['derivative'][i, j] = extract_derivative(func['function'][i, j], derivative_arguments)
                # else:
                    self.expressions[key]['derivative'] = extract_derivative(func['function'], derivative_arguments)

    def _set_parameters(self, parameters):
        """Add parameters to the model and initialize with given values."""
        for theta in self.variables['theta']:
            val = 1.0
            # TODO: improve approach for initializing parameters.
            if parameters is not None:
                if theta.name in parameters:
                    val = parameters[theta.name]
            # Add parameter.
            
            self.link_parameters(Param(theta.name, val, None))
            #self._set_attribute(theta.name, )

    def eval_parameters_changed(self):
        # TODO: place checks for inf/nan in here
        # do all the precomputation codes.
        self.eval_update_cache()

    def eval_update_cache(self, **kwargs):
        # TODO: place checks for inf/nan in here
        # for all provided keywords

        for var, code in self.variable_sort(self.code['parameters_changed']):
            self._set_attribute(var, eval(code, self.namespace))

        for var, value in kwargs.items():
            # update their cached values
            if value is not None:
                if var == 'X' or var == 'F' or var == 'M':
                    value = np.atleast_2d(value)
                    for i, theta in enumerate(self.variables[var]):
                        self._set_attribute(theta.name, value[:, i][:, None])
                elif var == 'Y':
                    # Y values can be missing.
                    value = np.atleast_2d(value)
                    for i, theta in enumerate(self.variables[var]):
                        self._set_attribute('missing' + str(i), np.isnan(value[:, i]))
                        self._set_attribute(theta.name, value[:, i][:, None])
                elif var == 'Z':
                    value = np.atleast_2d(value)
                    for i, theta in enumerate(self.variables[var]):
                        self._set_attribute(theta.name, value[:, i][None, :])
                else:
                    value = np.atleast_1d(value)
                    for i, theta in enumerate(self.variables[var]):
                        self._set_attribute(theta.name, value[i])
        for var, code in self.variable_sort(self.code['update_cache']):
            self._set_attribute(var, eval(code, self.namespace))

    def eval_update_gradients(self, function, partial, **kwargs):
        # TODO: place checks for inf/nan in here?
        self.eval_update_cache(**kwargs)
        gradient = {}
        for theta in self.variables['theta']:
            code = self.code[function]['derivative'][theta.name]
            gradient[theta.name] = (partial*eval(code, self.namespace)).sum()
        return gradient
        
    def eval_gradients_X(self, function, partial, **kwargs):
        if 'X' in kwargs:
            gradients_X = np.zeros_like(kwargs['X'])
        self.eval_update_cache(**kwargs)
        for i, theta in enumerate(self.variables['X']):
            code = self.code[function]['derivative'][theta.name]
            gradients_X[:, i:i+1] = partial*eval(code, self.namespace)
        return gradients_X

    def eval_function(self, function, **kwargs):
        self.eval_update_cache(**kwargs)
        return eval(self.code[function]['function'], self.namespace)

    def code_parameters_changed(self):
        # do all the precomputation codes.
        lcode = ''
        for variable, code in self.variable_sort(self.code['parameters_changed']):
            lcode += self._print_code(variable) + ' = ' + self._print_code(code) + '\n'
        return lcode
    
    def code_update_cache(self):
        lcode = ''
        for var in self.cacheable:
            lcode += 'if ' + var + ' is not None:\n'
            if var == 'X':
                reorder = '[:, None]'
            elif var == 'Z':
                reorder = '[None, :]'
            else:
                reorder = ''
            for i, theta in enumerate(self.variables[var]):
                lcode+= "\t" + var + '= np.atleast_2d(' + var + ')\n'
                lcode+= "\t" + self._print_code(theta.name) + ' = ' + var + '[:, ' + str(i) + "]" + reorder + "\n"
    
        for variable, code in self.variable_sort(self.code['update_cache']):
            lcode+= self._print_code(variable) + ' = ' + self._print_code(code) + "\n"

        return lcode

    def code_update_gradients(self, function):
        lcode = ''
        for theta in self.variables['theta']:
            code = self.code[function]['derivative'][theta.name]
            lcode += self._print_code(theta.name) + '.gradient = (partial*(' + self._print_code(code) + ')).sum()\n'
        return lcode

    def code_gradients_cacheable(self, function, variable):
        if variable not in self.cacheable:
            raise RuntimeError(variable + ' must be a cacheable.')
        lcode = 'gradients_' + variable + ' = np.zeros_like(' + variable + ')\n'
        lcode += 'self.update_cache(' + ', '.join(self.cacheable) + ')\n'
        for i, theta in enumerate(self.variables[variable]):
            code = self.code[function]['derivative'][theta.name]
            lcode += 'gradients_' + variable + '[:, ' + str(i) + ':' + str(i) + '+1] = partial*' + self._print_code(code) + '\n'
        lcode += 'return gradients_' + variable + '\n'
        return lcode

    def code_function(self, function):
        lcode = 'self.update_cache(' + ', '.join(self.cacheable) + ')\n'
        lcode += 'return ' + self._print_code(self.code[function]['function'])
        return lcode

    def stabilize(self, expr):
        """Stabilize the code in the model."""
        # this code is applied to expressions in the model in an attempt to sabilize them.
        return expr

    def global_stabilize(self):
        """Stabilize all code in the model."""
        pass

    def _set_attribute(self, name, value):
        """Make sure namespace gets updated when setting attributes."""
        setattr(self, name, value)
        self.namespace.update({name: getattr(self, name)})
        

    def update_expression_list(self):
        """Extract a list of expressions from the dictionary of expressions."""
        self.expression_list = [] # code arrives in dictionary, but is passed in this list
        self.expression_keys = [] # Keep track of the dictionary keys.
        self.expression_order = [] # This may be unecessary. It's to give ordering for cse
        for fname, fexpressions in self.expressions.items():
            for type, texpressions in fexpressions.items():
                if type == 'function':
                    self.expression_list.append(texpressions)            
                    self.expression_keys.append([fname, type])
                    self.expression_order.append(1) 
                elif type[-10:] == 'derivative':
                    for dtype, expression in texpressions.items():
                        self.expression_list.append(expression)
                        self.expression_keys.append([fname, type, dtype])
                        if type[:-10] == 'first_' or type[:-10] == '':
                            self.expression_order.append(3) #sym.count_ops(self.expressions[type][dtype]))
                        elif type[:-10] == 'second_':
                            self.expression_order.append(4) #sym.count_ops(self.expressions[type][dtype]))
                        elif type[:-10] == 'third_':
                            self.expression_order.append(5) #sym.count_ops(self.expressions[type][dtype]))
                else:
                    self.expression_list.append(fexpressions[type])            
                    self.expression_keys.append([fname, type])
                    self.expression_order.append(2) 

        # This step may be unecessary.
        # Not 100% sure if the sub expression elimination is order sensitive. This step orders the list with the 'function' code first and derivatives after.
        self.expression_order, self.expression_list, self.expression_keys = zip(*sorted(zip(self.expression_order, self.expression_list, self.expression_keys)))

    def extract_sub_expressions(self, cache_prefix='cache', sub_prefix='sub', prefix='XoXoXoX'):
        # Do the common sub expression elimination.
        common_sub_expressions, expression_substituted_list = sym.cse(self.expression_list, numbered_symbols(prefix=prefix))

        self.variables[cache_prefix] = []
        self.variables[sub_prefix] = []

        # Create dictionary of new sub expressions
        sub_expression_dict = {}
        for var, void in common_sub_expressions:
            sub_expression_dict[var.name] = var

        # Sort out any expression that's dependent on something that scales with data size (these are listed in cacheable).
        cacheable_list = []
        params_change_list = []
        # common_sube_expressions contains a list of paired tuples with the new variable and what it equals
        for var, expr in common_sub_expressions:
            arg_list = [e for e in expr.atoms() if e.is_Symbol]
            # List any cacheable dependencies of the sub-expression
            cacheable_symbols = [e for e in arg_list if e in cacheable_list or e in self.cacheable_vars]
            if cacheable_symbols:
                # list which ensures dependencies are cacheable.
                cacheable_list.append(var)
            else:
                params_change_list.append(var)

        replace_dict = {}
        for i, expr in enumerate(cacheable_list):
            sym_var = sym.var(cache_prefix + str(i))
            self.variables[cache_prefix].append(sym_var)
            replace_dict[expr.name] = sym_var
            
        for i, expr in enumerate(params_change_list):
            sym_var = sym.var(sub_prefix + str(i))
            self.variables[sub_prefix].append(sym_var)
            replace_dict[expr.name] = sym_var

        for replace, void in common_sub_expressions:
            for expr, keys in zip(expression_substituted_list, self.expression_keys):
                setInDict(self.expressions, keys, expr.subs(replace, replace_dict[replace.name]))
            for void, expr in common_sub_expressions:
                expr = expr.subs(replace, replace_dict[replace.name])

        # Replace original code with code including subexpressions.
        for keys in self.expression_keys:
            for replace, void in common_sub_expressions:
                setInDict(self.expressions, keys, getFromDict(self.expressions, keys).subs(replace, replace_dict[replace.name]))
        
        self.expressions['parameters_changed'] = {}
        self.expressions['update_cache'] = {}
        for var, expr in common_sub_expressions:
            for replace, void in common_sub_expressions:
                expr = expr.subs(replace, replace_dict[replace.name])
            if var in cacheable_list:
                self.expressions['update_cache'][replace_dict[var.name].name] = expr
            else:
                self.expressions['parameters_changed'][replace_dict[var.name].name] = expr
            

    def _gen_code(self):
        """Generate code for the list of expressions provided using the common sub-expression eliminator to separate out portions that are computed multiple times."""
        # This is the dictionary that stores all the generated code.

        self.code = {}
        def match_key(expr):
            if type(expr) is dict:
                code = {}
                for key in expr.keys():
                    code[key] = match_key(expr[key])
            else:
                arg_list = [e for e in expr.atoms() if e.is_Symbol]
                code = self._expr2code(arg_list, expr)
            return code

        self.code = match_key(self.expressions)
                            
 
    def _expr2code(self, arg_list, expr):
        """Convert the given symbolic expression into code."""
        code = lambdastr(arg_list, expr)
        function_code = code.split(':')[1].strip()
        #for arg in arg_list:
        #    function_code = function_code.replace(arg.name, 'self.'+arg.name)

        return function_code

    def _print_code(self, code):
        """Prepare code for string writing."""
        # This needs a rewrite --- it doesn't check for match clashes! So sub11 would be replaced by sub1 before being replaced with sub11!!
        for key in self.variables.keys():
            for arg in self.variables[key]:
                code = code.replace(arg.name, 'self.'+arg.name)
        return code

    def _display_expression(self, keys, user_substitutes={}):
        """Helper function for human friendly display of the symbolic components."""
        # Create some pretty maths symbols for the display.
        sigma, alpha, nu, omega, l, variance = sym.var('\sigma, \alpha, \nu, \omega, \ell, \sigma^2')
        substitutes = {'scale': sigma, 'shape': alpha, 'lengthscale': l, 'variance': variance}
        substitutes.update(user_substitutes)

        function_substitutes = {normcdfln : lambda arg : sym.log(normcdf(arg)),
                                logisticln : lambda arg : -sym.log(1+sym.exp(-arg)),
                                logistic : lambda arg : 1/(1+sym.exp(-arg)),
                                erfcx : lambda arg : erfc(arg)/sym.exp(arg*arg),
                                gammaln : lambda arg : sym.log(sym.gamma(arg))}
        expr = getFromDict(self.expressions, keys)
        for var_name, sub in self.variable_sort(self.expressions['update_cache'], reverse=True):
            for var in self.variables['cache']:
                if var_name == var.name:
                    expr = expr.subs(var, sub)
                    break
        for var_name, sub in self.variable_sort(self.expressions['parameters_changed'], reverse=True):
            for var in self.variables['sub']:
                if var_name == var.name:
                    expr = expr.subs(var, sub)
                    break

        for var_name, sub in self.variable_sort(substitutes, reverse=True):
            for var in self.variables['theta']:
                if var_name == var.name:
                    expr = expr.subs(var, sub)
                    break
        for m, r in function_substitutes.items():
            expr = expr.replace(m, r)#normcdfln, lambda arg : sym.log(normcdf(arg)))
        return expr.simplify()

    def variable_sort(self, var_dict, reverse=False):
        def sort_key(x):
            digits = re.findall(r'\d+$', x[0])
            if digits:
                return int(digits[0])
            else:
                return x[0]
            
        return sorted(var_dict.items(), key=sort_key, reverse=reverse)
