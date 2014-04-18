# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
from ..core.parameterization import Parameterized
import numpy as np
import sympy as sym
from ..core.parameterization import Param
from sympy.utilities.lambdify import lambdastr, _imp_namespace, _get_namespace
from sympy.utilities.iterables import numbered_symbols
from sympy import exp
from scipy.special import gammaln, gamma, erf, erfc, erfcx, polygamma
from GPy.util.functions import normcdf, normcdfln, logistic, logisticln

class Symbolic_core():
    """
    Base model symbolic class.
    """

    def __init__(self, expressions, cacheable, derivatives=None, parameters=None, func_modules=[]):
        # Base class init, do some basic derivatives etc.

        # Func_modules sets up the right mapping for functions.
        self.func_modules = func_modules
        self.func_modules += [{'gamma':gamma,
                               'gammaln':gammaln,
                               'erf':erf, 'erfc':erfc,
                               'erfcx':erfcx,
                               'polygamma':polygamma,
                               'normcdf':normcdf,
                               'normcdfln':normcdfln,
                               'logistic':logistic,
                               'logisticln':logisticln},
                              'numpy']

        self._set_expressions(expressions)
        self._set_variables(cacheable)
        self._set_derivatives(derivatives)
        self._set_parameters(parameters)
        self.namespace = [globals(), self.__dict__]
        self._gen_code()

    def _set_expressions(self, expressions):
        """Extract expressions and variables from the user provided expressions."""
        self.expressions = {}
        for key, item in expressions.items():
            self.expressions[key] = {'function': item}

    def _set_variables(self, cacheable):
        """Pull the variable names out of the provided expressions and separate into cacheable expressions and normal parameters. Those that are only stored in the cache, the parameters are stored in this object."""
        # pull the parameters and inputs out of the symbolic pdf
        self.cacheable = cacheable
        self.variables = {}
        vars = []
        for expression in self.expressions.values():
            vars += [e for e in expression['function'].atoms() if e.is_Symbol and e not in vars]
        print vars
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
        derivative_arguments = []
        if derivatives is not None:
            for derivative in derivatives:
                derivative_arguments += self.variables[derivative]

            # Do symbolic work to compute derivatives.        
            for key, func in self.expressions.items():
                self.expressions[key]['derivative'] = {theta.name : sym.diff(func['function'],theta) for theta in derivative_arguments}

    def _set_parameters(self, parameters):
        """Add parameters to the model and initialize with given values."""
        for theta in self.variables['theta']:
            val = 1.0
            # TODO: improve approach for initializing parameters.
            if parameters is not None:
                if parameters.has_key(theta.name):
                    val = parameters[theta.name]
            # Add parameter.
            
            self.add_parameters(Param(theta.name, val, None))
            #setattr(self, theta.name, )

    def eval_parameters_changed(self):
        # TODO: place checks for inf/nan in here
        # do all the precomputation codes.
        for variable, code in sorted(self.code['parameters_change'].iteritems()):
            setattr(self, variable, eval(code, *self.namespace))
        self.eval_update_cache()

    def eval_update_cache(self, **kwargs):
        # TODO: place checks for inf/nan in here
        # for all provided keywords
        for variable, value in kwargs.items():
            # update their cached values
            if value is not None:
                if variable == 'X' or variable == 'F' or variable == 'Mu':
                    for i, theta in enumerate(self.variables[variable]):
                        setattr(self, theta.name, value[:, i][:, None])
                elif variable.name == 'Z':
                    for i, theta in enumerate(self.variables[variable]):
                        setattr(self, theta.name, value[:, i][None, :])
                else:
                    setattr(self, theta.name, value[:, i])

        for variable, code in sorted(self.code['update_cache'].iteritems()):
            setattr(self, variable, eval(code, *self.namespace))

    def eval_update_gradients(self, function, partial, **kwargs):
        # TODO: place checks for inf/nan in here
        self.eval_update_cache(**kwargs)
        for theta in self.variables['theta']:
            code = self.code[function]['derivative'][theta.name]
            setattr(getattr(self, theta.name),
                    'gradient',
                    (partial*eval(code, *self.namespace)).sum())

    def eval_gradients_X(self, function, partial, **kwargs):
        if kwargs.has_key('X'):
            gradients_X = np.zeros_like(kwargs['X'])
        self.eval_update_cache(**kwargs)
        for i, theta in enumerate(self.variables['X']):
            code = self.code[function]['derivative'][theta.name]
            gradients_X[:, i:i+1] = partial*eval(code, *self.namespace)
        return gradients_X

    def eval_function(self, function, **kwargs):
        self.eval_update_cache(**kwargs)
        return eval(self.code[function]['function'], *self.namespace)

    def code_parameters_changed(self):
        # do all the precomputation codes.
        lcode = ''
        for variable, code in sorted(self.code['parameters_change'].iteritems()):
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
                lcode+= "\t" + self._print_code(theta.name) + ' = ' + var + '[:, ' + str(i) + "]" + reorder + "\n"
    
        for variable, code in sorted(self.code['update_cache'].iteritems()):
            lcode+= self._print_code(variable) + ' = ' + self._print_code(code) + "\n"

        return lcode

    def code_update_gradients(self, function):
        lcode = ''
        for theta in self.variables['theta']:
            code = self.code[function]['derivative'][theta.name]
            lcode += self._print_code(theta.name) + '.gradient = (partial*(' + self._print_code(code) + ')).sum()\n'
        return lcode

    def code_gradients_X(self, function):
        lcode = 'gradients_X = np.zeros_like(X)\n'
        lcode += 'self.update_cache(' + ', '.join(self.cacheable) + ')\n'
        for i, theta in enumerate(self.variables['X']):
            code = self.code[function]['derivative'][theta.name]
            lcode += 'gradients_X[:, ' + str(i) + ':' + str(i) + '+1] = partial*' + self._print_code(code) + '\n'
        lcode += 'return gradients_X\n'
        return lcode

    def code_function(self, function):
        lcode = 'self.update_cache(' + ', '.join(self.cacheable) + ')\n'
        lcode += 'return ' + self._print_code(self.code[function]['function'])
        return lcode

    def stabilise(self):
        """Stabilize the code in the model."""
        # this code is applied to all expressions in the model in an attempt to sabilize them.
        pass

    def _gen_namespace(self, modules=None, use_imps=True):
        """Gets the relevant namespaces for the given expressions."""
        from sympy.core.symbol import Symbol

        # If the user hasn't specified any modules, use what is available.
        module_provided = True
        if modules is None:
            module_provided = False
            # Use either numpy (if available) or python.math where possible.
            # XXX: This leads to different behaviour on different systems and
            #      might be the reason for irreproducible errors.
            modules = ["math", "mpmath", "sympy"]

            try:
                _import("numpy")
            except ImportError:
                pass
            else:
                modules.insert(1, "numpy")


        # Get the needed namespaces.
        namespaces = []
        # First find any function implementations
        if use_imps:
            for expr in self._expression_list:
                namespaces.append(_imp_namespace(expr))
        # Check for dict before iterating
        if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
            namespaces.append(modules)
        else:
            namespaces += list(modules)
        # fill namespace with first having highest priority
        namespace = {}
        for m in namespaces[::-1]:
            buf = _get_namespace(m)
            namespace.update(buf)
        for expr in self._expression_list:
            if hasattr(expr, "atoms"):
                #Try if you can extract symbols from the expression.
                #Move on if expr.atoms in not implemented.
                syms = expr.atoms(Symbol)
                for term in syms:
                    namespace.update({str(term): term})


        return namespace
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
                    self.code[fname] = {type: ''}
                elif type[-10:] == 'derivative':
                    self.code[fname] = {type:{}}
                    for dtype, expression in texpressions.items():
                        self.expression_list.append(expression)
                        self.expression_keys.append([fname, type, dtype])
                        if type[:-10] == 'first' or type[:-10] == '':
                            self.expression_order.append(3) #sym.count_ops(self.expressions[type][dtype]))
                        elif type[:-10] == 'second':
                            self.expression_order.append(4) #sym.count_ops(self.expressions[type][dtype]))
                        elif type[:-10] == 'third':
                            self.expression_order.append(5) #sym.count_ops(self.expressions[type][dtype]))
                        self.code[fname][type][dtype] = ''
                else:
                    self.expression_list.append(fexpressions[type])            
                    self.expression_keys.append([fname, type])
                    self.expression_order.append(2) 
                    self.code[fname][type] = ''

        # This step may be unecessary.
        # Not 100% sure if the sub expression elimination is order sensitive. This step orders the list with the 'function' code first and derivatives after.
        self.expression_order, self.expression_list, self.expression_keys = zip(*sorted(zip(self.expression_order, self.expression_list, self.expression_keys)))


    def _gen_code(self, cache_prefix = 'cache', sub_prefix = 'sub', prefix='XoXoXoX'):
        """Generate code for the list of expressions provided using the common sub-expression eliminator to separate out portions that are computed multiple times."""
        # This is the dictionary that stores all the generated code.
        self.code = {}

        # Convert the expressions to a list for common sub expression elimination
        # We should find the following type of expressions: 'function', 'derivative', 'second_derivative', 'third_derivative'. 
        self.update_expression_list()

        # Apply any global stabilisation operations to expressions.
        self.stabilise()

        # Helper functions to get data in and out of dictionaries.
        # this code from http://stackoverflow.com/questions/14692690/access-python-nested-dictionary-items-via-a-list-of-keys
        def getFromDict(dataDict, mapList):
            return reduce(lambda d, k: d[k], mapList, dataDict)
        def setInDict(dataDict, mapList, value):
            getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


        # Do the common sub expression elimination
        subexpressions, expression_substituted_list = sym.cse(self.expression_list, numbered_symbols(prefix=prefix))
        cacheable_list = []

        # Sort out any expression that's dependent on something that scales with data size (these are listed in cacheable).
        self.expressions['parameters_change'] = []
        self.expressions['update_cache'] = []
        cache_expressions_list = []
        sub_expression_list = []
        for expr in subexpressions:
            arg_list = [e for e in expr[1].atoms() if e.is_Symbol]
            cacheable_symbols = [e for e in arg_list if e in cacheable_list or e in self.cacheable_vars]
            if cacheable_symbols:
                self.expressions['update_cache'].append((expr[0].name, self._expr2code(arg_list, expr[1])))
                # list which ensures dependencies are cacheable.
                cacheable_list.append(expr[0])
                cache_expressions_list.append(expr[0].name)
            else:
                self.expressions['parameters_change'].append((expr[0].name, self._expr2code(arg_list, expr[1])))
                sub_expression_list.append(expr[0].name)

        # Replace original code with code including subexpressions.
        for expr, keys in zip(expression_substituted_list, self.expression_keys):
            arg_list = [e for e in expr.atoms() if e.is_Symbol]
            setInDict(self.code, keys, self._expr2code(arg_list, expr))
            setInDict(self.expressions, keys, expr)

        # Create variable names for cache and sub expression portions
        cache_dict = {}
        self.variables[cache_prefix] = []
        for i, sub in enumerate(cache_expressions_list):
            name = cache_prefix + str(i)
            cache_dict[sub] = name
            self.variables[cache_prefix].append(sym.var(name))

        sub_dict = {}
        self.variables[sub_prefix] = []
        for i, sub in enumerate(sub_expression_list):
            name = sub_prefix + str(i)
            sub_dict[sub] = name
            self.variables[sub_prefix].append(sym.var(name))

        # Replace sub expressions in main code with either cacheN or subN.
        for key, val in cache_dict.iteritems():
            for keys in self.expression_keys:
                setInDict(self.code, keys,
                          getFromDict(self.code,keys).replace(key, val))

        for key, val in sub_dict.iteritems():
            for keys in self.expression_keys:
                setInDict(self.code, keys,
                          getFromDict(self.code,keys).replace(key, val))

        # Set up precompute code as either cacheN or subN.
        self.code['update_cache'] = {}
        for key, val in self.expressions['update_cache']:
            expr = val
            for key2, val2 in cache_dict.iteritems():
                expr = expr.replace(key2, val2)
            for key2, val2 in sub_dict.iteritems():
                expr = expr.replace(key2, val2)
            self.code['update_cache'][cache_dict[key]] = expr

        self.expressions['update_cache'] = dict(self.expressions['update_cache'])
        self.code['parameters_change'] = {}
        for key, val in self.expressions['parameters_change']:
            expr = val
            for key2, val2 in cache_dict.iteritems():
                expr = expr.replace(key2, val2)
            for key2, val2 in sub_dict.iteritems():
                expr = expr.replace(key2, val2)
            self.code['parameters_change'][sub_dict[key]] = expr
        self.expressions['parameters_change'] = dict(self.expressions['parameters_change'])
 
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
