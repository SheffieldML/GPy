import sympy as sym
from sympy import Function, S, oo, I, cos, sin, asin, log, erf, pi, exp, sqrt, sign, gamma, polygamma
from sympy.utilities.lambdify import lambdastr
from sympy.utilities.iterables import numbered_symbols
def stabilise(e):
    """Attempt to find the most numerically stable form of an expression"""
    return e #sym.expand(e)


def gen_code(expressions, cache_prefix = 'cache', sub_prefix = 'sub', prefix='XoXoXoX', cacheable=[]):
    """Generate code for the list of expressions provided using the common sub-expression eliminator to separate out portions that are computed multiple times."""
    # First convert the expressions to a list.
    # We should find the following type of expressions: 'function', 'derivative', 'second_derivative', 'third_derivative'. 

    # Helper functions to get data in and out of dictionaries.
    # this code from http://stackoverflow.com/questions/14692690/access-python-nested-dictionary-items-via-a-list-of-keys
    def getFromDict(dataDict, mapList):
        return reduce(lambda d, k: d[k], mapList, dataDict)
    def setInDict(dataDict, mapList, value):
        getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

    # This is the return dictionary that stores all the generated code.
    code = {}
    expression_list = []
    key_list = []
    order_list = []
    code['main'] = {}
    for key in expressions.keys():
        if key == 'function':
            expression_list.append(expressions[key])            
            key_list.append([key])
            order_list.append(1) 
            code['main'][key] = ''
        elif key[-10:] == 'derivative':
            code['main'][key] = {}
            for dkey in expressions[key].keys():
                expression_list.append(expressions[key][dkey])
                key_list.append([key, dkey])
                if key[:-10] == 'first' or key[:-10] == '':
                    order_list.append(3) #sym.count_ops(expressions[key][dkey]))
                elif key[:-10] == 'second':
                    order_list.append(4) #sym.count_ops(expressions[key][dkey]))
                elif key[:-10] == 'third':
                    order_list.append(5) #sym.count_ops(expressions[key][dkey]))
                code['main'][key][dkey] = ''
        else:
            expression_list.append(expressions[key])            
            key_list.append([key])
            order_list.append(2) 
            code['main'][key] = ''

    # This step may be unecessary.
    # Not 100% sure if the sub expression elimination is order sensitive. This step orders the list with the 'function' code first and derivatives after.
    order_list, expression_list, key_list = zip(*sorted(zip(order_list, expression_list, key_list)))

    print expression_list

    subexpressions, expression_substituted_list = sym.cse(expression_list, numbered_symbols(prefix=prefix))
    cacheable_list = []
    # Create strings that lambda strings from the expressions. 
    code['params_change'] = []
    code['cache'] = []
    for expr in subexpressions:
        arg_list = [e for e in expr[1].atoms() if e.is_Symbol]
        cacheable_symbols = [e for e in arg_list if e in cacheable_list or e in cacheable]
        if cacheable_symbols:
            code['cacheable'].append((expr[0],expr2code(arg_list, expr[1])))
            # list which ensures dependencies are cacheable.
            cacheable_list.append(expr[0])
            code['cacheexpressions'].append(expr[0])
        else:
            code['params_change'].append((expr[0],expr2code(arg_list, expr[1])))
            code['subexpressions'].append(expr[0])
            
    for expr, keys in zip(expression_substituted_list, key_list):
        arg_list = [e for e in expr.atoms() if e.is_Symbol]
        setInDict(code['main'], keys, expr2code(arg_list, expr))
        setInDict(expressions, keys, expr)

    sub_dict = {}
    for i, sub in enumerate(code['cacheexpressions']):
        sub_dict[sub.name] = cache_prefix + str(i)
    for i, sub in enumerate(code['subexpressions']):
        sub_dict[sub.name] = sub_prefix + str(i)

    
    
    return code

def expr2code(arg_list, expr):
    """Convert the given symbolic expression into code."""
    code = lambdastr(arg_list, expr)
    function_code = code.split(':')[1]
    for arg in arg_list:
        function_code = function_code.replace(arg.name, 'self.'+arg.name)
        
    return function_code

class logistic(Function):
    """The logistic function as a symbolic function."""
    nargs = 1
    def fdiff(self, argindex=1):
        x = self.args[0]
        return logistic(x)*(1-logistic(x))

    @classmethod
    def eval(cls, x):
        if x.is_Number:
            return 1/(1+exp(-x))
    
class logisticln(Function):
    """The log logistic, which can often be computed with more precision than the simply taking log(logistic(x)) when x is small or large."""
    nargs = 1

    def fdiff(self, argindex=1):
        x = self.args[0]
        return 1-logistic(x)

    @classmethod
    def eval(cls, x):
        if x.is_Number:
            return -np.log(1+exp(-x))

class erfc(Function):
    """The complementary error function, erfc(x) = 1-erf(x). Used as a helper function, particularly for erfcx, the scaled complementary error function. and the normal distributions cdf."""
    nargs = 1
    
    @classmethod
    def eval(cls, arg):
        return 1-erf(arg)

class erfcx(Function):
    nargs = 1
    def fdiff(self, argindex=1):
        x = self.args[0]
        return x*erfcx(x)-2/sqrt(pi)
        
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            return exp(x**2)*erfc(x)

class gammaln(Function):
    """The log of the gamma function, which is often needed instead of log(gamma(x)) for better accuracy for large x."""
    nargs = 1

    def fdiff(self, argindex=1):
        x=self.args[0]
        return polygamma(0, x)

    @classmethod
    def eval(cls, x):
        if x.is_Number:
            return log(gamma(x))
    

class normcdfln(Function):
    """The log of the normal cdf. Can often be computed with better accuracy than log(normcdf(x)), particulary when x is either small or large."""
    nargs = 1

    def fdiff(self, argindex=1):
        x = self.args[0]
        #return -erfcx(-x/sqrt(2))/sqrt(2*pi)
        #return exp(-normcdfln(x) - 0.5*x*x)/sqrt(2*pi)
        return sqrt(2/pi)*1/erfcx(-x/sqrt(2))

    @classmethod
    def eval(cls, x):
        if x.is_Number:
            return log(normcdf(x))

    def _eval_is_real(self):
        return self.args[0].is_real

class normcdf(Function):
    """The cumulative distribution function of the standard normal. Provided as a convenient helper function. It is computed throught -0.5*erfc(-x/sqrt(2))."""
    nargs = 1
    def fdiff(self, argindex=1):
        x = self.args[0]
        return gaussian(x)

    @classmethod
    def eval(cls, x):
        if x.is_Number:
            return 0.5*(erfc(-x/sqrt(2)))

    def _eval_is_real(self):
        return self.args[0].is_real

class normalln(Function):
    """The log of the standard normal distribution."""
    nargs = 1
    def fdiff(self, argindex=1):
         x = self.args[0]
         return -x
     
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            return 0.5*sqrt(2*pi) - 0.5*x*x


    def _eval_is_real(self):
        return True


class normal(Function):
    """The standard normal distribution. Provided as a convenience function."""
    nargs = 1
    @classmethod
    def eval(cls, x):
        return 1/sqrt(2*pi)*exp(-0.5*x*x)

    def _eval_is_real(self):
        return True

class differfln(Function):
    nargs = 2

    def fdiff(self, argindex=2):
        if argindex == 2:
            x0, x1 = self.args
            return -2/(sqrt(pi)*(erfcx(x1)-exp(x1**2-x0**2)*erfcx(x0)))
        elif argindex == 1:
            x0, x1 = self.args
            return 2/(sqrt(pi)*(exp(x0**2-x1**2)*erfcx(x1)-erfcx(x0)))
        else:
            raise ArgumentIndexError(self, argindex)
        
    @classmethod
    def eval(cls, x0, x1):
        if x0.is_Number and x1.is_Number:            
            return log(erfc(x1)-erfc(x0))

class dh_dd_i(Function):
    nargs = 5
    @classmethod
    def eval(cls, t, tprime, d_i, d_j, l):
        if (t.is_Number
            and tprime.is_Number
            and d_i.is_Number
            and d_j.is_Number
            and l.is_Number):

            diff_t = (t-tprime)
            l2 = l*l
            h = h(t, tprime, d_i, d_j, l)
            half_l_di = 0.5*l*d_i
            arg_1 = half_l_di + tprime/l
            arg_2 = half_l_di - (t-tprime)/l
            ln_part_1 = ln_diff_erf(arg_1, arg_2)
            arg_1 = half_l_di 
            arg_2 = half_l_di - t/l
            sign_val = sign(t/l)
            ln_part_2 = ln_diff_erf(half_l_di, half_l_di - t/l)

            base = ((0.5*d_i*l2*(d_i+d_j)-1)*h 
                    + (-diff_t*sign_val*exp(half_l_di*half_l_di
                                          -d_i*diff_t
                                          +ln_part_1)
                       +t*sign_val*exp(half_l_di*half_l_di
                                          -d_i*t-d_j*tprime
                                          +ln_part_2))
                    + l/sqrt(pi)*(-exp(-diff_t*diff_t/l2)
                                     +exp(-tprime*tprime/l2-d_i*t)
                                     +exp(-t*t/l2-d_j*tprime)
                                     -exp(-(d_i*t + d_j*tprime))))
            return base/(d_i+d_j)

class dh_dd_j(Function):
    nargs = 5
    @classmethod
    def eval(cls, t, tprime, d_i, d_j, l):
        if (t.is_Number
            and tprime.is_Number
            and d_i.is_Number
            and d_j.is_Number
            and l.is_Number):
            diff_t = (t-tprime)
            l2 = l*l
            half_l_di = 0.5*l*d_i
            h = h(t, tprime, d_i, d_j, l)
            arg_1 = half_l_di + tprime/l
            arg_2 = half_l_di - (t-tprime)/l
            ln_part_1 = ln_diff_erf(arg_1, arg_2)
            arg_1 = half_l_di 
            arg_2 = half_l_di - t/l
            sign_val = sign(t/l)
            ln_part_2 = ln_diff_erf(half_l_di, half_l_di - t/l)
            sign_val = sign(t/l)
            base = tprime*sign_val*exp(half_l_di*half_l_di-(d_i*t+d_j*tprime)+ln_part_2)-h
            return base/(d_i+d_j)
    
class dh_dl(Function):
    nargs = 5
    @classmethod
    def eval(cls, t, tprime, d_i, d_j, l):
        if (t.is_Number
            and tprime.is_Number
            and d_i.is_Number
            and d_j.is_Number
            and l.is_Number):

            diff_t = (t-tprime)
            l2 = l*l
            h = h(t, tprime, d_i, d_j, l)
            return 0.5*d_i*d_i*l*h + 2./(sqrt(pi)*(d_i+d_j))*((-diff_t/l2-d_i/2.)*exp(-diff_t*diff_t/l2)+(-tprime/l2+d_i/2.)*exp(-tprime*tprime/l2-d_i*t)-(-t/l2-d_i/2.)*exp(-t*t/l2-d_j*tprime)-d_i/2.*exp(-(d_i*t+d_j*tprime)))

class dh_dt(Function):
    nargs = 5
    @classmethod
    def eval(cls, t, tprime, d_i, d_j, l):
        if (t.is_Number
            and tprime.is_Number
            and d_i.is_Number
            and d_j.is_Number
            and l.is_Number):
            if (t is S.NaN
                or tprime is S.NaN
                or d_i is S.NaN
                or d_j is S.NaN
                or l is S.NaN):
                return S.NaN
            else:
                half_l_di = 0.5*l*d_i
                arg_1 = half_l_di + tprime/l
                arg_2 = half_l_di - (t-tprime)/l
                ln_part_1 = ln_diff_erf(arg_1, arg_2)
                arg_1 = half_l_di 
                arg_2 = half_l_di - t/l
                sign_val = sign(t/l)
                ln_part_2 = ln_diff_erf(half_l_di, half_l_di - t/l)

                
                return (sign_val*exp(half_l_di*half_l_di
                                        - d_i*(t-tprime)
                                        + ln_part_1
                                        - log(d_i + d_j))
                        - sign_val*exp(half_l_di*half_l_di
                                          - d_i*t - d_j*tprime
                                          + ln_part_2
                                          - log(d_i + d_j))).diff(t)

class dh_dtprime(Function):
    nargs = 5
    @classmethod
    def eval(cls, t, tprime, d_i, d_j, l):
        if (t.is_Number
            and tprime.is_Number
            and d_i.is_Number
            and d_j.is_Number
            and l.is_Number):
            if (t is S.NaN
                or tprime is S.NaN
                or d_i is S.NaN
                or d_j is S.NaN
                or l is S.NaN):
                return S.NaN
            else:
                half_l_di = 0.5*l*d_i
                arg_1 = half_l_di + tprime/l
                arg_2 = half_l_di - (t-tprime)/l
                ln_part_1 = ln_diff_erf(arg_1, arg_2)
                arg_1 = half_l_di 
                arg_2 = half_l_di - t/l
                sign_val = sign(t/l)
                ln_part_2 = ln_diff_erf(half_l_di, half_l_di - t/l)

                
                return (sign_val*exp(half_l_di*half_l_di
                                        - d_i*(t-tprime)
                                        + ln_part_1
                                        - log(d_i + d_j))
                        - sign_val*exp(half_l_di*half_l_di
                                          - d_i*t - d_j*tprime
                                          + ln_part_2
                                          - log(d_i + d_j))).diff(tprime)


class h(Function):
    nargs = 5
    def fdiff(self, argindex=5):
        t, tprime, d_i, d_j, l = self.args
        if argindex == 1:
            return dh_dt(t, tprime, d_i, d_j, l)
        elif argindex == 2:
            return dh_dtprime(t, tprime, d_i, d_j, l)
        elif argindex == 3:
            return dh_dd_i(t, tprime, d_i, d_j, l)
        elif argindex == 4:
            return dh_dd_j(t, tprime, d_i, d_j, l)
        elif argindex == 5:
            return dh_dl(t, tprime, d_i, d_j, l)
                                                                
    
    @classmethod
    def eval(cls, t, tprime, d_i, d_j, l):
        if (t.is_Number
            and tprime.is_Number
            and d_i.is_Number
            and d_j.is_Number
            and l.is_Number):
            if (t is S.NaN
                or tprime is S.NaN
                or d_i is S.NaN
                or d_j is S.NaN
                or l is S.NaN):
                return S.NaN
            else:
                half_l_di = 0.5*l*d_i
                arg_1 = half_l_di + tprime/l
                arg_2 = half_l_di - (t-tprime)/l
                ln_part_1 = ln_diff_erf(arg_1, arg_2)
                arg_1 = half_l_di 
                arg_2 = half_l_di - t/l
                sign_val = sign(t/l)
                ln_part_2 = ln_diff_erf(half_l_di, half_l_di - t/l)

                
                return (sign_val*exp(half_l_di*half_l_di
                                        - d_i*(t-tprime)
                                        + ln_part_1
                                        - log(d_i + d_j))
                        - sign_val*exp(half_l_di*half_l_di
                                          - d_i*t - d_j*tprime
                                          + ln_part_2
                                          - log(d_i + d_j)))
            
                                  
                # return (exp((d_j/2.*l)**2)/(d_i+d_j)
                #         *(exp(-d_j*(tprime - t))
                #           *(erf((tprime-t)/l - d_j/2.*l)
                #             + erf(t/l + d_j/2.*l))
                #           - exp(-(d_j*tprime + d_i))
                #           *(erf(tprime/l - d_j/2.*l)
                #             + erf(d_j/2.*l))))



