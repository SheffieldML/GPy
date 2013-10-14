from sympy import Function, S, oo, I, cos, sin, asin, log, erf,pi,exp


class ln_diff_erf(Function):
    nargs = 2

    def fdiff(self, argindex=2):
        if argindex == 2:
            x0, x1 = self.args
            return -2*exp(-x1**2)/(sqrt(pi)*(erf(x0)-erf(x1)))
        elif argindex == 1:
            x0, x1 = self.args
            return 2*exp(-x0**2)/(sqrt(pi)*(erf(x0)-erf(x1)))
        else:
            raise ArgumentIndexError(self, argindex)
        
    @classmethod
    def eval(cls, x0, x1):
        if x0.is_Number and x1.is_Number:            
            return log(erf(x0)-erf(x1))

class sim_h(Function):
    nargs = 5

    def fdiff(self, argindex=1):
        pass
    
    @classmethod
    def eval(cls, t, tprime, d_i, d_j, l):
        # putting in the is_Number stuff forces it to look for a fdiff method for derivative.
        return (exp((d_j/2*l)**2)/(d_i+d_j)
                *(exp(-d_j*(tprime - t))
                  *(erf((tprime-t)/l - d_j/2*l)
                    + erf(t/l + d_j/2*l))
                  - exp(-(d_j*tprime + d_i))
                  *(erf(tprime/l - d_j/2*l)
                    + erf(d_j/2*l))))

class erfc(Function):
    nargs = 1
    
    @classmethod
    def eval(cls, arg):
        return 1-erf(arg)

class erfcx(Function):
    nargs = 1

    @classmethod
    def eval(cls, arg):
        return erfc(arg)*exp(arg*arg)

class sinc_grad(Function):
    nargs = 1
    
    def fdiff(self, argindex=1):
        if argindex==1:
            # Strictly speaking this should be computed separately, as it won't work when x=0. See http://calculus.subwiki.org/wiki/Sinc_function
            return ((2-x*x)*sin(self.args[0]) - 2*x*cos(x))/(x*x*x)
        else:
            raise ArgumentIndexError(self, argindex)

    
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            if x is S.NaN:
                return S.NaN
            elif x is S.Zero:
                return S.Zero
            else:
                return (x*cos(x) - sin(x))/(x*x)
            
class sinc(Function):
    
    nargs = 1
    
    def fdiff(self, argindex=1):
        if argindex==1:
            return sinc_grad(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    
    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Zero:
                return S.One
            else:
                return sin(arg)/arg

        if arg.func is asin:
            x = arg.args[0]
            return x / arg

    def _eval_is_real(self):
        return self.args[0].is_real

