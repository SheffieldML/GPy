from sympy import Function, S, oo, I, cos, sin


class sinc_grad(Function):
    nargs = 1
    
    def fdiff(self, argindex=1):
        return ((2-x*x)*sin(self.args[0]) - 2*x*cos(x))/(x*x*x)
    
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            if x is S.Zero:
                return S.Zero
            else:
                return (x*cos(x) - sin(x))/(x*x)
            
class sinc(Function):
    
    nargs = 1
    
    def fdiff(self, argindex=1):
        return sinc_grad(self.args[0])
    
    @classmethod
    def eval(cls, x):
        if x.is_Number:
            if x is S.Zero:
                return S.One
            else:
                return sin(x)/x
    
    def _eval_is_real(self):
        return self.args[0].is_real
