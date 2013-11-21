from sympy import Function, S, oo, I, cos, sin, asin, log, erf, pi, exp, sqrt, sign


class ln_diff_erf(Function):
    nargs = 2

    def fdiff(self, argindex=2):
        if argindex == 2:
            x0, x1 = self.args
            return -2*exp(-x1**2)/(sqrt(pi)*(erf(x0)-erf(x1)))
        elif argindex == 1:
            x0, x1 = self.args
            return 2.*exp(-x0**2)/(sqrt(pi)*(erf(x0)-erf(x1)))
        else:
            raise ArgumentIndexError(self, argindex)
        
    @classmethod
    def eval(cls, x0, x1):
        if x0.is_Number and x1.is_Number:            
            return log(erf(x0)-erf(x1))

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
        # putting in the is_Number stuff forces it to look for a fdiff method for derivative. If it's left out, then when asking for self.diff, it just does the diff on the eval symbolic terms directly. We want to avoid that because we are looking to ensure everything is numerically stable. Maybe it's because of the if statement that this happens? 
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

