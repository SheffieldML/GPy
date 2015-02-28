# Copyright I. Nabney, N.Lawrence and James Hensman (1996 - 2014)

# Scaled Conjuagte Gradients, originally in Matlab as part of the Netlab toolbox by I. Nabney, converted to python N. Lawrence and given a pythonic interface by James Hensman

#      THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
#      HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#      EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
#      NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#      MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#      PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#      REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
#      DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#      EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#      (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
#      OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#      DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#      HOWEVER CAUSED AND ON ANY THEORY OF
#      LIABILITY, WHETHER IN CONTRACT, STRICT
#      LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#      OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#      OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#      POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
import numpy as np
import sys

def print_out(len_maxiters, fnow, current_grad, beta, iteration):
    print('\r', end=' ')
    print('{0:>0{mi}g}  {1:> 12e}  {2:< 12.6e}  {3:> 12e}'.format(iteration, float(fnow), float(beta), float(current_grad), mi=len_maxiters), end=' ') # print 'Iteration:', iteration, ' Objective:', fnow, '  Scale:', beta, '\r',
    sys.stdout.flush()

def exponents(fnow, current_grad):
    exps = [np.abs(np.float(fnow)), current_grad]
    return np.sign(exps) * np.log10(exps).astype(int)

def SCG(f, gradf, x, optargs=(), maxiters=500, max_f_eval=np.inf, display=True, xtol=None, ftol=None, gtol=None):
    """
    Optimisation through Scaled Conjugate Gradients (SCG)

    f: the objective function
    gradf : the gradient function (should return a 1D np.ndarray)
    x : the initial condition

    Returns
    x the optimal value for x
    flog : a list of all the objective values
    function_eval number of fn evaluations
    status: string describing convergence status
    """
    if xtol is None:
        xtol = 1e-6
    if ftol is None:
        ftol = 1e-6
    if gtol is None:
        gtol = 1e-5

    sigma0 = 1.0e-7
    fold = f(x, *optargs) # Initial function value.
    function_eval = 1
    fnow = fold
    gradnew = gradf(x, *optargs) # Initial gradient.
    function_eval += 1
    #if any(np.isnan(gradnew)):
    #    raise UnexpectedInfOrNan, "Gradient contribution resulted in a NaN value"
    current_grad = np.dot(gradnew, gradnew)
    gradold = gradnew.copy()
    d = -gradnew # Initial search direction.
    success = True # Force calculation of directional derivs.
    nsuccess = 0 # nsuccess counts number of successes.
    beta = 1.0 # Initial scale parameter.
    betamin = 1.0e-15 # Lower bound on scale.
    betamax = 1.0e15 # Upper bound on scale.
    status = "Not converged"

    flog = [fold]

    iteration = 0

    len_maxiters = len(str(maxiters))
    if display:
        print(' {0:{mi}s}   {1:11s}    {2:11s}    {3:11s}'.format("I", "F", "Scale", "|g|", mi=len_maxiters))
        exps = exponents(fnow, current_grad)
        p_iter = iteration

    # Main optimization loop.
    while iteration < maxiters:

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            sigma = sigma0 / np.sqrt(kappa)
            xplus = x + sigma * d
            gplus = gradf(xplus, *optargs)
            function_eval += 1
            theta = np.dot(d, (gplus - gradnew)) / sigma

        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if delta <= 0:
            delta = beta * kappa
            beta = beta - theta / kappa

        alpha = -mu / delta

        # Calculate the comparison ratio.
        xnew = x + alpha * d
        fnew = f(xnew, *optargs)
        function_eval += 1

        if function_eval >= max_f_eval:
            status = "maximum number of function evaluations exceeded"
            break
            return x, flog, function_eval, status

        Delta = 2.*(fnew - fold) / (alpha * mu)
        if Delta >= 0.:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        # Store relevant variables
        flog.append(fnow) # Current function value

        iteration += 1
        if display:
            print_out(len_maxiters, fnow, current_grad, beta, iteration)
            n_exps = exponents(fnow, current_grad)
            if iteration - p_iter >= 20 * np.random.rand():
                a = iteration >= p_iter * 2.78
                b = np.any(n_exps < exps)
                if a or b:
                    p_iter = iteration
                    print('')
                if b:
                    exps = n_exps

        if success:
            # Test for termination

            if (np.abs(fnew - fold) < ftol):
                status = 'converged - relative reduction in objective'
                break
#                 return x, flog, function_eval, status
            elif (np.max(np.abs(alpha * d)) < xtol):
                status = 'converged - relative stepsize'
                break
            else:
                # Update variables for new position
                gradold = gradnew
                gradnew = gradf(x, *optargs)
                function_eval += 1
                current_grad = np.dot(gradnew, gradnew)
                fold = fnew
                # If the gradient is zero then we are done.
                if current_grad <= gtol:
                    status = 'converged - relative reduction in gradient'
                    break
                    # return x, flog, function_eval, status

        # Adjust beta according to comparison ratio.
        if Delta < 0.25:
            beta = min(4.0 * beta, betamax)
        if Delta > 0.75:
            beta = max(0.25 * beta, betamin)

        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if nsuccess == x.size:
            d = -gradnew
            beta = 1. # This is not in the original paper
            nsuccess = 0
        elif success:
            Gamma = np.dot(gradold - gradnew, gradnew) / (mu)
            d = Gamma * d - gradnew
    else:
        # If we get here, then we haven't terminated in the given number of
        # iterations.
        status = "maxiter exceeded"

    if display:
        print_out(len_maxiters, fnow, current_grad, beta, iteration)
        print("")
        print(status)
    return x, flog, function_eval, status
