#Copyright I. Nabney, N.Lawrence and James Hensman (1996 - 2012)

#Scaled Conjuagte Gradients, originally in Matlab as part of the Netlab toolbox by I. Nabney, converted to python N. Lawrence and given a pythonic interface by James Hensman

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


import numpy as np
import sys

def SCG(f, gradf, x, optargs=(), maxiters=500, max_f_eval=500, display=True, xtol=1e-6, ftol=1e-6):
    """
    Optimisation through Scaled Conjugate Gradients (SCG)

    f: the objective function
    gradf : the gradient function (should return a 1D np.ndarray)
    x : the initial condition

    Returns
    x the optimal value for x
    flog : a list of all the objective values

    """

    sigma0 = 1.0e-4
    fold = f(x, *optargs)	# Initial function value.
    function_eval = 1
    fnow = fold
    gradnew = gradf(x, *optargs)	# Initial gradient.
    gradold = gradnew.copy()
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0				# Initial scale parameter.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e100			# Upper bound on scale.
    status = "Not converged"

    flog = [fold]

    iteration = 0

    # Main optimization loop.
    while iteration < maxiters:

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            sigma = sigma0/np.sqrt(kappa)
            xplus = x + sigma*d
            gplus = gradf(xplus, *optargs)
            theta = np.dot(d, (gplus - gradnew))/sigma

        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta*kappa
        if delta <= 0:
            delta = beta*kappa
            beta = beta - theta/kappa

        alpha = - mu/delta

        # Calculate the comparison ratio.
        xnew = x + alpha*d
        fnew = f(xnew, *optargs)
        function_eval += 1

        if function_eval >= max_f_eval:
            status = "Maximum number of function evaluations exceeded"
            return x, flog, function_eval, status

        Delta = 2.*(fnew - fold)/(alpha*mu)
        if Delta  >= 0.:
            success = True
            nsuccess += 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold

        # Store relevant variables
        flog.append(fnow)		# Current function value

        iteration += 1
        if display:
            print '\r',
            print 'Iteration: {0:>5g}  Objective:{1:> 12e}  Scale:{2:> 12e}'.format(iteration, fnow, beta),
            # print 'Iteration:', iteration, ' Objective:', fnow, '  Scale:', beta, '\r',
            sys.stdout.flush()

        if success:
            # Test for termination
            if (np.max(np.abs(alpha*d)) < xtol) or (np.abs(fnew-fold) < ftol):
                status='converged'
                return x, flog, function_eval, status

            else:
                # Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *optargs)
                # If the gradient is zero then we are done.
                if np.dot(gradnew,gradnew) == 0:
                    return x, flog, function_eval, status

        # Adjust beta according to comparison ratio.
        if Delta < 0.25:
            beta = min(4.0*beta, betamax)
        if Delta > 0.75:
            beta = max(0.5*beta, betamin)

        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if nsuccess == x.size:
            d = -gradnew
            nsuccess = 0
        elif success:
            gamma = np.dot(gradold - gradnew,gradnew)/(mu)
            d = gamma*d - gradnew

    # If we get here, then we haven't terminated in the given number of
    # iterations.
    status = "maxiter exceeded"

    return x, flog, function_eval, status
