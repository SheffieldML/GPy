# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
from ...util.linalg import mdot, jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri, dpotri, dpotrs, symmetrify
from ...util import diag
from ...core.parameterization.variational import VariationalPosterior
import numpy as np
from ...util.misc import param_to_array
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)
import logging, itertools
logger = logging.getLogger('vardtc')

class VarDTC(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    const_jitter = 1e-6
    def __init__(self, limit=1):
        #self._YYTfactor_cache = caching.cache()
        from ...util.caching import Cacher
        self.limit = limit
        self.get_trYYT = Cacher(self._get_trYYT, limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, limit)

    def set_limit(self, limit):
        self.get_trYYT.limit = limit
        self.get_YYTfactor.limit = limit

    def _get_trYYT(self, Y):
        return param_to_array(np.sum(np.square(Y)))

    def __getstate__(self):
        # has to be overridden, as Cacher objects cannot be pickled.
        return self.limit

    def __setstate__(self, state):
        # has to be overridden, as Cacher objects cannot be pickled.
        self.limit = state
        from ...util.caching import Cacher
        self.get_trYYT = Cacher(self._get_trYYT, self.limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, self.limit)

    def _get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LLT = YYT.

        Note that L may have fewer columns than Y.
        """
        N, D = Y.shape
        if (N>=D):
            return param_to_array(Y)
        else:
            return jitchol(tdot(Y))

    def get_VVTfactor(self, Y, prec):
        return Y * prec # TODO chache this, and make it effective

    def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None):
        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
            psi0 = kern.psi0(Z, X)
            psi1 = kern.psi1(Z, X)
            psi2 = kern.psi2(Z, X)
        else:
            uncertain_inputs = False
            psi0 = kern.Kdiag(X)
            psi1 = kern.K(X, Z)
            psi2 = None

        #see whether we're using variational uncertain inputs

        _, output_dim = Y.shape

        #see whether we've got a different noise variance for each datum
        beta = 1./np.fmax(likelihood.gaussian_variance(Y_metadata), 1e-6)
        # VVT_factor is a matrix such that tdot(VVT_factor) = VVT...this is for efficiency!
        #self.YYTfactor = self.get_YYTfactor(Y)
        #VVT_factor = self.get_VVTfactor(self.YYTfactor, beta)
        VVT_factor = beta*Y
        #VVT_factor = beta*Y
        trYYT = self.get_trYYT(Y)

        # do the inference:
        het_noise = beta.size > 1
        num_inducing = Z.shape[0]
        num_data = Y.shape[0]
        # kernel computations, using BGPLVM notation

        Kmm = kern.K(Z).copy()
        diag.add(Kmm, self.const_jitter)
        Lm = jitchol(Kmm)

        # The rather complex computations of A
        if uncertain_inputs:
            if het_noise:
                psi2_beta = psi2 * (beta.flatten().reshape(num_data, 1, 1)).sum(0)
            else:
                psi2_beta = psi2.sum(0) * beta
            #if 0:
            #    evals, evecs = linalg.eigh(psi2_beta)
            #    clipped_evals = np.clip(evals, 0., 1e6) # TODO: make clipping configurable
            #    if not np.array_equal(evals, clipped_evals):
            #        pass # print evals
            #    tmp = evecs * np.sqrt(clipped_evals)
            #    tmp = tmp.T
            # no backsubstitution because of bound explosion on tr(A) if not...
            LmInv = dtrtri(Lm)
            A = LmInv.dot(psi2_beta.dot(LmInv.T))
        else:
            if het_noise:
                tmp = psi1 * (np.sqrt(beta.reshape(num_data, 1)))
            else:
                tmp = psi1 * (np.sqrt(beta))
            tmp, _ = dtrtrs(Lm, tmp.T, lower=1)
            A = tdot(tmp) #print A.sum()

        # factor B
        B = np.eye(num_inducing) + A
        LB = jitchol(B)
        psi1Vf = np.dot(psi1.T, VVT_factor)
        # back substutue C into psi1Vf
        tmp, _ = dtrtrs(Lm, psi1Vf, lower=1, trans=0)
        _LBi_Lmi_psi1Vf, _ = dtrtrs(LB, tmp, lower=1, trans=0)
        tmp, _ = dtrtrs(LB, _LBi_Lmi_psi1Vf, lower=1, trans=1)
        Cpsi1Vf, _ = dtrtrs(Lm, tmp, lower=1, trans=1)

        # data fit and derivative of L w.r.t. Kmm
        delit = tdot(_LBi_Lmi_psi1Vf)
        data_fit = np.trace(delit)
        DBi_plus_BiPBi = backsub_both_sides(LB, output_dim * np.eye(num_inducing) + delit)
        delit = -0.5 * DBi_plus_BiPBi
        delit += -0.5 * B * output_dim
        delit += output_dim * np.eye(num_inducing)
        # Compute dL_dKmm
        dL_dKmm = backsub_both_sides(Lm, delit)

        # derivatives of L w.r.t. psi
        dL_dpsi0, dL_dpsi1, dL_dpsi2 = _compute_dL_dpsi(num_inducing, num_data, output_dim, beta, Lm,
            VVT_factor, Cpsi1Vf, DBi_plus_BiPBi,
            psi1, het_noise, uncertain_inputs)

        # log marginal likelihood
        log_marginal = _compute_log_marginal_likelihood(likelihood, num_data, output_dim, beta, het_noise,
            psi0, A, LB, trYYT, data_fit, VVT_factor)

        #put the gradients in the right places
        dL_dR = _compute_dL_dR(likelihood,
            het_noise, uncertain_inputs, LB,
            _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A,
            psi0, psi1, beta,
            data_fit, num_data, output_dim, trYYT, Y)

        dL_dthetaL = likelihood.exact_inference_gradients(dL_dR,Y_metadata)

        if uncertain_inputs:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dpsi0':dL_dpsi0,
                         'dL_dpsi1':dL_dpsi1,
                         'dL_dpsi2':dL_dpsi2,
                         'dL_dthetaL':dL_dthetaL}
        else:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dKdiag':dL_dpsi0,
                         'dL_dKnm':dL_dpsi1,
                         'dL_dthetaL':dL_dthetaL}

        #get sufficient things for posterior prediction
        #TODO: do we really want to do this in  the loop?
        if VVT_factor.shape[1] == Y.shape[1]:
            woodbury_vector = Cpsi1Vf # == Cpsi1V
        else:
            print 'foobar'
            psi1V = np.dot(Y.T*beta, psi1).T
            tmp, _ = dtrtrs(Lm, psi1V, lower=1, trans=0)
            tmp, _ = dpotrs(LB, tmp, lower=1)
            woodbury_vector, _ = dtrtrs(Lm, tmp, lower=1, trans=1)
        Bi, _ = dpotri(LB, lower=1)
        symmetrify(Bi)
        Bi = -dpotri(LB, lower=1)[0]
        diag.add(Bi, 1)

        woodbury_inv = backsub_both_sides(Lm, Bi)

        #construct a posterior object
        post = Posterior(woodbury_inv=woodbury_inv, woodbury_vector=woodbury_vector, K=Kmm, mean=None, cov=None, K_chol=Lm)
        return post, log_marginal, grad_dict

class VarDTCMissingData(LatentFunctionInference):
    const_jitter = 1e-10
    def __init__(self, limit=1, inan=None):
        from ...util.caching import Cacher
        self._Y = Cacher(self._subarray_computations, limit)
        if inan is not None: self._inan = ~inan
        else: self._inan = None
        pass

    def set_limit(self, limit):
        self._Y.limit = limit

    def __getstate__(self):
        # has to be overridden, as Cacher objects cannot be pickled.
        return self._Y.limit, self._inan

    def __setstate__(self, state):
        # has to be overridden, as Cacher objects cannot be pickled.
        from ...util.caching import Cacher
        self.limit = state[0]
        self._inan = state[1]
        self._Y = Cacher(self._subarray_computations, self.limit)

    def _subarray_computations(self, Y):
        if self._inan is None:
            inan = np.isnan(Y)
            has_none = inan.any()
            self._inan = ~inan
        else:
            inan = self._inan
            has_none = True
        if has_none:
            #print "caching missing data slices, this can take several minutes depending on the number of unique dimensions of the data..."
            #csa = common_subarrays(inan, 1)
            size = Y.shape[1]
            #logger.info('preparing subarrays {:3.3%}'.format((i+1.)/size))
            Ys = []
            next_ten = [0.]
            count = itertools.count()
            for v, y in itertools.izip(inan.T, Y.T[:,:,None]):
                i = count.next()
                if ((i+1.)/size) >= next_ten[0]:
                    logger.info('preparing subarrays {:>6.1%}'.format((i+1.)/size))
                    next_ten[0] += .1
                Ys.append(y[v,:])

            next_ten = [0.]
            count = itertools.count()
            def trace(y):
                i = count.next()
                if ((i+1.)/size) >= next_ten[0]:
                    logger.info('preparing traces {:>6.1%}'.format((i+1.)/size))
                    next_ten[0] += .1
                y = y[inan[:,i],i:i+1]
                return np.einsum('ij,ij->', y,y)
            traces = [trace(Y) for _ in xrange(size)]
            return Ys, traces
        else:
            self._subarray_indices = [[slice(None),slice(None)]]
            return [Y], [(Y**2).sum()]

    def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None):
        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
            psi0_all = kern.psi0(Z, X)
            psi1_all = kern.psi1(Z, X)
            psi2_all = kern.psi2(Z, X)
        else:
            uncertain_inputs = False
            psi0_all = kern.Kdiag(X)
            psi1_all = kern.K(X, Z)
            psi2_all = None

        Ys, traces = self._Y(Y)
        beta_all = 1./np.fmax(likelihood.gaussian_variance(Y_metadata), 1e-6)
        het_noise = beta_all.size != 1

        num_inducing = Z.shape[0]

        dL_dpsi0_all = np.zeros(Y.shape[0])
        dL_dpsi1_all = np.zeros((Y.shape[0], num_inducing))
        if uncertain_inputs:
            dL_dpsi2_all = np.zeros((Y.shape[0], num_inducing, num_inducing))

        dL_dR = 0
        woodbury_vector = np.zeros((num_inducing, Y.shape[1]))
        woodbury_inv_all = np.zeros((num_inducing, num_inducing, Y.shape[1]))
        dL_dKmm = 0
        log_marginal = 0

        Kmm = kern.K(Z).copy()
        diag.add(Kmm, self.const_jitter)
        #factor Kmm
        Lm = jitchol(Kmm)
        if uncertain_inputs: LmInv = dtrtri(Lm)

        size = Y.shape[1]
        next_ten = 0
        for i, [y, v, trYYT] in enumerate(itertools.izip(Ys, self._inan.T, traces)):
            if ((i+1.)/size) >= next_ten:
                logger.info('inference {:> 6.1%}'.format((i+1.)/size))
                next_ten += .1
            if het_noise: beta = beta_all[i]
            else: beta = beta_all

            VVT_factor = (y*beta)
            output_dim = 1#len(ind)

            psi0 = psi0_all[v]
            psi1 = psi1_all[v, :]
            if uncertain_inputs: psi2 = psi2_all[v, :]
            else: psi2 = None
            num_data = psi1.shape[0]

            if uncertain_inputs:
                if het_noise: psi2_beta = psi2 * (beta.flatten().reshape(num_data, 1, 1)).sum(0)
                else: psi2_beta = psi2.sum(0) * beta
                A = LmInv.dot(psi2_beta.dot(LmInv.T))
            else:
                if het_noise: tmp = psi1 * (np.sqrt(beta.reshape(num_data, 1)))
                else: tmp = psi1 * (np.sqrt(beta))
                tmp, _ = dtrtrs(Lm, tmp.T, lower=1)
                A = tdot(tmp) #print A.sum()

            # factor B
            B = np.eye(num_inducing) + A
            LB = jitchol(B)

            psi1Vf = psi1.T.dot(VVT_factor)
            tmp, _ = dtrtrs(Lm, psi1Vf, lower=1, trans=0)
            _LBi_Lmi_psi1Vf, _ = dtrtrs(LB, tmp, lower=1, trans=0)
            tmp, _ = dtrtrs(LB, _LBi_Lmi_psi1Vf, lower=1, trans=1)
            Cpsi1Vf, _ = dtrtrs(Lm, tmp, lower=1, trans=1)

            # data fit and derivative of L w.r.t. Kmm
            delit = tdot(_LBi_Lmi_psi1Vf)
            data_fit = np.trace(delit)
            DBi_plus_BiPBi = backsub_both_sides(LB, output_dim * np.eye(num_inducing) + delit)
            delit = -0.5 * DBi_plus_BiPBi
            delit += -0.5 * B * output_dim
            delit += output_dim * np.eye(num_inducing)
            dL_dKmm += backsub_both_sides(Lm, delit)

            # derivatives of L w.r.t. psi
            dL_dpsi0, dL_dpsi1, dL_dpsi2 = _compute_dL_dpsi(num_inducing, num_data, output_dim, beta, Lm,
                VVT_factor, Cpsi1Vf, DBi_plus_BiPBi,
                psi1, het_noise, uncertain_inputs)

            dL_dpsi0_all[v] += dL_dpsi0
            dL_dpsi1_all[v, :] += dL_dpsi1
            if uncertain_inputs:
                dL_dpsi2_all[v, :] += dL_dpsi2

            # log marginal likelihood
            log_marginal += _compute_log_marginal_likelihood(likelihood, num_data, output_dim, beta, het_noise,
                psi0, A, LB, trYYT, data_fit,VVT_factor)

            #put the gradients in the right places
            dL_dR += _compute_dL_dR(likelihood,
                het_noise, uncertain_inputs, LB,
                _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A,
                psi0, psi1, beta,
                data_fit, num_data, output_dim, trYYT, Y)

            #if full_VVT_factor:
            woodbury_vector[:, i:i+1] = Cpsi1Vf
            #else:
            #    print 'foobar'
            #    tmp, _ = dtrtrs(Lm, psi1V, lower=1, trans=0)
            #    tmp, _ = dpotrs(LB, tmp, lower=1)
            #    woodbury_vector[:, ind] = dtrtrs(Lm, tmp, lower=1, trans=1)[0]

            #import ipdb;ipdb.set_trace()
            Bi, _ = dpotri(LB, lower=1)
            symmetrify(Bi)
            Bi = -dpotri(LB, lower=1)[0]
            diag.add(Bi, 1)
            woodbury_inv_all[:, :, i:i+1] = backsub_both_sides(Lm, Bi)[:,:,None]

        dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)

        # gradients:
        if uncertain_inputs:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dpsi0':dL_dpsi0_all,
                         'dL_dpsi1':dL_dpsi1_all,
                         'dL_dpsi2':dL_dpsi2_all,
                         'dL_dthetaL':dL_dthetaL}
        else:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dKdiag':dL_dpsi0_all,
                         'dL_dKnm':dL_dpsi1_all,
                         'dL_dthetaL':dL_dthetaL}

        post = Posterior(woodbury_inv=woodbury_inv_all, woodbury_vector=woodbury_vector, K=Kmm, mean=None, cov=None, K_chol=Lm)

        return post, log_marginal, grad_dict

def _compute_dL_dpsi(num_inducing, num_data, output_dim, beta, Lm, VVT_factor, Cpsi1Vf, DBi_plus_BiPBi, psi1, het_noise, uncertain_inputs):
    dL_dpsi0 = -0.5 * output_dim * (beta[:,None] * np.ones([num_data, 1])).flatten()
    dL_dpsi1 = np.dot(VVT_factor, Cpsi1Vf.T)
    dL_dpsi2_beta = 0.5 * backsub_both_sides(Lm, output_dim * np.eye(num_inducing) - DBi_plus_BiPBi)
    if het_noise:
        if uncertain_inputs:
            dL_dpsi2 = beta[:, None, None] * dL_dpsi2_beta[None, :, :]
        else:
            dL_dpsi1 += 2.*np.dot(dL_dpsi2_beta, (psi1 * beta.reshape(num_data, 1)).T).T
            dL_dpsi2 = None
    else:
        dL_dpsi2 = beta * dL_dpsi2_beta
        if uncertain_inputs:
            # repeat for each of the N psi_2 matrices
            dL_dpsi2 = np.repeat(dL_dpsi2[None, :, :], num_data, axis=0)
        else:
            # subsume back into psi1 (==Kmn)
            dL_dpsi1 += 2.*np.dot(psi1, dL_dpsi2)
            dL_dpsi2 = None

    return dL_dpsi0, dL_dpsi1, dL_dpsi2


def _compute_dL_dR(likelihood, het_noise, uncertain_inputs, LB, _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A, psi0, psi1, beta, data_fit, num_data, output_dim, trYYT, Y):
    # the partial derivative vector for the likelihood
    if likelihood.size == 0:
        # save computation here.
        dL_dR = None
    elif het_noise:
        if uncertain_inputs:
            raise NotImplementedError, "heteroscedatic derivates with uncertain inputs not implemented"
        else:
            #from ...util.linalg import chol_inv
            #LBi = chol_inv(LB)
            LBi, _ = dtrtrs(LB,np.eye(LB.shape[0]))

            Lmi_psi1, nil = dtrtrs(Lm, psi1.T, lower=1, trans=0)
            _LBi_Lmi_psi1, _ = dtrtrs(LB, Lmi_psi1, lower=1, trans=0)

            dL_dR = -0.5 * beta + 0.5 * (beta*Y)**2
            dL_dR += 0.5 * output_dim * (psi0 - np.sum(Lmi_psi1**2,0))[:,None] * beta**2

            dL_dR += 0.5*np.sum(mdot(LBi.T,LBi,Lmi_psi1)*Lmi_psi1,0)[:,None]*beta**2

            dL_dR += -np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T * Y * beta**2
            dL_dR += 0.5*np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T**2 * beta**2
    else:
        # likelihood is not heteroscedatic
        dL_dR = -0.5 * num_data * output_dim * beta + 0.5 * trYYT * beta ** 2
        dL_dR += 0.5 * output_dim * (psi0.sum() * beta ** 2 - np.trace(A) * beta)
        dL_dR += beta * (0.5 * np.sum(A * DBi_plus_BiPBi) - data_fit)
    return dL_dR

def _compute_log_marginal_likelihood(likelihood, num_data, output_dim, beta, het_noise, psi0, A, LB, trYYT, data_fit,Y):
    #compute log marginal likelihood
    if het_noise:
        lik_1 = -0.5 * num_data * output_dim * np.log(2. * np.pi) + 0.5 * np.sum(np.log(beta)) - 0.5 * np.sum(beta * np.square(Y).sum(axis=-1))
        lik_2 = -0.5 * output_dim * (np.sum(beta.flatten() * psi0) - np.trace(A))
    else:
        lik_1 = -0.5 * num_data * output_dim * (np.log(2. * np.pi) - np.log(beta)) - 0.5 * beta * trYYT
        lik_2 = -0.5 * output_dim * (np.sum(beta * psi0) - np.trace(A))
    lik_3 = -output_dim * (np.sum(np.log(np.diag(LB))))
    lik_4 = 0.5 * data_fit
    log_marginal = lik_1 + lik_2 + lik_3 + lik_4
    return log_marginal
