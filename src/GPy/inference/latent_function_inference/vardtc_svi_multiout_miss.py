# Copyright (c) 2017, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPy.util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri,pdinv, dpotri
from GPy.util import diag
from GPy.core.parameterization.variational import VariationalPosterior
import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior
from .vardtc_svi_multiout import PosteriorMultioutput
log_2_pi = np.log(2*np.pi)


class VarDTC_SVI_Multiout_Miss(LatentFunctionInference):
    """
    The VarDTC inference method for Multi-output GP regression with missing data (GPy.models.GPMultioutRegressionMD)
    """
    const_jitter = 1e-6

    def get_trYYT(self, Y):
        return np.sum(np.square(Y))

    def get_YYTfactor(self, Y):
        N, D = Y.shape
        if (N>=D):
            return Y.view(np.ndarray)
        else:
            return jitchol(tdot(Y))

    def gatherPsiStat(self, kern, X, Z, uncertain_inputs):

        if uncertain_inputs:
            psi0 = kern.psi0(Z, X)
            psi1 = kern.psi1(Z, X)
            psi2 = kern.psi2n(Z, X)
        else:
            psi0 = kern.Kdiag(X)
            psi1 = kern.K(X, Z)
            psi2 = psi1[:,:,None]*psi1[:,None,:]

        return psi0, psi1, psi2

    def _init_grad_dict(self, N, D, Mr, Mc):
        grad_dict = {
            'dL_dthetaL': np.zeros(D),
            'dL_dqU_mean': np.zeros((Mc,Mr)),
            'dL_dqU_var_c':np.zeros((Mc,Mc)),
            'dL_dqU_var_r':np.zeros((Mr,Mr)),
            'dL_dKuu_c': np.zeros((Mc,Mc)),
            'dL_dKuu_r': np.zeros((Mr,Mr)),
            'dL_dpsi0_c': np.zeros(N),
            'dL_dpsi1_c': np.zeros((N,Mc)),
            'dL_dpsi2_c': np.zeros((N,Mc,Mc)),
            'dL_dpsi0_r': np.zeros(D),
            'dL_dpsi1_r': np.zeros((D,Mr)),
            'dL_dpsi2_r': np.zeros((D,Mr,Mr)),
        }
        return grad_dict

    def inference_d(self, d, beta, Y, indexD, grad_dict, mid_res, uncertain_inputs_r, uncertain_inputs_c, Mr, Mc):

        idx_d = indexD==d
        Y = Y[idx_d]
        N, D = Y.shape[0], 1
        beta = beta[d]

        psi0_r, psi1_r, psi2_r = mid_res['psi0_r'], mid_res['psi1_r'], mid_res['psi2_r']
        psi0_c, psi1_c, psi2_c = mid_res['psi0_c'], mid_res['psi1_c'], mid_res['psi2_c']
        psi0_r, psi1_r, psi2_r = psi0_r[d], psi1_r[d:d+1], psi2_r[d]
        psi0_c, psi1_c, psi2_c = psi0_c[idx_d].sum(), psi1_c[idx_d], psi2_c[idx_d].sum(0)

        Lr = mid_res['Lr']
        Lc = mid_res['Lc']
        LcInvMLrInvT = mid_res['LcInvMLrInvT']
        LcInvScLcInvT = mid_res['LcInvScLcInvT']
        LrInvSrLrInvT = mid_res['LrInvSrLrInvT']


        LcInvPsi2_cLcInvT = backsub_both_sides(Lc, psi2_c,'right')
        LrInvPsi2_rLrInvT = backsub_both_sides(Lr, psi2_r,'right')
        LcInvPsi1_cT = dtrtrs(Lc, psi1_c.T)[0]
        LrInvPsi1_rT = dtrtrs(Lr, psi1_r.T)[0]

        tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT = (LrInvPsi2_rLrInvT*LrInvSrLrInvT).sum()
        tr_LcInvPsi2_cLcInvT_LcInvScLcInvT = (LcInvPsi2_cLcInvT*LcInvScLcInvT).sum()
        tr_LrInvPsi2_rLrInvT = np.trace(LrInvPsi2_rLrInvT)
        tr_LcInvPsi2_cLcInvT = np.trace(LcInvPsi2_cLcInvT)

        logL_A = - np.square(Y).sum() \
               - (LcInvMLrInvT.T.dot(LcInvPsi2_cLcInvT).dot(LcInvMLrInvT)*LrInvPsi2_rLrInvT).sum() \
               -  tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT* tr_LcInvPsi2_cLcInvT_LcInvScLcInvT \
               + 2 * (Y * LcInvPsi1_cT.T.dot(LcInvMLrInvT).dot(LrInvPsi1_rT)).sum() - psi0_c * psi0_r \
               + tr_LrInvPsi2_rLrInvT * tr_LcInvPsi2_cLcInvT

        logL = -N*D/2.*(np.log(2.*np.pi)-np.log(beta)) + beta/2.* logL_A

        # ======= Gradients =====

        tmp =  beta* LcInvPsi2_cLcInvT.dot(LcInvMLrInvT).dot(LrInvPsi2_rLrInvT).dot(LcInvMLrInvT.T) \
             + beta* tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT * LcInvPsi2_cLcInvT.dot(LcInvScLcInvT) \
             - beta* LcInvMLrInvT.dot(LrInvPsi1_rT).dot(Y.T).dot(LcInvPsi1_cT.T) \
             - beta/2. * tr_LrInvPsi2_rLrInvT* LcInvPsi2_cLcInvT

        dL_dKuu_c = backsub_both_sides(Lc, tmp, 'left')
        dL_dKuu_c += dL_dKuu_c.T
        dL_dKuu_c *= 0.5

        tmp =  beta* LcInvMLrInvT.T.dot(LcInvPsi2_cLcInvT).dot(LcInvMLrInvT).dot(LrInvPsi2_rLrInvT) \
             + beta* tr_LcInvPsi2_cLcInvT_LcInvScLcInvT * LrInvPsi2_rLrInvT.dot(LrInvSrLrInvT) \
             - beta* LrInvPsi1_rT.dot(Y.T).dot(LcInvPsi1_cT.T).dot(LcInvMLrInvT) \
             - beta/2. * tr_LcInvPsi2_cLcInvT * LrInvPsi2_rLrInvT

        dL_dKuu_r = backsub_both_sides(Lr, tmp, 'left')
        dL_dKuu_r += dL_dKuu_r.T
        dL_dKuu_r *= 0.5

        #======================================================================
        # Compute dL_dthetaL
        #======================================================================

        dL_dthetaL = -D*N*beta/2. - logL_A*beta*beta/2.

        #======================================================================
        # Compute dL_dqU
        #======================================================================

        tmp = -beta * LcInvPsi2_cLcInvT.dot(LcInvMLrInvT).dot(LrInvPsi2_rLrInvT)\
              + beta* LcInvPsi1_cT.dot(Y).dot(LrInvPsi1_rT.T)

        dL_dqU_mean = dtrtrs(Lc, dtrtrs(Lr, tmp.T, trans=1)[0].T, trans=1)[0]

        tmp = -beta/2.*tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT * LcInvPsi2_cLcInvT
        dL_dqU_var_c = backsub_both_sides(Lc, tmp, 'left')

        tmp = -beta/2.*tr_LcInvPsi2_cLcInvT_LcInvScLcInvT * LrInvPsi2_rLrInvT
        dL_dqU_var_r = backsub_both_sides(Lr, tmp, 'left')

        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        dL_dpsi0_r = - psi0_c * beta/2. * np.ones((D,))
        dL_dpsi0_c = - psi0_r * beta/2. * np.ones((N,))

        dL_dpsi1_c = beta * dtrtrs(Lc, (Y.dot(LrInvPsi1_rT.T).dot(LcInvMLrInvT.T)).T, trans=1)[0].T
        dL_dpsi1_r = beta * dtrtrs(Lr, (Y.T.dot(LcInvPsi1_cT.T).dot(LcInvMLrInvT)).T, trans=1)[0].T

        tmp = beta/2.*(-LcInvMLrInvT.dot(LrInvPsi2_rLrInvT).dot(LcInvMLrInvT.T) - tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT * LcInvScLcInvT
              +tr_LrInvPsi2_rLrInvT *np.eye(Mc))
        dL_dpsi2_c = backsub_both_sides(Lc, tmp, 'left')
        tmp = beta/2.*(-LcInvMLrInvT.T.dot(LcInvPsi2_cLcInvT).dot(LcInvMLrInvT) - tr_LcInvPsi2_cLcInvT_LcInvScLcInvT * LrInvSrLrInvT
              +tr_LcInvPsi2_cLcInvT *np.eye(Mr))
        dL_dpsi2_r = backsub_both_sides(Lr, tmp, 'left')

        grad_dict['dL_dthetaL'][d:d+1] = dL_dthetaL
        grad_dict['dL_dqU_mean'] += dL_dqU_mean
        grad_dict['dL_dqU_var_c'] += dL_dqU_var_c
        grad_dict['dL_dqU_var_r'] += dL_dqU_var_r
        grad_dict['dL_dKuu_c'] += dL_dKuu_c
        grad_dict['dL_dKuu_r'] += dL_dKuu_r

        # if not uncertain_inputs_r:
        #     dL_dpsi1_r += (dL_dpsi2_r * psi1_r[:,:,None]).sum(1) + (dL_dpsi2_r * psi1_r[:,None,:]).sum(2)
        # if not uncertain_inputs_c:
        #     dL_dpsi1_c += (dL_dpsi2_c * psi1_c[:,:,None]).sum(1) + (dL_dpsi2_c * psi1_c[:,None,:]).sum(2)

        if not uncertain_inputs_r:
            dL_dpsi1_r += psi1_r.dot(dL_dpsi2_r+dL_dpsi2_r.T)
        if not uncertain_inputs_c:
            dL_dpsi1_c += psi1_c.dot(dL_dpsi2_c+dL_dpsi2_c.T)

        grad_dict['dL_dpsi0_c'][idx_d] += dL_dpsi0_c
        grad_dict['dL_dpsi1_c'][idx_d] += dL_dpsi1_c
        grad_dict['dL_dpsi2_c'][idx_d] += dL_dpsi2_c

        grad_dict['dL_dpsi0_r'][d:d+1] += dL_dpsi0_r
        grad_dict['dL_dpsi1_r'][d:d+1] += dL_dpsi1_r
        grad_dict['dL_dpsi2_r'][d] += dL_dpsi2_r


        return logL


    def inference(self, kern_r, kern_c, Xr, Xc, Zr, Zc, likelihood, Y, qU_mean ,qU_var_r, qU_var_c, indexD, output_dim):
        """
        The SVI-VarDTC inference
        """

        N, D, Mr, Mc, Qr, Qc = Y.shape[0], output_dim,Zr.shape[0], Zc.shape[0], Zr.shape[1], Zc.shape[1]

        uncertain_inputs_r = isinstance(Xr, VariationalPosterior)
        uncertain_inputs_c = isinstance(Xc, VariationalPosterior)
        uncertain_outputs = isinstance(Y, VariationalPosterior)

        grad_dict = self._init_grad_dict(N,D,Mr,Mc)

        beta = 1./likelihood.variance
        if len(beta)==1:
            beta = np.zeros(D)+beta

        psi0_r, psi1_r, psi2_r = self.gatherPsiStat(kern_r, Xr, Zr, uncertain_inputs_r)
        psi0_c, psi1_c, psi2_c = self.gatherPsiStat(kern_c, Xc, Zc, uncertain_inputs_c)

        #======================================================================
        # Compute Common Components
        #======================================================================

        Kuu_r = kern_r.K(Zr).copy()
        diag.add(Kuu_r, self.const_jitter)
        Lr = jitchol(Kuu_r)

        Kuu_c = kern_c.K(Zc).copy()
        diag.add(Kuu_c, self.const_jitter)
        Lc = jitchol(Kuu_c)

        mu, Sr, Sc = qU_mean, qU_var_r, qU_var_c
        LSr = jitchol(Sr)
        LSc = jitchol(Sc)

        LcInvMLrInvT = dtrtrs(Lc,dtrtrs(Lr,mu.T)[0].T)[0]
        LcInvLSc = dtrtrs(Lc, LSc)[0]
        LrInvLSr = dtrtrs(Lr, LSr)[0]
        LcInvScLcInvT = tdot(LcInvLSc)
        LrInvSrLrInvT = tdot(LrInvLSr)
        tr_LrInvSrLrInvT = np.square(LrInvLSr).sum()
        tr_LcInvScLcInvT = np.square(LcInvLSc).sum()

        mid_res = {
            'psi0_r': psi0_r,
            'psi1_r': psi1_r,
            'psi2_r': psi2_r,
            'psi0_c': psi0_c,
            'psi1_c': psi1_c,
            'psi2_c': psi2_c,
            'Lr':Lr,
            'Lc':Lc,
            'LcInvMLrInvT': LcInvMLrInvT,
            'LcInvScLcInvT': LcInvScLcInvT,
            'LrInvSrLrInvT': LrInvSrLrInvT,
        }

        #======================================================================
        # Compute log-likelihood
        #======================================================================

        logL = 0.
        for d in range(D):
            logL += self.inference_d(d, beta, Y, indexD, grad_dict, mid_res, uncertain_inputs_r, uncertain_inputs_c, Mr, Mc)

        logL += -Mc * (np.log(np.diag(Lr)).sum()-np.log(np.diag(LSr)).sum())  -Mr * (np.log(np.diag(Lc)).sum()-np.log(np.diag(LSc)).sum()) \
               - np.square(LcInvMLrInvT).sum()/2. - tr_LrInvSrLrInvT * tr_LcInvScLcInvT/2. + Mr*Mc/2.

        #======================================================================
        # Compute dL_dKuu
        #======================================================================

        tmp =  tdot(LcInvMLrInvT)/2. + tr_LrInvSrLrInvT/2. * LcInvScLcInvT - Mr/2.*np.eye(Mc)

        dL_dKuu_c = backsub_both_sides(Lc, tmp, 'left')
        dL_dKuu_c += dL_dKuu_c.T
        dL_dKuu_c *= 0.5

        tmp =  tdot(LcInvMLrInvT.T)/2. + tr_LcInvScLcInvT/2. * LrInvSrLrInvT - Mc/2.*np.eye(Mr)

        dL_dKuu_r = backsub_both_sides(Lr, tmp, 'left')
        dL_dKuu_r += dL_dKuu_r.T
        dL_dKuu_r *= 0.5

        #======================================================================
        # Compute dL_dqU
        #======================================================================

        tmp = - LcInvMLrInvT
        dL_dqU_mean = dtrtrs(Lc, dtrtrs(Lr, tmp.T, trans=1)[0].T, trans=1)[0]

        LScInv = dtrtri(LSc)
        tmp = -tr_LrInvSrLrInvT/2.*np.eye(Mc)
        dL_dqU_var_c = backsub_both_sides(Lc, tmp, 'left') + tdot(LScInv.T) * Mr/2.

        LSrInv = dtrtri(LSr)
        tmp =  -tr_LcInvScLcInvT/2.*np.eye(Mr)
        dL_dqU_var_r = backsub_both_sides(Lr, tmp, 'left') + tdot(LSrInv.T) * Mc/2.

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================

        post = PosteriorMultioutput(LcInvMLrInvT=LcInvMLrInvT, LcInvScLcInvT=LcInvScLcInvT,
                LrInvSrLrInvT=LrInvSrLrInvT, Lr=Lr, Lc=Lc, kern_r=kern_r, Xr=Xr, Zr=Zr)

        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        grad_dict['dL_dqU_mean'] += dL_dqU_mean
        grad_dict['dL_dqU_var_c'] += dL_dqU_var_c
        grad_dict['dL_dqU_var_r'] += dL_dqU_var_r
        grad_dict['dL_dKuu_c'] += dL_dKuu_c
        grad_dict['dL_dKuu_r'] += dL_dKuu_r

        if not uncertain_inputs_c:
            grad_dict['dL_dKdiag_c'] = grad_dict['dL_dpsi0_c']
            grad_dict['dL_dKfu_c'] = grad_dict['dL_dpsi1_c']

        if not uncertain_inputs_r:
            grad_dict['dL_dKdiag_r'] = grad_dict['dL_dpsi0_r']
            grad_dict['dL_dKfu_r'] = grad_dict['dL_dpsi1_r']

        return post, logL, grad_dict
