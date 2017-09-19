#from .posterior import Posterior
from GPy.util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri,pdinv, dpotri
from GPy.util import diag
from GPy.core.parameterization.variational import VariationalPosterior
import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior
log_2_pi = np.log(2*np.pi)


class VarDTC_SVI_Multiout(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

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
            psi0 = kern.psi0(Z, X).sum()
            psi1 = kern.psi1(Z, X)
            psi2 = kern.psi2(Z, X)
        else:
            psi0 = kern.Kdiag(X).sum()
            psi1 = kern.K(X, Z)
            psi2 = tdot(psi1.T)

        return psi0, psi1, psi2

    def inference(self, kern_r, kern_c, Xr, Xc, Zr, Zc, likelihood, Y, qU_mean ,qU_var_r, qU_var_c):
        """
        The SVI-VarDTC inference
        """

        N, D, Mr, Mc, Qr, Qc = Y.shape[0], Y.shape[1], Zr.shape[0], Zc.shape[0], Zr.shape[1], Zc.shape[1]

        uncertain_inputs_r = isinstance(Xr, VariationalPosterior)
        uncertain_inputs_c = isinstance(Xc, VariationalPosterior)
        uncertain_outputs = isinstance(Y, VariationalPosterior)

        beta = 1./likelihood.variance

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
        LcInvPsi2_cLcInvT = backsub_both_sides(Lc, psi2_c,'right')
        LrInvPsi2_rLrInvT = backsub_both_sides(Lr, psi2_r,'right')
        LcInvLSc = dtrtrs(Lc, LSc)[0]
        LrInvLSr = dtrtrs(Lr, LSr)[0]
        LcInvScLcInvT = tdot(LcInvLSc)
        LrInvSrLrInvT = tdot(LrInvLSr)
        LcInvPsi1_cT = dtrtrs(Lc, psi1_c.T)[0]
        LrInvPsi1_rT = dtrtrs(Lr, psi1_r.T)[0]

        tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT = (LrInvPsi2_rLrInvT*LrInvSrLrInvT).sum()
        tr_LcInvPsi2_cLcInvT_LcInvScLcInvT = (LcInvPsi2_cLcInvT*LcInvScLcInvT).sum()
        tr_LrInvSrLrInvT = np.square(LrInvLSr).sum()
        tr_LcInvScLcInvT = np.square(LcInvLSc).sum()
        tr_LrInvPsi2_rLrInvT = np.trace(LrInvPsi2_rLrInvT)
        tr_LcInvPsi2_cLcInvT = np.trace(LcInvPsi2_cLcInvT)

        #======================================================================
        # Compute log-likelihood
        #======================================================================

        logL_A = - np.square(Y).sum() \
               - (LcInvMLrInvT.T.dot(LcInvPsi2_cLcInvT).dot(LcInvMLrInvT)*LrInvPsi2_rLrInvT).sum() \
               -  tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT* tr_LcInvPsi2_cLcInvT_LcInvScLcInvT \
               + 2 * (Y * LcInvPsi1_cT.T.dot(LcInvMLrInvT).dot(LrInvPsi1_rT)).sum() - psi0_c * psi0_r \
               + tr_LrInvPsi2_rLrInvT * tr_LcInvPsi2_cLcInvT 

        logL = -N*D/2.*(np.log(2.*np.pi)-np.log(beta)) + beta/2.* logL_A \
               -Mc * (np.log(np.diag(Lr)).sum()-np.log(np.diag(LSr)).sum())  -Mr * (np.log(np.diag(Lc)).sum()-np.log(np.diag(LSc)).sum()) \
               - np.square(LcInvMLrInvT).sum()/2. - tr_LrInvSrLrInvT * tr_LcInvScLcInvT/2. + Mr*Mc/2.

        #======================================================================
        # Compute dL_dKuu
        #======================================================================

        tmp =  beta* LcInvPsi2_cLcInvT.dot(LcInvMLrInvT).dot(LrInvPsi2_rLrInvT).dot(LcInvMLrInvT.T) \
             + beta* tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT * LcInvPsi2_cLcInvT.dot(LcInvScLcInvT) \
             - beta* LcInvMLrInvT.dot(LrInvPsi1_rT).dot(Y.T).dot(LcInvPsi1_cT.T) \
             - beta/2. * tr_LrInvPsi2_rLrInvT* LcInvPsi2_cLcInvT - Mr/2.*np.eye(Mc) \
             + tdot(LcInvMLrInvT)/2. + tr_LrInvSrLrInvT/2. * LcInvScLcInvT

        dL_dKuu_c = backsub_both_sides(Lc, tmp, 'left')
        dL_dKuu_c += dL_dKuu_c.T
        dL_dKuu_c *= 0.5

        tmp =  beta* LcInvMLrInvT.T.dot(LcInvPsi2_cLcInvT).dot(LcInvMLrInvT).dot(LrInvPsi2_rLrInvT) \
             + beta* tr_LcInvPsi2_cLcInvT_LcInvScLcInvT * LrInvPsi2_rLrInvT.dot(LrInvSrLrInvT) \
             - beta* LrInvPsi1_rT.dot(Y.T).dot(LcInvPsi1_cT.T).dot(LcInvMLrInvT) \
             - beta/2. * tr_LcInvPsi2_cLcInvT * LrInvPsi2_rLrInvT - Mc/2.*np.eye(Mr) \
             + tdot(LcInvMLrInvT.T)/2. + tr_LcInvScLcInvT/2. * LrInvSrLrInvT

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
              + beta* LcInvPsi1_cT.dot(Y).dot(LrInvPsi1_rT.T) - LcInvMLrInvT

        dL_dqU_mean = dtrtrs(Lc, dtrtrs(Lr, tmp.T, trans=1)[0].T, trans=1)[0]

        LScInv = dtrtri(LSc)
        tmp = -beta/2.*tr_LrInvPsi2_rLrInvT_LrInvSrLrInvT * LcInvPsi2_cLcInvT -tr_LrInvSrLrInvT/2.*np.eye(Mc)
        dL_dqU_var_c = backsub_both_sides(Lc, tmp, 'left') + tdot(LScInv.T) * Mr/2.

        LSrInv = dtrtri(LSr)
        tmp = -beta/2.*tr_LcInvPsi2_cLcInvT_LcInvScLcInvT * LrInvPsi2_rLrInvT -tr_LcInvScLcInvT/2.*np.eye(Mr)
        dL_dqU_var_r = backsub_both_sides(Lr, tmp, 'left') + tdot(LSrInv.T) * Mc/2.

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================

        post = PosteriorMultioutput(LcInvMLrInvT=LcInvMLrInvT, LcInvScLcInvT=LcInvScLcInvT,
                LrInvSrLrInvT=LrInvSrLrInvT, Lr=Lr, Lc=Lc, kern_r=kern_r, Xr=Xr, Zr=Zr)

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

        if not uncertain_inputs_r:
            dL_dpsi1_r += psi1_r.dot(dL_dpsi2_r+dL_dpsi2_r.T)
        if not uncertain_inputs_c:
            dL_dpsi1_c += psi1_c.dot(dL_dpsi2_c+dL_dpsi2_c.T)

        grad_dict = {
            'dL_dthetaL':dL_dthetaL,
            'dL_dqU_mean':dL_dqU_mean,
            'dL_dqU_var_c':dL_dqU_var_c,
            'dL_dqU_var_r':dL_dqU_var_r,
            'dL_dKuu_c': dL_dKuu_c,
            'dL_dKuu_r': dL_dKuu_r,
        }

        if uncertain_inputs_c:
            grad_dict['dL_dpsi0_c'] = dL_dpsi0_c
            grad_dict['dL_dpsi1_c'] = dL_dpsi1_c
            grad_dict['dL_dpsi2_c'] = dL_dpsi2_c
        else:
            grad_dict['dL_dKdiag_c'] = dL_dpsi0_c
            grad_dict['dL_dKfu_c'] = dL_dpsi1_c

        if uncertain_inputs_r:
            grad_dict['dL_dpsi0_r'] = dL_dpsi0_r
            grad_dict['dL_dpsi1_r'] = dL_dpsi1_r
            grad_dict['dL_dpsi2_r'] = dL_dpsi2_r
        else:
            grad_dict['dL_dKdiag_r'] = dL_dpsi0_r
            grad_dict['dL_dKfu_r'] = dL_dpsi1_r

        return post, logL, grad_dict

class PosteriorMultioutput(object):

    def __init__(self,LcInvMLrInvT, LcInvScLcInvT, LrInvSrLrInvT, Lr, Lc, kern_r, Xr, Zr):
        self.LcInvMLrInvT = LcInvMLrInvT
        self.LcInvScLcInvT = LcInvScLcInvT
        self.LrInvSrLrInvT = LrInvSrLrInvT
        self.Lr = Lr
        self.Lc = Lc
        self.kern_r = kern_r
        self.Xr = Xr
        self.Zr = Zr

    def _prepare(self):
        D, Mr, Mc = self.Xr.shape[0], self.Zr.shape[0], self.LcInvMLrInvT.shape[0]
        psi2_r_n = self.kern_r.psi2n(self.Zr, self.Xr)
        psi0_r = self.kern_r.psi0(self.Zr, self.Xr)
        psi1_r = self.kern_r.psi1(self.Zr, self.Xr)

        LrInvPsi1_rT = dtrtrs(self.Lr, psi1_r.T)[0]
        self.woodbury_vector = self.LcInvMLrInvT.dot(LrInvPsi1_rT)

        LrInvPsi2_r_nLrInvT = dtrtrs(self.Lr, np.swapaxes((dtrtrs(self.Lr, psi2_r_n.reshape(D*Mr,Mr).T)[0].T).reshape(D,Mr,Mr),1,2).reshape(D*Mr,Mr).T)[0].T.reshape(D,Mr,Mr)

        tr_LrInvPsi2_r_nLrInvT = LrInvPsi2_r_nLrInvT.reshape(D,Mr*Mr).sum(1)
        tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT = LrInvPsi2_r_nLrInvT.reshape(D,Mr*mr).dot(self.LrInvSrLrInvT.flat)

        tmp = LrInvPsi2_r_nLrInvT - LrInvPsi1_rT.T[:,:,None]*LrInvPsi1_rT.T[:,None,:]
        tmp = np.swapaxes(tmp.reshape(D*Mr,Mr).dot(self.LcInvMLrInvT.T).reshape(D,Mr,Mc), 1,2).reshape(D*Mc,Mr).dot(self.LcInvMLrInvT.T).reshape(D,Mc,Mc)

    def _raw_predict(self, kern, Xnew, pred_var, full_cov=False):

        N = Xnew.shape[0]
        psi1_c = kern.K(Xnew, pred_var)
        psi0_c = kern.Kdiag(Xnew)
        LcInvPsi1_cT = dtrtrs(self.Lc, psi1_c.T)[0]

        D, Mr, Mc = self.Xr.shape[0], self.Zr.shape[0], self.LcInvMLrInvT.shape[0]
        psi2_r_n = self.kern_r.psi2n(self.Zr, self.Xr)
        psi0_r = self.kern_r.psi0(self.Zr, self.Xr)
        psi1_r = self.kern_r.psi1(self.Zr, self.Xr)

        LrInvPsi1_rT = dtrtrs(self.Lr, psi1_r.T)[0]
        woodbury_vector = self.LcInvMLrInvT.dot(LrInvPsi1_rT)

        mu = np.dot(LcInvPsi1_cT.T, woodbury_vector)

        LrInvPsi2_r_nLrInvT = dtrtrs(self.Lr, np.swapaxes((dtrtrs(self.Lr, psi2_r_n.reshape(D*Mr,Mr).T)[0].T).reshape(D,Mr,Mr),1,2).reshape(D*Mr,Mr).T)[0].T.reshape(D,Mr,Mr)

        tr_LrInvPsi2_r_nLrInvT = np.diagonal(LrInvPsi2_r_nLrInvT,axis1=1,axis2=2).sum(1)
        tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT = LrInvPsi2_r_nLrInvT.reshape(D,Mr*Mr).dot(self.LrInvSrLrInvT.flat)

        tmp = LrInvPsi2_r_nLrInvT - LrInvPsi1_rT.T[:,:,None]*LrInvPsi1_rT.T[:,None,:]
        tmp = np.swapaxes(tmp.reshape(D*Mr,Mr).dot(self.LcInvMLrInvT.T).reshape(D,Mr,Mc), 1,2).reshape(D*Mc,Mr).dot(self.LcInvMLrInvT.T).reshape(D,Mc,Mc)

        var1 = (tmp.reshape(D*Mc,Mc).dot(LcInvPsi1_cT).reshape(D,Mc,N)*LcInvPsi1_cT[None,:,:]).sum(1).T
        var2 = psi0_c[:,None]*psi0_r[None,:]
        var3 = tr_LrInvPsi2_r_nLrInvT[None,:]*np.square(LcInvPsi1_cT).sum(0)[:,None]
        var4 = tr_LrInvPsi2_r_nLrInvT_LrInvSrLrInvT[None,:]* (self.LcInvScLcInvT.dot(LcInvPsi1_cT)*LcInvPsi1_cT).sum(0)[:,None]
        var = var1+var2-var3+var4
        return mu, var
