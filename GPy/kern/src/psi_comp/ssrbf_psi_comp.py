# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The package for the psi statistics computation
"""

import numpy as np

try:
    from scipy import weave

    def _psicomputations(variance, lengthscale, Z, variational_posterior):
        """
        Z - MxQ
        mu - NxQ
        S - NxQ
        gamma - NxQ
        """
        # here are the "statistics" for psi0, psi1 and psi2
        # Produced intermediate results:
        # _psi1                NxM
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob

        N,M,Q = mu.shape[0],Z.shape[0],mu.shape[1]
        l2 = np.square(lengthscale)
        log_denom1 = np.log(S/l2+1)
        log_denom2 = np.log(2*S/l2+1)
        log_gamma = np.log(gamma)
        log_gamma1 = np.log(1.-gamma)
        variance = float(variance)
        psi0 = np.empty(N)
        psi0[:] = variance
        psi1 = np.empty((N,M))
        psi2n = np.empty((N,M,M))

        from ....util.misc import param_to_array
        S = param_to_array(S)
        mu = param_to_array(mu)
        gamma = param_to_array(gamma)
        Z = param_to_array(Z)

        support_code = """
        #include <math.h>
        """
        code = """
        for(int n=0; n<N; n++) {
            for(int m1=0;m1<M;m1++) {
                double log_psi1=0;
                for(int m2=0;m2<=m1;m2++) {
                    double log_psi2_n=0;
                    for(int q=0;q<Q;q++) {
                        double Snq = S(n,q);
                        double lq = l2(q);
                        double Zm1q = Z(m1,q);
                        double Zm2q = Z(m2,q);

                        if(m2==0) {
                            // Compute Psi_1
                            double muZ = mu(n,q)-Z(m1,q);

                            double psi1_exp1 = log_gamma(n,q) - (muZ*muZ/(Snq+lq) +log_denom1(n,q))/2.;
                            double psi1_exp2 = log_gamma1(n,q) -Zm1q*Zm1q/(2.*lq);
                            log_psi1 += (psi1_exp1>psi1_exp2)?psi1_exp1+log1p(exp(psi1_exp2-psi1_exp1)):psi1_exp2+log1p(exp(psi1_exp1-psi1_exp2));
                        }
                        // Compute Psi_2
                        double muZhat = mu(n,q) - (Zm1q+Zm2q)/2.;
                        double Z2 = Zm1q*Zm1q+ Zm2q*Zm2q;
                        double dZ = Zm1q - Zm2q;

                        double psi2_exp1 = dZ*dZ/(-4.*lq)-muZhat*muZhat/(2.*Snq+lq) - log_denom2(n,q)/2. + log_gamma(n,q);
                        double psi2_exp2 = log_gamma1(n,q) - Z2/(2.*lq);
                        log_psi2_n += (psi2_exp1>psi2_exp2)?psi2_exp1+log1p(exp(psi2_exp2-psi2_exp1)):psi2_exp2+log1p(exp(psi2_exp1-psi2_exp2));
                    }
                    double exp_psi2_n = exp(log_psi2_n);
                    psi2n(n,m1,m2) = variance*variance*exp_psi2_n;
                    if(m1!=m2) { psi2n(n,m2,m1) = variance*variance*exp_psi2_n;}
                }
                psi1(n,m1) = variance*exp(log_psi1);
            }
        }
        """
        weave.inline(code, support_code=support_code, arg_names=['psi1','psi2n','N','M','Q','variance','l2','Z','mu','S','gamma','log_denom1','log_denom2','log_gamma','log_gamma1'], type_converters=weave.converters.blitz)

        psi2 = psi2n.sum(axis=0)
        return psi0,psi1,psi2,psi2n

    from GPy.util.caching import Cacher
    psicomputations = Cacher(_psicomputations, limit=3)

    def psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
        ARD = (len(lengthscale)!=1)

        _,psi1,_,psi2n = psicomputations(variance, lengthscale, Z, variational_posterior)

        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob
        N,M,Q = mu.shape[0],Z.shape[0],mu.shape[1]
        l2 = np.square(lengthscale)
        log_denom1 = np.log(S/l2+1)
        log_denom2 = np.log(2*S/l2+1)
        log_gamma = np.log(gamma)
        log_gamma1 = np.log(1.-gamma)
        variance = float(variance)

        dvar = np.zeros(1)
        dmu = np.zeros((N,Q))
        dS = np.zeros((N,Q))
        dgamma = np.zeros((N,Q))
        dl = np.zeros(Q)
        dZ = np.zeros((M,Q))
        dvar += np.sum(dL_dpsi0)

        from ....util.misc import param_to_array
        S = param_to_array(S)
        mu = param_to_array(mu)
        gamma = param_to_array(gamma)
        Z = param_to_array(Z)

        support_code = """
        #include <math.h>
        """
        code = """
        for(int n=0; n<N; n++) {
            for(int m1=0;m1<M;m1++) {
                double log_psi1=0;
                for(int m2=0;m2<M;m2++) {
                    double log_psi2_n=0;
                    for(int q=0;q<Q;q++) {
                        double Snq = S(n,q);
                        double lq = l2(q);
                        double Zm1q = Z(m1,q);
                        double Zm2q = Z(m2,q);
                        double gnq = gamma(n,q);
                        double mu_nq = mu(n,q);

                        if(m2==0) {
                            // Compute Psi_1
                            double lpsi1 = psi1(n,m1)*dL_dpsi1(n,m1);
                            if(q==0) {dvar(0) += lpsi1/variance;}

                            double Zmu = Zm1q - mu_nq;
                            double denom = Snq+lq;
                            double Zmu2_denom = Zmu*Zmu/denom;

                            double exp1 = log_gamma(n,q)-(Zmu*Zmu/(Snq+lq)+log_denom1(n,q))/(2.);
                            double exp2 = log_gamma1(n,q)-Zm1q*Zm1q/(2.*lq);
                            double d_exp1,d_exp2;
                            if(exp1>exp2) {
                                d_exp1 = 1.;
                                d_exp2 = exp(exp2-exp1);
                            } else {
                                d_exp1 = exp(exp1-exp2);
                                d_exp2 = 1.;
                            }
                            double exp_sum = d_exp1+d_exp2;

                            dmu(n,q) += lpsi1*Zmu*d_exp1/(denom*exp_sum);
                            dS(n,q) += lpsi1*(Zmu2_denom-1.)*d_exp1/(denom*exp_sum)/2.;
                            dgamma(n,q) += lpsi1*(d_exp1/gnq-d_exp2/(1.-gnq))/exp_sum;
                            dl(q) += lpsi1*((Zmu2_denom+Snq/lq)/denom*d_exp1+Zm1q*Zm1q/(lq*lq)*d_exp2)/(2.*exp_sum);
                            dZ(m1,q) += lpsi1*(-Zmu/denom*d_exp1-Zm1q/lq*d_exp2)/exp_sum;
                        }
                        // Compute Psi_2
                        double lpsi2 = psi2n(n,m1,m2)*dL_dpsi2(m1,m2);
                        if(q==0) {dvar(0) += lpsi2*2/variance;}

                        double dZm1m2 = Zm1q - Zm2q;
                        double Z2 = Zm1q*Zm1q+Zm2q*Zm2q;
                        double muZhat =  mu_nq - (Zm1q + Zm2q)/2.;
                        double denom = 2.*Snq+lq;
                        double muZhat2_denom = muZhat*muZhat/denom;

                        double exp1 = dZm1m2*dZm1m2/(-4.*lq)-muZhat*muZhat/(2.*Snq+lq) - log_denom2(n,q)/2. + log_gamma(n,q);
                        double exp2 = log_gamma1(n,q) - Z2/(2.*lq);
                        double d_exp1,d_exp2;
                        if(exp1>exp2) {
                            d_exp1 = 1.;
                            d_exp2 = exp(exp2-exp1);
                        } else {
                            d_exp1 = exp(exp1-exp2);
                            d_exp2 = 1.;
                        }
                        double exp_sum = d_exp1+d_exp2;

                        dmu(n,q) += -2.*lpsi2*muZhat/denom*d_exp1/exp_sum;
                        dS(n,q) += lpsi2*(2.*muZhat2_denom-1.)/denom*d_exp1/exp_sum;
                        dgamma(n,q) += lpsi2*(d_exp1/gnq-d_exp2/(1.-gnq))/exp_sum;
                        dl(q) += lpsi2*(((Snq/lq+muZhat2_denom)/denom+dZm1m2*dZm1m2/(4.*lq*lq))*d_exp1+Z2/(2.*lq*lq)*d_exp2)/exp_sum;
                        dZ(m1,q) += 2.*lpsi2*((muZhat/denom-dZm1m2/(2*lq))*d_exp1-Zm1q/lq*d_exp2)/exp_sum;
                    }
                }
            }
        }
        """
        weave.inline(code, support_code=support_code, arg_names=['dL_dpsi1','dL_dpsi2','psi1','psi2n','N','M','Q','variance','l2','Z','mu','S','gamma','log_denom1','log_denom2','log_gamma','log_gamma1','dvar','dl','dmu','dS','dgamma','dZ'], type_converters=weave.converters.blitz)

        dl *= 2.*lengthscale
        if not ARD:
            dl = dl.sum()

        return dvar, dl, dZ, dmu, dS, dgamma

except:

    def psicomputations(variance, lengthscale, Z, variational_posterior):
        """
        Z - MxQ
        mu - NxQ
        S - NxQ
        gamma - NxQ
        """
        # here are the "statistics" for psi0, psi1 and psi2
        # Produced intermediate results:
        # _psi1                NxM
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob

        psi0 = np.empty(mu.shape[0])
        psi0[:] = variance
        psi1 = _psi1computations(variance, lengthscale, Z, mu, S, gamma)
        psi2 = _psi2computations(variance, lengthscale, Z, mu, S, gamma)
        return psi0, psi1, psi2

    def _psi1computations(variance, lengthscale, Z, mu, S, gamma):
        """
        Z - MxQ
        mu - NxQ
        S - NxQ
        gamma - NxQ
        """
        # here are the "statistics" for psi1
        # Produced intermediate results:
        # _psi1                NxM

        lengthscale2 = np.square(lengthscale)

        # psi1
        _psi1_denom = S[:, None, :] / lengthscale2 + 1.  # Nx1xQ
        _psi1_denom_sqrt = np.sqrt(_psi1_denom) #Nx1xQ
        _psi1_dist = Z[None, :, :] - mu[:, None, :]  # NxMxQ
        _psi1_dist_sq = np.square(_psi1_dist) / (lengthscale2 * _psi1_denom) # NxMxQ
        _psi1_common = gamma[:,None,:] / (lengthscale2*_psi1_denom*_psi1_denom_sqrt) #Nx1xQ
        _psi1_exponent1 = np.log(gamma[:,None,:]) - (_psi1_dist_sq + np.log(_psi1_denom))/2. # NxMxQ
        _psi1_exponent2 = np.log(1.-gamma[:,None,:]) - (np.square(Z[None,:,:])/lengthscale2)/2. # NxMxQ
        _psi1_exponent_max = np.maximum(_psi1_exponent1,_psi1_exponent2)
        _psi1_exponent = _psi1_exponent_max+np.log(np.exp(_psi1_exponent1-_psi1_exponent_max) + np.exp(_psi1_exponent2-_psi1_exponent_max)) #NxMxQ
        _psi1_exp_sum = _psi1_exponent.sum(axis=-1) #NxM
        _psi1 = variance * np.exp(_psi1_exp_sum) # NxM

        return _psi1

    def _psi2computations(variance, lengthscale, Z, mu, S, gamma):
        """
        Z - MxQ
        mu - NxQ
        S - NxQ
        gamma - NxQ
        """
        # here are the "statistics" for psi2
        # Produced intermediate results:
        # _psi2                MxM

        lengthscale2 = np.square(lengthscale)

        _psi2_Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
        _psi2_Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
        _psi2_Zdist_sq = np.square(_psi2_Zdist / lengthscale) # M,M,Q
        _psi2_Z_sq_sum = (np.square(Z[:,None,:])+np.square(Z[None,:,:]))/lengthscale2 # MxMxQ

        # psi2
        _psi2_denom = 2.*S[:, None, None, :] / lengthscale2 + 1. # Nx1x1xQ
        _psi2_denom_sqrt = np.sqrt(_psi2_denom)
        _psi2_mudist = mu[:,None,None,:]-_psi2_Zhat #N,M,M,Q
        _psi2_mudist_sq = np.square(_psi2_mudist)/(lengthscale2*_psi2_denom)
        _psi2_common = gamma[:,None,None,:]/(lengthscale2 * _psi2_denom * _psi2_denom_sqrt) # Nx1x1xQ
        _psi2_exponent1 = -_psi2_Zdist_sq -_psi2_mudist_sq -0.5*np.log(_psi2_denom)+np.log(gamma[:,None,None,:]) #N,M,M,Q
        _psi2_exponent2 = np.log(1.-gamma[:,None,None,:]) - 0.5*(_psi2_Z_sq_sum) # NxMxMxQ
        _psi2_exponent_max = np.maximum(_psi2_exponent1, _psi2_exponent2)
        _psi2_exponent = _psi2_exponent_max+np.log(np.exp(_psi2_exponent1-_psi2_exponent_max) + np.exp(_psi2_exponent2-_psi2_exponent_max))
        _psi2_exp_sum = _psi2_exponent.sum(axis=-1) #NxM
        _psi2 = variance*variance * (np.exp(_psi2_exp_sum).sum(axis=0)) # MxM

        return _psi2

    def psiDerivativecomputations(dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, lengthscale, Z, variational_posterior):
        ARD = (len(lengthscale)!=1)

        dvar_psi1, dl_psi1, dZ_psi1, dmu_psi1, dS_psi1, dgamma_psi1 = _psi1compDer(dL_dpsi1, variance, lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
        dvar_psi2, dl_psi2, dZ_psi2, dmu_psi2, dS_psi2, dgamma_psi2 = _psi2compDer(dL_dpsi2, variance, lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)

        dL_dvar = np.sum(dL_dpsi0) + dvar_psi1 + dvar_psi2

        dL_dlengscale = dl_psi1 + dl_psi2
        if not ARD:
            dL_dlengscale = dL_dlengscale.sum()

        dL_dgamma = dgamma_psi1 + dgamma_psi2
        dL_dmu = dmu_psi1 + dmu_psi2
        dL_dS = dS_psi1 + dS_psi2
        dL_dZ = dZ_psi1 + dZ_psi2

        return dL_dvar, dL_dlengscale, dL_dZ, dL_dmu, dL_dS, dL_dgamma

    def _psi1compDer(dL_dpsi1, variance, lengthscale, Z, mu, S, gamma):
        """
        dL_dpsi1 - NxM
        Z - MxQ
        mu - NxQ
        S - NxQ
        gamma - NxQ
        """
        # here are the "statistics" for psi1
        # Produced intermediate results: dL_dparams w.r.t. psi1
        # _dL_dvariance     1
        # _dL_dlengthscale  Q
        # _dL_dZ            MxQ
        # _dL_dgamma        NxQ
        # _dL_dmu           NxQ
        # _dL_dS            NxQ

        lengthscale2 = np.square(lengthscale)

        # psi1
        _psi1_denom = S / lengthscale2 + 1.  # NxQ
        _psi1_denom_sqrt = np.sqrt(_psi1_denom) #NxQ
        _psi1_dist = Z[None, :, :] - mu[:, None, :]  # NxMxQ
        _psi1_dist_sq = np.square(_psi1_dist) / (lengthscale2 * _psi1_denom[:,None,:]) # NxMxQ
        _psi1_common = gamma / (lengthscale2*_psi1_denom*_psi1_denom_sqrt) #NxQ
        _psi1_exponent1 = np.log(gamma[:,None,:]) -0.5 * (_psi1_dist_sq + np.log(_psi1_denom[:, None,:])) # NxMxQ
        _psi1_exponent2 = np.log(1.-gamma[:,None,:]) -0.5 * (np.square(Z[None,:,:])/lengthscale2) # NxMxQ
        _psi1_exponent_max = np.maximum(_psi1_exponent1,_psi1_exponent2)
        _psi1_exponent = _psi1_exponent_max+np.log(np.exp(_psi1_exponent1-_psi1_exponent_max) + np.exp(_psi1_exponent2-_psi1_exponent_max)) #NxMxQ
        _psi1_exp_sum = _psi1_exponent.sum(axis=-1) #NxM
        _psi1_exp_dist_sq = np.exp(-0.5*_psi1_dist_sq) # NxMxQ
        _psi1_exp_Z = np.exp(-0.5*np.square(Z[None,:,:])/lengthscale2) # 1xMxQ
        _psi1_q = variance * np.exp(_psi1_exp_sum[:,:,None] - _psi1_exponent) # NxMxQ
        _psi1 = variance * np.exp(_psi1_exp_sum) # NxM
        _dL_dvariance = np.einsum('nm,nm->',dL_dpsi1, _psi1)/variance # 1
        _dL_dgamma = np.einsum('nm,nmq,nmq->nq',dL_dpsi1, _psi1_q, (_psi1_exp_dist_sq/_psi1_denom_sqrt[:,None,:]-_psi1_exp_Z)) # NxQ
        _dL_dmu = np.einsum('nm, nmq, nmq, nmq, nq->nq',dL_dpsi1,_psi1_q,_psi1_exp_dist_sq,_psi1_dist,_psi1_common)  # NxQ
        _dL_dS = np.einsum('nm,nmq,nmq,nq,nmq->nq',dL_dpsi1,_psi1_q,_psi1_exp_dist_sq,_psi1_common,(_psi1_dist_sq-1.))/2.  # NxQ
        _dL_dZ = np.einsum('nm,nmq,nmq->mq',dL_dpsi1,_psi1_q, (- _psi1_common[:,None,:] * _psi1_dist * _psi1_exp_dist_sq - (1-gamma[:,None,:])/lengthscale2*Z[None,:,:]*_psi1_exp_Z))
        _dL_dlengthscale = lengthscale* np.einsum('nm,nmq,nmq->q',dL_dpsi1,_psi1_q,(_psi1_common[:,None,:]*(S[:,None,:]/lengthscale2+_psi1_dist_sq)*_psi1_exp_dist_sq + (1-gamma[:,None,:])*np.square(Z[None,:,:]/lengthscale2)*_psi1_exp_Z))

        return _dL_dvariance, _dL_dlengthscale, _dL_dZ, _dL_dmu, _dL_dS, _dL_dgamma

    def _psi2compDer(dL_dpsi2, variance, lengthscale, Z, mu, S, gamma):
        """
        Z - MxQ
        mu - NxQ
        S - NxQ
        gamma - NxQ
        dL_dpsi2 - MxM
        """
        # here are the "statistics" for psi2
        # Produced the derivatives w.r.t. psi2:
        # _dL_dvariance      1
        # _dL_dlengthscale   Q
        # _dL_dZ             MxQ
        # _dL_dgamma         NxQ
        # _dL_dmu            NxQ
        # _dL_dS             NxQ

        lengthscale2 = np.square(lengthscale)

        _psi2_Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
        _psi2_Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
        _psi2_Zdist_sq = np.square(_psi2_Zdist / lengthscale) # M,M,Q
        _psi2_Z_sq_sum = (np.square(Z[:,None,:])+np.square(Z[None,:,:]))/lengthscale2 # MxMxQ

        # psi2
        _psi2_denom = 2.*S / lengthscale2 + 1. # NxQ
        _psi2_denom_sqrt = np.sqrt(_psi2_denom)
        _psi2_mudist = mu[:,None,None,:]-_psi2_Zhat #N,M,M,Q
        _psi2_mudist_sq = np.square(_psi2_mudist)/(lengthscale2*_psi2_denom[:,None,None,:])
        _psi2_common = gamma/(lengthscale2 * _psi2_denom * _psi2_denom_sqrt) # NxQ
        _psi2_exponent1 = -_psi2_Zdist_sq -_psi2_mudist_sq -0.5*np.log(_psi2_denom[:,None,None,:])+np.log(gamma[:,None,None,:]) #N,M,M,Q
        _psi2_exponent2 = np.log(1.-gamma[:,None,None,:]) - 0.5*(_psi2_Z_sq_sum) # NxMxMxQ
        _psi2_exponent_max = np.maximum(_psi2_exponent1, _psi2_exponent2)
        _psi2_exponent = _psi2_exponent_max+np.log(np.exp(_psi2_exponent1-_psi2_exponent_max) + np.exp(_psi2_exponent2-_psi2_exponent_max))
        _psi2_exp_sum = _psi2_exponent.sum(axis=-1) #NxM
        _psi2_q = variance*variance * np.exp(_psi2_exp_sum[:,:,:,None]-_psi2_exponent) # NxMxMxQ
        _psi2_exp_dist_sq = np.exp(-_psi2_Zdist_sq -_psi2_mudist_sq) # NxMxMxQ
        _psi2_exp_Z = np.exp(-0.5*_psi2_Z_sq_sum) # MxMxQ
        _psi2 = variance*variance * (np.exp(_psi2_exp_sum).sum(axis=0)) # MxM
        _dL_dvariance = np.einsum('mo,mo->',dL_dpsi2,_psi2)*2./variance
        _dL_dgamma = np.einsum('mo,nmoq,nmoq->nq',dL_dpsi2,_psi2_q,(_psi2_exp_dist_sq/_psi2_denom_sqrt[:,None,None,:] - _psi2_exp_Z))
        _dL_dmu = -2.*np.einsum('mo,nmoq,nq,nmoq,nmoq->nq',dL_dpsi2,_psi2_q,_psi2_common,_psi2_mudist,_psi2_exp_dist_sq)
        _dL_dS = np.einsum('mo,nmoq,nq,nmoq,nmoq->nq',dL_dpsi2,_psi2_q, _psi2_common, (2.*_psi2_mudist_sq-1.), _psi2_exp_dist_sq)
        _dL_dZ = 2.*np.einsum('mo,nmoq,nmoq->mq',dL_dpsi2,_psi2_q,(_psi2_common[:,None,None,:]*(-_psi2_Zdist*_psi2_denom[:,None,None,:]+_psi2_mudist)*_psi2_exp_dist_sq - (1-gamma[:,None,None,:])*Z[:,None,:]/lengthscale2*_psi2_exp_Z))
        _dL_dlengthscale = 2.*lengthscale* np.einsum('mo,nmoq,nmoq->q',dL_dpsi2,_psi2_q,(_psi2_common[:,None,None,:]*(S[:,None,None,:]/lengthscale2+_psi2_Zdist_sq*_psi2_denom[:,None,None,:]+_psi2_mudist_sq)*_psi2_exp_dist_sq+(1-gamma[:,None,None,:])*_psi2_Z_sq_sum*0.5/lengthscale2*_psi2_exp_Z))

        return _dL_dvariance, _dL_dlengthscale, _dL_dZ, _dL_dmu, _dL_dS, _dL_dgamma
