# Copyright (c) 2014, Cristian Guarnizo.
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import wofz
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this

class EQ_ODE2(Kern):
    """
    Covariance function for second order differential equation driven by an exponentiated quadratic covariance.

    This outputs of this kernel have the form
    .. math::
       \frac{\text{d}^2y_j(t)}{\text{d}^2t} + C_j\frac{\text{d}y_j(t)}{\text{d}t} + B_jy_j(t) = \sum_{i=1}^R w_{j,i} u_i(t)

    where :math:`R` is the rank of the system, :math:`w_{j,i}` is the sensitivity of the :math:`j`th output to the :math:`i`th latent function, :math:`d_j` is the decay rate of the :math:`j`th output and :math:`f_i(t)` and :math:`g_i(t)` are independent latent Gaussian processes goverened by an exponentiated quadratic covariance.

    :param output_dim: number of outputs driven by latent function.
    :type output_dim: int
    :param W: sensitivities of each output to the latent driving function.
    :type W: ndarray (output_dim x rank).
    :param rank: If rank is greater than 1 then there are assumed to be a total of rank latent forces independently driving the system, each with identical covariance.
    :type rank: int
    :param C: damper constant for the second order system.
    :type C: array of length output_dim.
    :param B: spring constant for the second order system.
    :type B: array of length output_dim.

    """
    #This code will only work for the sparseGP model, due to limitations in models for this kernel
    def __init__(self, input_dim=2, output_dim=1, rank=1, W=None, lengthscale=None, C=None, B=None, active_dims=None, name='eq_ode2'):
        #input_dim should be 1, but kern._slice_X is not returning index information required to evaluate kernels        
        assert input_dim == 2, "only defined for 1 input dims"
        super(EQ_ODE2, self).__init__(input_dim=input_dim, active_dims=active_dims, name=name)
        self.rank = rank
        self.output_dim = output_dim

        if lengthscale is None:
            lengthscale = .5+np.random.rand(self.rank)
        else:
            lengthscale = np.asarray(lengthscale)
            assert lengthscale.size in [1, self.rank], "Bad number of lengthscales"
            if lengthscale.size != self.rank:
                lengthscale = np.ones(self.input_dim)*lengthscale

        if W is None:
            #W = 0.5*np.random.randn(self.output_dim, self.rank)/np.sqrt(self.rank)
            W = np.ones((self.output_dim, self.rank))
        else:
            assert W.shape == (self.output_dim, self.rank)

        if C is None:
            C = np.ones(self.output_dim)

        if B is None:
            B = np.ones(self.output_dim)

        self.C = Param('C', C, Logexp())
        self.B = Param('B', B, Logexp())
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())
        self.W = Param('W', W)
        self.link_parameters(self.lengthscale, self.C, self.B, self.W)

    @Cache_this(limit=3)
    def K(self, X, X2=None):
        #This way is not working, indexes are lost after using k._slice_X
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        index = np.int_(X[:, 1])
        index = index.reshape(index.size,)
        X_flag = index[0] >= self.output_dim
        if X2 is None:
            if X_flag:
                #Calculate covariance function for the latent functions
                index -= self.output_dim
                return self._Kuu(X, index)
            else:
                raise NotImplementedError
        else:
            #This way is not working, indexes are lost after using k._slice_X
            #index2 = np.asarray(X2, dtype=np.int)
            #index2 = index2.reshape(index2.size,)
            if hasattr(X2, 'values'):
                X2 = X2.values
            index2 = np.int_(X2[:, 1])
            index2 = index2.reshape(index2.size,)
            X2_flag = index2[0] >= self.output_dim
            #Calculate cross-covariance function
            if not X_flag and X2_flag:
                index2 -= self.output_dim
                return self._Kfu(X, index, X2, index2) #Kfu
            else:
                index -= self.output_dim
                return self._Kfu(X2, index2, X, index).T #Kuf

    #Calculate the covariance function for diag(Kff(X,X))
    def Kdiag(self, X):
        #This way is not working, indexes are lost after using k._slice_X
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        index = np.int_(X[:, 1])
        index = index.reshape(index.size,)
        
        #terms that move along t
        t = X[:, 0].reshape(X.shape[0], 1)
        d = np.unique(index) #Output Indexes
        B = self.B.values[d]
        C = self.C.values[d]
        S = self.W.values[d, :]
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        #Check where wd becomes complex
        wbool = C*C >= 4.*B
        B = B.reshape(B.size, 1)
        C = C.reshape(C.size, 1)
        alpha = .5*C
        C2 = C*C

        wbool2 = wbool[index]
        ind2t = np.where(wbool2)
        ind3t = np.where(np.logical_not(wbool2))

        #Terms that move along q
        lq = self.lengthscale.values.reshape(1, self.lengthscale.size)
        S2 = S*S
        kdiag = np.empty((t.size, ))

        indD = np.arange(B.size)
        #(1) When wd is real
        if np.any(np.logical_not(wbool)):
            #Indexes of index and t related to (2)
            t1 = t[ind3t]
            ind = index[ind3t]
            d = np.asarray(np.where(np.logical_not(wbool))[0]) #Selection of outputs
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            S2lq = S2[d]*(.5*lq)
            c0 = S2lq*np.sqrt(np.pi)
            w = .5*np.sqrt(4.*B[d] - C2[d])
            alphad = alpha[d]
            w2 = w*w
            gam = alphad + 1j*w
            gamc = alphad - 1j*w
            c1 = .5/(alphad*w2)
            c2 = .5/(gam*w2)
            c = c1 - c2
            #DxQ terms
            nu = lq*(gam*.5)
            K01 = c0*c
            #Nx1 terms
            gamt = -gam[ind]*t1
            gamct = -gamc[ind]*t1
            egamt = np.exp(gamt)
            ec = egamt*c2[ind] - np.exp(gamct)*c1[ind]
            #NxQ terms
            t_lq = t1/lq

            # Upsilon Calculations
            # Using wofz
            wnu = wofz(1j*nu)
            lwnu = np.log(wnu)
            t2_lq2 = -t_lq*t_lq
            upm = wnu[ind] - np.exp(t2_lq2 + gamt + np.log(wofz(1j*(t_lq + nu[ind]))))
            upm[t1[:, 0] == 0, :] = 0.

            nu2 = nu*nu
            z1 = nu[ind] - t_lq
            indv1 = np.where(z1.real >= 0.)
            indv2 = np.where(z1.real < 0.)
            upv = -np.exp(lwnu[ind] + gamt)
            if indv1[0].shape > 0:
                upv[indv1] += np.exp(t2_lq2[indv1] + np.log(wofz(1j*z1[indv1])))
            if indv2[0].shape > 0:
                upv[indv2] += np.exp(nu2[ind[indv2[0]], indv2[1]] + gamt[indv2[0], 0] + np.log(2.))\
                             - np.exp(t2_lq2[indv2] + np.log(wofz(-1j*z1[indv2])))
            upv[t1[:, 0] == 0, :] = 0.

            #Covariance calculation
            kdiag[ind3t] = np.sum(np.real(K01[ind]*upm), axis=1)
            kdiag[ind3t] += np.sum(np.real((c0[ind]*ec)*upv), axis=1)

        #(2) When w_d is complex
        if np.any(wbool):
            t1 = t[ind2t]
            ind = index[ind2t]
            #Index transformation
            d = np.asarray(np.where(wbool)[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            S2lq = S2[d]*(lq*.25)
            c0 = S2lq*np.sqrt(np.pi)
            w = .5*np.sqrt(C2[d] - 4.*B[d])
            alphad = alpha[d]
            gam = alphad - w
            gamc = alphad + w
            w2 = -w*w
            c1 = .5/(alphad*w2)
            c21 = .5/(gam*w2)
            c22 = .5/(gamc*w2)
            c = c1 - c21
            c2 = c1 - c22
            #DxQ terms
            K011 = c0*c
            K012 = c0*c2
            nu = lq*(.5*gam)
            nuc = lq*(.5*gamc)
            #Nx1 terms
            gamt = -gam[ind]*t1
            gamct = -gamc[ind]*t1
            egamt = np.exp(gamt)
            egamct = np.exp(gamct)
            ec = egamt*c21[ind] - egamct*c1[ind]
            ec2 = egamct*c22[ind] - egamt*c1[ind]
            #NxQ terms
            t_lq = t1/lq

            #Upsilon Calculations using wofz
            t2_lq2 = -t_lq*t_lq #Required when using wofz
            wnu = wofz(1j*nu).real
            lwnu = np.log(wnu)
            upm = wnu[ind] - np.exp(t2_lq2 + gamt + np.log(wofz(1j*(t_lq + nu[ind])).real))
            upm[t1[:, 0] == 0., :] = 0.

            nu2 = nu*nu
            z1 = nu[ind] - t_lq
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            upv = -np.exp(lwnu[ind] + gamt)
            if indv1[0].shape > 0:
                upv[indv1] += np.exp(t2_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                upv[indv2] += np.exp(nu2[ind[indv2[0]], indv2[1]] + gamt[indv2[0], 0] + np.log(2.))\
                              - np.exp(t2_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
            upv[t1[:, 0] == 0, :] = 0.

            wnuc = wofz(1j*nuc).real
            lwnuc = np.log(wnuc)

            upmc = wnuc[ind] - np.exp(t2_lq2 + gamct + np.log(wofz(1j*(t_lq + nuc[ind])).real))
            upmc[t1[:, 0] == 0., :] = 0.

            nuc2 = nuc*nuc
            z1 = nuc[ind] - t_lq
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            upvc = - np.exp(lwnuc[ind] + gamct)
            if indv1[0].shape > 0:
                upvc[indv1] += np.exp(t2_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                upvc[indv2] += np.exp(nuc2[ind[indv2[0]], indv2[1]] + gamct[indv2[0], 0] + np.log(2.))\
                               - np.exp(t2_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
            upvc[t1[:, 0] == 0, :] = 0.

            #Covariance calculation
            kdiag[ind2t] = np.sum(K011[ind]*upm + K012[ind]*upmc + (c0[ind]*ec)*upv + (c0[ind]*ec2)*upvc, axis=1)
        return kdiag

    def update_gradients_full(self, dL_dK, X, X2 = None):
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        self.B.gradient = np.zeros(self.B.shape)
        self.C.gradient = np.zeros(self.C.shape)
        self.W.gradient = np.zeros(self.W.shape)
        self.lengthscale.gradient = np.zeros(self.lengthscale.shape)
        index = np.int_(X[:, 1])
        index = index.reshape(index.size,)
        X_flag = index[0] >= self.output_dim
        if X2 is None:
            if X_flag: #Kuu or Kmm
                index -= self.output_dim
                tmp = dL_dK*self._gkuu_lq(X, index)
                for q in np.unique(index):
                    ind = np.where(index == q)
                    self.lengthscale.gradient[q] = tmp[np.ix_(ind[0], ind[0])].sum()
            else:
                raise NotImplementedError
        else: #Kfu or Knm
            #index2 = np.asarray(X2, dtype=np.int)
            #index2 = index2.reshape(index2.size,)
            if hasattr(X2, 'values'):
                X2 = X2.values
            index2 = np.int_(X2[:, 1])
            index2 = index2.reshape(index2.size,)
            X2_flag = index2[0] >= self.output_dim
            if not X_flag and X2_flag:
                index2 -= self.output_dim
            else:
                dL_dK = dL_dK.T #so we obtaing dL_Kfu
                indtemp = index - self.output_dim
                Xtemp = X
                X = X2
                X2 = Xtemp
                index = index2
                index2 = indtemp
            glq, gSdq, gB, gC = self._gkfu(X, index, X2, index2)
            tmp = dL_dK*glq
            for q in np.unique(index2):
                ind = np.where(index2 == q)
                self.lengthscale.gradient[q] = tmp[:, ind].sum()
            tmpB = dL_dK*gB
            tmpC = dL_dK*gC
            tmp = dL_dK*gSdq
            for d in np.unique(index):
                ind = np.where(index == d)
                self.B.gradient[d] = tmpB[ind, :].sum()
                self.C.gradient[d] = tmpC[ind, :].sum()
                for q in np.unique(index2):
                    ind2 = np.where(index2 == q)
                    self.W.gradient[d, q] = tmp[np.ix_(ind[0], ind2[0])].sum()

    def update_gradients_diag(self, dL_dKdiag, X):
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        self.B.gradient = np.zeros(self.B.shape)
        self.C.gradient = np.zeros(self.C.shape)
        self.W.gradient = np.zeros(self.W.shape)
        self.lengthscale.gradient = np.zeros(self.lengthscale.shape)
        index = np.int_(X[:, 1])
        index = index.reshape(index.size,)
        
        glq, gS, gB, gC = self._gkdiag(X, index)
        tmp = dL_dKdiag.reshape(index.size, 1)*glq
        self.lengthscale.gradient = tmp.sum(0)
        #TODO: Avoid the reshape by a priori knowing the shape of dL_dKdiag
        tmpB = dL_dKdiag*gB.reshape(dL_dKdiag.shape)
        tmpC = dL_dKdiag*gC.reshape(dL_dKdiag.shape)
        tmp = dL_dKdiag.reshape(index.size, 1)*gS
        for d in np.unique(index):
            ind = np.where(index == d)
            self.B.gradient[d] = tmpB[ind].sum()
            self.C.gradient[d] = tmpC[ind].sum()
            self.W.gradient[d, :] = tmp[ind].sum(0)

    def gradients_X(self, dL_dK, X, X2=None):
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        index = np.int_(X[:, 1])
        index = index.reshape(index.size,)
        X_flag = index[0] >= self.output_dim
        #If input_dim == 1, use this
        #gX = np.zeros((X.shape[0], 1))
        #Cheat to allow gradient for input_dim==2
        gX = np.zeros(X.shape)
        if X2 is None: #Kuu or Kmm
            if X_flag:
                index -= self.output_dim
                gX[:, 0] = 2.*(dL_dK*self._gkuu_X(X, index)).sum(0)
                return gX
            else:
                raise NotImplementedError
        else: #Kuf or Kmn
            #index2 = np.asarray(X2, dtype=np.int)
            #index2 = index2.reshape(index2.size,)
            if hasattr(X2, 'values'):
                X2 = X2.values
            index2 = np.int_(X2[:, 1])
            index2 = index2.reshape(index2.size,)
            X2_flag = index2[0] >= self.output_dim
            if X_flag and not X2_flag: #gradient of Kuf(Z, X) wrt Z
                index -= self.output_dim
                gX[:, 0] = (dL_dK*self._gkfu_z(X2, index2, X, index).T).sum(1)
                return gX
            else:
                raise NotImplementedError

    #---------------------------------------#
    #             Helper functions          #
    #---------------------------------------#

    #Evaluation of squared exponential for LFM
    def _Kuu(self, X, index):
        index = index.reshape(index.size,)
        t = X[:, 0].reshape(X.shape[0],)
        lq = self.lengthscale.values.reshape(self.rank,)
        lq2 = lq*lq
        #Covariance matrix initialization
        kuu = np.zeros((t.size, t.size))
        #Assign 1. to diagonal terms
        kuu[np.diag_indices(t.size)] = 1.
        #Upper triangular indices
        indtri1, indtri2 = np.triu_indices(t.size, 1)
        #Block Diagonal indices among Upper Triangular indices
        ind = np.where(index[indtri1] == index[indtri2])
        indr = indtri1[ind]
        indc = indtri2[ind]
        r = t[indr] - t[indc]
        r2 = r*r
        #Calculation of  covariance function
        kuu[indr, indc] = np.exp(-r2/lq2[index[indr]])
        #Completation of lower triangular part
        kuu[indc, indr] = kuu[indr, indc]
        return kuu

    #Evaluation of cross-covariance function
    def _Kfu(self, X, index, X2, index2):
        #terms that move along t
        t = X[:, 0].reshape(X.shape[0], 1)
        d = np.unique(index) #Output Indexes
        B = self.B.values[d]
        C = self.C.values[d]
        S = self.W.values[d, :]
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        #Check where wd becomes complex
        wbool = C*C >= 4.*B
        #Output related variables must be column-wise
        C = C.reshape(C.size, 1)
        B = B.reshape(B.size, 1)
        C2 = C*C
        #Input related variables must be row-wise
        z = X2[:, 0].reshape(1, X2.shape[0])
        lq = self.lengthscale.values.reshape((1, self.rank))
        #print np.max(z), np.max(z/lq[0, index2])
        alpha = .5*C

        wbool2 = wbool[index]
        ind2t = np.where(wbool2)
        ind3t = np.where(np.logical_not(wbool2))

        kfu = np.empty((t.size, z.size))

        indD = np.arange(B.size)
        #(1) when wd is real
        if np.any(np.logical_not(wbool)):
            #Indexes of index and t related to (2)
            t1 = t[ind3t]
            ind = index[ind3t]
            #Index transformation
            d = np.asarray(np.where(np.logical_not(wbool))[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            w = .5*np.sqrt(4.*B[d] - C2[d])
            alphad = alpha[d]
            gam = alphad - 1j*w

            #DxQ terms
            Slq = (S[d]/w)*(.5*lq)
            c0 = Slq*np.sqrt(np.pi)
            nu = gam*(.5*lq)
            #1xM terms
            z_lq = z/lq[0, index2]
            #NxQ terms
            t_lq = t1/lq
            #NxM terms
            zt_lq = z_lq - t_lq[:, index2]

            # Upsilon Calculations
            #Using wofz
            tz = t1-z
            fullind = np.ix_(ind, index2)
            zt_lq2 = -zt_lq*zt_lq
            z_lq2 = -z_lq*z_lq
            gamt = -gam[ind]*t1

            upsi = - np.exp(z_lq2 + gamt + np.log(wofz(1j*(z_lq + nu[fullind]))))
            z1 = zt_lq + nu[fullind]
            indv1 = np.where(z1.real >= 0.)
            indv2 = np.where(z1.real < 0.)
            if indv1[0].shape > 0:
                upsi[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1])))
            if indv2[0].shape > 0:
                nua2 = nu[ind[indv2[0]], index2[indv2[1]]]**2
                upsi[indv2] += np.exp(nua2 - gam[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2])))
            upsi[t1[:, 0] == 0., :] = 0.

            #Covariance calculation
            kfu[ind3t] = c0[fullind]*upsi.imag

        #(2) when wd is complex
        if np.any(wbool):
            #Indexes of index and t related to (2)
            t1 = t[ind2t]
            ind = index[ind2t]
            #Index transformation
            d = np.asarray(np.where(wbool)[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            w = .5*np.sqrt(C2[d] - 4.*B[d])
            alphad = alpha[d]
            gam = alphad - w
            gamc = alphad + w
            #DxQ terms
            Slq = S[d]*(lq*.25)
            c0 = -Slq*(np.sqrt(np.pi)/w)
            nu = gam*(lq*.5)
            nuc = gamc*(lq*.5)
            #1xM terms
            z_lq = z/lq[0, index2]
            #NxQ terms
            t_lq = t1/lq[0, index2]
            #NxM terms
            zt_lq = z_lq - t_lq

            # Upsilon Calculations
            tz = t1-z
            z_lq2 = -z_lq*z_lq
            zt_lq2 = -zt_lq*zt_lq
            gamt = -gam[ind]*t1
            gamct = -gamc[ind]*t1
            fullind = np.ix_(ind, index2)
            upsi = np.exp(z_lq2 + gamt + np.log(wofz(1j*(z_lq + nu[fullind])).real))\
                   - np.exp(z_lq2 + gamct + np.log(wofz(1j*(z_lq + nuc[fullind])).real))

            z1 = zt_lq + nu[fullind]
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            if indv1[0].shape > 0:
                upsi[indv1] -= np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                nua2 = nu[ind[indv2[0]], index2[indv2[1]]]**2
                upsi[indv2] -= np.exp(nua2 - gam[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
            z1 = zt_lq + nuc[fullind]
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            if indv1[0].shape > 0:
                upsi[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                nuac2 = nuc[ind[indv2[0]], index2[indv2[1]]]**2
                upsi[indv2] += np.exp(nuac2 - gamc[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
            upsi[t1[:, 0] == 0., :] = 0.

            kfu[ind2t] = c0[np.ix_(ind, index2)]*upsi
        return kfu

    #Gradient of Kuu wrt lengthscale
    def _gkuu_lq(self, X, index):
        t = X[:, 0].reshape(X.shape[0],)
        index = index.reshape(X.shape[0],)
        lq = self.lengthscale.values.reshape(self.rank,)
        lq2 = lq*lq
        #Covariance matrix initialization
        glq = np.zeros((t.size, t.size))
        #Upper triangular indices
        indtri1, indtri2 = np.triu_indices(t.size, 1)
        #Block Diagonal indices among Upper Triangular indices
        ind = np.where(index[indtri1] == index[indtri2])
        indr = indtri1[ind]
        indc = indtri2[ind]
        r = t[indr] - t[indc]
        r2 = r*r
        r2_lq2 = r2/lq2[index[indr]]
        #Calculation of  covariance function
        er2_lq2 = np.exp(-r2_lq2)
        #Gradient wrt lq
        c = 2.*r2_lq2/lq[index[indr]]
        glq[indr, indc] = er2_lq2*c
        #Complete the lower triangular
        glq[indc, indr] = glq[indr, indc]
        return glq

    #Be careful this derivative should be transpose it
    def _gkuu_X(self, X, index): #Diagonal terms are always zero
        t = X[:, 0].reshape(X.shape[0],)
        index = index.reshape(index.size,)
        lq = self.lengthscale.values.reshape(self.rank,)
        lq2 = lq*lq
        #Covariance matrix initialization
        gt = np.zeros((t.size, t.size))
        #Upper triangular indices
        indtri1, indtri2 = np.triu_indices(t.size, 1) #Offset of 1 from the diagonal
        #Block Diagonal indices among Upper Triangular indices
        ind = np.where(index[indtri1] == index[indtri2])
        indr = indtri1[ind]
        indc = indtri2[ind]
        r = t[indr] - t[indc]
        r2 = r*r
        r2_lq2 = r2/(-lq2[index[indr]])
        #Calculation of  covariance function
        er2_lq2 = np.exp(r2_lq2)
        #Gradient wrt t
        c = 2.*r/lq2[index[indr]]
        gt[indr, indc] = er2_lq2*c
        #Complete the lower triangular
        gt[indc, indr] = -gt[indr, indc]
        return gt

    #Gradients for Diagonal Kff
    def _gkdiag(self, X, index):
        index = index.reshape(index.size,)
        #terms that move along t
        d = np.unique(index)
        B = self.B[d].values
        C = self.C[d].values
        S = self.W[d, :].values
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        #Check where wd becomes complex
        wbool = C*C >= 4.*B
        #Output related variables must be column-wise
        t = X[:, 0].reshape(X.shape[0], 1)
        B = B.reshape(B.size, 1)
        C = C.reshape(C.size, 1)
        alpha = .5*C
        C2 = C*C
        S2 = S*S

        wbool2 = wbool[index]
        ind2t = np.where(wbool2)
        ind3t = np.where(np.logical_not(wbool2))

        #Input related variables must be row-wise
        lq = self.lengthscale.values.reshape(1, self.rank)
        lq2 = lq*lq

        gB = np.empty((t.size,))
        gC = np.empty((t.size,))
        glq = np.empty((t.size, lq.size))
        gS = np.empty((t.size, lq.size))

        indD = np.arange(B.size)
        #(1) When wd is real
        if np.any(np.logical_not(wbool)):
            #Indexes of index and t related to (1)
            t1 = t[ind3t]
            ind = index[ind3t]
            #Index transformation
            d = np.asarray(np.where(np.logical_not(wbool))[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            S2lq = S2[d]*(.5*lq)
            c0 = S2lq*np.sqrt(np.pi)

            w = .5*np.sqrt(4.*B[d] - C2[d])
            alphad = alpha[d]
            alpha2 = alphad*alphad
            w2 = w*w
            gam = alphad + 1j*w
            gam2 = gam*gam
            gamc = alphad - 1j*w
            c1 = 0.5/alphad
            c2 = 0.5/gam
            c = c1 - c2

            #DxQ terms
            c0 = c0/w2
            nu = (.5*lq)*gam
            #Nx1 terms
            gamt = -gam[ind]*t1
            gamct = -gamc[ind]*t1
            egamt = np.exp(gamt)
            egamct = np.exp(gamct)
            ec = egamt*c2[ind] - egamct*c1[ind]

            #NxQ terms
            t_lq = t1/lq
            t2_lq2 = -t_lq*t_lq
            t_lq2 = t_lq/lq

            et2_lq2 = np.exp(t2_lq2)
            etlq2gamt = np.exp(t2_lq2 + gamt)

            ##Upsilon calculations
            #Using wofz
            wnu = wofz(1j*nu)
            lwnu = np.log(wnu)
            t2_lq2 = -t_lq*t_lq
            upm = wnu[ind] - np.exp(t2_lq2 + gamt + np.log(wofz(1j*(t_lq + nu[ind]))))
            upm[t1[:, 0] == 0, :] = 0.

            nu2 = nu*nu
            z1 = nu[ind] - t_lq
            indv1 = np.where(z1.real >= 0.)
            indv2 = np.where(z1.real < 0.)
            upv = -np.exp(lwnu[ind] + gamt)
            if indv1[0].shape > 0:
                upv[indv1] += np.exp(t2_lq2[indv1] + np.log(wofz(1j*z1[indv1])))
            if indv2[0].shape > 0:
                upv[indv2] += np.exp(nu2[ind[indv2[0]], indv2[1]] + gamt[indv2[0], 0] + np.log(2.))\
                             - np.exp(t2_lq2[indv2] + np.log(wofz(-1j*z1[indv2])))
            upv[t1[:, 0] == 0, :] = 0.

            #Gradient wrt S
            Slq = S[d]*lq #For grad wrt S
            c0_S = Slq*np.sqrt(np.pi)/w2
            K01 = c0_S*c

            gS[ind3t] = np.real(K01[ind]*upm) + np.real((c0_S[ind]*ec)*upv)

            #For B and C
            upmd = etlq2gamt - 1.
            upvd = egamt - et2_lq2

            # gradient wrt B
            dw_dB = 0.5/w
            dgam_dB = 1j*dw_dB

            Ba1 = c0*(0.5*dgam_dB/gam2 + (0.5*lq2*gam*dgam_dB - 2.*dw_dB/w)*c)
            Ba2_1 = c0*(dgam_dB*(0.5/gam2 - 0.25*lq2) + dw_dB/(w*gam))
            Ba2_2 = c0*dgam_dB/gam
            Ba3 = c0*(-0.25*lq2*gam*dgam_dB/alphad + dw_dB/(w*alphad))
            Ba4_1 = (S2lq*lq)*dgam_dB/w2
            Ba4 = Ba4_1*c

            gB[ind3t] = np.sum(np.real(Ba1[ind]*upm) - np.real(((Ba2_1[ind] + Ba2_2[ind]*t1)*egamt - Ba3[ind]*egamct)*upv)\
                + np.real(Ba4[ind]*upmd) + np.real((Ba4_1[ind]*ec)*upvd), axis=1)

            # gradient wrt C
            dw_dC = - alphad*dw_dB
            dgam_dC = 0.5 + 1j*dw_dC

            Ca1 = c0*(-0.25/alpha2 + 0.5*dgam_dC/gam2 + (0.5*lq2*gam*dgam_dC - 2.*dw_dC/w)*c)
            Ca2_1 = c0*(dgam_dC*(0.5/gam2 - 0.25*lq2) + dw_dC/(w*gam))
            Ca2_2 = c0*dgam_dC/gam
            Ca3_1 = c0*(0.25/alpha2 - 0.25*lq2*gam*dgam_dC/alphad + dw_dC/(w*alphad))
            Ca3_2 = 0.5*c0/alphad
            Ca4_1 = (S2lq*lq)*dgam_dC/w2
            Ca4 = Ca4_1*c

            gC[ind3t] = np.sum(np.real(Ca1[ind]*upm) - np.real(((Ca2_1[ind] + Ca2_2[ind]*t1)*egamt - (Ca3_1[ind] + Ca3_2[ind]*t1)*egamct)*upv)\
                + np.real(Ca4[ind]*upmd) + np.real((Ca4_1[ind]*ec)*upvd), axis=1)

            #Gradient wrt lengthscale
            #DxQ terms
            la = (1./lq + nu*gam)*c0
            la1 = la*c

            c0l = (S2[d]/w2)*lq
            la3 = c0l*c
            gam_2 = .5*gam
            glq[ind3t] = (la1[ind]*upm).real + ((la[ind]*ec)*upv).real\
                + (la3[ind]*(-gam_2[ind] + etlq2gamt*(-t_lq2 + gam_2[ind]))).real\
                + ((c0l[ind]*ec)*(-et2_lq2*(t_lq2 + gam_2[ind]) + egamt*gam_2[ind])).real

        #(2) When w_d is complex
        if np.any(wbool):
            t1 = t[ind2t]
            ind = index[ind2t]
            #Index transformation
            d = np.asarray(np.where(wbool)[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            S2lq = S2[d]*(.25*lq)
            c0 = S2lq*np.sqrt(np.pi)
            w = .5*np.sqrt(C2[d]-4.*B[d])
            w2 = -w*w
            alphad = alpha[d]
            alpha2 = alphad*alphad
            gam = alphad - w
            gamc = alphad + w
            gam2 = gam*gam
            gamc2 = gamc*gamc
            c1 = .5/alphad
            c21 = .5/gam
            c22 = .5/gamc
            c = c1 - c21
            c2 = c1 - c22
            #DxQ terms
            c0 = c0/w2
            nu = .5*lq*gam
            nuc = .5*lq*gamc

            #Nx1 terms
            gamt = -gam[ind]*t1
            gamct = -gamc[ind]*t1
            egamt = np.exp(gamt)
            egamct = np.exp(gamct)
            ec = egamt*c21[ind] - egamct*c1[ind]
            ec2 = egamct*c22[ind] - egamt*c1[ind]
            #NxQ terms
            t_lq = t1/lq
            t2_lq2 = -t_lq*t_lq

            et2_lq2 = np.exp(t2_lq2)
            etlq2gamct = np.exp(t2_lq2 + gamct)
            etlq2gamt = np.exp(t2_lq2 + gamt)

            #Upsilon Calculations using wofz
            t2_lq2 = -t_lq*t_lq #Required when using wofz
            wnu = np.real(wofz(1j*nu))
            lwnu = np.log(wnu)

            upm = wnu[ind] - np.exp(t2_lq2 + gamt + np.log(wofz(1j*(t_lq + nu[ind])).real))
            upm[t1[:, 0] == 0., :] = 0.

            nu2 = nu*nu
            z1 = nu[ind] - t_lq
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            upv = -np.exp(lwnu[ind] + gamt)
            if indv1[0].shape > 0:
                upv[indv1] += np.exp(t2_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                upv[indv2] += np.exp(nu2[ind[indv2[0]], indv2[1]] + gamt[indv2[0], 0] + np.log(2.)) - np.exp(t2_lq2[indv2]\
                    + np.log(wofz(-1j*z1[indv2]).real))
            upv[t1[:, 0] == 0, :] = 0.

            wnuc = wofz(1j*nuc).real
            upmc = wnuc[ind] - np.exp(t2_lq2 + gamct + np.log(wofz(1j*(t_lq + nuc[ind])).real))
            upmc[t1[:, 0] == 0., :] = 0.

            lwnuc = np.log(wnuc)
            nuc2 = nuc*nuc
            z1 = nuc[ind] - t_lq
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            upvc = -np.exp(lwnuc[ind] + gamct)
            if indv1[0].shape > 0:
                upvc[indv1] += np.exp(t2_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                upvc[indv2] += np.exp(nuc2[ind[indv2[0]], indv2[1]] + gamct[indv2[0], 0] + np.log(2.)) - np.exp(t2_lq2[indv2]\
                    + np.log(wofz(-1j*z1[indv2]).real))
            upvc[t1[:, 0] == 0, :] = 0.

            #Gradient wrt S
            #NxQ terms
            c0_S = (S[d]/w2)*(lq*(np.sqrt(np.pi)*.5))

            K011 = c0_S*c
            K012 = c0_S*c2

            gS[ind2t] = K011[ind]*upm + K012[ind]*upmc + (c0_S[ind]*ec)*upv + (c0_S[ind]*ec2)*upvc

            #Is required to cache this, C gradient also required them
            upmd = -1. + etlq2gamt
            upvd = -et2_lq2 + egamt
            upmdc = -1. + etlq2gamct
            upvdc = -et2_lq2 + egamct

            # Gradient wrt B
            dgam_dB = 0.5/w
            dgamc_dB = -dgam_dB

            Ba1 = c0*(0.5*dgam_dB/gam2 + (0.5*lq2*gam*dgam_dB - 1./w2)*c)
            Ba3 = c0*(-0.25*lq2*gam*dgam_dB/alphad + 0.5/(w2*alphad))
            Ba4_1 = (S2lq*lq)*dgam_dB/w2
            Ba4 = Ba4_1*c
            Ba2_1 = c0*(dgam_dB*(0.5/gam2 - 0.25*lq2) + 0.5/(w2*gam))
            Ba2_2 = c0*dgam_dB/gam

            Ba1c = c0*(0.5*dgamc_dB/gamc2 + (0.5*lq2*gamc*dgamc_dB - 1./w2)*c2)
            Ba3c = c0*(-0.25*lq2*gamc*dgamc_dB/alphad + 0.5/(w2*alphad))
            Ba4_1c = (S2lq*lq)*dgamc_dB/w2
            Ba4c = Ba4_1c*c2
            Ba2_1c = c0*(dgamc_dB*(0.5/gamc2 - 0.25*lq2) + 0.5/(w2*gamc))
            Ba2_2c = c0*dgamc_dB/gamc

            gB[ind2t] = np.sum(Ba1[ind]*upm - ((Ba2_1[ind] + Ba2_2[ind]*t1)*egamt - Ba3[ind]*egamct)*upv\
                + Ba4[ind]*upmd + (Ba4_1[ind]*ec)*upvd\
                + Ba1c[ind]*upmc - ((Ba2_1c[ind] + Ba2_2c[ind]*t1)*egamct - Ba3c[ind]*egamt)*upvc\
                + Ba4c[ind]*upmdc + (Ba4_1c[ind]*ec2)*upvdc, axis=1)

            ##Gradient wrt C
            dw_dC = 0.5*alphad/w
            dgam_dC = 0.5 - dw_dC
            dgamc_dC = 0.5 + dw_dC
            S2lq2 = S2lq*lq

            Ca1 = c0*(-0.25/alpha2 + 0.5*dgam_dC/gam2 + (0.5*lq2*gam*dgam_dC + alphad/w2)*c)
            Ca2_1 = c0*(dgam_dC*(0.5/gam2 - 0.25*lq2) - 0.5*alphad/(w2*gam))
            Ca2_2 = c0*dgam_dC/gam
            Ca3_1 = c0*(0.25/alpha2 - 0.25*lq2*gam*dgam_dC/alphad - 0.5/w2)
            Ca3_2 = 0.5*c0/alphad
            Ca4_1 = S2lq2*(dgam_dC/w2)
            Ca4 = Ca4_1*c

            Ca1c = c0*(-0.25/alpha2 + 0.5*dgamc_dC/gamc2 + (0.5*lq2*gamc*dgamc_dC + alphad/w2)*c2)
            Ca2_1c = c0*(dgamc_dC*(0.5/gamc2 - 0.25*lq2) - 0.5*alphad/(w2*gamc))
            Ca2_2c = c0*dgamc_dC/gamc
            Ca3_1c = c0*(0.25/alpha2 - 0.25*lq2*gamc*dgamc_dC/alphad - 0.5/w2)
            Ca3_2c = 0.5*c0/alphad
            Ca4_1c = S2lq2*(dgamc_dC/w2)
            Ca4c = Ca4_1c*c2

            gC[ind2t] = np.sum(Ca1[ind]*upm - ((Ca2_1[ind] + Ca2_2[ind]*t1)*egamt - (Ca3_1[ind] + Ca3_2[ind]*t1)*egamct)*upv\
                + Ca4[ind]*upmd + (Ca4_1[ind]*ec)*upvd\
                + Ca1c[ind]*upmc - ((Ca2_1c[ind] + Ca2_2c[ind]*t1)*egamct - (Ca3_1c[ind] + Ca3_2c[ind]*t1)*egamt)*upvc\
                + Ca4c[ind]*upmdc + (Ca4_1c[ind]*ec2)*upvdc, axis=1)

            #Gradient wrt lengthscale
            #DxQ terms
            la = (1./lq + nu*gam)*c0
            lac = (1./lq + nuc*gamc)*c0
            la1 = la*c
            la1c = lac*c2
            t_lq2 = t_lq/lq
            c0l = (S2[d]/w2)*(.5*lq)
            la3 = c0l*c
            la3c = c0l*c2
            gam_2 = .5*gam
            gamc_2 = .5*gamc
            glq[ind2t] = la1c[ind]*upmc + (lac[ind]*ec2)*upvc\
                + la3c[ind]*(-gamc_2[ind] + etlq2gamct*(-t_lq2 + gamc_2[ind]))\
                + (c0l[ind]*ec2)*(-et2_lq2*(t_lq2 + gamc_2[ind]) + egamct*gamc_2[ind])\
                + la1[ind]*upm + (la[ind]*ec)*upv\
                + la3[ind]*(-gam_2[ind] + etlq2gamt*(-t_lq2 + gam_2[ind]))\
                + (c0l[ind]*ec)*(-et2_lq2*(t_lq2 + gam_2[ind]) + egamt*gam_2[ind])

        return glq, gS, gB, gC

    def _gkfu(self, X, index, Z, index2):
        index = index.reshape(index.size,)
        #TODO: reduce memory usage
        #terms that move along t
        d = np.unique(index)
        B = self.B[d].values
        C = self.C[d].values
        S = self.W[d, :].values
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        #Check where wd becomes complex
        wbool = C*C >= 4.*B
        #t column
        t = X[:, 0].reshape(X.shape[0], 1)
        C = C.reshape(C.size, 1)
        B = B.reshape(B.size, 1)
        C2 = C*C
        #z row
        z = Z[:, 0].reshape(1, Z.shape[0])
        index2 = index2.reshape(index2.size,)
        lq = self.lengthscale.values.reshape((1, self.rank))
        lq2 = lq*lq

        alpha = .5*C

        wbool2 = wbool[index]
        ind2t = np.where(wbool2)
        ind3t = np.where(np.logical_not(wbool2))
        #kfu = np.empty((t.size, z.size))
        glq = np.empty((t.size, z.size))
        gSdq = np.empty((t.size, z.size))
        gB = np.empty((t.size, z.size))
        gC = np.empty((t.size, z.size))

        indD = np.arange(B.size)
        #(1) when wd is real
        if np.any(np.logical_not(wbool)):
            #Indexes of index and t related to (2)
            t1 = t[ind3t]
            ind = index[ind3t]
            #Index transformation
            d = np.asarray(np.where(np.logical_not(wbool))[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            w = .5*np.sqrt(4.*B[d] - C2[d])
            alphad = alpha[d]
            gam = alphad - 1j*w
            gam_2 = .5*gam
            S_w = S[d]/w
            S_wpi = S_w*(.5*np.sqrt(np.pi))
            #DxQ terms
            c0 = S_wpi*lq #lq*Sdq*sqrt(pi)/(2w)
            nu = gam*lq
            nu2 = 1.+.5*(nu*nu)
            nu *= .5

            #1xM terms
            z_lq = z/lq[0, index2]
            z_lq2 = -z_lq*z_lq
            #NxQ terms
            t_lq = t1/lq
            #DxM terms
            gamt = -gam[ind]*t1
            #NxM terms
            zt_lq = z_lq - t_lq[:, index2]
            zt_lq2 = -zt_lq*zt_lq
            ezt_lq2 = -np.exp(zt_lq2)
            ezgamt = np.exp(z_lq2 + gamt)

            # Upsilon calculations
            fullind = np.ix_(ind, index2)
            upsi = - np.exp(z_lq2 + gamt + np.log(wofz(1j*(z_lq + nu[fullind]))))
            tz = t1-z
            z1 = zt_lq + nu[fullind]
            indv1 = np.where(z1.real >= 0.)
            indv2 = np.where(z1.real < 0.)
            if indv1[0].shape > 0:
                upsi[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1])))
            if indv2[0].shape > 0:
                nua2 = nu[ind[indv2[0]], index2[indv2[1]]]**2
                upsi[indv2] += np.exp(nua2 - gam[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2])))
            upsi[t1[:, 0] == 0., :] = 0.

            #Gradient wrt S
            #DxQ term
            Sa1 = lq*(.5*np.sqrt(np.pi))/w

            gSdq[ind3t] = Sa1[np.ix_(ind, index2)]*upsi.imag

            #Gradient wrt lq
            la1 = S_wpi*nu2
            la2 = S_w*lq
            uplq = ezt_lq2*(gam_2[ind])
            uplq += ezgamt*(-z_lq/lq[0, index2] + gam_2[ind])

            glq[ind3t] = (la1[np.ix_(ind, index2)]*upsi).imag
            glq[ind3t] += la2[np.ix_(ind, index2)]*uplq.imag

            #Gradient wrt B
            #Dx1 terms
            dw_dB = .5/w
            dgam_dB = -1j*dw_dB
            #DxQ terms
            Ba1 = -c0*dw_dB/w #DXQ
            Ba2 = c0*dgam_dB #DxQ
            Ba3 = lq2*gam_2 #DxQ
            Ba4 = (dgam_dB*S_w)*(.5*lq2) #DxQ

            gB[ind3t] = ((Ba1[np.ix_(ind, index2)] + Ba2[np.ix_(ind, index2)]*(Ba3[np.ix_(ind, index2)] - (t1-z)))*upsi).imag\
                + (Ba4[np.ix_(ind, index2)]*(ezt_lq2 + ezgamt)).imag

            #Gradient wrt C (it uses some calculations performed in B)
            #Dx1 terms
            dw_dC = -.5*alphad/w
            dgam_dC = 0.5 - 1j*dw_dC
            #DxQ terms
            Ca1 = -c0*dw_dC/w #DXQ
            Ca2 = c0*dgam_dC #DxQ
            Ca4 = (dgam_dC*S_w)*(.5*lq2) #DxQ

            gC[ind3t] = ((Ca1[np.ix_(ind, index2)] + Ca2[np.ix_(ind, index2)]*(Ba3[np.ix_(ind, index2)] - (t1-z)))*upsi).imag\
                + (Ca4[np.ix_(ind, index2)]*(ezt_lq2 + ezgamt)).imag

        #(2) when wd is complex
        if np.any(wbool):
            #Indexes of index and t related to (2)
            t1 = t[ind2t]
            ind = index[ind2t]
            #Index transformation
            d = np.asarray(np.where(wbool)[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            w = .5*np.sqrt(C2[d] - 4.*B[d])
            w2 = w*w
            alphad = alpha[d]
            gam = alphad - w
            gamc = alphad + w
            #DxQ terms
            S_w= -S[d]/w #minus is given by j*j
            S_wpi = S_w*(.25*np.sqrt(np.pi))

            c0 = S_wpi*lq
            gam_2 = .5*gam
            gamc_2 = .5*gamc
            nu = gam*lq
            nuc = gamc*lq
            nu2 = 1.+.5*(nu*nu)
            nuc2 = 1.+.5*(nuc*nuc)
            nu *= .5
            nuc *= .5
            #1xM terms
            z_lq = z/lq[0, index2]
            z_lq2 = -z_lq*z_lq
            #Nx1
            gamt = -gam[ind]*t1
            gamct = -gamc[ind]*t1
            #NxQ terms
            t_lq = t1/lq[0, index2]
            #NxM terms
            zt_lq = z_lq - t_lq
            zt_lq2 = -zt_lq*zt_lq
            ezt_lq2 = -np.exp(zt_lq2)
            ezgamt = np.exp(z_lq2 + gamt)
            ezgamct = np.exp(z_lq2 + gamct)

            # Upsilon calculations
            fullind = np.ix_(ind, index2)
            upsi1 = - np.exp(z_lq2 + gamct + np.log(wofz(1j*(z_lq + nuc[fullind])).real))
            tz = t1-z
            z1 = zt_lq + nuc[fullind]
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            if indv1[0].shape > 0:
                upsi1[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                nuac2 = nuc[ind[indv2[0]], index2[indv2[1]]]**2
                upsi1[indv2] += np.exp(nuac2 - gamc[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
            upsi1[t1[:, 0] == 0., :] = 0.

            upsi2 = - np.exp(z_lq2 + gamt + np.log(wofz(1j*(z_lq + nu[fullind])).real))
            z1 = zt_lq + nu[fullind]
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            if indv1[0].shape > 0:
                upsi2[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                nua2 = nu[ind[indv2[0]], index2[indv2[1]]]**2
                upsi2[indv2] += np.exp(nua2 - gam[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
            upsi2[t1[:, 0] == 0., :] = 0.

            #Gradient wrt lq
            la1 = S_wpi*nu2
            la1c = S_wpi*nuc2
            la2 = S_w*(.5*lq)
            uplq = ezt_lq2*(gamc_2[ind]) + ezgamct*(-z_lq/lq[0, index2] + gamc_2[ind])\
                - ezt_lq2*(gam_2[ind]) - ezgamt*(-z_lq/lq[0, index2] + gam_2[ind])

            glq[ind2t] = la1c[np.ix_(ind, index2)]*upsi1 - la1[np.ix_(ind, index2)]*upsi2\
                + la2[np.ix_(ind, index2)]*uplq


            #Gradient wrt S
            Sa1 = (lq*(-.25*np.sqrt(np.pi)))/w

            gSdq[ind2t] = Sa1[np.ix_(ind, index2)]*(upsi1 - upsi2)

            #Gradient wrt B
            #Dx1 terms
            dgam_dB = .5/w
            dgamc_dB = -dgam_dB
            #DxQ terms
            Ba1 = .5*(c0/w2)
            Ba2 = c0*dgam_dB
            Ba3 = lq2*gam_2
            Ba4 = (dgam_dB*S_w)*(.25*lq2)

            Ba2c = c0*dgamc_dB
            Ba3c = lq2*gamc_2
            Ba4c = (dgamc_dB*S_w)*(.25*lq2)

            gB[ind2t] = (Ba1[np.ix_(ind, index2)] + Ba2c[np.ix_(ind, index2)]*(Ba3c[np.ix_(ind, index2)] - (t1-z)))*upsi1\
                + Ba4c[np.ix_(ind, index2)]*(ezt_lq2 + ezgamct)\
                - (Ba1[np.ix_(ind, index2)] + Ba2[np.ix_(ind, index2)]*(Ba3[np.ix_(ind, index2)] - (t1-z)))*upsi2\
                - Ba4[np.ix_(ind, index2)]*(ezt_lq2 + ezgamt)

            #Gradient wrt C
            #Dx1 terms
            dgam_dC = 0.5 - .5*(alphad/w)
            dgamc_dC = 0.5 + .5*(alphad/w)
            #DxQ terms
            Ca1 = -c0*(.5*alphad/w2)
            Ca2 = c0*dgam_dC
            Ca4 = (dgam_dC*S_w)*(.25*lq2)

            Ca2c = c0*dgamc_dC
            Ca4c = (dgamc_dC*S_w)*(.25*lq2)

            gC[ind2t] = (Ca1[np.ix_(ind, index2)] + Ca2c[np.ix_(ind, index2)]*(Ba3c[np.ix_(ind, index2)] - (t1-z)))*upsi1\
                + Ca4c[np.ix_(ind, index2)]*(ezt_lq2 + ezgamct)\
                - (Ca1[np.ix_(ind, index2)] + Ca2[np.ix_(ind, index2)]*(Ba3[np.ix_(ind, index2)] - (t1-z)))*upsi2\
                - Ca4[np.ix_(ind, index2)]*(ezt_lq2 + ezgamt)

        return glq, gSdq, gB, gC

    #TODO: reduce memory usage
    def _gkfu_z(self, X, index, Z, index2): #Kfu(t,z)
        index = index.reshape(index.size,)
        #terms that move along t
        d = np.unique(index)
        B = self.B[d].values
        C = self.C[d].values
        S = self.W[d, :].values
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        #Check where wd becomes complex
        wbool = C*C >= 4.*B
        wbool2 = wbool[index]
        ind2t = np.where(wbool2)
        ind3t = np.where(np.logical_not(wbool2))
        #t column
        t = X[:, 0].reshape(X.shape[0], 1)
        C = C.reshape(C.size, 1)
        B = B.reshape(B.size, 1)
        C2 = C*C
        alpha = .5*C
        #z row
        z = Z[:, 0].reshape(1, Z.shape[0])
        index2 = index2.reshape(index2.size,)
        lq = self.lengthscale.values.reshape((1, self.rank))

        #kfu = np.empty((t.size, z.size))
        gz = np.empty((t.size, z.size))
        indD = np.arange(B.size)
        #(1) when wd is real
        if np.any(np.logical_not(wbool)):
            #Indexes of index and t related to (2)
            t1 = t[ind3t]
            ind = index[ind3t]
            #TODO: Find a better way of doing this
            #Index transformation
            d = np.asarray(np.where(np.logical_not(wbool))[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            w = .5*np.sqrt(4.*B[d] - C2[d])
            alphad = alpha[d]
            gam = alphad - 1j*w
            S_w = S[d]/w
            S_wpi =S_w*(.5*np.sqrt(np.pi))
            #DxQ terms
            c0 = S_wpi*lq #lq*Sdq*sqrt(pi)/(2w)
            nu = (.5*gam)*lq

            #1xM terms
            z_lq = z/lq[0, index2]
            z_lq2 = -z_lq*z_lq
            #NxQ terms
            t_lq = t1/lq
            #DxM terms
            gamt = -gam[ind]*t1
            #NxM terms
            zt_lq = z_lq - t_lq[:, index2]
            zt_lq2 = -zt_lq*zt_lq
            #ezt_lq2 = -np.exp(zt_lq2)
            ezgamt = np.exp(z_lq2 + gamt)

            # Upsilon calculations
            fullind = np.ix_(ind, index2)
            upsi = - np.exp(z_lq2 + gamt + np.log(wofz(1j*(z_lq + nu[fullind]))))
            tz = t1-z
            z1 = zt_lq + nu[fullind]
            indv1 = np.where(z1.real >= 0.)
            indv2 = np.where(z1.real < 0.)
            if indv1[0].shape > 0:
                upsi[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1])))
            if indv2[0].shape > 0:
                nua2 = nu[ind[indv2[0]], index2[indv2[1]]]**2
                upsi[indv2] += np.exp(nua2 - gam[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2])))
            upsi[t1[:, 0] == 0., :] = 0.

            #Gradient wrt z
            za1 = c0*gam
            #za2 = S_w
            gz[ind3t] = (za1[np.ix_(ind, index2)]*upsi).imag + S_w[np.ix_(ind, index2)]*ezgamt.imag

        #(2) when wd is complex
        if np.any(wbool):
            #Indexes of index and t related to (2)
            t1 = t[ind2t]
            ind = index[ind2t]
            #Index transformation
            d = np.asarray(np.where(wbool)[0])
            indd = indD.copy()
            indd[d] = np.arange(d.size)
            ind = indd[ind]
            #Dx1 terms
            w = .5*np.sqrt(C2[d] - 4.*B[d])
            alphad = alpha[d]
            gam = alphad - w
            gamc = alphad + w
            #DxQ terms
            S_w = -S[d]/w #minus is given by j*j
            S_wpi = S_w*(.25*np.sqrt(np.pi))
            c0 = S_wpi*lq
            nu = .5*gam*lq
            nuc = .5*gamc*lq

            #1xM terms
            z_lq = z/lq[0, index2]
            z_lq2 = -z_lq*z_lq
            #Nx1
            gamt = -gam[ind]*t1
            gamct = -gamc[ind]*t1
            #NxQ terms
            t_lq = t1/lq
            #NxM terms
            zt_lq = z_lq - t_lq[:, index2]
            ezgamt = np.exp(z_lq2 + gamt)
            ezgamct = np.exp(z_lq2 + gamct)

            # Upsilon calculations
            zt_lq2 = -zt_lq*zt_lq
            fullind = np.ix_(ind, index2)
            upsi1 = - np.exp(z_lq2 + gamct + np.log(wofz(1j*(z_lq + nuc[fullind])).real))
            tz = t1-z
            z1 = zt_lq + nuc[fullind]
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            if indv1[0].shape > 0:
                upsi1[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                nuac2 = nuc[ind[indv2[0]], index2[indv2[1]]]**2
                upsi1[indv2] += np.exp(nuac2 - gamc[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
            upsi1[t1[:, 0] == 0., :] = 0.

            upsi2 = - np.exp(z_lq2 + gamt + np.log(wofz(1j*(z_lq + nu[fullind])).real))
            z1 = zt_lq + nu[fullind]
            indv1 = np.where(z1 >= 0.)
            indv2 = np.where(z1 < 0.)
            if indv1[0].shape > 0:
                upsi2[indv1] += np.exp(zt_lq2[indv1] + np.log(wofz(1j*z1[indv1]).real))
            if indv2[0].shape > 0:
                nua2 = nu[ind[indv2[0]], index2[indv2[1]]]**2
                upsi2[indv2] += np.exp(nua2 - gam[ind[indv2[0]], 0]*tz[indv2] + np.log(2.))\
                               - np.exp(zt_lq2[indv2] + np.log(wofz(-1j*z1[indv2]).real))
            upsi2[t1[:, 0] == 0., :] = 0.

            #Gradient wrt z
            za1 = c0*gam
            za1c = c0*gamc
            za2 = .5*S_w
            gz[ind2t] = za1c[np.ix_(ind, index2)]*upsi1 - za1[np.ix_(ind, index2)]*upsi2\
                + za2[np.ix_(ind, index2)]*(ezgamct - ezgamt)
        return gz
