# Copyright (c) 2014, Cristian Guarnizo.
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import erf, erfcx
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this

class EQ_ODE1(Kern):
    """
    Covariance function for first order differential equation driven by an exponentiated quadratic covariance.

    This outputs of this kernel have the form
    .. math::
       \frac{\text{d}y_j}{\text{d}t} = \sum_{i=1}^R w_{j,i} u_i(t-\delta_j) - d_jy_j(t)

    where :math:`R` is the rank of the system, :math:`w_{j,i}` is the sensitivity of the :math:`j`th output to the :math:`i`th latent function, :math:`d_j` is the decay rate of the :math:`j`th output and :math:`u_i(t)` are independent latent Gaussian processes goverened by an exponentiated quadratic covariance.
    
    :param output_dim: number of outputs driven by latent function.
    :type output_dim: int
    :param W: sensitivities of each output to the latent driving function. 
    :type W: ndarray (output_dim x rank).
    :param rank: If rank is greater than 1 then there are assumed to be a total of rank latent forces independently driving the system, each with identical covariance.
    :type rank: int
    :param decay: decay rates for the first order system. 
    :type decay: array of length output_dim.
    :param delay: delay between latent force and output response.
    :type delay: array of length output_dim.
    :param kappa: diagonal term that allows each latent output to have an independent component to the response.
    :type kappa: array of length output_dim.
    
    .. Note: see first order differential equation examples in GPy.examples.regression for some usage.
    """
    def __init__(self, input_dim=2, output_dim=1, rank=1, W = None, lengthscale=None,  decay=None, active_dims=None, name='eq_ode1'):
        assert input_dim == 2, "only defined for 1 input dims"
        super(EQ_ODE1, self).__init__(input_dim=input_dim, active_dims=active_dims, name=name)

        self.rank = rank
        self.output_dim = output_dim

        if lengthscale is None:
            lengthscale = .5 + np.random.rand(self.rank)
        else:
            lengthscale = np.asarray(lengthscale)
            assert lengthscale.size in [1, self.rank], "Bad number of lengthscales"
            if lengthscale.size != self.rank:
                lengthscale = np.ones(self.rank)*lengthscale
            
        if W is None:
            W = .5*np.random.randn(self.output_dim, self.rank)/np.sqrt(self.rank)
        else:
            assert W.shape == (self.output_dim, self.rank)
        
        if decay is None:
            decay = np.ones(self.output_dim)
        else:
            decay = np.asarray(decay)
            assert decay.size in [1, self.output_dim], "Bad number of decay"
            if decay.size != self.output_dim:
                decay = np.ones(self.output_dim)*decay

#        if kappa is None:
#            self.kappa = np.ones(self.output_dim)
#        else:
#            kappa = np.asarray(kappa)
#            assert kappa.size in [1, self.output_dim], "Bad number of kappa"
#            if decay.size != self.output_dim:
#                decay = np.ones(self.output_dim)*kappa

        #self.kappa = Param('kappa', kappa, Logexp())
        #self.delay = Param('delay', delay, Logexp())
        #self.is_normalized = True
        #self.is_stationary = False
        #self.gaussian_initial = False

        self.lengthscale = Param('lengthscale', lengthscale, Logexp())
        self.decay = Param('decay', decay, Logexp())
        self.W = Param('W', W)
        self.link_parameters(self.lengthscale, self.decay, self.W)

    @Cache_this(limit=3)
    def K(self, X, X2=None):
        #This way is not working, indexes are lost after using k._slice_X
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        index = np.int_(np.round(X[:, 1]))
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
            index2 = np.int_(np.round(X2[:, 1]))
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
        B = self.decay.values[d]
        S = self.W.values[d, :]
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        
        B = B.reshape(B.size, 1)
        #Terms that move along q
        lq = self.lengthscale.values.reshape(1, self.rank)
        S2 = S*S
        kdiag = np.empty((t.size, ))

        #Dx1 terms
        c0 = (S2/B)*((.5*np.sqrt(np.pi))*lq)

        #DxQ terms
        nu = lq*(B*.5)
        nu2 = nu*nu
        #Nx1 terms
        gamt = -2.*B
        gamt = gamt[index]*t

        #NxQ terms
        t_lq = t/lq

        # Upsilon Calculations
        # Using wofz
        #erfnu = erf(nu)
        
        upm = np.exp(nu2[index, :] + lnDifErf( nu[index, :] ,t_lq+nu[index,:] ))
        upm[t[:, 0] == 0, :] = 0.

        
        upv = np.exp(nu2[index, :] + gamt + lnDifErf( -t_lq+nu[index,:], nu[index, :] ) )
        upv[t[:, 0] == 0, :] = 0.

        #Covariance calculation
        #kdiag = np.sum(c0[index, :]*(upm-upv), axis=1)
        kdiag = c0[index, :]*(upm-upv)
        return kdiag

    def update_gradients_full(self, dL_dK, X, X2 = None):
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        self.decay.gradient = np.zeros(self.decay.shape)
        self.W.gradient = np.zeros(self.W.shape)
        self.lengthscale.gradient = np.zeros(self.lengthscale.shape)
        index = np.int_(np.round(X[:, 1]))
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
            index2 = np.int_(np.round(X2[:, 1]))
            index2 = index2.reshape(index2.size,)
            X2_flag = index2[0] >= self.output_dim
            if not X_flag and X2_flag: #Kfu
                index2 -= self.output_dim
            else: #Kuf
                dL_dK = dL_dK.T #so we obtaing dL_Kfu
                indtemp = index - self.output_dim
                Xtemp = X
                X = X2
                X2 = Xtemp
                index = index2
                index2 = indtemp
            glq, gSdq, gB = self._gkfu(X, index, X2, index2)
            tmp = dL_dK*glq
            for q in np.unique(index2):
                ind = np.where(index2 == q)
                self.lengthscale.gradient[q] = tmp[:, ind].sum()
            tmpB = dL_dK*gB
            tmp = dL_dK*gSdq
            for d in np.unique(index):
                ind = np.where(index == d)
                self.decay.gradient[d] = tmpB[ind, :].sum()
                for q in np.unique(index2):
                    ind2 = np.where(index2 == q)
                    self.W.gradient[d, q] = tmp[np.ix_(ind[0], ind2[0])].sum()

    def update_gradients_diag(self, dL_dKdiag, X):
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        self.decay.gradient = np.zeros(self.decay.shape)
        self.W.gradient = np.zeros(self.W.shape)
        self.lengthscale.gradient = np.zeros(self.lengthscale.shape)
        index = np.int_(X[:, 1])
        index = index.reshape(index.size,)
        
        glq, gS, gB = self._gkdiag(X, index)
        if dL_dKdiag.size == X.shape[0]:
            dL_dKdiag = np.reshape(dL_dKdiag, (index.size, 1))
        tmp = dL_dKdiag*glq
        self.lengthscale.gradient = tmp.sum(0)
        tmpB = dL_dKdiag*gB
        tmp = dL_dKdiag*gS
        for d in np.unique(index):
            ind = np.where(index == d)
            self.decay.gradient[d] = tmpB[ind, :].sum()
            self.W.gradient[d, :] = tmp[ind].sum(0)

    def gradients_X(self, dL_dK, X, X2=None):
        #index = np.asarray(X, dtype=np.int)
        #index = index.reshape(index.size,)
        if hasattr(X, 'values'):
            X = X.values
        index = np.int_(np.round(X[:, 1]))
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
            index2 = np.int_(np.round(X2[:, 1]))
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
        #Completion of lower triangular part
        kuu[indc, indr] = kuu[indr, indc]
        return kuu

    #Evaluation of cross-covariance function
    def _Kfu(self, X, index, X2, index2):
        #terms that move along t
        t = X[:, 0].reshape(X.shape[0], 1)
        d = np.unique(index) #Output Indexes
        B = self.decay.values[d]
        S = self.W.values[d, :]
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        #Output related variables must be column-wise
        B = B.reshape(B.size, 1)
        #Input related variables must be row-wise
        z = X2[:, 0].reshape(1, X2.shape[0])
        lq = self.lengthscale.values.reshape((1, self.rank))

        kfu = np.empty((t.size, z.size))

        #DxQ terms
        c0 = S*((.5*np.sqrt(np.pi))*lq)
        nu = B*(.5*lq)
        nu2 = nu**2
        #1xM terms
        z_lq = z/lq[0, index2]
        #NxM terms
        tz = t-z
        tz_lq = tz/lq[0, index2]

        # Upsilon Calculations
        fullind = np.ix_(index, index2)

        upsi = np.exp(nu2[fullind] - B[index]*tz + lnDifErf( -tz_lq + nu[fullind], z_lq+nu[fullind]))
        upsi[t[:, 0] == 0, :] = 0.
        #Covariance calculation
        kfu = c0[fullind]*upsi

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
        B = self.decay[d].values
        S = self.W[d, :].values
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        #Output related variables must be column-wise
        t = X[:, 0].reshape(X.shape[0], 1)
        B = B.reshape(B.size, 1)
        S2 = S*S

        #Input related variables must be row-wise
        lq = self.lengthscale.values.reshape(1, self.rank)

        gB = np.empty((t.size,))
        glq = np.empty((t.size, lq.size))
        gS = np.empty((t.size, lq.size))

        #Dx1 terms
        c0 = S2*lq*np.sqrt(np.pi)

        #DxQ terms
        nu = (.5*lq)*B
        nu2 = nu*nu
        
        #Nx1 terms
        gamt = -B[index]*t
        egamt = np.exp(gamt)
        e2gamt = egamt*egamt

        #NxQ terms
        t_lq = t/lq
        t2_lq2 = -t_lq*t_lq

        etlq2gamt = np.exp(t2_lq2 + gamt) #NXQ

        ##Upsilon calculations
        #erfnu = erf(nu) #TODO: This can be improved

        upm = np.exp(nu2[index, :] + lnDifErf( nu[index, :], t_lq + nu[index, :]) )
        upm[t[:, 0] == 0, :] = 0.

        upv = np.exp(nu2[index, :] + 2.*gamt + lnDifErf(-t_lq + nu[index, :], nu[index, :]) ) #egamt*upv
        upv[t[:, 0] == 0, :] = 0.

        #Gradient wrt S
        c0_S = (S/B)*(lq*np.sqrt(np.pi))

        gS = c0_S[index]*(upm - upv)

        #For B
        CB1 = (.5*lq)**2 - .5/B**2 #DXQ
        lq2_2B = (.5*lq**2)*(S2/B) #DXQ
        CB2 = 2.*etlq2gamt - e2gamt - 1. #NxQ
        
        # gradient wrt B NxZ
        gB = c0[index, :]*(CB1[index, :]*upm - (CB1[index, :] - t/B[index])*upv) + \
        lq2_2B[index, :]*CB2

        #Gradient wrt lengthscale
        #DxQ terms
        c0 = (.5*np.sqrt(np.pi))*(S2/B)*(1.+.5*(lq*B)**2)
        Clq1 = S2*(lq*.5)
        glq = c0[index]*(upm - upv) + Clq1[index]*CB2

        return glq, gS, gB

    def _gkfu(self, X, index, Z, index2):
        index = index.reshape(index.size,)
        #TODO: reduce memory usage
        #terms that move along t
        d = np.unique(index)
        B = self.decay[d].values
        S = self.W[d, :].values

        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]
        #t column
        t = X[:, 0].reshape(X.shape[0], 1)
        B = B.reshape(B.size, 1)
        #z row
        z = Z[:, 0].reshape(1, Z.shape[0])
        index2 = index2.reshape(index2.size,)
        lq = self.lengthscale.values.reshape((1, self.rank))

        #kfu = np.empty((t.size, z.size))
        glq = np.empty((t.size, z.size))
        gSdq = np.empty((t.size, z.size))
        gB = np.empty((t.size, z.size))

        #Dx1 terms
        B_2 = B*.5
        S_pi = S*(.5*np.sqrt(np.pi))
        #DxQ terms
        c0 = S_pi*lq #lq*Sdq*sqrt(pi)
        nu = B*lq*.5
        nu2 = nu*nu

        #1xM terms
        z_lq = z/lq[0, index2]
        
        #NxM terms
        tz = t-z
        tz_lq = tz/lq[0, index2]
        etz_lq2 = -np.exp(-tz_lq*tz_lq)
        ez_lq_Bt = np.exp(-z_lq*z_lq -B[index]*t)
        
        # Upsilon calculations
        fullind = np.ix_(index, index2)
        upsi = np.exp(nu2[fullind] - B[index]*tz + lnDifErf( -tz_lq + nu[fullind], z_lq+nu[fullind] ) )
        upsi[t[:, 0] == 0., :] = 0.

        #Gradient wrt S
        #DxQ term
        Sa1 = lq*(.5*np.sqrt(np.pi))

        gSdq = Sa1[0,index2]*upsi

        #Gradient wrt lq
        la1 = S_pi*(1. + 2.*nu2)
        Slq = S*lq
        uplq = etz_lq2*(tz_lq/lq[0, index2] + B_2[index])
        uplq += ez_lq_Bt*(-z_lq/lq[0, index2] + B_2[index])

        glq = la1[fullind]*upsi
        glq += Slq[fullind]*uplq

        #Gradient wrt B
        Slq = Slq*lq
        nulq = nu*lq
        upBd = etz_lq2 + ez_lq_Bt
        gB = c0[fullind]*(nulq[fullind] - tz)*upsi + .5*Slq[fullind]*upBd

        return glq, gSdq, gB

    #TODO: reduce memory usage
    def _gkfu_z(self, X, index, Z, index2): #Kfu(t,z)
        index = index.reshape(index.size,)
        #terms that move along t
        d = np.unique(index)
        B = self.decay[d].values
        S = self.W[d, :].values
        #Index transformation
        indd = np.arange(self.output_dim)
        indd[d] = np.arange(d.size)
        index = indd[index]

        #t column
        t = X[:, 0].reshape(X.shape[0], 1)
        B = B.reshape(B.size, 1)
        #z row
        z = Z[:, 0].reshape(1, Z.shape[0])
        index2 = index2.reshape(index2.size,)
        lq = self.lengthscale.values.reshape((1, self.rank))

        #kfu = np.empty((t.size, z.size))
        gz = np.empty((t.size, z.size))

        #Dx1 terms
        S_pi =S*(.5*np.sqrt(np.pi))
        #DxQ terms
        #Slq = S*lq
        c0 = S_pi*lq #lq*Sdq*sqrt(pi)
        nu = (.5*lq)*B
        nu2 = nu*nu

        #1xM terms
        z_lq = z/lq[0, index2]
        z_lq2 = -z_lq*z_lq
        #NxQ terms
        t_lq = t/lq
        #NxM terms
        zt_lq = z_lq - t_lq[:, index2]
        zt_lq2 = -zt_lq*zt_lq

        # Upsilon calculations
        fullind = np.ix_(index, index2)
        z2 = z_lq + nu[fullind]
        z1 = z2 - t_lq[:, index2]
        upsi = np.exp(nu2[fullind] - B[index]*(t-z) + lnDifErf(z1,z2) )
        upsi[t[:, 0] == 0., :] = 0.

        #Gradient wrt z
        za1 = c0*B
        #za2 = S_w
        gz = za1[fullind]*upsi + S[fullind]*( np.exp(z_lq2 - B[index]*t) -np.exp(zt_lq2) )

        return gz
        
def lnDifErf(z1,z2):
    #Z2 is always positive
    logdiferf = np.zeros(z1.shape)        
    ind = np.where(z1>0.)
    ind2 = np.where(z1<=0.)
    if ind[0].shape > 0:
        z1i = z1[ind]
        z12 = z1i*z1i
        z2i = z2[ind]
        logdiferf[ind] = -z12 + np.log(erfcx(z1i) - erfcx(z2i)*np.exp(z12-z2i**2))
    
    if ind2[0].shape > 0:
        z1i = z1[ind2]
        z2i = z2[ind2]
        logdiferf[ind2] = np.log(erf(z2i) - erf(z1i))
        
    return logdiferf