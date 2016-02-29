# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import numpy as np
from .independent_outputs import index_to_slices


class ODE_st(Kern):
    """
    kernel resultiong from a first order ODE with OU driving GP

    :param input_dim: the number of input dimension, has to be equal to one
    :type input_dim: int
    :param varianceU: variance of the driving GP
    :type varianceU: float
    :param lengthscaleU: lengthscale of the driving GP  (sqrt(3)/lengthscaleU)
    :type lengthscaleU: float
    :param varianceY: 'variance' of the transfer function
    :type varianceY: float
    :param lengthscaleY: 'lengthscale' of the transfer function (1/lengthscaleY)
    :type lengthscaleY: float
    :rtype: kernel object

    """
    
    def __init__(self, input_dim, a=1.,b=1., c=1.,variance_Yx=3.,variance_Yt=1.5, lengthscale_Yx=1.5, lengthscale_Yt=1.5, active_dims=None, name='ode_st'):
        assert input_dim ==3, "only defined for 3 input dims"
        super(ODE_st, self).__init__(input_dim, active_dims, name)

        self.variance_Yt = Param('variance_Yt', variance_Yt, Logexp())
        self.variance_Yx = Param('variance_Yx', variance_Yx, Logexp())
        self.lengthscale_Yt = Param('lengthscale_Yt', lengthscale_Yt, Logexp())
        self.lengthscale_Yx = Param('lengthscale_Yx', lengthscale_Yx, Logexp())        

        self.a= Param('a', a, Logexp())
        self.b = Param('b', b, Logexp())
        self.c = Param('c', c, Logexp())

        self.link_parameters(self.a, self.b, self.c, self.variance_Yt, self.variance_Yx, self.lengthscale_Yt,self.lengthscale_Yx)


    def K(self, X, X2=None):        
    # model :   -a d^2y/dx^2  + b dy/dt + c * y = U
    # kernel Kyy rbf spatiol temporal
    # vyt Y temporal variance  vyx Y spatiol variance   lyt Y temporal lengthscale   lyx Y spatiol lengthscale
    # kernel Kuu doper( doper(Kyy))
    # a   b    c    lyt   lyx    vyx*vyt
        """Compute the covariance matrix between X and X2."""        
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
            K = np.zeros((X.shape[0], X.shape[0]))
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
            K = np.zeros((X.shape[0], X2.shape[0]))


        tdist = (X[:,0][:,None] - X2[:,0][None,:])**2
        xdist = (X[:,1][:,None] - X2[:,1][None,:])**2

        ttdist = (X[:,0][:,None] - X2[:,0][None,:])
        #rdist = [tdist,xdist]
        #dist = np.abs(X - X2.T)
        vyt = self.variance_Yt
        vyx = self.variance_Yx
        
        lyt=1/(2*self.lengthscale_Yt)
        lyx=1/(2*self.lengthscale_Yx)

        a = self.a ## -a is used in the model, negtive diffusion
        b = self.b
        c = self.c

        kyy = lambda tdist,xdist: np.exp(-lyt*(tdist) -lyx*(xdist))

        k1 = lambda tdist: (2*lyt - 4*lyt**2 * (tdist) )

        k2 = lambda xdist: ( 4*lyx**2 * (xdist)  - 2*lyx )

        k3 = lambda xdist: ( 3*4*lyx**2 - 6*8*xdist*lyx**3 + 16*xdist**2*lyx**4 )

        k4 = lambda ttdist: 2*lyt*(ttdist)

        for i, s1 in enumerate(slices):
            for j, s2 in enumerate(slices2):
                for ss1 in s1:
                    for ss2 in s2:
                        if i==0 and j==0:
                            K[ss1,ss2] = vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                        elif i==0 and j==1:
                            K[ss1,ss2] = (-a*k2(xdist[ss1,ss2]) + b*k4(ttdist[ss1,ss2]) + c)*vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            #K[ss1,ss2]=  np.where(  rdist[ss1,ss2]>0 , kuyp(np.abs(rdist[ss1,ss2])), kuyn(np.abs(rdist[ss1,ss2]) )   )
                            #K[ss1,ss2]=  np.where(  rdist[ss1,ss2]>0 , kuyp(rdist[ss1,ss2]), kuyn(rdist[ss1,ss2] )   )
                        elif i==1 and j==1:
                            K[ss1,ss2] = ( b**2*k1(tdist[ss1,ss2]) - 2*a*c*k2(xdist[ss1,ss2]) + a**2*k3(xdist[ss1,ss2]) + c**2 )* vyt*vyx* kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                        else:
                            K[ss1,ss2] = (-a*k2(xdist[ss1,ss2]) - b*k4(ttdist[ss1,ss2]) + c)*vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            #K[ss1,ss2]= np.where(  rdist[ss1,ss2]>0 , kyup(np.abs(rdist[ss1,ss2])), kyun(np.abs(rdist[ss1,ss2]) )   )
                            #K[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , kyup(rdist[ss1,ss2]), kyun(rdist[ss1,ss2] )   )
        
        #stop
        return K

    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix associated to X."""
        vyt = self.variance_Yt
        vyx = self.variance_Yx

        lyt = 1./(2*self.lengthscale_Yt)
        lyx = 1./(2*self.lengthscale_Yx)

        a = self.a
        b = self.b
        c = self.c

        ## dk^2/dtdt'
        k1 = (2*lyt )*vyt*vyx
        ## dk^2/dx^2
        k2 = ( - 2*lyx )*vyt*vyx
        ## dk^4/dx^2dx'^2
        k3 = ( 4*3*lyx**2 )*vyt*vyx


        Kdiag = np.zeros(X.shape[0])
        slices = index_to_slices(X[:,-1])

        for i, ss1 in enumerate(slices):
            for s1 in ss1:
                if i==0:
                    Kdiag[s1]+= vyt*vyx
                elif i==1:
                    #i=1
                    Kdiag[s1]+= b**2*k1 - 2*a*c*k2 + a**2*k3 + c**2*vyt*vyx
                    #Kdiag[s1]+= Vu*Vy*(k1+k2+k3)
                else:
                    raise ValueError("invalid input/output index")

        return Kdiag
        

    def update_gradients_full(self, dL_dK, X, X2=None):
    #def dK_dtheta(self, dL_dK, X, X2, target):
        """derivative of the covariance matrix with respect to the parameters."""
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
            K = np.zeros((X.shape[0], X.shape[0]))
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
        
        vyt = self.variance_Yt
        vyx = self.variance_Yx

        lyt = 1./(2*self.lengthscale_Yt)
        lyx = 1./(2*self.lengthscale_Yx)

        a = self.a
        b = self.b
        c = self.c

        tdist = (X[:,0][:,None] - X2[:,0][None,:])**2
        xdist = (X[:,1][:,None] - X2[:,1][None,:])**2
        #rdist = [tdist,xdist]
        ttdist = (X[:,0][:,None] - X2[:,0][None,:])
        
        rd=tdist.shape[0]

        dka = np.zeros([rd,rd])
        dkb = np.zeros([rd,rd])
        dkc = np.zeros([rd,rd])
        dkYdvart = np.zeros([rd,rd])
        dkYdvarx = np.zeros([rd,rd])
        dkYdlent = np.zeros([rd,rd])
        dkYdlenx = np.zeros([rd,rd])


        kyy = lambda tdist,xdist: np.exp(-lyt*(tdist) -lyx*(xdist))
        #k1 = lambda tdist: (lyt - lyt**2 * (tdist) )
        #k2 = lambda xdist: ( lyx**2 * (xdist)  - lyx )
        #k3 = lambda xdist: ( 3*lyx**2 - 6*xdist*lyx**3 + xdist**2*lyx**4 )
        #k4 = lambda tdist: -lyt*np.sqrt(tdist)

        k1 = lambda tdist: (2*lyt - 4*lyt**2 * (tdist) )

        k2 = lambda xdist: ( 4*lyx**2 * (xdist)  - 2*lyx )

        k3 = lambda xdist: ( 3*4*lyx**2 - 6*8*xdist*lyx**3 + 16*xdist**2*lyx**4 )

        k4 = lambda ttdist: 2*lyt*(ttdist)

        dkyydlyx = lambda tdist,xdist: kyy(tdist,xdist)*(-xdist)
        dkyydlyt = lambda tdist,xdist: kyy(tdist,xdist)*(-tdist)

        dk1dlyt = lambda tdist: 2. - 4*2.*lyt*tdist
        dk2dlyx = lambda xdist: (4.*2.*lyx*xdist -2.)
        dk3dlyx = lambda xdist: (6.*4.*lyx - 18.*8*xdist*lyx**2 + 4*16*xdist**2*lyx**3)

        dk4dlyt = lambda ttdist: 2*(ttdist)

        for i, s1 in enumerate(slices):
            for j, s2 in enumerate(slices2):
                for ss1 in s1:
                    for ss2 in s2:
                        if i==0 and j==0:
                            dka[ss1,ss2] = 0
                            dkb[ss1,ss2] = 0
                            dkc[ss1,ss2] = 0
                            dkYdvart[ss1,ss2] = vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdvarx[ss1,ss2] = vyt*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdlenx[ss1,ss2] = vyt*vyx*dkyydlyx(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdlent[ss1,ss2] = vyt*vyx*dkyydlyt(tdist[ss1,ss2],xdist[ss1,ss2])
                        elif i==0 and j==1:
                            dka[ss1,ss2] = -k2(xdist[ss1,ss2])*vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkb[ss1,ss2] = k4(ttdist[ss1,ss2])*vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkc[ss1,ss2] = vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            #dkYdvart[ss1,ss2] = 0
                            #dkYdvarx[ss1,ss2] = 0
                            #dkYdlent[ss1,ss2] = 0
                            #dkYdlenx[ss1,ss2] = 0
                            dkYdvart[ss1,ss2] = (-a*k2(xdist[ss1,ss2])+b*k4(ttdist[ss1,ss2])+c)*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdvarx[ss1,ss2] = (-a*k2(xdist[ss1,ss2])+b*k4(ttdist[ss1,ss2])+c)*vyt*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdlent[ss1,ss2] = vyt*vyx*dkyydlyt(tdist[ss1,ss2],xdist[ss1,ss2])* (-a*k2(xdist[ss1,ss2])+b*k4(ttdist[ss1,ss2])+c)+\
                            vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])*b*dk4dlyt(ttdist[ss1,ss2])
                            dkYdlenx[ss1,ss2] = vyt*vyx*dkyydlyx(tdist[ss1,ss2],xdist[ss1,ss2])*(-a*k2(xdist[ss1,ss2])+b*k4(ttdist[ss1,ss2])+c)+\
                            vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])*(-a*dk2dlyx(xdist[ss1,ss2]))
                        elif i==1 and j==1:
                            dka[ss1,ss2] = (2*a*k3(xdist[ss1,ss2]) - 2*c*k2(xdist[ss1,ss2]))*vyt*vyx* kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkb[ss1,ss2] = 2*b*k1(tdist[ss1,ss2])*vyt*vyx* kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkc[ss1,ss2] = (-2*a*k2(xdist[ss1,ss2]) + 2*c )*vyt*vyx* kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdvart[ss1,ss2] = ( b**2*k1(tdist[ss1,ss2]) - 2*a*c*k2(xdist[ss1,ss2]) + a**2*k3(xdist[ss1,ss2]) + c**2 )*vyx* kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdvarx[ss1,ss2] = ( b**2*k1(tdist[ss1,ss2]) - 2*a*c*k2(xdist[ss1,ss2]) + a**2*k3(xdist[ss1,ss2]) + c**2 )*vyt* kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdlent[ss1,ss2] = vyt*vyx*dkyydlyt(tdist[ss1,ss2],xdist[ss1,ss2])*( b**2*k1(tdist[ss1,ss2]) - 2*a*c*k2(xdist[ss1,ss2]) + a**2*k3(xdist[ss1,ss2]) + c**2 ) +\
                            vyx*vyt*kyy(tdist[ss1,ss2],xdist[ss1,ss2])*b**2*dk1dlyt(tdist[ss1,ss2])
                            dkYdlenx[ss1,ss2] = vyt*vyx*dkyydlyx(tdist[ss1,ss2],xdist[ss1,ss2])*( b**2*k1(tdist[ss1,ss2]) - 2*a*c*k2(xdist[ss1,ss2]) + a**2*k3(xdist[ss1,ss2]) + c**2 ) +\
                            vyx*vyt*kyy(tdist[ss1,ss2],xdist[ss1,ss2])* (-2*a*c*dk2dlyx(xdist[ss1,ss2]) + a**2*dk3dlyx(xdist[ss1,ss2]) )
                        else:
                            dka[ss1,ss2] = -k2(xdist[ss1,ss2])*vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkb[ss1,ss2] = -k4(ttdist[ss1,ss2])*vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkc[ss1,ss2] = vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            #dkYdvart[ss1,ss2] = 0
                            #dkYdvarx[ss1,ss2] = 0
                            #dkYdlent[ss1,ss2] = 0
                            #dkYdlenx[ss1,ss2] = 0
                            dkYdvart[ss1,ss2] = (-a*k2(xdist[ss1,ss2])-b*k4(ttdist[ss1,ss2])+c)*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdvarx[ss1,ss2] = (-a*k2(xdist[ss1,ss2])-b*k4(ttdist[ss1,ss2])+c)*vyt*kyy(tdist[ss1,ss2],xdist[ss1,ss2])
                            dkYdlent[ss1,ss2] = vyt*vyx*dkyydlyt(tdist[ss1,ss2],xdist[ss1,ss2])* (-a*k2(xdist[ss1,ss2])-b*k4(ttdist[ss1,ss2])+c)+\
                            vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])*(-1)*b*dk4dlyt(ttdist[ss1,ss2])
                            dkYdlenx[ss1,ss2] = vyt*vyx*dkyydlyx(tdist[ss1,ss2],xdist[ss1,ss2])*(-a*k2(xdist[ss1,ss2])-b*k4(ttdist[ss1,ss2])+c)+\
                            vyt*vyx*kyy(tdist[ss1,ss2],xdist[ss1,ss2])*(-a*dk2dlyx(xdist[ss1,ss2])) 

        self.a.gradient = np.sum(dka * dL_dK)  

        self.b.gradient = np.sum(dkb * dL_dK) 

        self.c.gradient = np.sum(dkc * dL_dK)


        self.variance_Yt.gradient = np.sum(dkYdvart * dL_dK)  # Vy

        self.variance_Yx.gradient = np.sum(dkYdvarx * dL_dK)

        self.lengthscale_Yt.gradient = np.sum(dkYdlent*(-0.5*self.lengthscale_Yt**(-2)) * dL_dK)    #ly np.sum(dktheta2*(-self.lengthscale_Y**(-2)) * dL_dK) 

        self.lengthscale_Yx.gradient =  np.sum(dkYdlenx*(-0.5*self.lengthscale_Yx**(-2)) * dL_dK)

