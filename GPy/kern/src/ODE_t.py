from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import numpy as np
from ...util.multioutput import index_to_slices


class ODE_t(Kern):

        def __init__(self, input_dim, a=1., c=1.,variance_Yt=3., lengthscale_Yt=1.5,ubias =1., active_dims=None, name='ode_st'):
                assert input_dim ==2, "only defined for 2 input dims"
                super(ODE_t, self).__init__(input_dim, active_dims, name)

                self.variance_Yt = Param('variance_Yt', variance_Yt, Logexp())
                self.lengthscale_Yt = Param('lengthscale_Yt', lengthscale_Yt, Logexp())        

                self.a= Param('a', a, Logexp())
                self.c = Param('c', c, Logexp())
                self.ubias = Param('ubias', ubias, Logexp())
                self.link_parameters(self.a, self.c, self.variance_Yt, self.lengthscale_Yt,self.ubias)

        def K(self, X, X2=None):
                """Compute the covariance matrix between X and X2."""        
                X,slices = X[:,:-1],index_to_slices(X[:,-1])
                if X2 is None:
                        X2,slices2 = X,slices
                        K = np.zeros((X.shape[0], X.shape[0]))
                else:
                        X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
                        K = np.zeros((X.shape[0], X2.shape[0]))

                tdist = (X[:,0][:,None] - X2[:,0][None,:])**2
                ttdist = (X[:,0][:,None] - X2[:,0][None,:])
                
                vyt = self.variance_Yt
                
                lyt=1/(2*self.lengthscale_Yt)

                a = -self.a
                c = self.c

                kyy = lambda tdist: np.exp(-lyt*(tdist))

                k1 = lambda tdist: (2*lyt - 4*lyt**2 *(tdist) )

                k4 = lambda tdist: 2*lyt*(tdist)

                for i, s1 in enumerate(slices):
                        for j, s2 in enumerate(slices2):
                                for ss1 in s1:
                                    for ss2 in s2:
                                        if i==0 and j==0:
                                            K[ss1,ss2] = vyt*kyy(tdist[ss1,ss2])
                                        elif i==0 and j==1:
                                            K[ss1,ss2] = (k4(ttdist[ss1,ss2])+1)*vyt*kyy(tdist[ss1,ss2])
                                            #K[ss1,ss2] = (2*lyt*(ttdist[ss1,ss2])+1)*vyt*kyy(tdist[ss1,ss2])
                                        elif i==1 and j==1:
                                            K[ss1,ss2] = ( k1(tdist[ss1,ss2]) + 1. )*vyt* kyy(tdist[ss1,ss2])+self.ubias
                                        else:
                                            K[ss1,ss2] = (-k4(ttdist[ss1,ss2])+1)*vyt*kyy(tdist[ss1,ss2])
                                            #K[ss1,ss2] = (-2*lyt*(ttdist[ss1,ss2])+1)*vyt*kyy(tdist[ss1,ss2])
                #stop
                return K


        def Kdiag(self, X):

                vyt = self.variance_Yt
                lyt = 1./(2*self.lengthscale_Yt)

                a = -self.a
                c = self.c        
                
                k1 = (2*lyt )*vyt
                
                Kdiag = np.zeros(X.shape[0])
                slices = index_to_slices(X[:,-1])

                for i, ss1 in enumerate(slices):
                    for s1 in ss1:
                        if i==0:
                            Kdiag[s1]+= vyt
                        elif i==1:
                            #i=1
                            Kdiag[s1]+= k1 + vyt+self.ubias
                            #Kdiag[s1]+= Vu*Vy*(k1+k2+k3)
                        else:
                            raise ValueError("invalid input/output index")

                return Kdiag

        def update_gradients_full(self, dL_dK, X, X2=None):
                """derivative of the covariance matrix with respect to the parameters."""
                X,slices = X[:,:-1],index_to_slices(X[:,-1])
                if X2 is None:
                    X2,slices2 = X,slices
                    K = np.zeros((X.shape[0], X.shape[0]))
                else:
                    X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])


                vyt = self.variance_Yt

                lyt = 1./(2*self.lengthscale_Yt)

                tdist = (X[:,0][:,None] - X2[:,0][None,:])**2
                ttdist = (X[:,0][:,None] - X2[:,0][None,:])
                #rdist = [tdist,xdist]
                
                rd=tdist.shape[0]

                dka = np.zeros([rd,rd])
                dkc = np.zeros([rd,rd])
                dkYdvart = np.zeros([rd,rd])
                dkYdlent = np.zeros([rd,rd])

                dkdubias = np.zeros([rd,rd])

                kyy = lambda tdist: np.exp(-lyt*(tdist))
                dkyydlyt = lambda tdist: kyy(tdist)*(-tdist)

                k1 = lambda tdist: (2*lyt - 4*lyt**2 * (tdist) )

                k4 = lambda ttdist: 2*lyt*(ttdist)

                dk1dlyt = lambda tdist: 2. - 4*2.*lyt*tdist

                dk4dlyt = lambda ttdist: 2*(ttdist)

                for i, s1 in enumerate(slices):
                    for j, s2 in enumerate(slices2):
                        for ss1 in s1:
                            for ss2 in s2:
                                if i==0 and j==0:
                                    dkYdvart[ss1,ss2] = kyy(tdist[ss1,ss2])
                                    dkYdlent[ss1,ss2] = vyt*dkyydlyt(tdist[ss1,ss2])
                                    dkdubias[ss1,ss2] = 0
                                elif i==0 and j==1:
                                    dkYdvart[ss1,ss2] = (k4(ttdist[ss1,ss2])+1)*kyy(tdist[ss1,ss2])
                                    #dkYdvart[ss1,ss2] = ((2*lyt*ttdist[ss1,ss2])+1)*kyy(tdist[ss1,ss2])
                                    dkYdlent[ss1,ss2] = vyt*dkyydlyt(tdist[ss1,ss2])* (k4(ttdist[ss1,ss2])+1.)+\
                                    vyt*kyy(tdist[ss1,ss2])*(dk4dlyt(ttdist[ss1,ss2]))
                                    #dkYdlent[ss1,ss2] = vyt*dkyydlyt(tdist[ss1,ss2])* (2*lyt*(ttdist[ss1,ss2])+1.)+\
                                    #vyt*kyy(tdist[ss1,ss2])*(2*ttdist[ss1,ss2])
                                    dkdubias[ss1,ss2] = 0
                                elif i==1 and j==1:
                                    dkYdvart[ss1,ss2] = (k1(tdist[ss1,ss2]) + 1. )* kyy(tdist[ss1,ss2])
                                    dkYdlent[ss1,ss2] = vyt*dkyydlyt(tdist[ss1,ss2])*( k1(tdist[ss1,ss2]) + 1. ) +\
                          			vyt*kyy(tdist[ss1,ss2])*dk1dlyt(tdist[ss1,ss2])
                                    dkdubias[ss1,ss2] = 1
                                else:
                                    dkYdvart[ss1,ss2] = (-k4(ttdist[ss1,ss2])+1)*kyy(tdist[ss1,ss2])
                                    #dkYdvart[ss1,ss2] = (-2*lyt*(ttdist[ss1,ss2])+1)*kyy(tdist[ss1,ss2])
                                    dkYdlent[ss1,ss2] = vyt*dkyydlyt(tdist[ss1,ss2])* (-k4(ttdist[ss1,ss2])+1.)+\
                                    vyt*kyy(tdist[ss1,ss2])*(-dk4dlyt(ttdist[ss1,ss2]) )
                                    dkdubias[ss1,ss2] = 0
                                    #dkYdlent[ss1,ss2] = vyt*dkyydlyt(tdist[ss1,ss2])* (-2*lyt*(ttdist[ss1,ss2])+1.)+\
                                    #vyt*kyy(tdist[ss1,ss2])*(-2)*(ttdist[ss1,ss2])
   

                self.variance_Yt.gradient = np.sum(dkYdvart * dL_dK)

                self.lengthscale_Yt.gradient =  np.sum(dkYdlent*(-0.5*self.lengthscale_Yt**(-2)) * dL_dK)

                self.ubias.gradient = np.sum(dkdubias * dL_dK) 
