# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .kern import Kern
from GPy.util.multioutput import index_to_slices
from ...core.parameterization import Param
from paramz.transformations import Logexp
import numpy as np

class ODE_UY(Kern):
    def __init__(self, input_dim, variance_U=3., variance_Y=1., lengthscale_U=1., lengthscale_Y=1., active_dims=None, name='ode_uy'):
        assert input_dim ==2, "only defined for 2 input dims"
        super(ODE_UY, self).__init__(input_dim, active_dims, name)

        self.variance_Y = Param('variance_Y', variance_Y, Logexp())
        self.variance_U = Param('variance_U', variance_Y, Logexp())
        self.lengthscale_Y = Param('lengthscale_Y', lengthscale_Y, Logexp())
        self.lengthscale_U = Param('lengthscale_U', lengthscale_Y, Logexp())

        self.link_parameters(self.variance_Y, self.variance_U, self.lengthscale_Y, self.lengthscale_U)

    def K(self, X, X2=None):
        # model :   a * dy/dt + b * y = U
        #lu=sqrt(3)/theta1  ly=1/theta2  theta2= a/b :thetay   sigma2=1/(2ab) :sigmay

        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
            K = np.zeros((X.shape[0], X.shape[0]))
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
            K = np.zeros((X.shape[0], X2.shape[0]))


        #rdist = X[:,0][:,None] - X2[:,0][:,None].T
        rdist = X - X2.T
        ly=1/self.lengthscale_Y
        lu=np.sqrt(3)/self.lengthscale_U
        #iu=self.input_lengthU  #dimention of U
        Vu=self.variance_U
        Vy=self.variance_Y
        #Vy=ly/2
        #stop


        # kernel for kuu  matern3/2
        kuu = lambda dist:Vu * (1 + lu* np.abs(dist)) * np.exp(-lu * np.abs(dist))

        # kernel for kyy
        k1 = lambda dist:np.exp(-ly*np.abs(dist))*(2*lu+ly)/(lu+ly)**2
        k2 = lambda dist:(np.exp(-lu*dist)*(ly-2*lu+lu*ly*dist-lu**2*dist) + np.exp(-ly*dist)*(2*lu-ly) ) / (ly-lu)**2
        k3 = lambda dist:np.exp(-lu*dist) * ( (1+lu*dist)/(lu+ly) + (lu)/(lu+ly)**2 )
        kyy = lambda dist:Vu*Vy*(k1(dist) + k2(dist) + k3(dist))


        # cross covariance function
        kyu3 = lambda dist:np.exp(-lu*dist)/(lu+ly)*(1+lu*(dist+1/(lu+ly)))
        #kyu3 = lambda dist: 0

        k1cros = lambda dist:np.exp(ly*dist)/(lu-ly) * ( 1- np.exp( (lu-ly)*dist) + lu* ( dist*np.exp( (lu-ly)*dist ) + (1- np.exp( (lu-ly)*dist ) ) /(lu-ly)   )    )
        #k1cros = lambda dist:0

        k2cros = lambda dist:np.exp(ly*dist)*( 1/(lu+ly) + lu/(lu+ly)**2 )
        #k2cros = lambda dist:0

        Vyu=np.sqrt(Vy*ly*2)

        # cross covariance kuy
        kuyp = lambda dist:Vu*Vyu*(kyu3(dist))       #t>0 kuy
        kuyn = lambda dist:Vu*Vyu*(k1cros(dist)+k2cros(dist))      #t<0 kuy
        # cross covariance kyu
        kyup = lambda dist:Vu*Vyu*(k1cros(-dist)+k2cros(-dist))    #t>0 kyu
        kyun = lambda dist:Vu*Vyu*(kyu3(-dist))       #t<0 kyu


        for i, s1 in enumerate(slices):
            for j, s2 in enumerate(slices2):
                for ss1 in s1:
                    for ss2 in s2:
                        if i==0 and j==0:
                            K[ss1,ss2] = kuu(np.abs(rdist[ss1,ss2]))
                        elif i==0 and j==1:
                            #K[ss1,ss2]=  np.where(  rdist[ss1,ss2]>0 , kuyp(np.abs(rdist[ss1,ss2])), kuyn(np.abs(rdist[ss1,ss2]) )   )
                            K[ss1,ss2]=  np.where(  rdist[ss1,ss2]>0 , kuyp(rdist[ss1,ss2]), kuyn(rdist[ss1,ss2] )   )
                        elif i==1 and j==1:
                            K[ss1,ss2] = kyy(np.abs(rdist[ss1,ss2]))
                        else:
                            #K[ss1,ss2]= 0
                            #K[ss1,ss2]= np.where(  rdist[ss1,ss2]>0 , kyup(np.abs(rdist[ss1,ss2])), kyun(np.abs(rdist[ss1,ss2]) )   )
                            K[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , kyup(rdist[ss1,ss2]), kyun(rdist[ss1,ss2] )   )
        return K



    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix associated to X."""
        Kdiag = np.zeros(X.shape[0])
        ly=1/self.lengthscale_Y
        lu=np.sqrt(3)/self.lengthscale_U

        Vu = self.variance_U
        Vy=self.variance_Y

        k1 = (2*lu+ly)/(lu+ly)**2
        k2 = (ly-2*lu + 2*lu-ly ) / (ly-lu)**2
        k3 = 1/(lu+ly) + (lu)/(lu+ly)**2

        slices = index_to_slices(X[:,-1])

        for i, ss1 in enumerate(slices):
            for s1 in ss1:
                if i==0:
                    Kdiag[s1]+= self.variance_U
                elif i==1:
                    Kdiag[s1]+= Vu*Vy*(k1+k2+k3)
                else:
                    raise ValueError("invalid input/output index")
        #Kdiag[slices[0][0]]+= self.variance_U   #matern32 diag
        #Kdiag[slices[1][0]]+= self.variance_U*self.variance_Y*(k1+k2+k3)  #  diag
        return Kdiag


    def update_gradients_full(self, dL_dK, X, X2=None):
        """derivative of the covariance matrix with respect to the parameters."""
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
        #rdist = X[:,0][:,None] - X2[:,0][:,None].T

        rdist = X - X2.T
        ly=1/self.lengthscale_Y
        lu=np.sqrt(3)/self.lengthscale_U

        Vu=self.variance_U
        Vy=self.variance_Y
        Vyu = np.sqrt(Vy*ly*2)
        dVdly = 0.5/np.sqrt(ly)*np.sqrt(2*Vy)
        dVdVy = 0.5/np.sqrt(Vy)*np.sqrt(2*ly)

        rd=rdist.shape
        dktheta1 = np.zeros(rd)
        dktheta2 = np.zeros(rd)
        dkUdvar = np.zeros(rd)
        dkYdvar = np.zeros(rd)

        # dk dtheta for UU
        UUdtheta1 = lambda dist: np.exp(-lu* dist)*dist + (-dist)*np.exp(-lu* dist)*(1+lu*dist)
        UUdtheta2 = lambda dist: 0
        #UUdvar = lambda dist: (1 + lu*dist)*np.exp(-lu*dist)
        UUdvar = lambda dist: (1 + lu* np.abs(dist)) * np.exp(-lu * np.abs(dist))

        # dk dtheta for YY

        dk1theta1 = lambda dist: np.exp(-ly*dist)*2*(-lu)/(lu+ly)**3

        dk2theta1 = lambda dist: (1.0)*(
            np.exp(-lu*dist)*dist*(-ly+2*lu-lu*ly*dist+dist*lu**2)*(ly-lu)**(-2) + np.exp(-lu*dist)*(-2+ly*dist-2*dist*lu)*(ly-lu)**(-2)
            +np.exp(-dist*lu)*(ly-2*lu+ly*lu*dist-dist*lu**2)*2*(ly-lu)**(-3)
            +np.exp(-dist*ly)*2*(ly-lu)**(-2)
            +np.exp(-dist*ly)*2*(2*lu-ly)*(ly-lu)**(-3)
            )

        dk3theta1 = lambda dist: np.exp(-dist*lu)*(lu+ly)**(-2)*((2*lu+ly+dist*lu**2+lu*ly*dist)*(-dist-2/(lu+ly))+2+2*lu*dist+ly*dist)

        #dktheta1 = lambda dist: self.variance_U*self.variance_Y*(dk1theta1+dk2theta1+dk3theta1)




        dk1theta2 = lambda dist: np.exp(-ly*dist) * ((lu+ly)**(-2)) * (  (-dist)*(2*lu+ly)  +  1  +  (-2)*(2*lu+ly)/(lu+ly)  )

        dk2theta2 =lambda dist:  1*(
            np.exp(-dist*lu)*(ly-lu)**(-2) * ( 1+lu*dist+(-2)*(ly-2*lu+lu*ly*dist-dist*lu**2)*(ly-lu)**(-1) )
            +np.exp(-dist*ly)*(ly-lu)**(-2) * ( (-dist)*(2*lu-ly) -1+(2*lu-ly)*(-2)*(ly-lu)**(-1) )
            )

        dk3theta2 = lambda dist: np.exp(-dist*lu) * (-3*lu-ly-dist*lu**2-lu*ly*dist)/(lu+ly)**3

        #dktheta2 = lambda dist: self.variance_U*self.variance_Y*(dk1theta2 + dk2theta2 +dk3theta2)

        # kyy kernel

        k1 = lambda dist: np.exp(-ly*dist)*(2*lu+ly)/(lu+ly)**2
        k2 = lambda dist: (np.exp(-lu*dist)*(ly-2*lu+lu*ly*dist-lu**2*dist) + np.exp(-ly*dist)*(2*lu-ly) ) / (ly-lu)**2
        k3 = lambda dist: np.exp(-lu*dist) * ( (1+lu*dist)/(lu+ly) + (lu)/(lu+ly)**2 )
        #dkdvar = k1+k2+k3



        # cross covariance function
        kyu3 = lambda dist:np.exp(-lu*dist)/(lu+ly)*(1+lu*(dist+1/(lu+ly)))

        k1cros = lambda dist:np.exp(ly*dist)/(lu-ly) * ( 1- np.exp( (lu-ly)*dist) + lu* ( dist*np.exp( (lu-ly)*dist ) + (1- np.exp( (lu-ly)*dist ) ) /(lu-ly)   )    )

        k2cros = lambda dist:np.exp(ly*dist)*( 1/(lu+ly) + lu/(lu+ly)**2 )
        # cross covariance kuy
        kuyp = lambda dist:(kyu3(dist))       #t>0 kuy
        kuyn = lambda dist:(k1cros(dist)+k2cros(dist))      #t<0 kuy
        # cross covariance kyu
        kyup = lambda dist:(k1cros(-dist)+k2cros(-dist))    #t>0 kyu
        kyun = lambda dist:(kyu3(-dist))       #t<0 kyu

        # dk dtheta for UY


        dkyu3dtheta2 = lambda dist: np.exp(-lu*dist) * ( (-1)*(lu+ly)**(-2)*(1+lu*dist+lu*(lu+ly)**(-1)) + (lu+ly)**(-1)*(-lu)*(lu+ly)**(-2) )
        dkyu3dtheta1 = lambda dist: np.exp(-lu*dist)*(lu+ly)**(-1)* ( (-dist)*(1+dist*lu+lu*(lu+ly)**(-1)) -\
         (lu+ly)**(-1)*(1+dist*lu+lu*(lu+ly)**(-1)) +dist+(lu+ly)**(-1)-lu*(lu+ly)**(-2) )

        dkcros2dtheta1 = lambda dist: np.exp(ly*dist)* ( -(ly+lu)**(-2) + (ly+lu)**(-2) + (-2)*lu*(lu+ly)**(-3)  )
        dkcros2dtheta2 = lambda dist: np.exp(ly*dist)*dist* ( (ly+lu)**(-1) + lu*(lu+ly)**(-2) ) + \
                                      np.exp(ly*dist)*( -(lu+ly)**(-2) + lu*(-2)*(lu+ly)**(-3)  )

        dkcros1dtheta1 = lambda dist: np.exp(ly*dist)*(     -(lu-ly)**(-2)*(  1-np.exp((lu-ly)*dist) + lu*dist*np.exp((lu-ly)*dist)+ \
          lu*(1-np.exp((lu-ly)*dist))/(lu-ly)  )  +  (lu-ly)**(-1)*(  -np.exp( (lu-ly)*dist )*dist + dist*np.exp( (lu-ly)*dist)+\
          lu*dist**2*np.exp((lu-ly)*dist)+(1-np.exp((lu-ly)*dist))/(lu-ly) - lu*np.exp((lu-ly)*dist)*dist/(lu-ly) -\
          lu*(1-np.exp((lu-ly)*dist))/(lu-ly)**2  )   )

        dkcros1dtheta2 = lambda t: np.exp(ly*t)*t/(lu-ly)*( 1-np.exp((lu-ly)*t) +lu*t*np.exp((lu-ly)*t)+\
            lu*(1-np.exp((lu-ly)*t))/(lu-ly)  )+\
            np.exp(ly*t)/(lu-ly)**2* ( 1-np.exp((lu-ly)*t) +lu*t*np.exp((lu-ly)*t) + lu*( 1-np.exp((lu-ly)*t) )/(lu-ly)  )+\
            np.exp(ly*t)/(lu-ly)*( np.exp((lu-ly)*t)*t -lu*t*t*np.exp((lu-ly)*t) +lu*t*np.exp((lu-ly)*t)/(lu-ly)+\
            lu*( 1-np.exp((lu-ly)*t) )/(lu-ly)**2 )

        dkuypdtheta1 = lambda dist:(dkyu3dtheta1(dist))       #t>0 kuy
        dkuyndtheta1 = lambda dist:(dkcros1dtheta1(dist)+dkcros2dtheta1(dist))      #t<0 kuy
        # cross covariance kyu
        dkyupdtheta1 = lambda dist:(dkcros1dtheta1(-dist)+dkcros2dtheta1(-dist))    #t>0 kyu
        dkyundtheta1 = lambda dist:(dkyu3dtheta1(-dist))       #t<0 kyu

        dkuypdtheta2 = lambda dist:(dkyu3dtheta2(dist))       #t>0 kuy
        dkuyndtheta2 = lambda dist:(dkcros1dtheta2(dist)+dkcros2dtheta2(dist))      #t<0 kuy
        # cross covariance kyu
        dkyupdtheta2 = lambda dist:(dkcros1dtheta2(-dist)+dkcros2dtheta2(-dist))    #t>0 kyu
        dkyundtheta2 = lambda dist:(dkyu3dtheta2(-dist))       #t<0 kyu


        for i, s1 in enumerate(slices):
            for j, s2 in enumerate(slices2):
                for ss1 in s1:
                    for ss2 in s2:
                        if i==0 and j==0:
                            #target[ss1,ss2] = kuu(np.abs(rdist[ss1,ss2]))
                            dktheta1[ss1,ss2] = Vu*UUdtheta1(np.abs(rdist[ss1,ss2]))
                            dktheta2[ss1,ss2] = 0
                            dkUdvar[ss1,ss2] = UUdvar(np.abs(rdist[ss1,ss2]))
                            dkYdvar[ss1,ss2] = 0
                        elif i==0 and j==1:
                            ########target[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , kuyp(np.abs(rdist[ss1,ss2])), kuyn(np.abs(rdist[s1[0],s2[0]]) )   )
                            #np.where(  rdist[ss1,ss2]>0 , kuyp(np.abs(rdist[ss1,ss2])), kuyn(np.abs(rdist[s1[0],s2[0]]) )   )
                            #dktheta1[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , self.variance_U*self.variance_Y*dkcrtheta1(np.abs(rdist[ss1,ss2])) ,self.variance_U*self.variance_Y*(dk1theta1(np.abs(rdist[ss1,ss2]))+dk2theta1(np.abs(rdist[ss1,ss2])))    )
                            #dktheta2[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , self.variance_U*self.variance_Y*dkcrtheta2(np.abs(rdist[ss1,ss2])) ,self.variance_U*self.variance_Y*(dk1theta2(np.abs(rdist[ss1,ss2]))+dk2theta2(np.abs(rdist[ss1,ss2])))    )
                            dktheta1[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*Vyu*dkuypdtheta1(rdist[ss1,ss2]),Vu*Vyu*dkuyndtheta1(rdist[ss1,ss2]) )
                            dkUdvar[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vyu*kuyp(rdist[ss1,ss2]), Vyu* kuyn(rdist[ss1,ss2])  )
                            dktheta2[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*Vyu*dkuypdtheta2(rdist[ss1,ss2])+Vu*dVdly*kuyp(rdist[ss1,ss2]),Vu*Vyu*dkuyndtheta2(rdist[ss1,ss2])+Vu*dVdly*kuyn(rdist[ss1,ss2]) )
                            dkYdvar[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*dVdVy*kuyp(rdist[ss1,ss2]), Vu*dVdVy* kuyn(rdist[ss1,ss2])  )
                        elif i==1 and j==1:
                            #target[ss1,ss2] = kyy(np.abs(rdist[ss1,ss2]))
                            dktheta1[ss1,ss2] = self.variance_U*self.variance_Y*(dk1theta1(np.abs(rdist[ss1,ss2]))+dk2theta1(np.abs(rdist[ss1,ss2]))+dk3theta1(np.abs(rdist[ss1,ss2])))
                            dktheta2[ss1,ss2] = self.variance_U*self.variance_Y*(dk1theta2(np.abs(rdist[ss1,ss2])) + dk2theta2(np.abs(rdist[ss1,ss2])) +dk3theta2(np.abs(rdist[ss1,ss2])))
                            dkUdvar[ss1,ss2] = self.variance_Y*(k1(np.abs(rdist[ss1,ss2]))+k2(np.abs(rdist[ss1,ss2]))+k3(np.abs(rdist[ss1,ss2])) )
                            dkYdvar[ss1,ss2] = self.variance_U*(k1(np.abs(rdist[ss1,ss2]))+k2(np.abs(rdist[ss1,ss2]))+k3(np.abs(rdist[ss1,ss2])) )
                        else:
                            #######target[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , kyup(np.abs(rdist[ss1,ss2])), kyun(np.abs(rdist[s1[0],s2[0]]) )   )
                            #dktheta1[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 ,self.variance_U*self.variance_Y*(dk1theta1(np.abs(rdist[ss1,ss2]))+dk2theta1(np.abs(rdist[ss1,ss2]))) , self.variance_U*self.variance_Y*dkcrtheta1(np.abs(rdist[ss1,ss2])) )
                            #dktheta2[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 ,self.variance_U*self.variance_Y*(dk1theta2(np.abs(rdist[ss1,ss2]))+dk2theta2(np.abs(rdist[ss1,ss2]))) , self.variance_U*self.variance_Y*dkcrtheta2(np.abs(rdist[ss1,ss2])) )
                            dktheta1[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*Vyu*dkyupdtheta1(rdist[ss1,ss2]),Vu*Vyu*dkyundtheta1(rdist[ss1,ss2])  )
                            dkUdvar[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vyu*kyup(rdist[ss1,ss2]),Vyu*kyun(rdist[ss1,ss2]))
                            dktheta2[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*Vyu*dkyupdtheta2(rdist[ss1,ss2])+Vu*dVdly*kyup(rdist[ss1,ss2]),Vu*Vyu*dkyundtheta2(rdist[ss1,ss2])+Vu*dVdly*kyun(rdist[ss1,ss2])  )
                            dkYdvar[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*dVdVy*kyup(rdist[ss1,ss2]), Vu*dVdVy*kyun(rdist[ss1,ss2]))

        #stop
        self.variance_U.gradient = np.sum(dkUdvar * dL_dK)     # Vu

        self.variance_Y.gradient = np.sum(dkYdvar * dL_dK)     # Vy

        self.lengthscale_U.gradient = np.sum(dktheta1*(-np.sqrt(3)*self.lengthscale_U**(-2))* dL_dK)     #lu

        self.lengthscale_Y.gradient = np.sum(dktheta2*(-self.lengthscale_Y**(-2)) * dL_dK)              #ly

