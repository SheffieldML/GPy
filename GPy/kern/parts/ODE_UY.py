# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# model :   a * dy/dt + b * y = U
# inverse lengthscale for U : lu=sqrt(3)/theta1
# inverse lengthscale for Y : ly=1/theta2    theta2= a/b      :thetay  
# variance for Y  :  sigma2=1/(2ab)                           :sigmay  
# variance for the cross variance :  Vyu=1/a   

from kernpart import Kernpart
import numpy as np

def index_to_slices(index):
    """
    take a numpy array of integers (index) and return a  nested list of slices such that the slices describe the start, stop points for each integer in the index. 

    e.g.
    >>> index = np.asarray([0,0,0,1,1,1,2,2,2])
    returns
    >>> [[slice(0,3,None)],[slice(3,6,None)],[slice(6,9,None)]]

    or, a more complicated example
    >>> index = np.asarray([0,0,1,1,0,2,2,2,1,1])
    returns
    >>> [[slice(0,2,None),slice(4,5,None)],[slice(2,4,None),slice(8,10,None)],[slice(5,8,None)]]
    """

    #contruct the return structure
    ind = np.asarray(index,dtype=np.int64)
    ret = [[] for i in range(ind.max()+1)]

    #find the switchpoints
    ind_ = np.hstack((ind,ind[0]+ind[-1]+1))
    switchpoints = np.nonzero(ind_ - np.roll(ind_,+1))[0]

    [ret[ind_i].append(slice(*indexes_i)) for ind_i,indexes_i in zip(ind[switchpoints[:-1]],zip(switchpoints,switchpoints[1:]))]
    return ret

class ODE_UY(Kernpart):
    """
    kernel resultiong from a first order ODE with OU driving GP

    :param input_dim: the number of input dimension, has to be equal to one
    :type input_dim: int
    :param input_lengthU: the number of input U length
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




    def __init__(self, input_dim=2, varianceU=None, varianceY=None, lengthscaleU=None, lengthscaleY=None):
        assert input_dim==2, "Only defined for input_dim = 1"
        self.input_dim = input_dim
        self.num_params = 4
        self.name = 'ODE_UY'

   

        if varianceU is not None:
            varianceU = np.asarray(varianceU)
            assert varianceU.size == 1, "lengthscaleU should be one dimensional"
        else:
            varianceU = 3


        if lengthscaleU is not None:
            lengthscaleU = np.asarray(lengthscaleU)
            assert lengthscaleU.size == 1, "lengthscaleU should be one dimensional"
        else:
            #lengthscaleU = np.ones(1)
            lengthscaleU = 1

        if lengthscaleY is not None:
            lengthscaleY = np.asarray(lengthscaleY)
            assert lengthscaleY.size == 1, "lengthscaleY should be one dimensional"
        else:
            #lengthscaleY = np.ones(1)
            lengthscaleY = 1

        print varianceU
        print varianceY
            # model :   a * dy/dt + b * y = U   x[1]=a   x[3]=b
        #aaa=varianceY               #variance = 1/(2ab)  
        #bbb=lengthscaleY            # lengthscale=a/b   
        #varianceY=1/(2*aaa*bbb) 
        #lengthscaleY=aaa/bbb

        self._set_params(np.hstack((varianceU, varianceY, lengthscaleU,lengthscaleY)))
        print self.varianceU
        print self.varianceY

    def _get_params(self):
        """return the value of the parameters."""
        return np.hstack((self.varianceU,self.varianceY, self.lengthscaleU,self.lengthscaleY))

    def _set_params(self, x):
        """set the value of the parameters."""
        assert x.size == self.num_params

        self.varianceU = x[0]
        self.lengthscaleU = x[2]

        self.varianceY = x[1]   
        self.lengthscaleY = x[3]   
        #stop

    def _get_param_names(self):
        """return parameter names."""
        return ['varianceU','varianceY', 'lengthscaleU', 'lengthscaleY']


    def K(self, X, X2, target):
        """Compute the covariance matrix between X and X2."""
        # model :   a * dy/dt + b * y = U
        #lu=sqrt(3)/theta1  ly=1/theta2  theta2= a/b :thetay   sigma2=1/(2ab) :sigmay   
        
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])


        #rdist = X[:,0][:,None] - X2[:,0][:,None].T
        rdist = X - X2.T
        ly=1/self.lengthscaleY
        lu=np.sqrt(3)/self.lengthscaleU
        #iu=self.input_lengthU  #dimention of U
        Vu=self.varianceU
        Vy=self.varianceY
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
                            target[ss1,ss2] = kuu(np.abs(rdist[ss1,ss2]))
                        elif i==0 and j==1:
                            #target[ss1,ss2]=  np.where(  rdist[ss1,ss2]>0 , kuyp(np.abs(rdist[ss1,ss2])), kuyn(np.abs(rdist[ss1,ss2]) )   )
                            target[ss1,ss2]=  np.where(  rdist[ss1,ss2]>0 , kuyp(rdist[ss1,ss2]), kuyn(rdist[ss1,ss2] )   )
                        elif i==1 and j==1:
                            target[ss1,ss2] = kyy(np.abs(rdist[ss1,ss2]))
                        else:
                            #target[ss1,ss2]= 0
                            #target[ss1,ss2]= np.where(  rdist[ss1,ss2]>0 , kyup(np.abs(rdist[ss1,ss2])), kyun(np.abs(rdist[ss1,ss2]) )   )
                            target[ss1,ss2]= np.where(  rdist[ss1,ss2]>0 , kyup(rdist[ss1,ss2]), kyun(rdist[ss1,ss2] )   )


    def Kdiag(self, X, target):
        """Compute the diagonal of the covariance matrix associated to X."""
        ly=1/self.lengthscaleY
        lu=np.sqrt(3)/self.lengthscaleU

        Vu = self.varianceU
        Vy=self.varianceY

        k1 = (2*lu+ly)/(lu+ly)**2
        k2 = (ly-2*lu + 2*lu-ly ) / (ly-lu)**2 
        k3 = 1/(lu+ly) + (lu)/(lu+ly)**2 

        slices = index_to_slices(X[:,-1])

        for i, ss1 in enumerate(slices):
            for s1 in ss1:
                if i==0:
                    target[s1]+= self.varianceU 
                elif i==1:
                    target[s1]+= Vu*Vy*(k1+k2+k3)
                else:
                    raise ValueError, "invalid input/output index"
        
        #target[slices[0][0]]+= self.varianceU   #matern32 diag
        #target[slices[1][0]]+= self.varianceU*self.varianceY*(k1+k2+k3)  #  diag


    def dK_dtheta(self, dL_dK, X, X2, target):
        """derivative of the covariance matrix with respect to the parameters."""
       
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
        #rdist = X[:,0][:,None] - X2[:,0][:,None].T

        rdist = X - X2.T
        ly=1/self.lengthscaleY
        lu=np.sqrt(3)/self.lengthscaleU

        Vu=self.varianceU
        Vy=self.varianceY
        Vyu = np.sqrt(Vy*ly*2)
        dVdly = 0.5/np.sqrt(ly)*np.sqrt(2*Vy)
        dVdVy = 0.5/np.sqrt(Vy)*np.sqrt(2*ly)

        rd=rdist.shape[0]
        dktheta1 = np.zeros([rd,rd])
        dktheta2 = np.zeros([rd,rd])
        dkUdvar = np.zeros([rd,rd])
        dkYdvar = np.zeros([rd,rd])

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

        #dktheta1 = lambda dist: self.varianceU*self.varianceY*(dk1theta1+dk2theta1+dk3theta1)




        dk1theta2 = lambda dist: np.exp(-ly*dist) * ((lu+ly)**(-2)) * (  (-dist)*(2*lu+ly)  +  1  +  (-2)*(2*lu+ly)/(lu+ly)  )

        dk2theta2 =lambda dist:  1*(
            np.exp(-dist*lu)*(ly-lu)**(-2) * ( 1+lu*dist+(-2)*(ly-2*lu+lu*ly*dist-dist*lu**2)*(ly-lu)**(-1) )
            +np.exp(-dist*ly)*(ly-lu)**(-2) * ( (-dist)*(2*lu-ly) -1+(2*lu-ly)*(-2)*(ly-lu)**(-1) )
            )

        dk3theta2 = lambda dist: np.exp(-dist*lu) * (-3*lu-ly-dist*lu**2-lu*ly*dist)/(lu+ly)**3

        #dktheta2 = lambda dist: self.varianceU*self.varianceY*(dk1theta2 + dk2theta2 +dk3theta2)

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
                            #dktheta1[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , self.varianceU*self.varianceY*dkcrtheta1(np.abs(rdist[ss1,ss2])) ,self.varianceU*self.varianceY*(dk1theta1(np.abs(rdist[ss1,ss2]))+dk2theta1(np.abs(rdist[ss1,ss2])))    )  
                            #dktheta2[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , self.varianceU*self.varianceY*dkcrtheta2(np.abs(rdist[ss1,ss2])) ,self.varianceU*self.varianceY*(dk1theta2(np.abs(rdist[ss1,ss2]))+dk2theta2(np.abs(rdist[ss1,ss2])))    )  
                            dktheta1[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*Vyu*dkuypdtheta1(rdist[ss1,ss2]),Vu*Vyu*dkuyndtheta1(rdist[ss1,ss2]) )
                            dkUdvar[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vyu*kuyp(rdist[ss1,ss2]), Vyu* kuyn(rdist[ss1,ss2])  )  
                            dktheta2[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*Vyu*dkuypdtheta2(rdist[ss1,ss2])+Vu*dVdly*kuyp(rdist[ss1,ss2]),Vu*Vyu*dkuyndtheta2(rdist[ss1,ss2])+Vu*dVdly*kuyn(rdist[ss1,ss2]) )
                            dkYdvar[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*dVdVy*kuyp(rdist[ss1,ss2]), Vu*dVdVy* kuyn(rdist[ss1,ss2])  )  
                        elif i==1 and j==1:
                            #target[ss1,ss2] = kyy(np.abs(rdist[ss1,ss2]))
                            dktheta1[ss1,ss2] = self.varianceU*self.varianceY*(dk1theta1(np.abs(rdist[ss1,ss2]))+dk2theta1(np.abs(rdist[ss1,ss2]))+dk3theta1(np.abs(rdist[ss1,ss2])))
                            dktheta2[ss1,ss2] = self.varianceU*self.varianceY*(dk1theta2(np.abs(rdist[ss1,ss2])) + dk2theta2(np.abs(rdist[ss1,ss2])) +dk3theta2(np.abs(rdist[ss1,ss2])))
                            dkUdvar[ss1,ss2] = self.varianceY*(k1(np.abs(rdist[ss1,ss2]))+k2(np.abs(rdist[ss1,ss2]))+k3(np.abs(rdist[ss1,ss2])) )
                            dkYdvar[ss1,ss2] = self.varianceU*(k1(np.abs(rdist[ss1,ss2]))+k2(np.abs(rdist[ss1,ss2]))+k3(np.abs(rdist[ss1,ss2])) )
                        else:
                            #######target[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , kyup(np.abs(rdist[ss1,ss2])), kyun(np.abs(rdist[s1[0],s2[0]]) )   )
                            #dktheta1[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 ,self.varianceU*self.varianceY*(dk1theta1(np.abs(rdist[ss1,ss2]))+dk2theta1(np.abs(rdist[ss1,ss2]))) , self.varianceU*self.varianceY*dkcrtheta1(np.abs(rdist[ss1,ss2])) )  
                            #dktheta2[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 ,self.varianceU*self.varianceY*(dk1theta2(np.abs(rdist[ss1,ss2]))+dk2theta2(np.abs(rdist[ss1,ss2]))) , self.varianceU*self.varianceY*dkcrtheta2(np.abs(rdist[ss1,ss2])) )  
                            dktheta1[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*Vyu*dkyupdtheta1(rdist[ss1,ss2]),Vu*Vyu*dkyundtheta1(rdist[ss1,ss2])  )
                            dkUdvar[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vyu*kyup(rdist[ss1,ss2]),Vyu*kyun(rdist[ss1,ss2]))  
                            dktheta2[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*Vyu*dkyupdtheta2(rdist[ss1,ss2])+Vu*dVdly*kyup(rdist[ss1,ss2]),Vu*Vyu*dkyundtheta2(rdist[ss1,ss2])+Vu*dVdly*kyun(rdist[ss1,ss2])  )
                            dkYdvar[ss1,ss2] = np.where(  rdist[ss1,ss2]>0 , Vu*dVdVy*kyup(rdist[ss1,ss2]), Vu*dVdVy*kyun(rdist[ss1,ss2])) 
                            
        #stop
        target[0] += np.sum(dkUdvar * dL_dK)     # Vu

        target[1] += np.sum(dkYdvar * dL_dK)     # Vy

        target[2] += np.sum(dktheta1*(-np.sqrt(3)*self.lengthscaleU**(-2))* dL_dK)     #lu

        target[3] += np.sum(dktheta2*(-self.lengthscaleY**(-2)) * dL_dK)              #ly



    # def dKdiag_dtheta(self, dL_dKdiag, X, target):
    #     """derivative of the diagonal of the covariance matrix with respect to the parameters."""
    #     # NB: derivative of diagonal elements wrt lengthscale is 0
    #     target[0] += np.sum(dL_dKdiag)

    # def dK_dX(self, dL_dK, X, X2, target):
    #     """derivative of the covariance matrix with respect to X."""
    #     if X2 is None: X2 = X
    #     dist = np.sqrt(np.sum(np.square((X[:, None, :] - X2[None, :, :]) / self.lengthscale), -1))[:, :, None]
    #     ddist_dX = (X[:, None, :] - X2[None, :, :]) / self.lengthscale ** 2 / np.where(dist != 0., dist, np.inf)
    #     dK_dX = -np.transpose(self.variance * np.exp(-dist) * ddist_dX, (1, 0, 2))
    #     target += np.sum(dK_dX * dL_dK.T[:, :, None], 0)

    # def dKdiag_dX(self, dL_dKdiag, X, target):
    #     pass
