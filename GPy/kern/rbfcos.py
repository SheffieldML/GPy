
# Copyright (c) 2012, James Hensman and Andrew Gordon Wilson
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np

class rbfcos(Kernpart):
    def __init__(self,input_dim,variance=1.,frequencies=None,bandwidths=None,ARD=False):
        self.input_dim = input_dim
        self.name = 'rbfcos'
        if self.input_dim>10:
            print "Warning: the rbfcos kernel requires a lot of memory for high dimensional inputs"
        self.ARD = ARD

        #set the default frequencies and bandwidths, appropriate num_params
        if ARD:
            self.num_params = 2*self.input_dim + 1
            if frequencies is not None:
                frequencies = np.asarray(frequencies)
                assert frequencies.size == self.input_dim, "bad number of frequencies"
            else:
                frequencies = np.ones(self.input_dim)
            if bandwidths is not None:
                bandwidths = np.asarray(bandwidths)
                assert bandwidths.size == self.input_dim, "bad number of bandwidths"
            else:
                bandwidths = np.ones(self.input_dim)
        else:
            self.num_params = 3
            if frequencies is not None:
                frequencies = np.asarray(frequencies)
                assert frequencies.size == 1, "Exactly one frequency needed for non-ARD kernel"
            else:
                frequencies = np.ones(1)

            if bandwidths is not None:
                bandwidths = np.asarray(bandwidths)
                assert bandwidths.size == 1, "Exactly one bandwidth needed for non-ARD kernel"
            else:
                bandwidths = np.ones(1)

        #initialise cache
        self._X, self._X2, self._params = np.empty(shape=(3,1))

        self._set_params(np.hstack((variance,frequencies.flatten(),bandwidths.flatten())))


    def _get_params(self):
        return np.hstack((self.variance,self.frequencies, self.bandwidths))

    def _set_params(self,x):
        assert x.size==(self.num_params)
        if self.ARD:
            self.variance = x[0]
            self.frequencies = x[1:1+self.input_dim]
            self.bandwidths = x[1+self.input_dim:]
        else:
            self.variance, self.frequencies, self.bandwidths = x

    def _get_param_names(self):
        if self.num_params == 3:
            return ['variance','frequency','bandwidth']
        else:
            return ['variance']+['frequency_%i'%i for i in range(self.input_dim)]+['bandwidth_%i'%i for i in range(self.input_dim)]

    def K(self,X,X2,target):
        self._K_computations(X,X2)
        target += self.variance*self._dvar

    def Kdiag(self,X,target):
        np.add(target,self.variance,target)

    def dK_dtheta(self,dL_dK,X,X2,target):
        self._K_computations(X,X2)
        target[0] += np.sum(dL_dK*self._dvar)
        if self.ARD:
            for q in xrange(self.input_dim):
                target[q+1] += -2.*np.pi*self.variance*np.sum(dL_dK*self._dvar*np.tan(2.*np.pi*self._dist[:,:,q]*self.frequencies[q])*self._dist[:,:,q])
                target[q+1+self.input_dim] += -2.*np.pi**2*self.variance*np.sum(dL_dK*self._dvar*self._dist2[:,:,q])
        else:
            target[1] += -2.*np.pi*self.variance*np.sum(dL_dK*self._dvar*np.sum(np.tan(2.*np.pi*self._dist*self.frequencies)*self._dist,-1))
            target[2] += -2.*np.pi**2*self.variance*np.sum(dL_dK*self._dvar*self._dist2.sum(-1))


    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        target[0] += np.sum(dL_dKdiag)

    def dK_dX(self,dL_dK,X,X2,target):
        #TODO!!!
        raise NotImplementedError

    def dKdiag_dX(self,dL_dKdiag,X,target):
        pass

    def _K_computations(self,X,X2):
        if not (np.all(X==self._X) and np.all(X2==self._X2)):
            if X2 is None: X2 = X
            self._X = X.copy()
            self._X2 = X2.copy()

            #do the distances: this will be high memory for large input_dim
            #NB: we don't take the abs of the dist because cos is symmetric
            self._dist = X[:,None,:] - X2[None,:,:]
            self._dist2 = np.square(self._dist)

            #ensure the next section is computed:
            self._params = np.empty(self.num_params)

        if not np.all(self._params == self._get_params()):
            self._params == self._get_params().copy()

            self._rbf_part = np.exp(-2.*np.pi**2*np.sum(self._dist2*self.bandwidths,-1))
            self._cos_part = np.prod(np.cos(2.*np.pi*self._dist*self.frequencies),-1)
            self._dvar = self._rbf_part*self._cos_part

