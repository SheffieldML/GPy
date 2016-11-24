# Copyright (c) 2016, Mike Smith
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import numpy as np

def get_log_likelihood_diff(inputs,data,clusti,clustj,common_kern,noise_scale):
    
    #offset indicies
    ind_offset = np.vstack([np.zeros([N,1]),np.ones([N,1]),nans])

    #independent output indicies
    ind_indpoutputs = np.vstack([np.zeros([N*2,1]),np.ones([N,1]),np.ones([N,1])*2])
    X = np.hstack([X,ind_offset,ind_indpoutputs])
    Y1 = np.sin((X[0:N,0])/10.0)[:,None]
    #Y2 = np.sin((X[0:N,0])/10.0)[:,None]
    Y2 = np.cos((X[0:N,0])/10.0)[:,None]
    Y1 += np.random.randn(Y1.shape[0],Y1.shape[1])*0.1
    Y2 += np.random.randn(Y2.shape[0],Y2.shape[1])*0.1
    Y = np.vstack([Y1,Y2,Y1,Y2])

    #Structure of inputs:
    # actual input | offset_kernel_index | indp_output_index
    #      2.4              0                     0
    #      2.9              0                     0
    #      3.4              1                     0
    #      3.9              1                     0
    #      2.4              nan                   1
    #      2.9              nan                   1
    #      3.4              nan                   2
    #      3.9              nan                   2
    #print X
    #print Y

    #base kernel to explain all time series with
    common_kern = GPy.kern.Matern32(input_dim=1)

    #the offset kernel, that can shift one time series wrt another
    offset_kern = GPy.kern.Offset(common_kern,2,[0])

    #we want to discourage massive offsets, which can achieve good fits by simply moving the two datasets far apart
    offset_kern.offset.set_prior(GPy.priors.Gaussian(0,4.0))

    #our overall kernel contains our offset kernel and two common kernels
    independent_kern = GPy.kern.IndependentOutputs([offset_kern,common_kern.copy(),common_kern.copy()],index_dim=2)

    tiekern = GPy.kern.Tie(independent_kern,3,[['.*lengthscale'],['.*variance']])

    model = GPy.models.GPRegression(X,Y,tiekern)
    model.optimize()




    #base kernel to explain all time series with
    common_kern = GPy.kern.Matern32(input_dim=1)

    #the offset kernel, that can shift one time series wrt another
    offset_kern = GPy.kern.Offset(common_kern,2,[0])

    #we want to discourage massive offsets, which can achieve good fits by simply moving the two datasets far apart
    offset_kern.offset.set_prior(GPy.priors.Gaussian(0,4.0))

    #our overall kernel contains our offset kernel and two common kernels
    independent_kern = GPy.kern.IndependentOutputs([common_kern.copy(),common_kern.copy()],index_dim=1)
    independent_model = GPy.models.GPRegression(indepX,indepY,independent_kern)

    offset_model = GPy.models.GPRegression(offsetX,offsetY,offset_kern)



def cluster(data,inputs,common_kern,noise_scale=1.0,verbose=False):
    """Clusters data
    
    Using the new offset model, this method uses a greedy algorithm to cluster
    the data. It starts with all the data points in separate clusters and tests
    whether combining them increases the overall log-likelihood (LL). It then
    iteratively joins pairs of clusters which cause the greatest increase in
    the LL, until no join increases the LL.
    
    arguments:
    inputs -- the 'X's in a list, one item per cluster
    data -- the 'Y's in a list, one item per cluster
    
    returns a list of the clusters.    
    """
    N=len(data)
    
    #Define a set of N active cluster
    active = []
    for p in range(0,N):
        active.append([p])

    diffloglikes = np.zeros([len(active),len(active)])
    diffloglikes[:] = None
    offsets = np.zeros([len(active),len(active)])

    it = 0
    while True:
        if verbose:
            it +=1
            print("Iteration %d" % it)
                
        for clusti in range(len(active)):        
            for clustj in range(clusti): #count from 0 to clustj-1
                if np.isnan(diffloglikes[clusti,clustj]):
                    diffloglikes[clusti,clustj],offsets[clusti,clustj] = get_log_likelihood_diff(inputs,data,[clusti,clustj],common_kern,noise_scale)

        #np.fill_diagonal(diffloglikes,np.nan)
        
        top = np.unravel_index(np.nanargmax(diffloglikes), diffloglikes.shape)

        if diffloglikes[top[0],top[1]]>0:
            active[top[0]].extend(active[top[1]])
            offset=offsets[top[0],top[1]]
            inputs[top[0]] = np.vstack([inputs[top[0]],inputs[top[1]]-offset])
            data[top[0]] = np.hstack([data[top[0]],data[top[1]]])
            del inputs[top[1]]
            del data[top[1]]
            del active[top[1]]
            
            #None = we need to recalculate
            diffloglikes[:,top[0]] = None
            diffloglikes[top[0],:] = None
            diffloglikes = np.delete(diffloglikes,top[1],0)
            diffloglikes = np.delete(diffloglikes,top[1],1)
        else:
            break
            
    #TODO Add a way to return the offsets applied to all the time series
    return active
