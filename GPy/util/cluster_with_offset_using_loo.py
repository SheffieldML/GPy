# Copyright (c) 2016, Mike Smith
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import numpy as np

def add_index_column(inputs,data,clust):

    S = data[0].shape[0] #number of time series
        
    X = np.zeros([0,2]) #notice the extra column, this is for the cluster index
    Y = np.zeros([0,S])
    
    #for each cluster, add their inputs and data to the new
    #dataset. Note we add an index identifying which person is which data point.
    #This is for the offset model to use, to allow it to know which data points
    #to shift.
    for i,p in enumerate(clust):
        idx = i*np.ones([inputs[p].shape[0],1])
        X = np.vstack([X,np.hstack([inputs[p],idx])])
        Y = np.vstack([Y,data[p].T])
    return X,Y
    
def get_individual_log_likelihood_offset(inputs,data,clust,common_kern,noise_scale):
    m = GPy.models.GPRegression(inputs[clust],data[clust].T,common_kern.copy())
    m.Gaussian_noise.fix(noise_scale)
    m.optimize()
    #m.optimize_restarts(2,verbose=False)
    ll = np.sum(m.LOO())
    return ll

def get_shared_log_likelihood_offset(inputs,data,clust,common_kern,noise_scale):
    """Get the log likelihood of a combined set of clusters, fitting the offsets
    
    arguments:
    inputs -- the 'X's in a list, one item per cluster
    data -- the 'Y's in a list, one item per cluster
    clust -- list of clusters to use
    
    returns a tuple:
    log likelihood and the offset
    """    
    X,Y = add_index_column(inputs,data,clust)
    m = GPy.models.GPOffsetRegression(X,Y,common_kern.copy())
    m.offset.set_prior(GPy.priors.Gaussian(0,20)) 
    m.Gaussian_noise.fix(noise_scale)
    try:
        m.optimize()
        #m.optimize_restarts(2,verbose=False)
    except np.linalg.LinAlgError:
        print("Optimization failed: assuming poor fit")
        return -10000,0 #not being able to fit this is probably a sign this isn't a good fit?? TODO
    ll = np.sum(m.LOO())
    offset = m.offset.values[0]
    return ll,offset

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

    individualloglikes = np.zeros([len(active)])
    individualloglikes[:] = None
    sharedloglikes = np.zeros([len(active),len(active)])
    sharedloglikes[:] = None
    sharedoffset = np.zeros([len(active),len(active)])

    it = 0
    while True:
    
        if verbose:
            it +=1
            print("Iteration %d" % it)
        
        #Compute the log-likelihood of each cluster
        for clusti in range(len(active)):
            individualloglikes[clusti] = get_individual_log_likelihood_offset(inputs,data,clusti,common_kern,noise_scale)

            #try combining with each other cluster...        
        for clusti in range(len(active)):        
            for clustj in range(clusti): #count from 0 to clustj-1
                temp = [clusti,clustj]
                if np.isnan(sharedloglikes[clusti,clustj]):
                    sharedloglikes[clusti,clustj],sharedoffset[clusti,clustj] = get_shared_log_likelihood_offset(inputs,data,temp,common_kern,noise_scale)

        #how much likelihood improves with clustering
        loglikeimprovement = sharedloglikes - (individualloglikes[:,None] + individualloglikes[None,:]) 
        np.fill_diagonal(loglikeimprovement,np.nan)
        
        top = np.unravel_index(np.nanargmax(sharedloglikes-individualloglikes), sharedloglikes.shape)
        #print loglikeimprovement
        if loglikeimprovement[top[0],top[1]]>-10:
            active[top[0]].extend(active[top[1]])
            offset=sharedoffset[top[0],top[1]]
            inputs[top[0]] = np.vstack([inputs[top[0]],inputs[top[1]]-offset])
            data[top[0]] = np.hstack([data[top[0]],data[top[1]]])
            del inputs[top[1]]
            del data[top[1]]
            del active[top[1]]

            #None = we need to recalculate
            sharedloglikes[:,top[0]] = None 
            sharedloglikes[top[0],:] = None 
            sharedloglikes = np.delete(sharedloglikes,top[1],0)
            sharedloglikes = np.delete(sharedloglikes,top[1],1)
            individualloglikes[top[0]] = None 
            individualloglikes[top[1]] = None 
            individualloglikes = np.delete(individualloglikes,top[1])
            #print len(individualloglikes)
        else:
            break
            
    #TODO Add a way to return the offsets applied to all the time series
    return active
