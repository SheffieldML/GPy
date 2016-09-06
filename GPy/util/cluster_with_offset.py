# Copyright (c) 2016, Mike Smith
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import numpy as np
import sys #so I can print dots

def get_individual_log_likelihood_offset(inputs,data,clust,common_kern):
    """Get the LL of a pair of clusters, but having them independent
    
    Get the log likelihood of a cluster pair.
    
    arguments:
    inputs -- the 'X's in a list, one item per cluster
    data -- the 'Y's in a list, one item per cluster
    clust -- list of clusters to use
    
    returns a tuple:
    log likelihood and the offset
    """
 
    S = data[0].shape[0] #number of time series
    
    #build a new dataset from the clusters, by combining all clusters together
    X = np.zeros([0,1])
    Y = np.zeros([0,S])
    
    #for each person in the cluster,
    #add their inputs and data to the new dataset
    idx = np.zeros([0,1])
    for it,p in enumerate(clust):
        X = np.vstack([X,inputs[p]])
        Y = np.vstack([Y,data[p].T])
        idx = np.r_[idx,np.ones([data[p].shape[1],1])*it]

    X = np.c_[X,idx]
    #print(X)
    k_independent = GPy.kern.IndependentOutputs(common_kern.copy(),index_dim=1)

    m = GPy.models.GPRegression(X,Y,k_independent)
    m.optimize()
    ll=m.log_likelihood()    
    return ll,0

def get_shared_log_likelihood_offset(inputs,data,clust,common_kern):
    """Get the log likelihood of a combined set of clusters, fitting the offsets
    
    arguments:
    inputs -- the 'X's in a list, one item per cluster
    data -- the 'Y's in a list, one item per cluster
    clust -- list of clusters to use
    
    returns a tuple:
    log likelihood and the offset
    """    

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
                      
    m = GPy.models.GPOffsetRegression(X,Y,common_kern.copy())

    #TODO: How to select a sensible prior?
   ## m.offset.set_prior(GPy.priors.Gaussian(0,20)) 
    #TODO: Set a sensible start value for the length scale,
    #make it long to help the offset fit.
    
    m.optimize()
    
    ll = m.log_likelihood()
    offset = m.offset.values[0]
    return ll,offset

def cluster(data,inputs,common_kern,verbose=False):
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

    individualloglikes = np.zeros([len(active),len(active)])
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
            #try combining with each other cluster...
            for clustj in range(clusti): #count from 0 to clustj-1
                temp = [clusti,clustj]
                if np.isnan(sharedloglikes[clusti,clustj]):
                    sharedloglikes[clusti,clustj],sharedoffset[clusti,clustj] = get_shared_log_likelihood_offset(inputs,data,temp,common_kern)    
                    individualloglikes[clusti,clustj], unused_offset = get_individual_log_likelihood_offset(inputs,data,temp,common_kern)

        loglikeimprovement = sharedloglikes - individualloglikes #how much likelihood improves with clustering
        top = np.unravel_index(np.nanargmax(sharedloglikes-individualloglikes), sharedloglikes.shape)

        #if loglikeimprovement.shape[0]<3:
        # #no more clustering to do - this shouldn't happen really unless
        # #we've set the threshold to apply clustering to less than 0
        #    break 
        
        #if theres further clustering to be done...
       # print("SHARED")
       # print(sharedloglikes)
       # print("IND")
       # print(individualloglikes)        
       # print("----")
        if loglikeimprovement[top[0],top[1]]>0:
            #print(loglikeimprovement[top[0],top[1]])
            active[top[0]].extend(active[top[1]])
            offset=sharedoffset[top[0],top[1]]
            inputs[top[0]] = np.vstack([inputs[top[0]],inputs[top[1]]-offset])
            data[top[0]] = np.hstack([data[top[0]],data[top[1]]])
            del inputs[top[1]]
            del data[top[1]]
            del active[top[1]]

            #None = message to say we need to recalculate
            sharedloglikes[:,top[0]] = None 
            sharedloglikes[top[0],:] = None 
            sharedloglikes = np.delete(sharedloglikes,top[1],0)
            sharedloglikes = np.delete(sharedloglikes,top[1],1)
            individualloglikes[:,top[0]] = None 
            individualloglikes[top[0],:] = None 
            individualloglikes = np.delete(individualloglikes,top[1],0)
            individualloglikes = np.delete(individualloglikes,top[1],1)
        else:
            break
            
        #if loglikeimprovement[top[0],top[1]]>0:
        #    print "joined"
        #    print top
        #    print offset
        #    print offsets
        #    print offsets[top[1]]-offsets[top[0]]

    #TODO Add a way to return the offsets applied to all the time series
    return active 

#starttime = time.time()
#active = cluster(data,inputs)
#endtime = time.time()
#print "TOTAL TIME %0.4f" % (endtime-starttime)
