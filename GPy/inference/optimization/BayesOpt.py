import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 


####### Preliminar BO with standad acquisition functions ###############################
# Types of BO
# MM: Maximum (or minimum) mean
# MPI: Maximum posterior improvement  
# MUI: Maximum upper interval

def BOacquisition(X,Y,model,type_bo="MPI",type_objective="max",par_mpi = 0,z_mui=1.96,plot=True,n_eval = 500):

    # Only works in dimension 1
    # Grid where the GP will be evaluated
    X_star 	= np.linspace(min(X)-10,max(X)+10,n_eval)
    X_star 	= X_star[:,None]
        
    # Posterior GP evaluated on the grid
    fest = model.predict(X_star)
    
    # Calculate the acquisition function
    ## IF Maximize
    if type_objective == "max":
        if type_bo == "MPI": # add others here
            acqu =  norm.cdf((fest[0]-(1+par_mpi)*max(fest[0])) / fest[1])
            acqu = acqu/(2*max(acqu))
        if type_bo == "MM":    
        	acqu = fest[0]/max(fest[0])
        	acqu = acqu/(2*max(acqu))
        if type_bo == "MUI":
        	acqu = fest[0]+z_mui*np.sqrt(fest[1])
        	acqu = acqu/(2*max(acqu))
        optimal_loc = np.argmax(acqu)
        x_new = X_star[optimal_loc]
    
    ## IF Minimize    
    if type_objective == "min":
        if type_bo == "MPI":  # add others here
            acqu = 1-norm.cdf((fest[0]-(1+par_mpi)*min(fest[0])) / fest[1])   
            acqu = acqu/(2*max(acqu))
        if type_bo == "MM":    
        	acqu = 1-fest[0]/max(fest[0])
        	acqu = acqu/(2*max(acqu))
       	if type_bo == "MUI":
        	acqu = -fest[0]+z_mui*np.sqrt(fest[1])
        	acqu = acqu/(2*max(acqu))
        optimal_loc = np.argmax(acqu)
        x_new = X_star[optimal_loc]
          
    # Plot GP posterior, collected data and the acquisition function
    if plot:
        plt.plot(X,Y , 'p')
        plt.title('Acquisition function')
        model.plot()
        plt.plot(X_star,  acqu, 'r--')

    
    # Return the point where we shoould take the new sample
    return x_new
    ###############################################################


