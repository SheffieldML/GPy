import numpy as np
from scipy.stats import norm 
import unittest
import GPy

class LOOTest(unittest.TestCase):
    def test_LOO(self):
        """
        Tests if the log likelihoods for each element of the LOO x-validation
        from the LOO method match the log likelihoods.
        """

        #simple model
        X = np.arange(0,10,1)[:,None]
        Y = np.sin(X/5.0)+np.random.randn(X.shape[0],X.shape[1])*0.1
        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(X,Y,k)
        m.optimize()

        #variables to store the predictions of the GP model.
        preds = []
        variances = []
        for it in range(len(X)):
            #we create a new dataset with all but one element left out
            Xloo = np.delete(X,it,0)
            Yloo = np.delete(Y,it,0)
            #the location we have left out (and that we want to predict)
            predX = X[it].copy() 
            #configure the model using the left-one-out data
            k2 = GPy.kern.RBF(1)
            m2 = GPy.models.GPRegression(Xloo,Yloo,k2)
            #copy the hyperparameters over (these don't change between LOOs)
            m2.param_array[:]=m.param_array[:].copy()
            m2.update_model(True)

            #make a prediction for the left-out data
            pred, var = m2.predict(np.array(predX[:,None]))
            preds.append(pred[0,0])
            variances.append(var[0,0])
        preds = np.array(preds)[:,None]
        variances = np.array(variances)[:,None]


        expected = np.log(norm.pdf(preds,Y,np.sqrt(variances)))
        actual = m.LOO()

        for e,a in zip(expected,actual):
            print "OK"
            assert np.isclose(e,a,atol=0.0001,rtol=0.0001), "%s, %s" % (e,a)
