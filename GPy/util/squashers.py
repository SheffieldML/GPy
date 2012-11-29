import numpy as np

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    ex = np.exp(x-x.max(1)[:,None])
    return ex/ex.sum(1)[:,np.newaxis]

def single_softmax(x):
    ex = np.exp(x)
    return ex/ex.sum()


