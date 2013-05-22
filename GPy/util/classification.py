import numpy as np

def conf_matrix(p,labels,names=['1','0'],threshold=.5,show=True):
    """
    Returns true and false positives in a binary classification problem
    - Column names: true class of the examples
    - Row names: classification assigned by the model

    p: probabilities estimated for observation of belonging to class '1'
    labels: observations' class
    names: classes' names
    threshold: probability value at which the model allocate an element to each class
    show: whether the matrix should be shown or not
    """
    p = p.flatten()
    labels = labels.flatten()
    N = p.size
    C = np.ones(N)
    C[p<threshold] = 0
    True_1 = float((labels - C)[labels-C==0].shape[0] )
    False_1 = float((labels - C)[labels-C==-2].shape[0] )
    True_0 = float((labels - C)[labels-C==-1].shape[0] )
    False_0 = float((labels - C)[labels-C==1].shape[0] )
    if show:
        print (True_1 + True_0 + 0.)/N * 100,'% instances correctly classified'
        print '%-10s|  %-10s|  %-10s| ' % ('',names[0],names[1])
        print '----------|------------|------------|'
        print '%-10s|  %-10s|  %-10s| ' % (names[0],True_1,False_0)
        print '%-10s|  %-10s|  %-10s| ' % (names[1],False_1,True_0)
    return True_1, False_1, True_0, False_0
