import Tango
import pylab as pb
import numpy as np

def gpplot(x,mu,var,edgecol=Tango.coloursHex['darkBlue'],fillcol=Tango.coloursHex['lightBlue'],axes=None,**kwargs):
    if axes is None:
        axes = pb.gca()
    mu = mu.flatten()
    x = x.flatten()

    #here's the mean
    axes.plot(x,mu,color=edgecol,linewidth=2)

    #ensure variance is a vector
    if len(var.shape)>1:
        err = 2*np.sqrt(np.diag(var))
    else:
        err = 2*np.sqrt(var)

    #here's the 2*std box
    kwargs['linewidth']=0.5
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.3
    axes.fill(np.hstack((x,x[::-1])),np.hstack((mu+err,mu[::-1]-err[::-1])),color=fillcol,**kwargs)

    #this is the edge:
    axes.plot(x,mu+err,color=edgecol,linewidth=0.2)
    axes.plot(x,mu-err,color=edgecol,linewidth=0.2)

def removeRightTicks(ax=None):
    ax = ax or pb.gca()
    for i, line in enumerate(ax.get_yticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)
def removeUpperTicks(ax=None):
    ax = ax or pb.gca()
    for i, line in enumerate(ax.get_xticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)
def fewerXticks(ax=None,divideby=2):
    ax = ax or pb.gca()
    ax.set_xticks(ax.get_xticks()[::divideby])

def align_subplots(N,M,xlim=None, ylim=None):
    """make all of the subplots have the same limits, turn off unnecessary ticks"""
    #find sensible xlim,ylim
    if xlim is None:
        xlim = [np.inf,-np.inf]
        for i in range(N*M):
            pb.subplot(N,M,i+1)
            xlim[0] = min(xlim[0],pb.xlim()[0])
            xlim[1] = max(xlim[1],pb.xlim()[1])
    if ylim is None:
        ylim = [np.inf,-np.inf]
        for i in range(N*M):
            pb.subplot(N,M,i+1)
            ylim[0] = min(ylim[0],pb.ylim()[0])
            ylim[1] = max(ylim[1],pb.ylim()[1])

    for i in range(N*M):
        pb.subplot(N,M,i+1)
        pb.xlim(xlim)
        pb.ylim(ylim)
        if (i)%M:
            pb.yticks([])
        else:
            removeRightTicks()
        if i<(M*(N-1)):
            pb.xticks([])
        else:
            removeUpperTicks()


