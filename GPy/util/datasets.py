import os
import pylab as pb
import numpy as np
import GPy
import scipy.sparse
import scipy.io
data_path = os.path.join(os.path.dirname(__file__),'datasets')
default_seed =10000

# Some general utilities.
def sample_class(f):
    p = 1./(1.+np.exp(-f))
    c = np.random.binomial(1,p)
    c = np.where(c,1,-1)
    return c

def della_gatta_TRP63_gene_expression(gene_number=None):
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'DellaGattadata.mat'))
    X = np.double(mat_data['timepoints'])
    if gene_number == None:
        Y = mat_data['exprs_tp53_RMA']
    else:
        Y = mat_data['exprs_tp53_RMA'][:, gene_number]
        if len(Y.shape) == 1:
            Y = Y[:, None]
    return {'X': X, 'Y': Y, 'info': "The full gene expression data set from della Gatta et al (http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2413161/) processed by RMA."}


# The data sets
def oil():
    fid = open(os.path.join(data_path, 'oil', 'DataTrn.txt'))
    X = np.fromfile(fid, sep='\t').reshape((-1, 12))
    fid.close()
    fid = open(os.path.join(data_path, 'oil', 'DataTrnLbls.txt'))
    Y = np.fromfile(fid, sep='\t').reshape((-1, 3))*2.-1.
    fid.close()
    return {'X': X, 'Y': Y, 'info': "The oil data from Bishop and James (1993)."}

def oil_100(seed=default_seed):
    np.random.seed(seed=seed)
    data = oil()
    indices = np.random.permutation(1000)
    indices = indices[0:100]
    X = data['X'][indices, :]
    Y = data['Y'][indices, :]
    return {'X': X, 'Y': Y, 'info': "Subsample of the oil data extracting 100 values randomly without replacement."}

def pumadyn(seed=default_seed):
    # Data is variance 1, no need to normalize.
    data = np.loadtxt(os.path.join(data_path, 'pumadyn-32nm/Dataset.data.gz'))
    indices = np.random.permutation(data.shape[0])
    indicesTrain = indices[0:7168]
    indicesTest = indices[7168:-1]
    indicesTrain.sort(axis=0)
    indicesTest.sort(axis=0)
    X = data[indicesTrain, 0:-2]
    Y = data[indicesTrain, -1][:, None]
    Xtest = data[indicesTest, 0:-2]
    Ytest = data[indicesTest, -1][:, None]
    return {'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest, 'info': "The puma robot arm data with 32 inputs. This data is the non linear case with medium noise (pumadyn-32nm). For training 7,168 examples are sampled without replacement."}


def brendan_faces():
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'frey_rawface.mat'))
    Y = mat_data['ff'].T
    return {'Y': Y, 'info': "Face data made available by Brendan Frey"}




def silhouette():
    # Ankur Agarwal and Bill Trigg's silhoutte data.
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'mocap', 'ankur', 'ankurDataPoseSilhouette.mat'))
    inMean = np.mean(mat_data['Y'])
    inScales = np.sqrt(np.var(mat_data['Y']))
    X = mat_data['Y'] - inMean
    X = X/inScales
    Xtest = mat_data['Y_test'] - inMean
    Xtest = Xtest/inScales
    Y = mat_data['Z']
    Ytest = mat_data['Z_test']
    return {'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest, 'info': "Artificial silhouette simulation data developed from Agarwal and Triggs (2004)."}

def stick():
    Y, connect = GPy.util.mocap.load_text_data('run1', data_path)
    Y = Y[0:-1:4, :]
    lbls = 'connect'
    return {'Y': Y, 'connect' : connect, 'info': "Stick man data from Ohio."}


def swiss_roll_1000():
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'swiss_roll_data'))
    Y = mat_data['X_data'][:, 0:1000].transpose()
    return {'Y': Y, 'info': "Subsample of the swiss roll data extracting only the first 1000 values."}

def swiss_roll():
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'swiss_roll_data.mat'))
    Y = mat_data['X_data'][:, 0:3000].transpose()
    return {'Y': Y, 'info': "The first 3,000 points from the swiss roll data of Tennenbaum, de Silva and Langford (2001)."}

def toy_rbf_1d(seed=default_seed):
    np.random.seed(seed=seed)
    numIn = 1
    N = 500
    X = np.random.uniform(low=-1.0, high=1.0, size=(N, numIn))
    X.sort(axis=0)
    rbf = GPy.kern.rbf(numIn, variance=1., lengthscale=np.array((0.25,)))
    white = GPy.kern.white(numIn, variance=1e-2)
    kernel = rbf + white
    K = kernel.K(X)
    y = np.reshape(np.random.multivariate_normal(np.zeros(N), K), (N,1))
    return {'X':X, 'Y':y, 'info': "Samples 500 values of a function from an RBF covariance with very small noise for inputs uniformly distributed between -1 and 1."}

def toy_rbf_1d_50(seed=default_seed):
    np.random.seed(seed=seed)
    data = toy_rbf_1d()
    indices = np.random.permutation(data['X'].shape[0])
    indices = indices[0:50]
    indices.sort(axis=0)
    X = data['X'][indices, :]
    Y = data['Y'][indices, :]
    return {'X': X, 'Y': Y, 'info': "Subsamples the toy_rbf_sample with 50 values randomly taken from the original sample."}


def toy_linear_1d_classification(seed=default_seed):
    np.random.seed(seed=seed)
    x1 = np.random.normal(-3,5,20)
    x2 = np.random.normal(3,5,20)
    X = (np.r_[x1,x2])[:,None]
    return {'X': X, 'Y':  sample_class(2.*X), 'F': 2.*X}

def rogers_girolami_olympics():
    olympic_data = scipy.io.loadmat(os.path.join(data_path, 'olympics.mat'))['male100']
    X = olympic_data[:, 0][:, None]
    Y= olympic_data[:, 1][:, None]
    return {'X': X, 'Y': Y, 'info': "Olympic sprint times for 100 m men from 1896 until 2008. Example is from Rogers and Girolami's First Course in Machine Learning."}
# def movielens_small(partNo=1,seed=default_seed):
#     np.random.seed(seed=seed)

#     fileName = os.path.join(data_path, 'movielens', 'small', 'u' + str(partNo) + '.base')
#     fid = open(fileName)
#     uTrain = np.fromfile(fid, sep='\t', dtype=np.int16).reshape((-1, 4))
#     fid.close()
#     maxVals = np.amax(uTrain, axis=0)
#     numUsers = maxVals[0]
#     numFilms = maxVals[1]
#     numRatings = uTrain.shape[0]

#     Y = scipy.sparse.lil_matrix((numFilms, numUsers), dtype=np.int8)
#     for i in range(numUsers):
#         ind = pb.mlab.find(uTrain[:, 0]==i+1)
#         Y[uTrain[ind, 1]-1, i] = uTrain[ind, 2]

#     fileName = os.path.join(data_path, 'movielens', 'small', 'u' + str(partNo) + '.test')
#     fid = open(fileName)
#     uTest = np.fromfile(fid, sep='\t', dtype=np.int16).reshape((-1, 4))
#     fid.close()
#     numTestRatings = uTest.shape[0]

#     Ytest = scipy.sparse.lil_matrix((numFilms, numUsers), dtype=np.int8)
#     for i in range(numUsers):
#         ind = pb.mlab.find(uTest[:, 0]==i+1)
#         Ytest[uTest[ind, 1]-1, i] = uTest[ind, 2]

#     lbls = np.empty((1,1))
#     lblstest = np.empty((1,1))
#     return {'Y':Y, 'lbls':lbls, 'Ytest':Ytest, 'lblstest':lblstest}




def crescent_data(num_data=200,seed=default_seed):
    """Data set formed from a mixture of four Gaussians. In each class two of the Gaussians are elongated at right angles to each other and offset to form an approximation to the crescent data that is popular in semi-supervised learning as a toy problem.
    :param num_data_part: number of data to be sampled (default is 200).
    :type num_data: int
    :param seed: random seed to be used for data generation.
    :type seed: int"""
    np.random.seed(seed=seed)
    sqrt2 = np.sqrt(2)
    # Rotation matrix
    R = np.array([[sqrt2/2, -sqrt2/2], [sqrt2/2, sqrt2/2]])
    # Scaling matrices
    scales = []
    scales.append(np.array([[3, 0], [0, 1]]))
    scales.append(np.array([[3, 0], [0, 1]]))
    scales.append([[1, 0], [0, 3]])
    scales.append([[1, 0], [0, 3]])
    means = []
    means.append(np.array([4, 4]))
    means.append(np.array([0, 4]))
    means.append(np.array([-4, -4]))
    means.append(np.array([0, -4]))

    Xparts = []
    num_data_part = []
    num_data_total = 0
    for i in range(0, 4):
        num_data_part.append(round(((i+1)*num_data)/4.))
        num_data_part[i] -= num_data_total
        #print num_data_part[i]
        part = np.random.normal(size=(num_data_part[i], 2))
        part = np.dot(np.dot(part, scales[i]), R) + means[i]
        Xparts.append(part)
        num_data_total += num_data_part[i]
    X = np.vstack((Xparts[0], Xparts[1], Xparts[2], Xparts[3]))


    Y = np.vstack((np.ones((num_data_part[0]+num_data_part[1], 1)), -np.ones((num_data_part[2]+num_data_part[3], 1))))
    return {'X':X, 'Y':Y, 'info': "Two separate classes of data formed approximately in the shape of two crescents."}


def creep_data():
    all_data = np.loadtxt(os.path.join(data_path, 'creep', 'taka'))
    y = all_data[:, 1:2].copy()
    features = [0]
    features.extend(range(2, 31))
    X = all_data[:,features].copy()
    return {'X': X, 'y' : y}

