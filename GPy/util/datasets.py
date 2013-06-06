import os
import pylab as pb
import numpy as np
import GPy
import scipy.sparse
import scipy.io
import cPickle as pickle
import urllib2 as url

data_path = os.path.join(os.path.dirname(__file__), 'datasets')
default_seed = 10000
neil_url = 'http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/'

def prompt_user():
    # raw_input returns the empty string for "enter"
    yes = set(['yes', 'y'])
    no = set(['no','n'])

    choice = raw_input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'yes', 'y' or 'no', 'n'")
        return prompt_user()

def download_data(dataset_name=None):
    """Helper function which contains the resource locations for each data set in one place"""

    # Note: there may be a better way of doing this. One of the pythonistas will need to take a look. Neil
    data_resources = {'oil': {'urls' : [neil_url + 'oil_data/'],
                              'files' : [['DataTrnLbls.txt', 'DataTrn.txt']],
                              'citation' : 'Bishop, C. M. and G. D. James (1993). Analysis of multiphase flows using dual-energy gamma densitometry and neural networks. Nuclear Instruments and Methods in Physics Research A327, 580-593',
                              'details' : """The three phase oil data used initially for demonstrating the Generative Topographic mapping.""",
                              'agreement' : None},
                      'brendan_faces' : {'url' : ['http://www.cs.nyu.edu/~roweis/data/'],
                                         'files': [['frey_rawface.mat']],
                                         'citation' : 'Frey, B. J., Colmenarez, A and Huang, T. S. Mixtures of Local Linear Subspaces for Face Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 1998, 32-37, June 1998. Computer Society Press, Los Alamitos, CA.',
                                         'details' : """A video of Brendan Frey's face popularized as a benchmark for visualization by the Locally Linear Embedding.""",
                                         'agreement': None}
                      }


    print('Acquiring resource: ' + dataset_name)
    # TODO, check resource is in dictionary!
    dr = data_resources[dataset_name]
    print('Details of data: ')
    print(dr['details'])
    if dr['citation']:
        print('Please cite:')
        print(dr['citation'])
    if dr['agreement']:
        print('You must also agree to the following:')
        print(dr['agreement'])
    print('Do you wish to proceed with the download? [yes/no]')
    if prompt_user()==False:
        return False

    for url, files in zip(dr['urls'], dr['files']):
        for file in files:
            download_resource(url + file)
    return True
                  

        

# Some general utilities.
def sample_class(f):
    p = 1. / (1. + np.exp(-f))
    c = np.random.binomial(1, p)
    c = np.where(c, 1, -1)
    return c

def download_resource(resource, save_name = None, save_file = True, messages = True):
    if messages:
        print "Downloading resource: " , resource, " ... ",
    response = url.urlopen(resource)
    # TODO: Some error checking...
    # ...
    html = response.read()
    response.close()
    if save_file:
        # TODO: Check if already exists...
        # ...
        with open(save_name, "w") as text_file:
            text_file.write("%s"%html)
            if messages:
                print "Done!"
    return html

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

def simulation_BGPLVM():
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'BGPLVMSimulation.mat'))
    Y = np.array(mat_data['Y'], dtype=float)
    S = np.array(mat_data['initS'], dtype=float)
    mu = np.array(mat_data['initMu'], dtype=float)
    return {'Y': Y, 'S': S,
            'mu' : mu,
            'info': "Simulated test dataset generated in MATLAB to compare BGPLVM between python and MATLAB"}


# The data sets
def oil():
    #if download_data('oil'):
    oil_train_file = os.path.join(data_path, 'oil', 'DataTrn.txt')
    oil_trainlbls_file = os.path.join(data_path, 'oil', 'DataTrnLbls.txt')
    fid = open(oil_train_file)
    X = np.fromfile(fid, sep='\t').reshape((-1, 12))
    fid.close()
    fid = open(oil_trainlbls_file)
    Y = np.fromfile(fid, sep='\t').reshape((-1, 3)) * 2. - 1.
    fid.close()
    return {'X': X, 'Y': Y, 'info': "The oil data from Bishop and James (1993)."}
    #else:
    # throw an error
    
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
    X = X / inScales
    Xtest = mat_data['Y_test'] - inMean
    Xtest = Xtest / inScales
    Y = mat_data['Z']
    Ytest = mat_data['Z_test']
    return {'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest, 'info': "Artificial silhouette simulation data developed from Agarwal and Triggs (2004)."}

def stick():
    #if download_data('stick'):
    Y, connect = GPy.util.mocap.load_text_data('run1', data_path)
    Y = Y[0:-1:4, :]
    lbls = 'connect'
    return {'Y': Y, 'connect' : connect, 'info': "Stick man data from Ohio."}
    # else:
    # throw an error.

def swiss_roll_generated(N=1000, sigma=0.0):
    with open(os.path.join(data_path, 'swiss_roll.pickle')) as f:
        data = pickle.load(f)
    Na = data['Y'].shape[0]
    perm = np.random.permutation(np.r_[:Na])[:N]
    Y = data['Y'][perm, :]
    t = data['t'][perm]
    c = data['colors'][perm, :]
    so = np.argsort(t)
    Y = Y[so, :]
    t = t[so]
    c = c[so, :]
    return {'Y':Y, 't':t, 'colors':c}

def swiss_roll_1000():
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'swiss_roll_data'))
    Y = mat_data['X_data'][:, 0:1000].transpose()
    return {'Y': Y, 'info': "Subsample of the swiss roll data extracting only the first 1000 values."}

def swiss_roll(N=3000):
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'swiss_roll_data.mat'))
    Y = mat_data['X_data'][:, 0:N].transpose()
    return {'Y': Y, 'X': mat_data['X_data'], 'info': "The first 3,000 points from the swiss roll data of Tennenbaum, de Silva and Langford (2001)."}

def toy_rbf_1d(seed=default_seed):
    np.random.seed(seed=seed)
    numIn = 1
    N = 500
    X = np.random.uniform(low= -1.0, high=1.0, size=(N, numIn))
    X.sort(axis=0)
    rbf = GPy.kern.rbf(numIn, variance=1., lengthscale=np.array((0.25,)))
    white = GPy.kern.white(numIn, variance=1e-2)
    kernel = rbf + white
    K = kernel.K(X)
    y = np.reshape(np.random.multivariate_normal(np.zeros(N), K), (N, 1))
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
    x1 = np.random.normal(-3, 5, 20)
    x2 = np.random.normal(3, 5, 20)
    X = (np.r_[x1, x2])[:, None]
    return {'X': X, 'Y':  sample_class(2.*X), 'F': 2.*X}

def rogers_girolami_olympics():
    olympic_data = scipy.io.loadmat(os.path.join(data_path, 'olympics.mat'))['male100']
    X = olympic_data[:, 0][:, None]
    Y = olympic_data[:, 1][:, None]
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




def crescent_data(num_data=200, seed=default_seed):
    """Data set formed from a mixture of four Gaussians. In each class two of the Gaussians are elongated at right angles to each other and offset to form an approximation to the crescent data that is popular in semi-supervised learning as a toy problem.
    :param num_data_part: number of data to be sampled (default is 200).
    :type num_data: int
    :param seed: random seed to be used for data generation.
    :type seed: int"""
    np.random.seed(seed=seed)
    sqrt2 = np.sqrt(2)
    # Rotation matrix
    R = np.array([[sqrt2 / 2, -sqrt2 / 2], [sqrt2 / 2, sqrt2 / 2]])
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
        num_data_part.append(round(((i + 1) * num_data) / 4.))
        num_data_part[i] -= num_data_total
        # print num_data_part[i]
        part = np.random.normal(size=(num_data_part[i], 2))
        part = np.dot(np.dot(part, scales[i]), R) + means[i]
        Xparts.append(part)
        num_data_total += num_data_part[i]
    X = np.vstack((Xparts[0], Xparts[1], Xparts[2], Xparts[3]))


    Y = np.vstack((np.ones((num_data_part[0] + num_data_part[1], 1)), -np.ones((num_data_part[2] + num_data_part[3], 1))))
    return {'X':X, 'Y':Y, 'info': "Two separate classes of data formed approximately in the shape of two crescents."}

def creep_data():
    all_data = np.loadtxt(os.path.join(data_path, 'creep', 'taka'))
    y = all_data[:, 1:2].copy()
    features = [0]
    features.extend(range(2, 31))
    X = all_data[:, features].copy()
    return {'X': X, 'y' : y}

def cmu_mocap_49_balance():
    """Load CMU subject 49's one legged balancing motion that was used by Alvarez, Luengo and Lawrence at AISTATS 2009."""
    train_motions = ['18', '19']
    test_motions = ['20']
    data = cmu_mocap('49', train_motions, test_motions, sample_every=4)
    data['info'] = "One legged balancing motions from CMU data base subject 49. As used in Alvarez, Luengo and Lawrence at AISTATS 2009. It consists of " + data['info']
    return data

def cmu_mocap_35_walk_jog():
    """Load CMU subject 35's walking and jogging motions, the same data that was used by Taylor, Roweis and Hinton at NIPS 2007. but without their preprocessing. Also used by Lawrence at AISTATS 2007."""
    train_motions = ['01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
                '13', '14', '15', '16', '17', '19',
                '20', '21', '22', '23', '24', '25',
                '26', '28', '30', '31', '32', '33', '34']
    test_motions = ['18', '29']
    data = cmu_mocap('35', train_motions, test_motions, sample_every=4)
    data['info'] = "Walk and jog data from CMU data base subject 35. As used in Tayor, Roweis and Hinton at NIPS 2007, but without their pre-processing (i.e. as used by Lawrence at AISTATS 2007). It consists of " + data['info']
    return data

def cmu_mocap(subject, train_motions, test_motions=[], sample_every=4):
    """Load a given subject's training and test motions from the CMU motion capture data."""

    # Load in subject skeleton.
    subject_dir = os.path.join(data_path, 'mocap', 'cmu', subject)

    # Make sure the data is downloaded.
    mocap.fetch_cmu(([subject], [train_motions]), skel_store_dir=subject_dir,motion_store_dir=subject_dir)

    skel = GPy.util.mocap.acclaim_skeleton(os.path.join(subject_dir, subject + '.asf'))

    # Set up labels for each sequence
    exlbls = np.eye(len(train_motions))

    # Load sequences
    tot_length = 0
    temp_Y = []
    temp_lbls = []
    for i in range(len(train_motions)):
        temp_chan = skel.load_channels(os.path.join(subject_dir, subject + '_' + train_motions[i] + '.amc'))
        temp_Y.append(temp_chan[::sample_every, :])
        temp_lbls.append(np.tile(exlbls[i, :], (temp_Y[i].shape[0], 1)))
        tot_length += temp_Y[i].shape[0]

    Y = np.zeros((tot_length, temp_Y[0].shape[1]))
    lbls = np.zeros((tot_length, temp_lbls[0].shape[1]))

    end_ind = 0
    for i in range(len(temp_Y)):
        start_ind = end_ind
        end_ind += temp_Y[i].shape[0]
        Y[start_ind:end_ind, :] = temp_Y[i]
        lbls[start_ind:end_ind, :] = temp_lbls[i]
    if len(test_motions) > 0:
        temp_Ytest = []
        temp_lblstest = []

        testexlbls = np.eye(len(test_motions))
        tot_test_length = 0
        for i in range(len(test_motions)):
            temp_chan = skel.load_channels(os.path.join(subject_dir, subject + '_' + test_motions[i] + '.amc'))
            temp_Ytest.append(temp_chan[::sample_every, :])
            temp_lblstest.append(np.tile(testexlbls[i, :], (temp_Ytest[i].shape[0], 1)))
            tot_test_length += temp_Ytest[i].shape[0]

        # Load test data
        Ytest = np.zeros((tot_test_length, temp_Ytest[0].shape[1]))
        lblstest = np.zeros((tot_test_length, temp_lblstest[0].shape[1]))

        end_ind = 0
        for i in range(len(temp_Ytest)):
            start_ind = end_ind
            end_ind += temp_Ytest[i].shape[0]
            Ytest[start_ind:end_ind, :] = temp_Ytest[i]
            lblstest[start_ind:end_ind, :] = temp_lblstest[i]
    else:
        Ytest = None
        lblstest = None

    info = 'Subject: ' + subject + '. Training motions: '
    for motion in train_motions:
        info += motion + ', '
    info = info[:-2]
    if len(test_motions) > 0:
        info += '. Test motions: '
        for motion in test_motions:
            info += motion + ', '
        info = info[:-2] + '.'
    else:
        info += '.'
    if sample_every != 1:
        info += ' Data is sub-sampled to every ' + str(sample_every) + ' frames.'
    return {'Y': Y, 'lbls' : lbls, 'Ytest': Ytest, 'lblstest' : lblstest, 'info': info, 'skel': skel}
