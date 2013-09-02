import os
import numpy as np
import GPy
import scipy.io
import cPickle as pickle
import urllib as url
import zipfile
import tarfile
import datetime
    

import sys, urllib
def reporthook(a,b,c): 
    # ',' at the end of the line is important!
    #print "% 3.1f%% of %d bytes\r" % (min(100, float(a * b) / c * 100), c),
    #you can also use sys.stdout.write
    sys.stdout.write("\r% 3.1f%% of %d bytes" % (min(100, float(a * b) / c * 100), c))
    sys.stdout.flush()
     
# Global variables
data_path = os.path.join(os.path.dirname(__file__), 'datasets')
default_seed = 10000
overide_manual_authorize=False
neil_url = 'http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/dataset_mirror/'
cmu_url = 'http://mocap.cs.cmu.edu/subjects/'
# Note: there may be a better way of storing data resources. One of the pythonistas will need to take a look.
data_resources = {'ankur_pose_data' : {'urls' : [neil_url + 'ankur_pose_data/'],
                                       'files' : [['ankurDataPoseSilhouette.mat']],
                                       'license' : None,
                                       'citation' : """3D Human Pose from Silhouettes by Relevance Vector Regression (In CVPR'04). A. Agarwal and B. Triggs.""",
                                       'details' : """Artificially generated data of silhouettes given poses. Note that the data does not display a left/right ambiguity because across the entire data set one of the arms sticks out more the the other, disambiguating the pose as to which way the individual is facing."""},
                   
                  'boston_housing' : {'urls' : ['http://archive.ics.uci.edu/ml/machine-learning-databases/housing/'],
                                      'files' : [['Index', 'housing.data', 'housing.names']],
                                      'citation' : """Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.""",
                                      'details' : """The Boston Housing data relates house values in Boston to a range of input variables.""",
                                      'license' : None,
                                      'size' : 51276
                                      },
                  'brendan_faces' : {'urls' : ['http://www.cs.nyu.edu/~roweis/data/'],
                                     'files': [['frey_rawface.mat']],
                                     'citation' : 'Frey, B. J., Colmenarez, A and Huang, T. S. Mixtures of Local Linear Subspaces for Face Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 1998, 32-37, June 1998. Computer Society Press, Los Alamitos, CA.',
                                     'details' : """A video of Brendan Frey's face popularized as a benchmark for visualization by the Locally Linear Embedding.""",
                                     'license': None,
                                     'size' : 1100584},
                  'cmu_mocap_full' : {'urls' : ['http://mocap.cs.cmu.edu'],
                                 'files' : [['allasfamc.zip']],
                                 'citation' : """Please include this in your acknowledgements: The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217.""",
                                 'details' : """CMU Motion Capture data base. Captured by a Vicon motion capture system consisting of 12 infrared MX-40 cameras, each of which is capable of recording at 120 Hz with images of 4 megapixel resolution. Motions are captured in a working volume of approximately 3m x 8m. The capture subject wears 41 markers and a stylish black garment.""",
                                 'license' : """From http://mocap.cs.cmu.edu. This data is free for use in research projects. You may include this data in commercially-sold products, but you may not resell this data directly, even in converted form. If you publish results obtained using this data, we would appreciate it if you would send the citation to your published paper to jkh+mocap@cs.cmu.edu, and also would add this text to your acknowledgments section: The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.""",
                                 'size' : None},
                  'creep_rupture' : {'urls' : ['http://www.msm.cam.ac.uk/map/data/tar/'],
                                     'files' : [['creeprupt.tar']],
                                     'citation' : 'Materials Algorithms Project Data Library: MAP_DATA_CREEP_RUPTURE. F. Brun and T. Yoshida.',
                                     'details' : """Provides 2066 creep rupture test results of steels (mainly of two kinds of steels: 2.25Cr and 9-12 wt% Cr ferritic steels). See http://www.msm.cam.ac.uk/map/data/materials/creeprupt-b.html.""",
                                     'license' : None,
                                     'size' : 602797},
                  'della_gatta' : {'urls' : [neil_url + 'della_gatta/'],
                                   'files': [['DellaGattadata.mat']],
                                   'citation' : 'Direct targets of the TRP63 transcription factor revealed by a combination of gene expression profiling and reverse engineering. Giusy Della Gatta, Mukesh Bansal, Alberto Ambesi-Impiombato, Dario Antonini, Caterina Missero, and Diego di Bernardo, Genome Research 2008',
                                   'details': "The full gene expression data set from della Gatta et al (http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2413161/) processed by RMA.",
                                   'license':None,
                                   'size':3729650},
                  'epomeo_gpx' : {'urls' : [neil_url + 'epomeo_gpx/'],
                                   'files': [['endomondo_1.gpx', 'endomondo_2.gpx', 'garmin_watch_via_endomondo.gpx','viewranger_phone.gpx','viewranger_tablet.gpx']],
                                   'citation' : '',
                                   'details': "Five different GPS traces of the same run up Mount Epomeo in Ischia. The traces are from different sources. endomondo_1 and endomondo_2 are traces from the mobile phone app Endomondo, with a split in the middle. garmin_watch_via_endomondo is the trace from a Garmin watch, with a segment missing about 4 kilometers in. viewranger_phone and viewranger_tablet are traces from a phone and a tablet through the viewranger app. The viewranger_phone data comes from the same mobile phone as the Endomondo data (i.e. there are 3 GPS devices, but one device recorded two traces).",
                                   'license':None,
                                   'size': 2031872},
                  'three_phase_oil_flow': {'urls' : [neil_url + 'three_phase_oil_flow/'],
                                           'files' : [['DataTrnLbls.txt', 'DataTrn.txt', 'DataTst.txt', 'DataTstLbls.txt', 'DataVdn.txt', 'DataVdnLbls.txt']],
                                           'citation' : 'Bishop, C. M. and G. D. James (1993). Analysis of multiphase flows using dual-energy gamma densitometry and neural networks. Nuclear Instruments and Methods in Physics Research A327, 580-593',
                                           'details' : """The three phase oil data used initially for demonstrating the Generative Topographic mapping.""",
                                           'license' : None,
                                           'size' : 712796},
                  'rogers_girolami_data' : {'urls' : ['https://www.dropbox.com/sh/7p6tu1t29idgliq/_XqlH_3nt9/'],
                                            'files' : [['firstcoursemldata.tar.gz']],
                                            'suffices' : [['?dl=1']],
                                            'citation' : 'A First Course in Machine Learning. Simon Rogers and Mark Girolami: Chapman & Hall/CRC, ISBN-13: 978-1439824146',
                                            'details' : """Data from the textbook 'A First Course in Machine Learning'. Available from http://www.dcs.gla.ac.uk/~srogers/firstcourseml/.""",
                                            'license' : None,
                                            'size' : 21949154},
                  'olympic_marathon_men' : {'urls' : [neil_url + 'olympic_marathon_men/'],
                                            'files' : [['olympicMarathonTimes.csv']],
                                            'citation' : None,
                                            'details' : """Olympic mens' marathon gold medal winning times from 1896 to 2012. Time given in pace (minutes per kilometer). Data is originally downloaded and collated from Wikipedia, we are not responsible for errors in the data""",
                                            'license': None,
                                            'size' : 584},
                  'osu_run1' : {'urls': ['http://accad.osu.edu/research/mocap/data/', neil_url + 'stick/'],
                                'files': [['run1TXT.ZIP'],['connections.txt']],
                                'details' : "Motion capture data of a stick man running from the Open Motion Data Project at Ohio State University.",
                                'citation' : 'The Open Motion Data Project by The Ohio State University Advanced Computing Center for the Arts and Design, http://accad.osu.edu/research/mocap/mocap_data.htm.',
                                'license' : 'Data is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/).',
                                'size': 338103},
                  'osu_accad' : {'urls': ['http://accad.osu.edu/research/mocap/data/', neil_url + 'stick/'],
                                'files': [['swagger1TXT.ZIP','handspring1TXT.ZIP','quickwalkTXT.ZIP','run1TXT.ZIP','sprintTXT.ZIP','dogwalkTXT.ZIP','camper_04TXT.ZIP','dance_KB3_TXT.ZIP','per20_TXT.ZIP','perTWO07_TXT.ZIP','perTWO13_TXT.ZIP','perTWO14_TXT.ZIP','perTWO15_TXT.ZIP','perTWO16_TXT.ZIP'],['connections.txt']],
                                'details' : "Motion capture data of different motions from the Open Motion Data Project at Ohio State University.",
                                'citation' : 'The Open Motion Data Project by The Ohio State University Advanced Computing Center for the Arts and Design, http://accad.osu.edu/research/mocap/mocap_data.htm.',
                                'license' : 'Data is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License (http://creativecommons.org/licenses/by-nc-sa/3.0/).',
                                'size': 15922790},
                  'pumadyn-32nm' : {'urls' : ['ftp://ftp.cs.toronto.edu/pub/neuron/delve/data/tarfiles/pumadyn-family/'],
                                    'files' : [['pumadyn-32nm.tar.gz']],
                                    'details' : """Pumadyn non linear 32 input data set with moderate noise. See http://www.cs.utoronto.ca/~delve/data/pumadyn/desc.html for details.""",
                                    'citation' : """Created by Zoubin Ghahramani using the Matlab Robotics Toolbox of Peter Corke. Corke, P. I. (1996). A Robotics Toolbox for MATLAB. IEEE Robotics and Automation Magazine, 3 (1): 24-32.""",
                                    'license' : """Data is made available by the Delve system at the University of Toronto""",
                                    'size' : 5861646},
                  'robot_wireless' : {'urls' : [neil_url + 'robot_wireless/'],
                                      'files' : [['uw-floor.txt']],
                                      'citation' : """WiFi-SLAM using Gaussian Process Latent Variable Models by Brian Ferris, Dieter Fox and Neil Lawrence in IJCAI'07 Proceedings pages 2480-2485. Data used in A Unifying Probabilistic Perspective for Spectral Dimensionality Reduction: Insights and New Models by Neil D. Lawrence, JMLR 13 pg 1609--1638, 2012.""",
                                      'details' : """Data created by Brian Ferris and Dieter Fox. Consists of WiFi access point strengths taken during a circuit of the Paul Allen building at the University of Washington.""",
                                      'license' : None,
                                      'size' : 284390},
                  'swiss_roll' : {'urls' : ['http://isomap.stanford.edu/'],
                                  'files' : [['swiss_roll_data.mat']],
                                  'details' : """Swiss roll data made available by Tenenbaum, de Silva and Langford to demonstrate isomap, available from http://isomap.stanford.edu/datasets.html.""",
                                  'citation' : 'A Global Geometric Framework for Nonlinear Dimensionality Reduction, J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000',
                                  'license' : None,
                                  'size' : 800256},
                  'ripley_prnn_data' : {'urls' : ['http://www.stats.ox.ac.uk/pub/PRNN/'],
                                        'files' : [['Cushings.dat', 'README', 'crabs.dat', 'fglass.dat', 'fglass.grp', 'pima.te', 'pima.tr', 'pima.tr2', 'synth.te', 'synth.tr', 'viruses.dat', 'virus3.dat']],
                                        'details' : """Data sets from Brian Ripley's Pattern Recognition and Neural Networks""",
                                        'citation': """Pattern Recognition and Neural Networks by B.D. Ripley (1996) Cambridge University Press ISBN 0 521 46986 7""",
                                        'license' : None,
                                        'size' : 93565},
                  'isomap_face_data' : {'urls' : [neil_url + 'isomap_face_data/'],
                                        'files' : [['face_data.mat']],
                                        'details' : """Face data made available by Tenenbaum, de Silva and Langford to demonstrate isomap, available from http://isomap.stanford.edu/datasets.html.""",
                                        'citation' : 'A Global Geometric Framework for Nonlinear Dimensionality Reduction, J. B. Tenenbaum, V. de Silva and J. C. Langford, Science 290 (5500): 2319-2323, 22 December 2000',
                                        'license' : None,
                                        'size' : 24229368},
                  }
                  
def prompt_user():
    """Ask user for agreeing to data set licenses."""
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

def data_available(dataset_name=None):
    """Check if the data set is available on the local machine already."""
    for file_list in data_resources[dataset_name]['files']:
        for file in file_list:
            if not os.path.exists(os.path.join(data_path, dataset_name, file)):
                return False
    return True
            
def download_url(url, store_directory, save_name = None, messages = True, suffix=''):
    """Download a file from a url and save it to disk."""
    i = url.rfind('/')
    file = url[i+1:]
    print file
    dir_name = os.path.join(data_path, store_directory)
    save_name = os.path.join(dir_name, file)
    print "Downloading ", url, "->", os.path.join(store_directory, file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    urllib.urlretrieve(url+suffix, save_name, reporthook)

def authorize_download(dataset_name=None):
    """Check with the user that the are happy with terms and conditions for the data set."""
    print('Acquiring resource: ' + dataset_name)
    # TODO, check resource is in dictionary!
    print('')
    dr = data_resources[dataset_name]
    print('Details of data: ')
    print(dr['details'])
    print('')
    if dr['citation']:
        print('Please cite:')
        print(dr['citation'])
        print('')
    if dr['size']:
        print('After downloading the data will take up ' + str(dr['size']) + ' bytes of space.')
        print('')
    print('Data will be stored in ' + os.path.join(data_path, dataset_name) + '.')
    print('')
    if overide_manual_authorize:
        if dr['license']:
            print('You have agreed to the following license:')
            print(dr['license'])
            print('')
        return True
    else:
        if dr['license']:
            print('You must also agree to the following license:')
            print(dr['license'])
            print('')
        print('Do you wish to proceed with the download? [yes/no]')
        return prompt_user()

def download_data(dataset_name=None):
    """Check with the user that the are happy with terms and conditions for the data set, then download it."""

    dr = data_resources[dataset_name]
    if not authorize_download(dataset_name):
        return False

    if dr.has_key('suffices'):
        for url, files, suffices in zip(dr['urls'], dr['files'], dr['suffices']):
            for file, suffix in zip(files, suffices):
                download_url(os.path.join(url,file), dataset_name, dataset_name, suffix=suffix)
    else:
        for url, files in zip(dr['urls'], dr['files']):
            for file in files:
                download_url(os.path.join(url,file), dataset_name, dataset_name)
    return True
                  
def data_details_return(data, data_set):
    """Update the data component of the data dictionary with details drawn from the data_resources."""
    data.update(data_resources[data_set])
    return data

    
def cmu_urls_files(subj_motions, messages = True):
    '''
    Find which resources are missing on the local disk for the requested CMU motion capture motions. 
    '''
    
    subjects_num = subj_motions[0]
    motions_num = subj_motions[1]

    resource = {'urls' : [], 'files' : []}
    # Convert numbers to strings
    subjects = []
    motions = [list() for _ in range(len(subjects_num))]
    for i in range(len(subjects_num)):
        curSubj = str(int(subjects_num[i]))
        if int(subjects_num[i]) < 10:
            curSubj = '0' + curSubj
        subjects.append(curSubj)
        for j in range(len(motions_num[i])):
            curMot = str(int(motions_num[i][j]))
            if int(motions_num[i][j]) < 10:
                curMot = '0' + curMot
            motions[i].append(curMot)

    all_skels = []
    
    assert len(subjects) == len(motions)
    
    all_motions = []
            
    for i in range(len(subjects)):
        skel_dir = os.path.join(data_path, 'cmu_mocap')
        cur_skel_file = os.path.join(skel_dir, subjects[i] + '.asf')
        
        url_required = False
        file_download = []
        if not os.path.exists(cur_skel_file):
            # Current skel file doesn't exist.
            if not os.path.isdir(skel_dir):
                os.mkdir(skel_dir)
            # Add skel file to list.
            url_required = True
            file_download.append(subjects[i] + '.asf')
        for j in range(len(motions[i])):
            file_name = subjects[i] + '_' + motions[i][j] + '.amc'
            cur_motion_file = os.path.join(skel_dir, file_name)
            if not os.path.exists(cur_motion_file):
                url_required = True
                file_download.append(subjects[i] + '_' + motions[i][j] + '.amc')
        if url_required:
            resource['urls'].append(cmu_url + subjects[i] + '/')
            resource['files'].append(file_download)
    return resource

try:
    import gpxpy
    import gpxpy.gpx
    gpxpy_available = True

except ImportError:
    gpxpy_available = False

if gpxpy_available:
    def epomeo_gpx(data_set='epomeo_gpx', sample_every=4):
        if not data_available(data_set):
            download_data(data_set)
        files = ['endomondo_1', 'endomondo_2', 'garmin_watch_via_endomondo','viewranger_phone', 'viewranger_tablet']

        X = []
        for file in files:
            gpx_file = open(os.path.join(data_path, 'epomeo_gpx', file + '.gpx'), 'r')

            gpx = gpxpy.parse(gpx_file)
            segment = gpx.tracks[0].segments[0]
            points = [point for track in gpx.tracks for segment in track.segments for point in segment.points]
            data = [[(point.time-datetime.datetime(2013,8,21)).total_seconds(), point.latitude, point.longitude, point.elevation] for point in points]
            X.append(np.asarray(data)[::sample_every, :])
            gpx_file.close()        
        return data_details_return({'X' : X, 'info' : 'Data is an array containing time in seconds, latitude, longitude and elevation in that order.'}, data_set)

del gpxpy_available



# Some general utilities.
def sample_class(f):
    p = 1. / (1. + np.exp(-f))
    c = np.random.binomial(1, p)
    c = np.where(c, 1, -1)
    return c

def boston_housing(data_set='boston_housing'):
    if not data_available(data_set):
        download_data(data_set)
    all_data = np.genfromtxt(os.path.join(data_path, data_set, 'housing.data'))
    X = all_data[:, 0:13]
    Y = all_data[:, 13:14]
    return data_details_return({'X' : X, 'Y': Y}, data_set)

def brendan_faces(data_set='brendan_faces'):
    if not data_available(data_set):
        download_data(data_set)
    mat_data = scipy.io.loadmat(os.path.join(data_path, data_set, 'frey_rawface.mat'))
    Y = mat_data['ff'].T
    return data_details_return({'Y': Y}, data_set)

def della_gatta_TRP63_gene_expression(data_set='della_gatta', gene_number=None):
    if not data_available(data_set):
        download_data(data_set)
    mat_data = scipy.io.loadmat(os.path.join(data_path, data_set, 'DellaGattadata.mat'))
    X = np.double(mat_data['timepoints'])
    if gene_number == None:
        Y = mat_data['exprs_tp53_RMA']
    else:
        Y = mat_data['exprs_tp53_RMA'][:, gene_number]
        if len(Y.shape) == 1:
            Y = Y[:, None]
    return data_details_return({'X': X, 'Y': Y, 'gene_number' : gene_number}, data_set)



# The data sets
def oil(data_set='three_phase_oil_flow'):
    """The three phase oil data from Bishop and James (1993)."""
    if not data_available(data_set):
        download_data(data_set)
    oil_train_file = os.path.join(data_path, data_set, 'DataTrn.txt')
    oil_trainlbls_file = os.path.join(data_path, data_set, 'DataTrnLbls.txt')
    oil_test_file = os.path.join(data_path, data_set, 'DataTst.txt')
    oil_testlbls_file = os.path.join(data_path, data_set, 'DataTstLbls.txt')
    oil_valid_file = os.path.join(data_path, data_set, 'DataVdn.txt')
    oil_validlbls_file = os.path.join(data_path, data_set, 'DataVdnLbls.txt')
    fid = open(oil_train_file)
    X = np.fromfile(fid, sep='\t').reshape((-1, 12))
    fid.close()
    fid = open(oil_test_file)
    Xtest = np.fromfile(fid, sep='\t').reshape((-1, 12))
    fid.close()
    fid = open(oil_valid_file)
    Xvalid = np.fromfile(fid, sep='\t').reshape((-1, 12))
    fid.close()
    fid = open(oil_trainlbls_file)
    Y = np.fromfile(fid, sep='\t').reshape((-1, 3)) * 2. - 1.
    fid.close()
    fid = open(oil_testlbls_file)
    Ytest = np.fromfile(fid, sep='\t').reshape((-1, 3)) * 2. - 1.
    fid.close()
    fid = open(oil_validlbls_file)
    Yvalid = np.fromfile(fid, sep='\t').reshape((-1, 3)) * 2. - 1.
    fid.close()
    return data_details_return({'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest, 'Xtest' : Xtest, 'Xvalid': Xvalid, 'Yvalid': Yvalid}, data_set)
    #else:
    # throw an error
    
def oil_100(seed=default_seed, data_set = 'three_phase_oil_flow'):
    np.random.seed(seed=seed)
    data = oil()
    indices = np.random.permutation(1000)
    indices = indices[0:100]
    X = data['X'][indices, :]
    Y = data['Y'][indices, :]
    return data_details_return({'X': X, 'Y': Y, 'info': "Subsample of the full oil data extracting 100 values randomly without replacement, here seed was " + str(seed)}, data_set)

def pumadyn(seed=default_seed, data_set='pumadyn-32nm'):
    if not data_available(data_set):
        download_data(data_set)
        path = os.path.join(data_path, data_set)
        tar = tarfile.open(os.path.join(path, 'pumadyn-32nm.tar.gz'))
        print('Extracting file.')
        tar.extractall(path=path)
        tar.close()
    # Data is variance 1, no need to normalize.
    data = np.loadtxt(os.path.join(data_path, data_set, 'pumadyn-32nm', 'Dataset.data.gz'))
    indices = np.random.permutation(data.shape[0])
    indicesTrain = indices[0:7168]
    indicesTest = indices[7168:-1]
    indicesTrain.sort(axis=0)
    indicesTest.sort(axis=0)
    X = data[indicesTrain, 0:-2]
    Y = data[indicesTrain, -1][:, None]
    Xtest = data[indicesTest, 0:-2]
    Ytest = data[indicesTest, -1][:, None]
    return data_details_return({'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest, 'seed': seed}, data_set)

def robot_wireless(data_set='robot_wireless'):
    # WiFi access point strengths on a tour around UW Paul Allen building.
    if not data_available(data_set):
        download_data(data_set)
    file_name = os.path.join(data_path, data_set, 'uw-floor.txt')
    all_time = np.genfromtxt(file_name, usecols=(0))
    macaddress = np.genfromtxt(file_name, usecols=(1), dtype='string')
    x = np.genfromtxt(file_name, usecols=(2))
    y = np.genfromtxt(file_name, usecols=(3))
    strength = np.genfromtxt(file_name, usecols=(4))
    addresses = np.unique(macaddress)
    times = np.unique(all_time)
    addresses.sort()
    times.sort()
    allY = np.zeros((len(times), len(addresses)))
    allX = np.zeros((len(times), 2))
    allY[:]=-92.
    strengths={}
    for address, j in zip(addresses, range(len(addresses))):
        ind = np.nonzero(address==macaddress)
        temp_strengths=strength[ind]
        temp_x=x[ind]
        temp_y=y[ind]
        temp_times = all_time[ind]
        for time in temp_times:
            vals = time==temp_times
            if any(vals):
                ind2 = np.nonzero(vals)
                i = np.nonzero(time==times)
                allY[i, j] = temp_strengths[ind2]
                allX[i, 0] = temp_x[ind2]
                allX[i, 1] = temp_y[ind2]
    allY = (allY + 85.)/15.

    X = allX[0:215, :]
    Y = allY[0:215, :]

    Xtest = allX[215:, :]
    Ytest = allY[215:, :]
    return data_details_return({'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest, 'addresses' : addresses, 'times' : times}, data_set)

def silhouette(data_set='ankur_pose_data'):
    # Ankur Agarwal and Bill Trigg's silhoutte data.
    if not data_available(data_set):
        download_data(data_set)
    mat_data = scipy.io.loadmat(os.path.join(data_path, data_set, 'ankurDataPoseSilhouette.mat'))
    inMean = np.mean(mat_data['Y'])
    inScales = np.sqrt(np.var(mat_data['Y']))
    X = mat_data['Y'] - inMean
    X = X / inScales
    Xtest = mat_data['Y_test'] - inMean
    Xtest = Xtest / inScales
    Y = mat_data['Z']
    Ytest = mat_data['Z_test']
    return data_details_return({'X': X, 'Y': Y, 'Xtest': Xtest, 'Ytest': Ytest}, data_set)

def ripley_synth(data_set='ripley_prnn_data'):
    if not data_available(data_set):
        download_data(data_set)
    train = np.genfromtxt(os.path.join(data_path, data_set, 'synth.tr'), skip_header=1)
    X = train[:, 0:2]
    y = train[:, 2:3]
    test = np.genfromtxt(os.path.join(data_path, data_set, 'synth.te'), skip_header=1)
    Xtest = test[:, 0:2]
    ytest = test[:, 2:3]
    return data_details_return({'X': X, 'y': y, 'Xtest': Xtest, 'ytest': ytest, 'info': 'Synthetic data generated by Ripley for a two class classification problem.'}, data_set)

def osu_run1(data_set='osu_run1', sample_every=4):
    if not data_available(data_set):
        download_data(data_set)
    zip = zipfile.ZipFile(os.path.join(data_path, data_set, 'sprintTXT.ZIP'), 'r')
    path = os.path.join(data_path, data_set)
    for name in zip.namelist():
        zip.extract(name, path)
    Y, connect = GPy.util.mocap.load_text_data('Aug210107', path)
    Y = Y[0:-1:sample_every, :]
    return data_details_return({'Y': Y, 'connect' : connect}, data_set)

def swiss_roll_generated(num_samples=1000, sigma=0.0):
    with open(os.path.join(data_path, 'swiss_roll.pickle')) as f:
        data = pickle.load(f)
    Na = data['Y'].shape[0]
    perm = np.random.permutation(np.r_[:Na])[:num_samples]
    Y = data['Y'][perm, :]
    t = data['t'][perm]
    c = data['colors'][perm, :]
    so = np.argsort(t)
    Y = Y[so, :]
    t = t[so]
    c = c[so, :]
    return {'Y':Y, 't':t, 'colors':c}

def swiss_roll_1000():
    return swiss_roll(num_samples=1000)

def swiss_roll(num_samples=3000, data_set='swiss_roll'):
    if not data_available(data_set):
        download_data(data_set)
    mat_data = scipy.io.loadmat(os.path.join(data_path, data_set, 'swiss_roll_data.mat'))
    Y = mat_data['X_data'][:, 0:num_samples].transpose()
    return data_details_return({'Y': Y, 'X': mat_data['X_data'], 'info': "The first " + str(num_samples) + " points from the swiss roll data of Tennenbaum, de Silva and Langford (2001)."}, data_set)

def isomap_faces(num_samples=698, data_set='isomap_face_data'):
    if not data_available(data_set):
        download_data(data_set)
    mat_data = scipy.io.loadmat(os.path.join(data_path, data_set, 'face_data.mat'))
    Y = mat_data['images'][:, 0:num_samples].transpose()
    return data_details_return({'Y': Y, 'poses' : mat_data['poses'], 'lights': mat_data['lights'], 'info': "The first " + str(num_samples) + " points from the face data of Tennenbaum, de Silva and Langford (2001)."}, data_set)

def simulation_BGPLVM():
    mat_data = scipy.io.loadmat(os.path.join(data_path, 'BGPLVMSimulation.mat'))
    Y = np.array(mat_data['Y'], dtype=float)
    S = np.array(mat_data['initS'], dtype=float)
    mu = np.array(mat_data['initMu'], dtype=float)
    return data_details_return({'S': S, 'Y': Y, 'mu': mu}, data_set)
    return {'Y': Y, 'S': S,
            'mu' : mu,
            'info': "Simulated test dataset generated in MATLAB to compare BGPLVM between python and MATLAB"}

def toy_rbf_1d(seed=default_seed, num_samples=500):
    """Samples values of a function from an RBF covariance with very small noise for inputs uniformly distributed between -1 and 1.
    :param seed: seed to use for random sampling.
    :type seed: int
    :param num_samples: number of samples to sample in the function (default 500).
    :type num_samples: int
    """
    np.random.seed(seed=seed)
    num_in = 1
    X = np.random.uniform(low= -1.0, high=1.0, size=(num_samples, num_in))
    X.sort(axis=0)
    rbf = GPy.kern.rbf(num_in, variance=1., lengthscale=np.array((0.25,)))
    white = GPy.kern.white(num_in, variance=1e-2)
    kernel = rbf + white
    K = kernel.K(X)
    y = np.reshape(np.random.multivariate_normal(np.zeros(num_samples), K), (num_samples, 1))
    return {'X':X, 'Y':y, 'info': "Sampled " + str(num_samples) + " values of a function from an RBF covariance with very small noise for inputs uniformly distributed between -1 and 1."}

def toy_rbf_1d_50(seed=default_seed):
    np.random.seed(seed=seed)
    data = toy_rbf_1d()
    indices = np.random.permutation(data['X'].shape[0])
    indices = indices[0:50]
    indices.sort(axis=0)
    X = data['X'][indices, :]
    Y = data['Y'][indices, :]
    return {'X': X, 'Y': Y, 'info': "Subsamples the toy_rbf_sample with 50 values randomly taken from the original sample.", 'seed' : seed}


def toy_linear_1d_classification(seed=default_seed):
    np.random.seed(seed=seed)
    x1 = np.random.normal(-3, 5, 20)
    x2 = np.random.normal(3, 5, 20)
    X = (np.r_[x1, x2])[:, None]
    return {'X': X, 'Y':  sample_class(2.*X), 'F': 2.*X, 'seed' : seed}

def olympic_100m_men(data_set='rogers_girolami_data'):
    if not data_available(data_set):
        download_data(data_set)
        path = os.path.join(data_path, data_set)
        tar_file = os.path.join(path, 'firstcoursemldata.tar.gz')
        tar = tarfile.open(tar_file)
        print('Extracting file.')
        tar.extractall(path=path)
        tar.close()
    olympic_data = scipy.io.loadmat(os.path.join(data_path, data_set, 'data', 'olympics.mat'))['male100']

    X = olympic_data[:, 0][:, None]
    Y = olympic_data[:, 1][:, None]
    return data_details_return({'X': X, 'Y': Y, 'info': "Olympic sprint times for 100 m men from 1896 until 2008. Example is from Rogers and Girolami's First Course in Machine Learning."}, data_set)

def olympic_100m_women(data_set='rogers_girolami_data'):
    if not data_available(data_set):
        download_data(data_set)
        path = os.path.join(data_path, data_set)
        tar_file = os.path.join(path, 'firstcoursemldata.tar.gz')
        tar = tarfile.open(tar_file)
        print('Extracting file.')
        tar.extractall(path=path)
        tar.close()
    olympic_data = scipy.io.loadmat(os.path.join(data_path, data_set, 'data', 'olympics.mat'))['female100']

    X = olympic_data[:, 0][:, None]
    Y = olympic_data[:, 1][:, None]
    return data_details_return({'X': X, 'Y': Y, 'info': "Olympic sprint times for 100 m women from 1896 until 2008. Example is from Rogers and Girolami's First Course in Machine Learning."}, data_set)

def olympic_marathon_men(data_set='olympic_marathon_men'):
    if not data_available(data_set):
        download_data(data_set)
    olympics = np.genfromtxt(os.path.join(data_path, data_set, 'olympicMarathonTimes.csv'), delimiter=',')
    X = olympics[:, 0:1]
    Y = olympics[:, 1:2]
    return data_details_return({'X': X, 'Y': Y}, data_set)

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
        part = np.random.normal(size=(num_data_part[i], 2))
        part = np.dot(np.dot(part, scales[i]), R) + means[i]
        Xparts.append(part)
        num_data_total += num_data_part[i]
    X = np.vstack((Xparts[0], Xparts[1], Xparts[2], Xparts[3]))

    Y = np.vstack((np.ones((num_data_part[0] + num_data_part[1], 1)), -np.ones((num_data_part[2] + num_data_part[3], 1))))
    return {'X':X, 'Y':Y, 'info': "Two separate classes of data formed approximately in the shape of two crescents."}

def creep_data(data_set='creep_rupture'):
    """Brun and Yoshida's metal creep rupture data."""
    if not data_available(data_set):
        download_data(data_set)
        path = os.path.join(data_path, data_set)
        tar_file = os.path.join(path, 'creeprupt.tar')
        tar = tarfile.open(tar_file)
        print('Extracting file.')
        tar.extractall(path=path)
        tar.close()
    all_data = np.loadtxt(os.path.join(data_path, data_set, 'taka'))
    y = all_data[:, 1:2].copy()
    features = [0]
    features.extend(range(2, 31))
    X = all_data[:, features].copy()
    return data_details_return({'X': X, 'y': y}, data_set)

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

def cmu_mocap(subject, train_motions, test_motions=[], sample_every=4, data_set='cmu_mocap'):
    """Load a given subject's training and test motions from the CMU motion capture data."""
    # Load in subject skeleton.
    subject_dir = os.path.join(data_path, data_set)

    # Make sure the data is downloaded.
    all_motions = train_motions + test_motions
    resource = cmu_urls_files(([subject], [all_motions]))
    data_resources[data_set] = data_resources['cmu_mocap_full']
    data_resources[data_set]['files'] = resource['files']
    data_resources[data_set]['urls'] = resource['urls']
    if resource['urls']:
        download_data(data_set)

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
    return data_details_return({'Y': Y, 'lbls' : lbls, 'Ytest': Ytest, 'lblstest' : lblstest, 'info': info, 'skel': skel}, data_set)
