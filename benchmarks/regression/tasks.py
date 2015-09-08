# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import abc
import os
import numpy as np

class RegressionTask(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, datapath='./'):
        self.datapath = datapath
    
    @abc.abstractmethod
    def load_data(self):
        """Download the dataset if not exist. Return True if successful"""
        return True
    
    @abc.abstractmethod
    def get_training_data(self):
        """Return the training data: training data and labels"""
        return None
    
    @abc.abstractmethod
    def get_test_data(self):
        """Return the test data: training data and labels"""
        return None
    
class Housing(RegressionTask):
    
    name='Housing'
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    filename = 'housing.data'
    
    def load_data(self):
        from GPy.util.datasets import download_url, data_path
        if not os.path.exists(os.path.join(data_path,self.datapath, self.filename)):
            download_url(Housing.url, self.datapath, messages=True)
            if not os.path.exists(os.path.join(data_path, self.datapath, self.filename)):
                return False
        
        data = np.loadtxt(os.path.join(data_path, self.datapath, self.filename))
        self.data = data
        data_train = data[:250,:-1]
        label_train = data[:250, -1:]
        self.train = (data_train, label_train)
        data_test = data[250:,:-1]
        label_test = data[250:,-1:]
        self.test = (data_test, label_test)
        return True
    
    def get_training_data(self):
        return self.train
    
    def get_test_data(self):
        return self.test
    
class WineQuality(RegressionTask):
    
    name='WineQuality'
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    filename = 'winequality-red.csv'
    
    def load_data(self):
        from GPy.util.datasets import download_url, data_path
        if not os.path.exists(os.path.join(data_path,self.datapath, self.filename)):
            download_url(self.url, self.datapath, messages=True)
            if not os.path.exists(os.path.join(data_path, self.datapath, self.filename)):
                return False
        
        data = np.loadtxt(os.path.join(data_path, self.datapath, self.filename),skiprows=1,delimiter=';')
        self.data = data
        data_train = data[:1000,:-1]
        label_train = data[:1000, -1:]
        self.train = (data_train, label_train)
        data_test = data[1000:,:-1]
        label_test = data[1000:,-1:]
        self.test = (data_test, label_test)
        return True
    
    def get_training_data(self):
        return self.train
    
    def get_test_data(self):
        return self.test
        
