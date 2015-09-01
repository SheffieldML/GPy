# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function
import abc
import os
import numpy as np

class Output(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def output(self, config, results):
        """Return the test data: training data and labels"""
        return None

class ScreenOutput(Output):
            
    def output(self, config, results):
        print('='*10+'Report'+'='*10)
        print('\t'.join([' ']+[m.name+'('+e+')' for m in config['methods'] for e in [a.name for a in config['evaluations']]+['time']]))
        for task_i in range(len(config['tasks'])):
            print(config['tasks'][task_i].name+'\t', end='')

            outputs = []
            for method_i in range(len(config['methods'])):
                for ei in range(len(config['evaluations'])+1):
                    m,s = results[task_i, method_i, ei].mean(), results[task_i, method_i, ei].std()
                    outputs.append('%e(%e)'%(m,s))
            print('\t'.join(outputs))

class CSVOutput(Output):
    
    def __init__(self, outpath, prjname):
        self.fname = os.path.join(outpath, prjname+'.csv')
        
    def output(self, config, results):
        with open(self.fname,'w') as f:
            f.write(','.join([' ']+[m.name+'('+e+')' for m in config['methods'] for e in [a.name for a in config['evaluations']]+['time']])+'\n')
            for task_i in range(len(config['tasks'])):
                f.write(config['tasks'][task_i].name+',')

                outputs = []
                for method_i in range(len(config['methods'])):
                    for ei in range(len(config['evaluations'])+1):
                        m,s = results[task_i, method_i, ei].mean(), results[task_i, method_i, ei].std()
                        outputs.append('%e (%e)'%(m,s))
                f.write(','.join(outputs)+'\n')
            f.close()
            
class H5Output(Output):
    
    def __init__(self, outpath, prjname):
        self.fname = os.path.join(outpath, prjname+'.h5')
        
    def output(self, config, results):
            try:
                import h5py
                f = h5py.File(self.fname,'w')
                d = f.create_dataset('results',results.shape, dtype=results.dtype)
                d[:] = results
                f.close()
            except:
                raise 'Fails to write the parameters into a HDF5 file!'
