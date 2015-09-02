# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function
from evaluation import RMSE
from methods import GP_RBF, SVIGP_RBF, SparseGP_RBF
from tasks import Housing, WineQuality
from outputs import ScreenOutput, CSVOutput, H5Output
import numpy as np
import time

outpath = '.'
prjname = 'regression'
config = {
          'evaluations':[RMSE],
          'methods':[GP_RBF, SVIGP_RBF, SparseGP_RBF],
          'tasks':[WineQuality,Housing],
          'repeats':2,
          'outputs': [ScreenOutput(), CSVOutput(outpath, prjname), H5Output(outpath, prjname)]
          }

if __name__=='__main__':
    results = np.zeros((len(config['tasks']), len(config['methods']), len(config['evaluations'])+1, config['repeats']))
    
    for task_i in range(len(config['tasks'])):
        dataset = config['tasks'][task_i]()
        print('Benchmarking on '+dataset.name)
        res = dataset.load_data()
        if not res: print('Fail to load '+config['tasks'][task_i].name); continue
        train = dataset.get_training_data()
        test = dataset.get_test_data()
        
        for method_i in range(len(config['methods'])):
            method = config['methods'][method_i]
            print('With the method '+method.name, end='')
            for ri in range(config['repeats']):
                m = method()
                t_st = time.time()
                m.fit(train)
                pred = m.predict(test[0])
                t_pd = time.time() - t_st
                for ei in range(len(config['evaluations'])):
                    evalu = config['evaluations'][ei]()
                    results[task_i, method_i, ei, ri] = evalu.evaluate(test[1], pred)
                results[task_i, method_i, -1, ri] = t_pd
                print('.',end='')
            print()
                    
    [out.output(config, results) for out in config['outputs']]


            
            
