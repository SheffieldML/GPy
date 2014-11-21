# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

try:
    import matplot_dep
except (ImportError, NameError):
    print 'Fail to load GPy.plotting.matplot_dep.'