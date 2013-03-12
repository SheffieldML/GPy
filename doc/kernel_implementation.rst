
***************************
List of implemented kernels
***************************

The following table shows the implemented kernels in GPy and gives the details of the implemented function for each kernel.

==================== =========== =====  ===========  ======  ======= =========== =============== ======= =========== ====== ====== =======
NAME                  Dimension   ARD   get/set      K       Kdiag   dK_dtheta   dKdiag_dtheta   dK_dX   dKdiag_dX   psi0   psi1   psi2
==================== =========== =====  ===========  ======  ======= =========== =============== ======= =========== ====== ====== =======
bias                 n                  |tick|       |tick|  |tick|  |tick|      |tick|          |tick|  |tick|      |tick| |tick| |tick|
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
Brownian             1                  |tick|       |tick|  |tick|  |tick|      |tick|          |tick|  |tick|                 
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
exponential          n           yes    |tick|       |tick|  |tick|  |tick|      |tick|          |tick|  |tick|
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
finite_dimensional   n                  |tick|       |tick|  |tick|  |tick|      |tick| 
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
linear               n           yes    |tick|       |tick|  |tick|  |tick|      |tick|          |tick|              |tick| |tick| |tick|
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
Matern32             n           yes    |tick|       |tick|  |tick|  |tick|      |tick|          |tick|  |tick|        
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
Matern52             n           yes    |tick|       |tick|  |tick|  |tick|      |tick|          |tick|  |tick|
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
periodic_exponential 1                  |tick|       |tick|  |tick|  |tick|      |tick|
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
periodic_Matern32    1                  |tick|       |tick|  |tick|  |tick|      |tick|
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
periodic_Matern52    1                  |tick|       |tick|  |tick|  |tick|      |tick|
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
rational quadratic   1                  |tick|       |tick|  |tick|  |tick|      |tick|          |tick|  |tick|                           
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
rbf                  n           yes    |tick|       |tick|  |tick|  |tick|      |tick|          |tick|  |tick|      |tick| |tick| |tick|
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
spline               1                  |tick|       |tick|  |tick|  |tick|      |tick|                  |tick|     
-------------------- ----------- -----  -----------  ------  ------- ----------- --------------- ------- ----------- ------ ------ -------
white                n                  |tick|       |tick|  |tick|  |tick|      |tick|          |tick|  |tick|      |tick| |tick| |tick|
==================== =========== =====  ===========  ======  ======= =========== =============== ======= =========== ====== ====== =======

Depending on the use, all functions may not be required

    * ``get/set, K, Kdiag``: compulsory
    * ``dK_dtheta``: necessary to optimize the model
    * ``dKdiag_dtheta``: sparse models, BGPLVM, GPs with uncertain inputs
    * ``dK_dX``: sparse models, GPLVM, BGPLVM, GPs with uncertain inputs
    * ``dKdiag_dX``: sparse models, BGPLVM, GPs with uncertain inputs
    * ``psi0, psi1, psi2``: BGPLVM, GPs with uncertain inputs

..  |tick| image:: Figures/tick.png
