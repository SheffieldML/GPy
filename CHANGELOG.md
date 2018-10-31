# Changelog

## v1.9.6 (2018-10-30)

## New

* Added a new class that enables using multiple likelihoods [@esiivola]

* Alex grig kalman new [@AlexGrig, @mzwiessele]

## Fix

* fix typo in docstring for GP.opimize() [@RobRomijnders]

* Updates to posterior sampling [@lawrennd]

* Jayanthkoushik cython fix [@jayanthkoushik, @mzwiessele]

* Added missing columns (:), fixed indentation [@vlavorini]

* Fix the bug in the prediction of full covariance matrix [@zhenwendai]

## v1.9.5 (2018-09-02)

### New

* Student-t processes #525. [mzwiessele]

### Fix

* Merge #514. [mzwiessele]

* Merge. [mzwiessele]

### Other

* Bump version: 1.9.4 → 1.9.5. [mzwiessele]

* SDE: removed some unnecessary comments. [Alex Grigorievskiy]

* SDE: Remove sde kernels from the _src folder. [Alex Grigorievskiy]

* STATE-SPACE: Recent modifications to state-space inference, including bug fixes in state-space kernels. [Alex Grigorievskiy]

* TEST: Correcting message for test skipping. This is normal that this test does not work and hence skipped. [Alex Grigorievskiy]

* Solved incorrect parameter assignments (causing test faillure) [Joachim van der Herten]

* Added some shifts to the degrees of freedom parameter. [Joachim van der Herten]

* Removal of print statements. [Joachim van der Herten]

* Implementation of student-t processes. [Joachim van der Herten]

* Pkg: changelog. [mzwiessele]


## v1.9.4 (2018-09-02)

### Fix

* Bdist_dumb and bdist_rpm. [mzwiessele]

* Samples tests and plotting, multioutput. [mzwiessele]

* Py37 cython not compatible. [mzwiessele]

### Other

* Bump version: 1.9.3 → 1.9.4. [mzwiessele]

* Pkg: merged and tested, no py37. [mzwiessele]

* Pkg: no py37 still does not work. [mzwiessele]

* Add py37. [mzwiessele]

* Add py37. [mzwiessele]

* Merge branch 'cython-fix' of git://github.com/jayanthkoushik/GPy into jayanthkoushik-cython-fix. [mzwiessele]

* Ensure numpy version is used in coregionalize cython test. [Jayanth Koushik]

* Use explicity cython/numpy variants in coregionalize test. [Jayanth Koushik]

* Fix cython check in linalg. [Jayanth Koushik]

* Don't override global cython config in tests. [Jayanth Koushik]

* Refactor checking for cython availability. [Jayanth Koushik]

* Use correct cython check in kernel_tests.py. [Jayanth Koushik]

* Merge branch 'devel' into esiivola-feature-multioutput. [mzwiessele]

* Merge branch 'devel' into devel. [Neil Lawrence]

* Rewrite poster_samples_f to return NxDxsize. [Neil Lawrence]

* Testing for dims should be checking whether 2nd dim is greater than 1. [Neil Lawrence]

* Update gp.py. [Neil Lawrence]

  Sample return seemed to have been based on number of training data, not number of posterior samples requested.

* Merge pull request #668 from RobRomijnders/devel. [Zhenwen Dai]

  fix typo in docstring for GP.opimize()

* Fix typo in docstring for GP.opimize() [robromijnders]

* Merge pull request #648 from marpulli/symmetric_kernel. [Zhenwen Dai]

  Add Symmetric kernel

* Expand class description and some speed improvements. [Mark Pullin]

* Make symmetric kernel work with python 2.7. [Mark Pullin]

* Add param descriptions. [Mark Pullin]

* Add symmetric kernel. [Mark Pullin]

* Merge pull request #654 from palindromik/devel. [Zhenwen Dai]

  Return deserialized models with actual type instead of base type

* Return deserialized models with actual type instead of base type. [Keerthana Elango]

* Merge branch 'feature-multioutput' of https://github.com/esiivola/GPy into esiivola-feature-multioutput. [mzwiessele]

* Merge remote-tracking branch 'origin/devel' into feature-multioutput. [Eero Siivola]

* Modified likelihoods test to better test multioutput likelihood. [Eero Siivola]

* Added a notice of the correspondence of the likelihood structure to the one of GPstuff due to their request. [Eero Siivola]

* Added a new class that enables using multiple likelihoods for multioutput case (previously, Mixed noise only allowed use of multiple gaussians) [Siivola Eero]

* Pkg: CHANGELOG. [mzwiessele]


## v1.9.3 (2018-07-27)

### Fix

* Python=3.7. [mzwiessele]

### Other

* Bump version: 1.9.2 → 1.9.3. [mzwiessele]

* Merge pull request #655 from davidsmf/patch-2. [Zhenwen Dai]

  Allow setup.py to be parsed without numpy

* Allow setup.py to be parsed without numpy. [David Sheldon]

  If numpy isn't available, don't define ext_mods, pip will then determine numpy is required, install it, then call us again.

  Fixes #653

* Merge pull request #640 from pgmoren/devel. [Zhenwen Dai]

  Sparse GP serialization

* Serialization: Add docstrings. [Moreno]

* Sparse GP serialization. [Moreno]

* Merge pull request #613 from dtorrejo/Multi_sample_bug. [Max Zwiessele]

  Fixes the dimensions of the samples output

* Maintains consistency with numpy arrays. [Diego Torrejon]

* Fixes the dimensions of the samples output. [Diego Torrejon]

* Merge pull request #607 from pgmoren/devel. [Zhenwen Dai]

  Add serialization functions for EPDTC

* Add serialization functions for EPDTC. [Moreno]

* Merge pull request #604 from SheffieldML/deploy. [Max Zwiessele]

  Deploy

* Use old deploy pypi behavior. [Max Zwiessele]

  Until skip_existing option exists, use the old travis dpl behaviour to not fail on existing files.

* Don’t build docs anymore in travis. [Max Zwiessele]

* Merge pull request #603 from SheffieldML/devel. [Max Zwiessele]

  1.9.*


## v1.9.2 (2018-02-22)

### Fix

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

* Rtd. [mzwiessele]

### Other

* Bump version: 1.9.1 → 1.9.2. [mzwiessele]


## v1.9.1 (2018-02-22)

### Fix

* Paramz newest version. [mzwiessele]

### Other

* Bump version: 1.9.0 → 1.9.1. [mzwiessele]


## v1.9.0 (2018-02-22)

### Other

* Bump version: 1.8.7 → 1.9.0. [mzwiessele]


## v1.8.7 (2018-02-22)

### Fix

* Merge deploy back into devel. [mzwiessele]

### Other

* Bump version: 1.8.6 → 1.8.7. [mzwiessele]

* Deploy version 1.8.5. [Zhenwen Dai]

  * added extended version of MLP function with multiple hidden layers and different activation functions

  * Update mapping_tests.py

  Make output of gradient check verbose to diagnose error

  * Update mapping_tests.py

  Remove verbosity again after gradient checks passed without problem with verbosity

  * the implementation of SVI-MOGP

  * Try to fix the issue with model_tests

  * updated mapping test to pass gradient checks

  * Fix random seed for reproducible results in tests

  * Add mean function functionality to dtc inference method

  * Fix DSYR function (See https://github.com/scipy/scipy/issues/8155)

  * Updated sde_kern to work with scipy=1.0.0

  * Trying to fix tests for Matplotlib plotting issue

  * Testing Again #575

  * Figured it must be a matplotlib import error #575

  New import matplotlib must be missing a package

  * Removed ImageComparisonFailure #575

  ImageComparisonFailure no longer exists which causes issues with travis testing using the most recent matplotlib

  * Fix EP for non-zero mean GP priors

  * improve the documentation for LVMOGP

  * remove non-ascii characters

  * Small correction to doc

  * add type into docstring

  * update changelog for 1.8.5

  * bump the version: 1.8.4 -> 1.8.5


## v1.8.6 (2018-02-22)

### Fix

* Gamma prior no assignment after init. [mzwiessele]

* #568, product kernel resolution. [mzwiessele]

* #590. [Max Zwiessele]

  Y_normalized was not used for running optimization

* Appveyor comment missing. [mzwiessele]

### Other

* Bump version: 1.8.5 → 1.8.6. [mzwiessele]

* Merge pull request #597 from marpulli/devel. [Max Zwiessele]

  Allow calculation of full predictive covariance matrices with multipl…

* Allow calculation of full predictive covariance matrices with multiple outputs and normalization. [Mark Pullin]

* Merge pull request #600 from marpulli/plotting_fix. [Max Zwiessele]

  Fix visible dimensions for plotting inducing points

* Fix visible dimensions for plotting inducing points. [Mark Pullin]

* Merge pull request #599 from marpulli/grads_efficiency. [Zhenwen Dai]

  Make predictive_gradients more efficient

* Make predictive_gradients more efficient. [Mark Pullin]

* Merge pull request #587 from esiivola/feature-multioutput. [Zhenwen Dai]

  Merge the implementation of Multioutput kernel

* Changed two function names so that they follow the python naming convention. [Siivola Eero]

* Merge remote-tracking branch 'origin' into feature-multioutput. [Eero Siivola]

* Merge pull request #592 from SheffieldML/sparsegp-normalization. [Zhenwen Dai]

  fix: #590

* Merge pull request #589 from apaleyes/devel. [Zhenwen Dai]

  Implemented utility function to compute covariance between points in GP Model

* Moved posterior_covariance to Posterior class. [Andrei Paleyes]

* Implemented utility function to compute covariance between points in GP Model. [Andrei Paleyes]

* Changed the structure of multioutput kernel so that it doesn't change the API of Kernels + documented the class. [Eero Siivola]

* Merge remote-tracking branch 'origin/devel' into feature-multioutput. [Eero Siivola]

* Merge pull request #585 from YoshikawaMasashi/devel. [Zhenwen Dai]

  modify the MLP kernel equation

* Modify the MLP kernel equation. [masashi yoshikawa]

* Added multioutput kern and tests. [Eero Siivola]

* Multioutput kernel + initial test. [Siivola Eero]

* Multioutput kernel + initial test. [Siivola Eero]

* Change dtype for Python 3 in robot_wirelss. [Neil Lawrence]

* Bump the version: 1.8.4 -> 1.8.5. [Zhenwen Dai]

* Update changelog for 1.8.5. [Zhenwen Dai]

* Merge pull request #579 from SheffieldML/multi_out_doc. [Zhenwen Dai]

  Improve the documentation for LVMOGP

* Add type into docstring. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into multi_out_doc. [Zhenwen Dai]

* Remove non-ascii characters. [Zhenwen Dai]

* Improve the documentation for LVMOGP. [Zhenwen Dai]

* Merge pull request #580 from marpulli/devel. [Zhenwen Dai]

  Small correction to doc

* Small correction to doc. [Mark Pullin]

* Merge pull request #578 from pgmoren/devel. [Zhenwen Dai]

  Fix EP for non-zero mean GP priors (binary classification)

* Fix EP for non-zero mean GP priors. [Moreno]

* Merge pull request #572 from marpulli/devel. [Alan Saul]

  Add mean function functionality to dtc inference method

* Add mean function functionality to dtc inference method. [Mark Pullin]

* Merge pull request #573 from pgmoren/devel. [Zhenwen Dai]

  Fix DSYR function (See https://github.com/scipy/scipy/issues/8155)

* Fix DSYR function (See https://github.com/scipy/scipy/issues/8155) [Moreno]

* Merge pull request #574 from alansaul/lyapunov_fix. [Alan Saul]

  Fixing scipy=1.0.0 incompatibility of lyapunov discovered in PR #573. Coverage issue should be resolved by PR #575.

* Updated sde_kern to work with scipy=1.0.0. [Alan Saul]

* Merge pull request #575 from SheffieldML/matplotlib_testing. [Alan Saul]

  Fixing tests for Matplotlib plotting issue

* Removed ImageComparisonFailure #575. [Alan Saul]

  ImageComparisonFailure no longer exists which causes issues with travis testing using the most recent matplotlib

* Figured it must be a matplotlib import error #575. [Alan Saul]

  New import matplotlib must be missing a package

* Testing Again #575. [Alan Saul]

* Trying to fix tests for Matplotlib plotting issue. [Alan Saul]

* Merge pull request #526 from msbauer/mlp_extended. [Zhenwen Dai]

  added extended version of MLP function

* Fix random seed for reproducible results in tests. [msbauer]

* Updated mapping test to pass gradient checks. [msbauer]

* Update mapping_tests.py. [msbauer]

  Remove verbosity again after gradient checks passed without problem with verbosity

* Update mapping_tests.py. [msbauer]

  Make output of gradient check verbose to diagnose error

* Added extended version of MLP function with multiple hidden layers and different activation functions. [Bauer]

* Merge pull request #562 from SheffieldML/external-mo. [Zhenwen Dai]

  Release the implementation of LVMOGP

* Try to fix the issue with model_tests. [Zhenwen Dai]

* Merge with new changes from devel. [Zhenwen Dai]

* Merge pull request #561 from SheffieldML/deploy. [Max Zwiessele]

  Deploy

* Merge pull request #560 from SheffieldML/devel. [Max Zwiessele]

  appveyor twine upload error fix

* Merge branch 'deploy' into devel. [Max Zwiessele]

* Merge pull request #558 from SheffieldML/devel. [Max Zwiessele]

  Uniform prior fix for other domains

* Merge pull request #559 from SheffieldML/PS-upload-error. [Max Zwiessele]

  Update appveyor.yml

* The implementation of SVI-MOGP. [Zhenwen Dai]


## v1.8.4 (2017-10-06)

### Other

* Bump version: 1.8.3 → 1.8.4. [mzwiessele]

* Update appveyor.yml. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* Merge branch 'deploy' into devel. [Max Zwiessele]

* Merge pull request #557 from SheffieldML/devel. [Max Zwiessele]

  Paramz 0.8 update

* Merge pull request #544 from SheffieldML/devel. [Zhenwen Dai]

  Release GPy 1.8.x


## v1.8.3 (2017-10-02)

### Fix

* Uniform prior instantiation. [mzwiessele]

* Use paramz 0.8.5. [mzwiessele]

### Other

* Bump version: 1.8.2 → 1.8.3. [mzwiessele]


## v1.8.2 (2017-10-02)

### Fix

* Uniform prior tests. [mzwiessele]

* Uniform prior can be positive and negative, depending on lower and upper bound. [mzwiessele]

### Other

* Bump version: 1.8.1 → 1.8.2. [mzwiessele]


## v1.8.1 (2017-10-02)

### Fix

* Paramz 0.8. [mzwiessele]

* Creating new kernel was missing active dims. [mzwiessele]

* Naming of ard contribution in ARD plot. [Max Zwiessele]

* Replacing np.power with np.square for efficiency. [Chris Tomaszewski]

* Slight modification to MLP mapping to reduce potential for numpy overflows and unnecessary computation. [Chris Tomaszewski]

### Other

* Bump version: 1.8.0 → 1.8.1. [mzwiessele]

* Paramz 0.8. [Max Zwiessele]

* Paramz 0.8. [Max Zwiessele]

* Merge pull request #549 from christomaszewski/devel. [Zhenwen Dai]

  fix: slight modification to MLP mapping to reduce potential for numpy…

* Merge pull request #545 from pgmoren/devel. [Zhenwen Dai]

  Thanks Pablo for the changes about Basic framework for serializing GPy models. I think we need the corresponding docstrings as developer documentation.

* Basic framework for serializing GPy models. [Moreno]

* Merge pull request #543 from icdishb/devel. [Zhenwen Dai]

  Merged in the changes about Input warping using Kumar warping.

  Thanks for the contribution, Huibin!

* Input warping using Kumar warping. [Shen]

* Merge branch 'deploy' into devel. [Max Zwiessele]

* Merge branch 'devel' into deploy. [mzwiessele]

* Merge branch 'devel' into deploy. [mzwiessele]

* Merge branch 'deploy' of github.com:SheffieldML/GPy into deploy. [mzwiessele]

* Merge pull request #497 from SheffieldML/devel. [Max Zwiessele]

  New fixes for parallel optimize and minor fixes


## v1.8.0 (2017-09-11)

### Fix

* Updated keywords. [mzwiessele]

* Beiwang will add GMM in full. [mzwiessele]

* Kernel tests and variational tests. [mzwiessele]

* Plotting tests for new matplotlib. [mzwiessele]

* Model tests numpy integer error. [mzwiessele]

* Replot with new matplotlib. [mzwiessele]

* Offline plotting workaround with squeezing arrays. [mzwiessele]

* Fixed numpy 1.12 indexing and shape preservation. [mzwiessele]

### Other

* Bump version: 1.7.7 → 1.8.0. [mzwiessele]

* Merge pull request #527 from adhaka/ll-surv. [Zhenwen Dai]

  merged in the changes about Ll surv

* Merging with the main devel branch. [Akash Kumar Dhaka]

* Correcting weibull- changing parameterisation from f to exp(f) similar to loglogistic. [Akash Kumar Dhaka]

* Change import statements for calling locally. [Akash Kumar Dhaka]

* Fresh branch for these new likelihood functions along with test code- will work atleast with La[lace approximation ... [Akash Kumar Dhaka]

* Merge pull request #542 from SheffieldML/EP_refactor. [Zhenwen Dai]

  Merge in the EP refactoring

* Merge pull request #534 from adhaka/EP_refactor. [Zhenwen Dai]

  Ep refactor

* Some more merge conflicts resolved for syncing with the current devel branch of main GPy. [Akash Kumar Dhaka]

* Bug fixes in test cases due to changes in api of ep functions.. [Akash Kumar Dhaka]

* Robustified binomial likelihood. [Siivola Eero]

* Minor bugfix. [Siivola Eero]

* Fixed two small lbugs. [Siivola Eero]

* Putting exact_inference_gradients again and calling in instead from ep_gradients which is the genreic method for calculating expected gradients used in ep. [Akash Kumar Dhaka]

* Commiting code for some changes to api for calculating ep_gradients, also fixing some issues with gaussian hermite quadrature, no we have both avaialable ... [Akash Kumar Dhaka]

* Adding file for gaussian-kronrod integration, test cases and calculating gradients of log marginal wrt theta-likelihood params. [Akash Kumar Dhaka]

* Fixing a typo-bug in the last commit for ep test case. [Akash Kumar Dhaka]

* Merging with the gpy devel branch to be in sync with the latest code and make pull request again .. [Akash Kumar Dhaka]

* Bug fixes in test cases due to changes in api of ep functions.. [Akash Kumar Dhaka]

* Robustified binomial likelihood. [Siivola Eero]

* Minor bugfix. [Siivola Eero]

* Fixed two small lbugs. [Siivola Eero]

* Putting exact_inference_gradients again and calling in instead from ep_gradients which is the genreic method for calculating expected gradients used in ep. [Akash Kumar Dhaka]

* Commiting code for some changes to api for calculating ep_gradients, also fixing some issues with gaussian hermite quadrature, no we have both avaialable ... [Akash Kumar Dhaka]

* Adding file for gaussian-kronrod integration, test cases and calculating gradients of log marginal wrt theta-likelihood params. [Akash Kumar Dhaka]

* Fixing a typo-bug in the last commit for ep test case. [Akash Kumar Dhaka]

* Removing pods dependency and a few print commands. [Akash Kumar Dhaka]

* Bug fix to prevent infinite loop because of incorrect stopping condition. [Akash Kumar Dhaka]

* Adding some test cases for EP.. more work needs to be done, these are some high level test cases .. [Akash Kumar Dhaka]

* Refactor EP and EPDTC. [Moreno]

* Added initial version of the refactored EP. [esiivola]

* Added initial version of the refactored EP. [esiivola]

* Added initial version of the refactored EP. [esiivola]

* Bump version: 1.6.1 → 1.6.2. [mzwiessele]

* Fix in sparse_gp_mpi optimizer. [Alex Feldstein]

* Fix for parallel optimization. [Alex Feldstein]

* Changes in EP/EPDTC to fix numerical issues and increase the flexibility of the inference. [Moreno]

  Changes to avoid numerical issues and improve the performance:
      - Keep value of the EP parameters between calls
      - Enforce positivity of tau_tilde
      - Stable computation of the EP moments for the Bernoulli likelihood
      - Compute marginal in the GP model without directly inverting tau_tilde

      Changes to improve the flexibility:
      - Add parameter for maximum number of iterations
      - Distinguish between alternated/nested mode
      - Distinguish between sequential/parallel updates in EP

* Cython in linalg. [Max Zwiessele]

  did set cython to working if linalg_cython was importable.

* Bump version: 1.6.0 → 1.6.1. [mzwiessele]

* Bump version: 1.5.9 → 1.6.0. [mzwiessele]

* Provide two classes for plotly plots to remove global variable. [Alex Feldstein]

* Bump version: 1.5.8 → 1.5.9. [mzwiessele]

* Bump version: 1.5.7 → 1.5.8. [mzwiessele]

* Add offline plotting for plotly. [Alex Feldstein]

* Merge pull request #513 from adhaka/EP_refactor. [Zhenwen Dai]

  adding some test cases for EP.. more work needs to be done, these are…

* Removing pods dependency and a few print commands. [Akash Kumar Dhaka]

* Bug fix to prevent infinite loop because of incorrect stopping condition. [Akash Kumar Dhaka]

* Adding some test cases for EP.. more work needs to be done, these are some high level test cases .. [Akash Kumar Dhaka]

* Merge pull request #512 from pgmoren/EP_refactor. [Zhenwen Dai]

  Refactor EP/EPDTC

* Refactor EP and EPDTC. [Moreno]

* Merge pull request #508 from esiivola/EP_refactor. [Zhenwen Dai]

  Bring the ongoing working about EP refactoring

* Added initial version of the refactored EP. [esiivola]

* Added initial version of the refactored EP. [esiivola]

* Added initial version of the refactored EP. [esiivola]


## v1.7.7 (2017-06-19)

### Other

* Bump version: 1.7.6 → 1.7.7. [mzwiessele]


## v1.7.6 (2017-06-19)

### Fix

* Appveyor not uploading to testpypi for now. [mzwiessele]

### Other

* Bump version: 1.7.5 → 1.7.6. [mzwiessele]


## v1.7.5 (2017-06-19)

### Fix

* Splitting forecast tests into 3 to circumvent 10 minute stop of travis. [mzwiessele]

### Other

* Bump version: 1.7.4 → 1.7.5. [mzwiessele]


## v1.7.4 (2017-06-19)

### Fix

* Paramz version for parallel optimization fix. [mzwiessele]

### Other

* Bump version: 1.7.3 → 1.7.4. [mzwiessele]


## v1.7.3 (2017-06-19)

### Fix

* Appveyor build failing. [mzwiessele]

### Other

* Bump version: 1.7.2 → 1.7.3. [mzwiessele]


## v1.7.2 (2017-06-17)

### Fix

* Appveyor build python 3.6. [mzwiessele]

### Other

* Bump version: 1.7.1 → 1.7.2. [mzwiessele]


## v1.7.1 (2017-06-17)

### Fix

* Appveyor build python 3.6. [mzwiessele]

### Other

* Bump version: 1.7.0 → 1.7.1. [mzwiessele]


## v1.7.0 (2017-06-17)

### Fix

* Support for 3.5 and higher now that 3.6 is out. [mzwiessele]

### Other

* Bump version: 1.6.3 → 1.7.0. [mzwiessele]


## v1.6.3 (2017-06-17)

### Other

* Bump version: 1.6.2 → 1.6.3. [mzwiessele]

* Merge pull request #504 from rmcantin/devel. [Max Zwiessele]

* Fix python 2-3 compatibility. [Ruben Martinez-Cantin]

* Merge pull request #511 from dirmeier/devel. [Max Zwiessele]

* Added LICENSE file to MANIFEST.in. [dirmeier]


## v1.6.2 (2017-04-12)

### Fix

* Updated keywords. [mzwiessele]

### Other

* Bump version: 1.6.1 → 1.6.2. [mzwiessele]

* Merge pull request #491 from alexfeld/parallel_opt. [Max Zwiessele]

  fix for parallel optimization

* Fix in sparse_gp_mpi optimizer. [Alex Feldstein]

* Fix for parallel optimization. [Alex Feldstein]

* Merge pull request #492 from pgmoren/devel. [Zhenwen Dai]

  We did some benchmarking on classification. These changes should be fine. Let's merge it in.

* Changes in EP/EPDTC to fix numerical issues and increase the flexibility of the inference. [Moreno]

  Changes to avoid numerical issues and improve the performance:
      - Keep value of the EP parameters between calls
      - Enforce positivity of tau_tilde
      - Stable computation of the EP moments for the Bernoulli likelihood
      - Compute marginal in the GP model without directly inverting tau_tilde

      Changes to improve the flexibility:
      - Add parameter for maximum number of iterations
      - Distinguish between alternated/nested mode
      - Distinguish between sequential/parallel updates in EP

* Merge pull request #489 from SheffieldML/linalg_cython-1. [Max Zwiessele]

  cython in linalg fix #458

* Cython in linalg. [Max Zwiessele]

  did set cython to working if linalg_cython was importable.

* Merge pull request #486 from SheffieldML/deploy. [Max Zwiessele]

  Merge pull request #471 from SheffieldML/devel

* Merge pull request #471 from SheffieldML/devel. [Max Zwiessele]

  new version


## v1.6.1 (2017-02-28)

### Fix

* Beiwang will add GMM in full. [mzwiessele]

### Other

* Bump version: 1.6.0 → 1.6.1. [mzwiessele]


## v1.6.0 (2017-02-28)

### Fix

* Kernel tests and variational tests. [mzwiessele]

* Plotting tests for new matplotlib. [mzwiessele]

* Model tests numpy integer error. [mzwiessele]

* Replot with new matplotlib. [mzwiessele]

* Offline plotting workaround with squeezing arrays. [mzwiessele]

* Fixed numpy 1.12 indexing and shape preservation. [mzwiessele]

### Other

* Bump version: 1.5.9 → 1.6.0. [mzwiessele]

* Merge branch 'devel' into alexfeld-offline_plotly. [mzwiessele]

* Merge branch 'devel' into alexfeld-offline_plotly. [mzwiessele]

* Merge branch 'offline_plotly' of git://github.com/alexfeld/GPy into alexfeld-offline_plotly. [mzwiessele]

* Provide two classes for plotly plots to remove global variable. [Alex Feldstein]

* Add offline plotting for plotly. [Alex Feldstein]


## v1.5.9 (2017-02-23)

### Other

* Bump version: 1.5.8 → 1.5.9. [mzwiessele]

* Merge remote-tracking branch 'origin/deploy' into devel. [mzwiessele]

* Merge pull request #455 from SheffieldML/devel. [Max Zwiessele]

  1.5.6


## v1.5.8 (2017-02-23)

### Fix

* Predictive_gradients for new posterior class. [mzwiessele]

* Removed additional dict line. [mzwiessele]

* Plotting also allows 3D (capitals) [mzwiessele]

* Fallback for when no environment variables are set (#467) [Safrone]

  * fix: dev: add or in home directory getting

  adds another or when getting the home directory with os.getenv() so that if neither $HOME nor $USERPROFILE environment variable is set, os.path.join() will not fail by getting a None and the config will revert to the default configuration file.

  * fix: remove extra statement

### Other

* Bump version: 1.5.7 → 1.5.8. [mzwiessele]

* Update ss_gplvm.py. [Zhenwen Dai]

  resolve the future warning: FutureWarning:comparison to `None` will result in an elementwise object comparison in the future.

* Merge pull request #472 from SheffieldML/predictive_gradients. [Max Zwiessele]

  fix: predictive_gradients for new posterior class

* Merge pull request #470 from SheffieldML/plotting_fix. [Max Zwiessele]

  Plotting fix

* Bump version: 1.5.6 → 1.5.7. [mzwiessele]

* Changed the order of the operations, ensuring that the covariance matrix is symmetric despite numerical precision issues. Suggested by Alan. [Teo de Campos]

* Delete gmm_bayesian_gplvm.py. [beiwang]

* Gmm_creation. [beiwang]

* Gmm_creation. [beiwang]


## v1.5.6 (2016-11-07)

### New

* Added ploy basis kernel tests and import. [mzwiessele]

* Gitchangelogrc. [mzwiessele]

### Changes

* Added polynomial basis func kernel. [mzwiessele]

### Fix

* Installation #451. [Max Zwiessele]

* Pandoc install under travis osx. [mzwiessele]

* Pandoc install under travis osx. [mzwiessele]

* Pypi changing to pypi.org. [mzwiessele]

### Other

* Bump version: 1.5.5 → 1.5.6. [mzwiessele]

* Merge pull request #448 from thangbui/devel. [Max Zwiessele]

  Added pep.py -- Sparse Gaussian processes using Power Expectation Propagation

* Renamed pep test scripts. [Thang Bui]

* Fixed seed in pep test script #448. [Thang Bui]

* Added tests. [Thang Bui]

* Added pep.py -- Sparse Gaussian processes using Power Expectation Propagation. [Thang Bui]

  This allows interpolation between FITC (EP or alpha = 1), and Titsias's variational (VarDTC, VFE when alpha = 0).

* Merge pull request #452 from SheffieldML/setupreq. [Max Zwiessele]

  fix: Installation #451

* Merge pull request #447 from SheffieldML/polinomial. [Max Zwiessele]

  Polynomial

* Merge branch 'devel' into polinomial. [mzwiessele]

* Merge pull request #449 from SheffieldML/deploy. [Max Zwiessele]

  Deploy

* Update setup.py. [Mike Croucher]

* Merge pull request #446 from SheffieldML/devel. [Max Zwiessele]

  newest patch fixing some issues

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* Merge branch 'deploy' into devel. [Max Zwiessele]

* Merge pull request #442 from SheffieldML/devel. [Max Zwiessele]

  New Major for GPy

* Merge pull request #426 from SheffieldML/devel. [Max Zwiessele]

  some fixes from issues and beckdaniels warped gp improvements


## v1.5.5 (2016-10-03)

### Other

* Bump version: 1.5.4 → 1.5.5. [Max Zwiessele]


## v1.5.4 (2016-10-03)

### Changes

* Version update on paramz. [Max Zwiessele]

* Fixed naming in variational priors : [Max Zwiessele]

### Fix

* Bug in dataset (in fn download_url) which wrongly interprets the Content-Length meta data, and just takes first character. [Michael T Smith]

### Other

* Bump version: 1.5.3 → 1.5.4. [Max Zwiessele]

* Merge pull request #443 from SheffieldML/dataset_download_url_bugfix. [Max Zwiessele]

  fix: Bug in datasets.py

* Merge branch 'kurtCutajar-devel' into devel. [mzwiessele]


## v1.5.3 (2016-09-06)

### Other

* Bump version: 1.5.2 → 1.5.3. [mzwiessele]

* Merge branch 'devel' into kurtCutajar-devel. [mzwiessele]

* [doc] cleanup. [mzwiessele]

* [merge] into new devel. [Max Zwiessele]

* Making consistent with python 3. [kcutajar]

* Fixed incorrect import. [kcutajar]

* Fixed incorrect import. [kcutajar]

* Removed erreneous lines indicating merge conflicts. [kcutajar]

* Fixed conflicts. [kcutajar]

* Minor fix. [kcutajar]

* Removed SSM functionality - updated Kronecker grid case. [kcutajar]

* Added kernels for GpGrid and GpSsm regression. [kcutajar]

* Added core code for GpSSM and GpGrid. [kcutajar]

* Added fixes to repo + rebased. [kcutajar]

* Minor fix. [kcutajar]

* Removed SSM functionality - updated Kronecker grid case. [kcutajar]

* Added kernels for GpGrid and GpSsm regression. [kcutajar]

* Added core code for GpSSM and GpGrid. [kcutajar]


## v1.5.2 (2016-09-06)

### New

* Added deployment pull request instructions for developers. [mzwiessele]

### Other

* Bump version: 1.5.1 → 1.5.2. [mzwiessele]

* Minor readme changes. [mzwiessele]


## v1.5.1 (2016-09-06)

### Fix

* What's new update fix #425 in changelog. [mzwiessele]

### Other

* Bump version: 1.5.0 → 1.5.1. [mzwiessele]


## v1.5.0 (2016-09-06)

### New

* Using gitchangelog to keep track of changes and log new features. [mzwiessele]

### Other

* Bump version: 1.4.3 → 1.5.0. [mzwiessele]


## v1.4.3 (2016-09-06)

### Changes

* Changelog update. [mzwiessele]

### Other

* Bump version: 1.4.2 → 1.4.3. [mzwiessele]


## v1.4.2 (2016-09-06)

### Other

* Bump version: 1.4.1 → 1.4.2. [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* [kern] fix #440. [mzwiessele]


## v1.4.1 (2016-09-06)

### Other

* Bump version: 1.4.0 → 1.4.1. [mzwiessele]

* [setup] added bumpversion tagging again. [mzwiessele]

* [mrd] init updates and added tests. [mzwiessele]

* Merge pull request #356 from SheffieldML/binomial_laplace. [Max Zwiessele]

  Binomial laplace #352

* Merge branch 'devel' into binomial_laplace. [Max Zwiessele]

* Added binomial derivative and test. [Alan Saul]

* Merge branch 'devel' into fixed_inputs. [Alan Saul]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Alan Saul]

* Update README.md. [Max Zwiessele]

* Merge branch 'bwengals-devel' into devel. [Max Zwiessele]

* [basisfunc] kernel tests and model tests. [Max Zwiessele]

* Didnt realize last 2 changes went to PR, undoing. [Bill]

* Fixed import issue. [Bill]

* Trying to make gp_kronecker models savable. [Bill]

* Removed Logsumexp() from LogisticBasisFuncKernel, allowing slope parameter to be negative.  Also removed unnecessary scipy import. [Bill]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge pull request #433 from SheffieldML/update_initialize_paramz. [Max Zwiessele]

  [init] updated readme

* [init] updated readme. [mzwiessele]

* Merge pull request #432 from SheffieldML/mathDR-studentTprior. [Max Zwiessele]

  @Mathdr student t prior

* [prior] singleton tested. [mzwiessele]

* [studentT] prior by @mathDR. [mzwiessele]

* Removed spectral mixture stuff. [mathDR]

* [inference] rename wrong precision into variance. [Max Zwiessele]

* [plots] rerun baseline. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Bump version: 1.3.2 → 1.4.0. [mzwiessele]

* [warped stuff] plotting and normalizer in warped gps. [mzwiessele]

* Bump version: 1.3.1 → 1.3.2. [mzwiessele]

* [paramz] update included. [mzwiessele]

* Bump version: 1.3.0 → 1.3.1. [mzwiessele]

* [appveyor] skip existing twine upload. [mzwiessele]

* Bump version: 1.2.1 → 1.3.0. [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* Merge pull request #421 from SheffieldML/beckdaniel-wgps_improvements. [Max Zwiessele]

  Beckdaniel wgps improvements

* Merge branch 'devel' into beckdaniel-wgps_improvements. [mzwiessele]

* [merge] devel. [mzwiessele]

* Revert "Revert "[kern] Add kernel was swallowing parts #fix #412"" [mzwiessele]

  This reverts commit 0abb9b835ffeb020410bdf9a1e0532139ffa5cfc.

* Revert "[kern] Add kernel was swallowing parts #fix #412" [mzwiessele]

  This reverts commit b8867f1552c05244dcd5ba38a7a57b6f1056312c.

* [kern] Add kernel was swallowing parts #fix #412. [mzwiessele]

* Merge branch 'devel' into beckdaniel-wgps_improvements. [mzwiessele]

* Merge branch 'wgps_improvements' of https://github.com/beckdaniel/GPy into beckdaniel-wgps_improvements. [mzwiessele]

* Improving coverage and removing py2 print. [beckdaniel]

* Merged last devel. [beckdaniel]

* Removed the check on f(y), it was only useful in logtanh. [beckdaniel]

* Added tests for closed inverse in identity and log. [beckdaniel]

* Changed imports to relative ones. [beckdaniel]

* Replicated the cubic sine example into warped_gp tests for code coverage. [beckdaniel]

* Removed logtanh, put into a new branch. [beckdaniel]

* Added an exception when you input 0 or negative values to logtanh function. [beckdaniel]

* Added a log-tanh function. [beckdaniel]

* Cleaning. [beckdaniel]

* Moved cubic sine from tests to examples. [beckdaniel]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into wgps_improvements. [beckdaniel]

  Merging new devel

* Refactored the numeric inverse into the mother class, to test Identity and Log. [beckdaniel]

* Cleaning. [beckdaniel]

* Added a rate to inverse calculation. [beckdaniel]

* Logistic seems working but more tests are needed. [beckdaniel]

* First try on logistic function. [beckdaniel]

* Doctest on TanhFunction. [beckdaniel]

* Renamed TanhWarpingFunction to TanhFunction. [beckdaniel]

* Some cleaning on warped gp. [beckdaniel]

* Deleted old tanh_warp and renamed warp_tanh_d to warp_tanh. [beckdaniel]

* Cleaning on warping functions. [beckdaniel]

* Cleaning on warping functions. [beckdaniel]

* Added a log warping function. [Daniel Beck]

* Stuff. [beckdaniel]

* Skipping the wgps Snelson's test (comment the skip line to see the plots) [beckdaniel]

* (wpgs) fixing newton-raphson for f_inv and fixing plotting stuff. [beckdaniel]

* [priorizable] small edit to be usable with paramz 0.6.1 and greater. [mzwiessele]

* [priorizable] small edit to be usable with paramz 0.6.1 and greater. [mzwiessele]

* [appveyor] warning on existing whls. [mzwiessele]

* [plotting] small edit. [mzwiessele]

* [readme] added deploy status to readme. [mzwiessele]

* [Add] add kernel swallowed parts fix #412. [mzwiessele]

* Merge pull request #422 from SheffieldML/offset_and_clustering. [Mike Smith]

  Offset and clustering: Utility to do clustering (greedy, pairs best clusters until likelihood stops increasing). Also includes a model that has an offset parameter to allow aligning of time series.

* Removing 'threaded' version. [Michael T Smith]

* Push just to rerun testing. [Michael T Smith]

* Don't use message added to cluster code. [Michael T Smith]

* Added threaded option - but this doesn't work due to the global interpreter lock. [Ubuntu]

* Made initial lengthscale!=1 to ensure we're properly testing gradients. [Michael T Smith]

* Modified set code in test to work with python 2 and python 3. [Michael T Smith]

* Corrected mistake in gradients: Was finding d(Xi-Xj)/dOffset instead of dr/dOffset. Fixed by scaling by kernel lengthscale. [Michael T Smith]

* Added GPy import. [Michael T Smith]

* More useful message from testing re offset estimate. [Michael T Smith]

* Corrected v2 missing print brackets. Added test code for new model and util. [Michael T Smith]

* Offset model and clustering utility. [Michael T Smith]

* Adding refs to new clustering and offset model and utility. [Michael T Smith]

* Merge pull request #420 from SheffieldML/deploy. [Max Zwiessele]

  Deploy

* Merge pull request #419 from SheffieldML/devel. [Max Zwiessele]

  fixed issues

* [merge] [mzwiessele]

* Merge pull request #418 from SheffieldML/plotting_tests. [Max Zwiessele]

  Plotting tests and mean funcs

* [secret] coveralls only on travis. [mzwiessele]

* [prod] fix #388. [mzwiessele]

* [stationary] hint at confusing definition in GPy. [mzwiessele]

* [coverage] both coveralls and codecov, showing codecov. [mzwiessele]

* Coveralls token. [mzwiessele]

* Coveralls token in appveyor.yml. [mzwiessele]

* Revert "Revert "[coverage reports] change to coveralls as test"" [mzwiessele]

  This reverts commit ee23da6dd9405120bec62402abf7aaa228a87a19.

* Revert "[coverage reports] change to coveralls as test" [mzwiessele]

  This reverts commit 040ac72b82b6aa39720abe9817619103892b27a1.

* [coverage reports] change to coveralls as test. [mzwiessele]

* [plotting] last full upadate of baselinge. [mzwiessele]

* [plotting] more test updates and check for errors. [mzwiessele]

* [tests] all tests. [mzwiessele]

* [merge] devel and plotting tests from meanfunc. [mzwiessele]

* [plotting] updated for test skips. [mzwiessele]

* [mean_func] added parameters in additive mean func and tests for mean functions. [mzwiessele]

* Merge branch 'devel' into plotting_tests. [mzwiessele]

* [datasets] rnaseq changed up. [Max Zwiessele]

* [baseline] images adjusted and checked for testing including gplvm. [Max Zwiessele]

* [testsave] saved the testmodel for quicker and more robust plotting. [Max Zwiessele]

* Merge branch 'devel' into plotting_tests. [Max Zwiessele]

* [plotting] adjusting tests for quicker plotting. [Max Zwiessele]

* Bump version: 1.2.0 → 1.2.1. [mzwiessele]

* Update README.md. [Max Zwiessele]

* Merge pull request #415 from SheffieldML/devel. [Max Zwiessele]

  deploy

* Bump version: 1.1.3 → 1.2.0. [mzwiessele]

* [travis] not tagged. [mzwiessele]

* Merge pull request #413 from SheffieldML/devel. [Max Zwiessele]

  appveyor deploy

* Bump version: 1.1.2 → 1.1.3. [mzwiessele]

* [coverage] default to devel. [mzwiessele]

* Bump version: 1.1.1 → 1.1.2. [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* Merge branch 'deploy' into devel. [Max Zwiessele]

* Merge pull request #402 from SheffieldML/devel. [Max Zwiessele]

  1.0.9 on deploy

* [codecov] default branch. [mzwiessele]

* [appveyor] twine? [mzwiessele]

* [appveyor] elsif ps. [mzwiessele]

* [appveyor] elsif ps. [mzwiessele]

* [appveyor] stop deploy on github. [mzwiessele]

* [appveyor] stop deploy on github. [mzwiessele]

* Bump version: 1.1.0 → 1.1.1. [mzwiessele]

* [appveyor] test deploy and full deploy. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [deploy] branch check. [mzwiessele]

* [version] for github releases? [mzwiessele]

* [secure] yes but not on prs. [mzwiessele]

* [secure] does it actually work? [mzwiessele]

* [secure] does it actually work? [mzwiessele]

* [secure] does it actually work? [mzwiessele]

* [secure] does it actually work? [mzwiessele]

* [appveyor] secure password 3. [mzwiessele]

* [appveyor] secure password 3. [mzwiessele]

* [appveyor] secure password 2. [mzwiessele]

* [appveyor] secure password 2. [mzwiessele]

* [appveyor] secure password. [mzwiessele]

* [appveyor] twine. [mzwiessele]

* [appveyor] echo empty line. [mzwiessele]

* [appveyor] pypirc. [mzwiessele]

* [appveyor] pypirc. [mzwiessele]

* [appveyor] pypirc. [mzwiessele]

* [appveyor] pypirc. [mzwiessele]

* [appveyor] twine upload. [mzwiessele]

* [setup] more setup changes. [mzwiessele]

* [setup] more setup changes. [mzwiessele]

* [setup] more setup changes. [mzwiessele]

* [appveyor] upload is not working? [mzwiessele]

* [appveyor] version in gh releases. [mzwiessele]

* [appveyor] version in gh releases. [mzwiessele]

* [appveyor] use twine. [mzwiessele]

* [appveyor] use twine. [mzwiessele]

* [appveyor] trying out github relesaessese. [mzwiessele]

* [appveyor] on_success. [mzwiessele]

* [tests] skip on climin import error. [mzwiessele]

* [appveyor] build script. [mzwiessele]

* [py33] removed from builds. [mzwiessele]

* [appveyor] tests. [mzwiessele]

* [appveyor] user. [mzwiessele]

* Bump version: 1.0.9 → 1.1.0. [mzwiessele]

* [secure] password added to appveyor. [mzwiessele]

* Merge pull request #408 from mikecroucher/devel. [Mike Croucher]

  Automated Windows builds using Appveyor

* Appveyor: Py27 and Py35 builds. [Mike Croucher]

* Another appveyor bug fix. [Mike Croucher]

* Appveyor bug fix. [Mike Croucher]

* Appveyor: Use Miniconda for numpy and scipy. [Mike Croucher]

* Appveyor: Only build the deploy branch. [Mike Croucher]

* First attempt to use appveyor for windows builds. [Mike Croucher]

* Trying to be more specific. [Ricardo Andrade]

* Trying to be more specific. [Ricardo Andrade]

* Merge branch 'deploy' into devel. [Max Zwiessele]

* [py3] iterator .next fixes. [Max Zwiessele]

* [imports] fix #392. [Max Zwiessele]

* [#403] fix of inconsistent config naming. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge pull request #398 from SheffieldML/integral. [lionfish0]

  Integral kernels added

* Modified testing to allow not implemented exceptions in update_gradients_full. [Michael T Smith]

* References integral kernel classes. [Michael T Smith]

* Merge remote-tracking branch 'origin/devel' into integral. [Michael T Smith]

* Merge pull request #397 from avehtari/dev-python3. [Alan Saul]

  Python2->Python3

* Merge pull request #4 from alansaul/dev-python3. [Aki Vehtari]

  Added test for subarray in util to dev-python3

* Added test for subarray in util. [Alan Saul]

* More Python 3 compatibility fixes. [Aki Vehtari]

* Python2->Python3. [Aki Vehtari]

* Removed references to integral kernels from __init__ [Michael T Smith]

* Improved comments. import future added. Fixed exception. [Michael T Smith]

* New tests for kernel. [Michael T Smith]

* Integral kernels added. [Michael T Smith]

* Integral kernels removed from index (allows proper pull request) [Michael T Smith]

* [travis] updates for the coverage reports. [Max Zwiessele]

* Merge pull request #374 from SheffieldML/gradientsxx. [Max Zwiessele]

  Changes in kern.gradients_XX

* [integral] py3 compat. [Max Zwiessele]

* [integral] py3 compatability. [Max Zwiessele]

* [grads x] [Max Zwiessele]

* [grads x] diagonal entries fixed and add kernel adjusted. [Max Zwiessele]

* [dxxdiag] some steps towards the diagonal gradients in xx. [Max Zwiessele]

* [gradsxx] putting tests in, not complete yet! [Max Zwiessele]

* Merge branch 'devel' into gradientsxx. [Max Zwiessele]

* Merged __init__ [Michael T Smith]

* Updates for eq_ode1 and eq_ode2 kernels. [cdguarnizo]

* Add eq_ode1 kern and ibp_lfm model. [cdguarnizo]

* Integral kernels added, these allow 'histogram' or 'binned' data to be modelled. [Michael T Smith]

* Fixed bug, replaced for loops with einsum. [alessandratosi]

* [gradxx] not working with X,X... [mzwiessele]

* [dxx] faster numpy version of the gradients_XX. [mzwiessele]

* Fixed gradients_XX_diag. [alessandratosi]

* Merge branch 'gradientsxx' of github.com:SheffieldML/GPy into gradientsxx. [Max Zwiessele]

* Fixed covariance computation in predict_jacobian. [alessandratosi]

* [gradients xx] getting there. [Max Zwiessele]

* Merge branch devel into gradientsxx. [alessandratosi]

* Fixed bug in kernel_tests for gradients_XX. [alessandratosi]

* [slicing] fixed slicing for second order derivatives. [mzwiessele]

* [slicing] fixed slicing for second order derivatives. [mzwiessele]

* Merge branch 'devel' into gradientsxx. [mzwiessele]

* Bug fix. [alessandratosi]

* Syntax fix. [alessandratosi]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into gradientsxx. [alessandratosi]

* Modified kernel tests for gradients_XX. [alessandratosi]

* Added kernel tests for gradients_XX. [alessandratosi]

* Update function kern.gradients_XX() to compute cross-covariance terms. [alessandratosi]


## v1.0.9 (2016-05-11)

### Other

* Bump version: 1.0.8 → 1.0.9. [mzwiessele]

* [setxy] was bugged. [mzwiessele]


## v1.0.8 (2016-05-11)

### Other

* Bump version: 1.0.7 → 1.0.8. [mzwiessele]

* [examples] dim reduction plotting changes. [Max Zwiessele]

* [fix #380] reloading ep. [mzwiessele]

* [fix #380] reloading ep. [mzwiessele]

* Merge branch 'devel' into kenokabe-devel. [mzwiessele]

* [statespace] omg. [mzwiessele]

* [statespace] omg. [mzwiessele]

* [open] backwards compatibility. [mzwiessele]

* Merge branch 'devel' of git://github.com/kenokabe/GPy into kenokabe-devel. [mzwiessele]

* Suppress UnicodeDecodeError: ascii codec - when import GPy. [kenokabe]

* [kernel addition] in statespace is bugged for py33 on mac, deactivating it. [mzwiessele]

* [statespace] less restrictive test for regular statespace model. [Max Zwiessele]

* [travis] condition. [Max Zwiessele]

* [static] added fixed tests. [Max Zwiessele]

* Merge branch 'devel' of git://github.com/vsaase/GPy into vsaase-devel. [Max Zwiessele]

* Added precomputed kernel class. [vsaase]

* [readme] added landscape for code cleanines. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'state_space' into devel. [mzwiessele]

* [setup] pypi restrictions. [mzwiessele]

* Update setup.cfg. [Max Zwiessele]

* [tests] show skipped. [Max Zwiessele]


## v1.0.7 (2016-04-12)

### Other

* Merge pull request #368 from SheffieldML/devel. [Max Zwiessele]

  README of pypi now directly in setup

* [tests] classification tests less strict (sporadic fails) [Max Zwiessele]

* [tests] verbose in travis. [Max Zwiessele]

* [plotting] tests across platform adjustments. [Max Zwiessele]

* [plotting] tests. [Max Zwiessele]

* [plotting] tests now compare the arrays of the figure, instead of the platform dependend png images. [Max Zwiessele]

* Bump version: 1.0.6 → 1.0.7. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* [setup] readme in setup. [Max Zwiessele]

* []README. [Max Zwiessele]


## v1.0.6 (2016-04-11)

### Other

* Merge pull request #367 from SheffieldML/devel. [Max Zwiessele]

  Update setup to fix problems with slicing

* Update __version__.py. [Max Zwiessele]

* Update setup.cfg. [Max Zwiessele]

* Update setup.py. [Max Zwiessele]

* [coverage] some more restrictions. [Max Zwiessele]


## v1.0.5 (2016-04-08)

### Other

* Merge pull request #365 from SheffieldML/devel. [Max Zwiessele]

  patch 1.0.5

* [config] softfail when config cannot be written. [Max Zwiessele]

* Bump version: 1.0.4 → 1.0.5. [Max Zwiessele]

* [rtd] last removal of rtd. [Max Zwiessele]

* [rtfd] removeing rtfd support IT DOES NOT WORK, using travis to upload to pypi instead. [Max Zwiessele]

* [rtfd] weirdness. [Max Zwiessele]

* [rtfd] weirdness. [Max Zwiessele]

* [doc] readthedocs strangeness. [Max Zwiessele]

* [doc] readthedocs strangeness. [Max Zwiessele]

* [doc] readthedocs strangeness. [Max Zwiessele]

* [doc] readthedocs strangeness. [Max Zwiessele]

* Bump version: 1.0.3 → 1.0.4. [Max Zwiessele]

* [kern] commented out skip tests. [Max Zwiessele]

* [optimize] optional parameters taken over to GPy. [Max Zwiessele]

* Merge pull request #364 from SheffieldML/state_space. [Max Zwiessele]

  State space

* [plotting] lost a baselinge plot. [mzwiessele]

* [statespace] tests even more reliable. [mzwiessele]

* [statespace] tests mote thorough and numerically stable. [mzwiessele]

* [exponential] fixed exponential *2 error. [mzwiessele]

* Merge branch 'AlexGrig-kalman_new' into state_space. [mzwiessele]

* FIX: Fixing bugs and innacuracies in state-space inference. [Alexander Grigorievskiy]

* Merge branch 'devel' into state_space. [mzwiessele]

* Merge branch 'devel' into deploy. [Max Zwiessele]

* Bump version: 1.0.2 → 1.0.3. [Max Zwiessele]

* Merge pull request #359 from SheffieldML/devel. [Max Zwiessele]

  Minor patch

* [readme] now supporting 2.7, 3.4 and above. [Max Zwiessele]

* Merge pull request #358 from SheffieldML/devel. [Max Zwiessele]

  AUTHORS

* [statespace] changed tests to check model integrity with GP model. [Max Zwiessele]

* [model tests] added seeds for model tests for stability. [Max Zwiessele]

* [sde stationary] whitespaces. [Max Zwiessele]

* [statespace] make predict comply to gpy standards (no confidence interval) [Max Zwiessele]

* [plotting] reran magnification and added plottting to statespace. [Max Zwiessele]

* [magnification] prediction now accepts dimensions. [Max Zwiessele]

* [setup] fix #360. [Max Zwiessele]

* [readme] added depsy badge. [Max Zwiessele]

* [readme] added depsy badge. [Max Zwiessele]

* Bump version: 1.0.1 → 1.0.2. [Max Zwiessele]

* [plotting] make sure that install through pip does not have the baseline images included. [Max Zwiessele]

* Update setup.py. [Max Zwiessele]

* Update setup.py. [Max Zwiessele]

* Create README.rst. [Max Zwiessele]

* Update __version__.py. [Max Zwiessele]

* Update setup.cfg. [Max Zwiessele]

* Update MANIFEST.in. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* [readme] added stub rst readme, so that pypi shows a link to the github page. [Max Zwiessele]

* [verion] mismatch. [Max Zwiessele]

* Bump version: 1.0.0 → 2.0.0. [Max Zwiessele]

* Bump version: 0.9.8 → 1.0.0. [Max Zwiessele]

* Revert "Bump version: 0.9.8 → 1.0.0" [Max Zwiessele]

  This reverts commit b63af98f1fe86d9c065227e113c7da7f19163ad5.

* Revert "Revert "[predict] added noiseless convenience function to gp, bc of whining about it..."" [Max Zwiessele]

  This reverts commit 7c95076b9fd8ad327ae46766b30cc9657883941e.

* Revert "[predict] added noiseless convenience function to gp, bc of whining about it..." [Max Zwiessele]

  This reverts commit 2001cd6dfd77300e1286245cf68897c17d3f0af0.

* Bump version: 0.9.8 → 1.0.0. [Max Zwiessele]

* [predict] added noiseless convenience function to gp, bc of whining about it... [Max Zwiessele]

* Merge pull request #354 from SheffieldML/gpy_one_fixes. [Max Zwiessele]

  Gpy one fixes

* [release] calling release branch deploy. [Max Zwiessele]

* Merge branch 'jameshensman-master' into gpy_one_fixes. [Max Zwiessele]

* Independent outputs kernel now works correctly for symmetrical arguments. [James Hensman]

* ENH improved the stability of variational_expectations in the Binomial likelihood. [James Hensman]

* BUG allowed the var_gauss model to take Y_metadata. [James Hensman]

* [release] branch for releases now protected and automatically updated. [Max Zwiessele]

* Bump version: 0.9.7 → 0.9.8. [Max Zwiessele]

* Merge branch 'devel' into gpy_one_fixes. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge pull request #261 from AlexGrig/kalman_new. [Max Zwiessele]

  Kalman new

* TEST: Modifying constraints of the standard periodic kernel in order to pass tests on different platforms. [Alexander Grigorievskiy]

* TEST: Relaxing some test constraints for passing tests on all platforms. [Alexander Grigorievskiy]

* KERN: sde_standard_periodic kernel change parameters names. [Alexander Grigorievskiy]

* TEST: Rename parameters is test function. [Alexander Grigorievskiy]

* TEST: Tests use 'lbfgsb' optimization function. Also some syntactic changes. [Alexander Grigorievskiy]

* FIX: Some fixes which prevented tests passing on python3.5. [Alexander Grigorievskiy]

* FIX: SDE inference. Couple of bug fixes and minor syntactic madifications. [Alexander Grigorievskiy]

* FIX: Fixint the bug with matrix exponential computation. [Alexander Grigorievskiy]

* FIX: Get rid of unused imports in state_space_model file. [Alexander Grigorievskiy]

* ENH: Copying sde kernels to the '/src' directory. [Alexander Grigorievskiy]

* TEST: Modifying tests so that their ruunig time is shorter. [Alexander Grigorievskiy]

* FIX: Fixing the unit test which gave an error in Travis. [Alexander Grigorievskiy]

* BUG: change import from absolute to relative. [Alexander Grigorievskiy]

* UPD: Added testing, and bug fixing. [Alexander Grigorievskiy]

* UPD: Major update, changed interface of the module, Cython support added. Although cython gives almost no speed-up. [Alexander Grigorievskiy]

* UPD: Added SVD Kalman Filter, EM algorithm for gradient calculation (only for discrete KF) [Alexander Grigorievskiy]

* TEST: Remove test file which is incompatible with other tests in GPy. [Alexander Grigorievskiy]

* ENH: Adding tests for SDE kernels. [Alexander Grigorievskiy]

* ENH: Added SDE for all basic kernels except Rationale Quadratic. Some necessary modifications for the previous code are performed. [Alexander Grigorievskiy]

* ENH: Adding SDE representation of addition, sumation and standard periodic kernel. [Alexander Grigorievskiy]

  All changes have been tested tests are added in later commits.

* ENH: Added templates for state-space kernels. [Alexander Grigorievskiy]

  Those are childs of regular kernels with extra "sde" function.

* FIX: Fixe bug with "expm" function in "state_space_new". Also some minor changes. [Alexander Grigorievskiy]

  Test function has been modified also.

* EXT: State-Space modelling functionality is untied with the GPy models. [Alexander Grigorievskiy]

  Currently, these new functionality is added on the side, not intervening
      the old state-space functionality. Example file has been changed and minimal
      example where descripancies appear is cunstructed.

* Example of sde_Matern covarince function is added, along with other small changes. [Alexander Grigorievskiy]

  State-space example is slightly modified.
  Imports are corrected accordingly.

* Added summation of kernels under the state space formalism. [Arno Solin]

* Add the SDE for one kernel. [Arno Solin]

* Updated Kalman filter implementation to new GPy. [Arno Solin]

* Merge pull request #350 from SheffieldML/fixed_inputs. [Max Zwiessele]

  Fixed inputs and BGPLVM prediction tests

* Small convenience function for extracting fixed_inputs, fixed inputs can be set to their mean, median, or zero. [Alan Saul]

* Test for BGPLVM predictions, for linear case which is possible to do analytically. [Alan Saul]

* Added back fixed_inputs. [Alan Saul]

* [coverage] dont complain if tests dont hit defensive assertion code. [Max Zwiessele]

* [plotting] was not covered in tests, now is. [Max Zwiessele]

* [kernel] combination kernel and hierarchical independent gradient updates. [Max Zwiessele]

* [plotting] new baseling. [Max Zwiessele]

* [plotting] skip on fail. [Max Zwiessele]

* [kernel] added structural tests for ind outputs kernel, but problem with gradients persist. [Max Zwiessele]

* [plotting] still testing the testing. [Max Zwiessele]

* [plotting] skipping on fail. [Max Zwiessele]

* [plotting] back to png... [Max Zwiessele]

* [plotting] svg? [Max Zwiessele]

* [plotting] test for pdf? [Max Zwiessele]

* [plotting] tests. [Max Zwiessele]

* [plotting] tests. [Max Zwiessele]

* [Merge] merge devel. [Max Zwiessele]

* Merge pull request #339 from zhenwendai/devel_pullrequest. [Zhenwen Dai]

  numerical stable implementation of rational qudratic

* Implement sensitivie of periodic kernel for plot_ARD. [Zhenwen Dai]

* Merge from Sheffield/GPy devel branch. [Zhenwen Dai]

* Merge pull request #314 from zhenwendai/devel_pullrequest. [Max Zwiessele]

  Fix the issue of negative predicted variance of normal GP

* Implement the gradient_X for standard_periodic kernel. [Zhenwen Dai]

* Numerical stable implementation of rational qudratic. [Zhenwen Dai]

* Move _raw_predict into posterior object. [Zhenwen Dai]

* Bug fix for mcmc sampler and add test case. [Zhenwen Dai]

* Merge with upstream. [Zhenwen Dai]

* Merge pull request #324 from AlexGrig/std_periodic_kernel. [Max Zwiessele]

  [kern] Standard periodic kernel. Changes parameter name from 'waveleng…

* [kern] Standard periodic kernel. Changes paramter name from 'wavelenght' to 'period'. This seems to be more clear. Also some minor modifications in the same file. [Alexander Grigorievskiy]

* Merge pull request #326 from SheffieldML/kern. [Max Zwiessele]

  [kernel] fix #218 and #325

* [plotting] added plotting for missing data. [mzwiessele]

* [plotting] and ignoring it again. [mzwiessele]

* [plotting] adding plotting tests, due to many problems with plotting, when not checked. [mzwiessele]

* [util] tests for util/debug.py. [mzwiessele]

* [util] tests for util/debug.py. [mzwiessele]

* [kernel] fix #218 and #325. [mzwiessele]

* Merge pull request #323 from SheffieldML/stochastics. [Max Zwiessele]

  [stochastics] update for new stochastic iptimizers in gpy

* [kern] added doc string. [mzwiessele]

* [autograd] added install instr for autograd. [mzwiessele]

* [sparse gp] commented out print statements, which are never used. [mzwiessele]

* [minibatch] added coverage for branching, spottet bug in X_variance. [mzwiessele]

* [climin] added tests and install directions for travis. [mzwiessele]

* [stochastics] added optimization for a few runs. [mzwiessele]

* [stochastics] update for new stochastic iptimizers in gpy. [mzwiessele]

* [white hetero] additional check for prediction. [mzwiessele]

* [white hetero] additional check for prediction. [mzwiessele]

* Merge pull request #322 from SheffieldML/minibatch. [Max Zwiessele]

  [sparsegplvm] added sparsegplvm and tests for minibatch sparsegplvm

* Merge branch 'devel' into minibatch. [mzwiessele]

* Merge pull request #321 from SheffieldML/limit=3. [Max Zwiessele]

  [chaching] changing all chacher limits to 3

* [chaching] changing all chacher limits to 3. [mzwiessele]

* Merge pull request #320 from SheffieldML/white_hetero. [Max Zwiessele]

  Heteroscedastic White Kernel

* [white] added heteroscedastic white kernel for specific number of samples. [mzwiessele]

* [sparsegplvm] added sparsegplvm and tests for minibatch sparsegplvm. [mzwiessele]

* Merge pull request #318 from SheffieldML/gpy_one_fixes. [Max Zwiessele]

  Gpy one fixes

* [plotting] info heatmap plotly. [Max Zwiessele]

* [deprecated] deprecated spelling mistake in wishart embedding. [Max Zwiessele]

* [kern] inner transformation to types, start for the multitype pandas arrays. [Max Zwiessele]

* Merge pull request #317 from SheffieldML/fix-276. [Max Zwiessele]

  Last minute adjustements for plotly

* [plotly] fixes for mrd. [Max Zwiessele]

* [mrd] matplotlib specific fig_kwargs matplotlib specific. [Max Zwiessele]

* Merge pull request #316 from SheffieldML/fix-276. [Max Zwiessele]

  Fix for #276

* [mrd] plot_scales and plot_latent added. [Max Zwiessele]

* Add ssgplvm model test. [Zhenwen Dai]

* Merge GPy upstream. [Zhenwen Dai]

* Fallback original slvm kl divergence. [Zhenwen Dai]

* Fix gpu initialziation. [Zhenwen Dai]

* Fix gpu initialziation. [Zhenwen Dai]

* Get rid of mpi4py import. [Zhenwen Dai]

* Merge remote-tracking branch 'upstream/devel' into devel. [Zhenwen Dai]

* Fix the issue of negative prediction variance of normal GP. [Zhenwen Dai]

* Slvm gamma mean-field. [Zhenwen Dai]

* Merge remote-tracking branch 'upstream/devel' into devel. [Zhenwen Dai]

* Implement slvm. [Zhenwen Dai]

* Add the import of transformation __fixed__ [Zhenwen Dai]

* [plotting util] faster generation of grid. [Max Zwiessele]

* Merge pull request #309 from SheffieldML/travis_scripts. [Max Zwiessele]

  [travis_scripts] loading scripts from github repo

* [travis_scripts] loading scripts from github repo. [Max Zwiessele]

* [svgp] python 3.x fix for next. [Max Zwiessele]

* [latent plots] legend was always plotted. [Max Zwiessele]

* [latent plots] legend was always plotted. [Max Zwiessele]

* [latent plots] legend was always plotted. [Max Zwiessele]

* [latent plots] legend was always plotted. [Max Zwiessele]

* [latent plots] legend was always plotted. [Max Zwiessele]

* [travis] added retries for installing conda, as it seemed to fail on 404 errors. [Max Zwiessele]

* [Poly] added bias and scale. [Max Zwiessele]

* [Poly] added bias and scale. [Max Zwiessele]

* Bump version: 0.9.6 → 0.9.7. [Max Zwiessele]

* [gp_plots] transposed plotting of 2d contours. [Max Zwiessele]

* [matplotlib_dep] added the baseplots utility for backcompatibility. [Max Zwiessele]

* [matplotlib_dep] added the baseplots utility for backcompatibility. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* [plotting] limits added. [Max Zwiessele]

* Add test case for hmc sampler. [Zhenwen Dai]

* [plotting] xlim setting. [Max Zwiessele]

* [plotting] subsampling print waring corrected. [Max Zwiessele]

* Bump version: 0.9.5 → 0.9.6. [Max Zwiessele]

* [plotting&kern] bugfixes in plotting and kernel size. [Max Zwiessele]

* Apidoc in conf. [Alan Saul]

* Added paramz to requirement file for docs. [Alan Saul]

* [plotly] scatter plotting was defaulting to color='white' [Max Zwiessele]

* [vardtc] these two lines are overridden by the next two lines... [Max Zwiessele]

* [plotting] catching 3d error for plotting latent space in other then 2 dimensions. [Max Zwiessele]

* Bump version: 0.9.4 → 0.9.5. [Max Zwiessele]

* [plotting] skipping plotting tests, as they are inconsistent across platforms. [Max Zwiessele]

* [plotting] baseline updates for 3d plotting. [Max Zwiessele]

* [plotting] was failing on some 3 dimensional plots (latent) [Max Zwiessele]

* Changed add_paraters to link_paramters. [cdguarnizo]

* Merge pull request #271 from zhenwendai/devel. [Zhenwen Dai]

  Fix the dL_dK and dL_dpsi2 symmetric issue. The bug of the Independent_output kernel and Hierarchical kernel from the original implementation still remains.

* Mark the kernel test for independent kernel and hierarchical kernel as expectedFailure. [Zhenwen Dai]

* Merge with current GPy devel. [Zhenwen Dai]

* Bump version: 0.9.3 → 0.9.4. [Max Zwiessele]

* [examples] added bgplvm stochastic example and parameter for dimensionality. [Max Zwiessele]

* Bump version: 0.9.2 → 0.9.3. [mzwiessele]

* [plotting] latent plotting had dimension mix up in it. [mzwiessele]

* [plotting] latent plotting had dimension mix up in it. [mzwiessele]

* [plotting] latent plotting had dimension mix up in it. [mzwiessele]

* Bump version: 0.9.1 → 0.9.2. [mzwiessele]

* [plotting] latent plotting had dimension mix up in it. [mzwiessele]

* Bump version: 0.9.0 → 0.9.1. [mzwiessele]

* [readme] Reinstall update. [mzwiessele]

* [testing] testing the error messages for plotting. [mzwiessele]

* [testing] testing the error messages for plotting. [mzwiessele]

* [defaults.cfg] updated so we can ship it. [mzwiessele]

* [plotting] import updated so that the config is handled better. [mzwiessele]

* Bug fixed. [Ricardo Andrade]

  ICM-RBF is used as default Kernel, but the user should be able to define a multiple output kernel outside and pass it to the model.

* Bug fixed. [Ricardo Andrade]

* Bump version: 0.8.31 → 0.9.0. [mzwiessele]

* Merge pull request #280 from SheffieldML/manifest-include. [Max Zwiessele]

  Update MANIFEST.in

* Update MANIFEST.in. [Max Zwiessele]

* Update MANIFEST.in. [Max Zwiessele]

* Merge pull request #279 from SheffieldML/fixing_likelihoods. [Max Zwiessele]

  Fixing likelihoods and EP

* Documentation. [Alan Saul]

* Merge branch 'devel' into fixing_likelihoods. [Alan Saul]

* Bump version: 0.8.30 → 0.8.31. [Max Zwiessele]

* [travis] register failes.. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Bump version: 0.8.29 → 0.8.30. [mzwiessele]

* [travis] testing deployment code. [mzwiessele]

* [readme] added paramz update into readme. [Max Zwiessele]

* Removed old code. [Alan Saul]

* Allow EP to have a auto reset option. [Alan Saul]

* Merge branch 'devel' into fixing_likelihoods. [Alan Saul]

* Bump version: 0.8.28 → 0.8.29. [mzwiessele]

* [travis] testing deployment code. [mzwiessele]

* [travis] testing deployment code. [mzwiessele]

* Bump version: 0.8.27 → 0.8.28. [mzwiessele]

* [pypi. [mzwiessele]

* Bump version: 0.8.26 → 0.8.27. [mzwiessele]

* Bump version: 0.8.25 → 0.8.26. [mzwiessele]

* Bump version: 0.8.24 → 0.8.25. [mzwiessele]

* [coverage] updated coveragerc. [mzwiessele]

* Bump version: 0.8.23 → 0.8.24. [mzwiessele]

* [coverage] updated coveragerc. [mzwiessele]

* Bump version: 0.8.22 → 0.8.23. [mzwiessele]

* Bump version: 0.8.21 → 0.8.22. [mzwiessele]

* Merge. [Max Zwiessele]

* [optimization] deleted and backwardscompatible. [Max Zwiessele]

* [pickle] load errors bc of kernel changes, backwards compatibility issues fixed. [Max Zwiessele]

* Merge. [Max Zwiessele]

* Merge pull request #269 from SheffieldML/paramz. [Neil Lawrence]

  Paramz

* Merge pull request #274 from SheffieldML/bumpversion. [Max Zwiessele]

  Have pypi show the link to github

* Update __version__.py. [Max Zwiessele]

* Update setup.cfg. [Max Zwiessele]

* Have pypi show the link to github. [Max Zwiessele]

  rendering of the rst file is way too strict to fuzz about with.

* Merge pull request #273 from SheffieldML/bumpversion. [Neil Lawrence]

  Update setup.cfg

* Update __version__.py. [Max Zwiessele]

* Update setup.cfg. [Max Zwiessele]

* Bump version: 0.8.18 → 0.8.19. [mzwiessele]

* [paramz] changes in regexp. [mzwiessele]

* Bump version: 0.8.17 → 0.8.18. [mzwiessele]

* [description] was not being converted to rst. [mzwiessele]

* Bump version: 0.8.16 → 0.8.17. [mzwiessele]

* [description] was not being converted to rst. [mzwiessele]

* Bump version: 0.8.15 → 0.8.16. [mzwiessele]

* [description] was still in. [mzwiessele]

* Bump version: 0.8.14 → 0.8.15. [mzwiessele]

* Bump version: 0.8.13 → 0.8.14. [mzwiessele]

* [rst] pypandoc. [mzwiessele]

* Bump version: 0.8.12 → 0.8.13. [mzwiessele]

* [rst] pypandoc. [mzwiessele]

* Bump version: 0.8.11 → 0.8.12. [mzwiessele]

* [desc] no long desc. [mzwiessele]

* Bump version: 0.8.10 → 0.8.11. [mzwiessele]

* [plotting_tests] failed, because of same name? [mzwiessele]

* [plotting_tests] failed, because of same name? [mzwiessele]

* [plotting_tests] failed, because of same name? [mzwiessele]

* Bump version: 0.8.9 → 0.8.10. [mzwiessele]

* [doc] manifest includes tutorials. [mzwiessele]

* [paramz] updated. [mzwiessele]

* [MANIFEST] README added. [mzwiessele]

* [paramz] test file update. [mzwiessele]

* [paramz] test verbose error. [mzwiessele]

* [paramz] test warning catches. [mzwiessele]

* [merge] devel changes to regression objects. [mzwiessele]

* [paramz] right imports. [mzwiessele]

* [pickling] pickling causes seg fault, wtf? [mzwiessele]

* [core] import parameterization. [mzwiessele]

* Bump version: 0.8.8 → 0.8.9. [mzwiessele]

* [travis] using testpypi. [mzwiessele]

* [manifest] include right files. [mzwiessele]

* [unpickle] with latin1 encoding. [mzwiessele]

* [devel] pickling files... [mzwiessele]

* Merge branch 'devel' into paramz. [mzwiessele]

  Conflicts:
    GPy/core/parameterization/parameter_core.py
    GPy/testing/pickle_tests.py

* [paramz] wrapping - todo: deprecation warnings. [mzwiessele]

* Merge branch 'devel' into paramz. [mzwiessele]

* [dir] structure preserved. [mzwiessele]

* [setup] paramz integrated. [mzwiessele]

* [python3] paramz integration. [mzwiessele]

* [paramz] fully integrated all tests running. [mzwiessele]

* [paramz] started to pull paramz out. [mzwiessele]

* [gpyload] loading pickle with restructured kern src. [Max Zwiessele]

* [testing] getting rid of warnings. [mzwiessele]

* [coregionalized] ICM did not build a multioutput kernel correctly if passed a kernel. [mzwiessele]

* Update __init__.py. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* Started looking at quadrature code for moments. [Alan Saul]

* Added Y_metadata to moments_match_ep, and removed log-concave student-t test, and added EP test for bernoulli. [Alan Saul]

* Added Z_tilde contribution for EP, now log_marginal is correct, need to check for var_dtc case. [Alan Saul]

* Fixing bernoulli likelihood for Laplace, fixing Zep for EP, and starting working on quadrature limits. [Alan Saul]

* Increase default quadrature points. [Zhenwen Dai]

* Enable rbf gpu to support psi2n. [Zhenwen Dai]

* Enhance optimize parallel. [Zhenwen Dai]

* Add sample_W to SSGPLVM. [Zhenwen Dai]

* Bug fix for lbfgs with max_iters big than 15000. [Zhenwen Dai]

* Add save channel function to mocap lib. [Zhenwen Dai]

* Fix the dL_dK symmetric issue for linear kernel and set dL_dK in the kernel test to be random. [Zhenwen Dai]

* Resolve the requirement of dL_dpsi2 to be symmetric. [Zhenwen Dai]

* [pickling] have the pickling test against a model, which is now being shipped with the distro. [Max Zwiessele]

* [pickling] have the pickling test against a model, which is now being shipped with the distro. [Max Zwiessele]

* Revert "[pickling] _src -> src" [Max Zwiessele]

  This reverts commit 4cd16a86b48b03d4a6edd56a969242296ab66f4d.

* [pickling] _src -> src. [Max Zwiessele]

* [plotting] got the old way in again. [Max Zwiessele]

* [pickling] wrote warning for using pickling. [Max Zwiessele]

* [travis] make more builds. [mzwiessele]

* Merge branch 'beckdaniel-warped_gps_fixes' into devel. [mzwiessele]

* [beckdaniel] merge in warped gp changes. [mzwiessele]

* Merge remote-tracking branch 'ShefML/devel' into warped_gps_fixes. [beckdaniel]

* Merged master. [beckdaniel]

* Merging last master. [beckdaniel]

* Commenting plotting test. [beckdaniel]

* Some cleaning on WarpedGP code. [beckdaniel]

* Code cleaning on warping_functions. [beckdaniel]

* Added a new test which tries to replicate Snelson's toy 1D but NR seems to diverge... [beckdaniel]

* Added initial test for warped gps using identity function. [beckdaniel]

* Added identity warp function and change np.argmin to np.nanargmin in optimize_restarts. [beckdaniel]

* Implemented variance using gauss-hermite. [beckdaniel]

* First try in implementing warped mean. [beckdaniel]

* If the plot is a warped gp, then it plots Y before warping. [beckdaniel]

* Added predict_quantiles method to warped gps. [beckdaniel]

* Added keywords to predict in warped gps because they are used in the plotting method. [beckdaniel]

* Commented out has_uncertain_inputs in warped gps since it breaks plotting. [beckdaniel]

* Added a small test for warped gps. [beckdaniel]

* [merged] master. [mzwiessele]

* Update pickle_tests.py. [Max Zwiessele]

* Update pickle_tests.py. [Max Zwiessele]

* Update pickle_tests.py. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Add RADIANT grant to funding acknowledgements. [Zhenwen Dai]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Add adadelta as an optimizer. [Zhenwen Dai]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Master branch showing in readme. [Max Zwiessele]

* [README] updated readme instructions, added troubleshooting. [Max Zwiessele]

* [geepie] ? [Max Zwiessele]

* Merge pull request #265 from SheffieldML/issue_fixing. [Max Zwiessele]

  Fixing issues for latest patch

* [#198] checking input dim versus X dim and raising a warning if there is a missmatch. [mzwiessele]

* [rtfd] [mzwiessele]

* [rtfd] [mzwiessele]

* [rtfd] [mzwiessele]

* [travis] oops. [mzwiessele]

* Merge pull request #262 from SheffieldML/plot_density. [Max Zwiessele]

  Plot library

* [rtfd] [mzwiessele]

* [rtfd] [mzwiessele]

* [rtfd] [mzwiessele]

* [rtfd] lets see... [mzwiessele]

* [readthedocs] almost there. [mzwiessele]

* [readthedocs] almost there. [mzwiessele]

* [readthedocs] almost there. [mzwiessele]

* [readthedocs] almost there. [mzwiessele]

* [holy] example was run in examples. [mzwiessele]

* [debploy] on testpypi. [mzwiessele]

* [debploy] on testpypi. [mzwiessele]

* [debploy] on testpypi. [mzwiessele]

* [debploy] on testpypi. [mzwiessele]

* [debploy] on testpypi. [mzwiessele]

* [debploy] on testpypi. [mzwiessele]

* [doc] updated how to plot in gpy. [mzwiessele]

* [readme] updated. [mzwiessele]

* [travis] deploying. [mzwiessele]

* [kern] covariance plot also testing no limit. [mzwiessele]

* [kern] covariance plot also testing no limit. [mzwiessele]

* [kernel] plotting ard for prod and covariance plots added. [mzwiessele]

* [rst] oops. [mzwiessele]

* [rst] unicode is a pain. [mzwiessele]

* [readtorst] rst was not returned before. [mzwiessele]

* [readtorst] rst was not returned before. [mzwiessele]

* [readtorst] rst was not returned before. [mzwiessele]

* [codecs] to read file. [mzwiessele]

* [codecs] to read file. [mzwiessele]

* [codecs] to read file. [mzwiessele]

* Update README.md. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Merge pull request #264 from SheffieldML/pre0.8.8. [Max Zwiessele]

  Pre0.8.8

* Update .travis.yml. [Max Zwiessele]

* Update __version__.py. [Max Zwiessele]

* Update setup.cfg. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Merge pull request #263 from SheffieldML/plot_rc. [Max Zwiessele]

  prerelease testing

* Update __version__.py. [Max Zwiessele]

* Update setup.cfg. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* [codecs] to read file. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] pypandoc. [mzwiessele]

* [doc] added. [mzwiessele]

* [doc] added. [mzwiessele]

* [docs] had to rename kern._src to kern.src. PLEASE CHECK YOUR CODE, if it is running smoothly. [mzwiessele]

* [shit] tags. [mzwiessele]

* [travis] testing building docs. [mzwiessele]

* [build the docs?] [mzwiessele]

* [devel] merged. [mzwiessele]

* [devel] merged. [mzwiessele]

* [devel] merged. [mzwiessele]

* [devel] merged. [mzwiessele]

* Merge branch 'devel' into plot_density. [mzwiessele]

* Revert "change the inverse lengthscale of rbf" [Zhenwen Dai]

  This reverts commit 326ed31fbfff2907bc92d2d444c74d5a24b22691.

* Change the inverse lengthscale of rbf. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* A more efficient implementation of prediction with uncertain inputs. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Add original bfgs optimizer and add rbf with inverse lengthscale. [Zhenwen Dai]

* New difference method for laplace. [Alan Saul]

* [build the docs?] [mzwiessele]

* [docs] autogen. [mzwiessele]

* [build the docs?] [mzwiessele]

* [conf.py] deleted mocking for now. [mzwiessele]

* [conf.py] deleted mocking for now. [mzwiessele]

* [coverage] covering all of gpy_plot. [mzwiessele]

* [tests] increasing coverage. [mzwiessele]

* [testing] travis. [mzwiessele]

* Update __init__.py. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update plotting_tests.py. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update pickle_tests.py. [Max Zwiessele]

* Update pickle_tests.py. [Max Zwiessele]

* Update pickle_tests.py. [Max Zwiessele]

* Update pickle_tests.py. [Max Zwiessele]

* Update model_tests.py. [Max Zwiessele]

* Update pickle_tests.py. [Max Zwiessele]

* Update model_tests.py. [Max Zwiessele]

* [travis] upload to testpypi. [mzwiessele]

* [pypi] upload to pypi automatically. [mzwiessele]

* [tests] failing only on pull request, wtf? [mzwiessele]

* [test] coverage increased. [mzwiessele]

* [test] coverage increased. [mzwiessele]

* [config] default stub install verbose. [mzwiessele]

* [config] default stub install verbose. [mzwiessele]

* [config] default stub install verbose. [mzwiessele]

* [config] default stub install verbose. [mzwiessele]

* [config] default stub install verbose. [mzwiessele]

* [config] default stub install verbose. [mzwiessele]

* [inferenceX] test consistency. [mzwiessele]

* [plotly] density done. [mzwiessele]

* [inducing] 3d added. [mzwiessele]

* Testing everything. [Max Zwiessele]

* [kernel] plot_ard added (some other fixes as well) [Max Zwiessele]

* [plotly] last minute change for ipython notebook. [Max Zwiessele]

* [plotly] last minute change for ipython notebook. [Max Zwiessele]

* [baseline] tests. [mzwiessele]

* [baseline] tests. [mzwiessele]

* [plotly] everything working, except gradient. [mzwiessele]

* [plotly] todos: fill_gradient. [mzwiessele]

* [testing] updates again and plotly is going forward. [mzwiessele]

* [testing] updates again and plotly is going forward. [mzwiessele]

* [unicode] error in setup. [mzwiessele]

* [setup] [mzwiessele]

* [setup] [mzwiessele]

* [unicode] error in setup. [mzwiessele]

* [plotly] starting. [mzwiessele]

* [plotly] starting plotly. [mzwiessele]

* ['tests'] assert array equal. [mzwiessele]

* [tests] now working? [mzwiessele]

* [tests] now working? [mzwiessele]

* [tests] now working? [mzwiessele]

* [tests working now?] [mzwiessele]

* [tests] failing although the same... [mzwiessele]

* [tests] failing although the same... [mzwiessele]

* [tests] failing although the same... [mzwiessele]

* [tests] failing although the same... [mzwiessele]

* [tests] failing although the same... [mzwiessele]

* [matplotlib] plot updates and testing. [mzwiessele]

* [testing] BGPLVM. [mzwiessele]

* [testing] BGPLVM. [mzwiessele]

* [testing] BGPLVM. [mzwiessele]

* [docs] updated and testing. [mzwiessele]

* [rcparams] default added for rc not failing the tests. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] gradient plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] magnification plot added. [mzwiessele]

* [plotting] testing only pngs because macosx cant convert pdf to png.... [mzwiessele]

* [plotting] testing only pngs because macosx cant convert pdf to png.... [mzwiessele]

* [libraries] added for plotting. [mzwiessele]

* [libraries] added for plotting. [mzwiessele]

* [libraries] added for plotting. [mzwiessele]

* [libraries] added for plotting. [mzwiessele]

* [libraries] added for plotting. [mzwiessele]

* [libraries] added for plotting. [mzwiessele]

* [plotting] tests now working? [mzwiessele]

* [testing] harder then expected to test image files against each other.... [mzwiessele]

* [testing] more restructuring, almost ready to ship, added some tests for testing with travis. [mzwiessele]

* [testing] more restructuring, almost ready to ship, added some tests for testing with travis. [mzwiessele]

* [testing] more restructuring, almost ready to ship, added some tests for testing with travis. [mzwiessele]

* [testing] more restructuring, almost ready to ship, added some tests for testing with travis. [mzwiessele]

* [testing] more restructuring, almost ready to ship, added some tests for testing with travis. [mzwiessele]

* [testing] more restructuring, almost ready to ship, added some tests for testing with travis. [mzwiessele]

* [testing] more restructuring, almost ready to ship, added some tests for testing with travis. [mzwiessele]

* [testing] more restructuring, almost ready to ship, added some tests for testing with travis. [mzwiessele]

* [plotting] restructuring more and more. [mzwiessele]

* Update plotting_tests.py. [Max Zwiessele]

* Update gp_plots.py. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* [density] rewritten for consistent coloring. [mzwiessele]

* [plotting] added samples plot. [mzwiessele]

* [plotting] getting there, plots to go: dim red, kern, mapping etc. [mzwiessele]

* [active dims] kernel active dims now the real active dims. [mzwiessele]

* [plotting] library is unfolding and should be working tonight. [mzwiessele]

* [tests] running all tests. [mzwiessele]

* [tests] running all tests. [mzwiessele]

* [added testing and plotting] restructuring the plotting library. [mzwiessele]

* [plotting] cleanup first commit, this cleans the plotting library and adds plotting tests. [mzwiessele]

* [density] plotting of likelihoods permitted. [mzwiessele]

* [density] plot added. [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* Merge pull request #248 from SheffieldML/travis_testing. [Max Zwiessele]

  Travis update to test Linux & MacOSX

  Python 2.7, 3.5 on Ubuntu and MacOSX

* Merge branch 'devel' into travis_testing. [mzwiessele]

* [copyrighting] and testing. [mzwiessele]

* [rv tests] Gradient not checking right, @jameshensman what is going on here? [mzwiessele]

* [travis] oops. [mzwiessele]

* [travis] oops. [mzwiessele]

* [codecov] added, trying to merge in readme from master. [mzwiessele]

* Merge branch 'devel' into travis_testing. [mzwiessele]

* [travis] testing codecoverage. [mzwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Skipping Logexp test for now. [Max Zwiessele]

  creating issue for fixing

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update model.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* More transformation checks. [Max Zwiessele]

  Need to check, which ones can be checked by kde?

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* [master] readme takeover. [mzwiessele]

* [mrd] print statement py3, OMG. [mzwiessele]

* Added factorize_space function which returns a segmentation to shared and private dims. [Andreas]

* [plotting] py3 compatibility, is it right, that relative imports always have to be in the format from . import <.> [mzwiessele]

* [plotting] py3 compatibility, is it right, that relative imports always have to be in the format from . import <.> [mzwiessele]

* [plotting] py3 compatibility, is it right, that relative imports always have to be in the format from . import <.> [mzwiessele]

* [plotting] py3 compatibility, is it right, that relative imports always have to be in the format from . import <.> [mzwiessele]

* [plotting] py3 compatibility, is it right, that relative imports always have to be in the format from . import <.> [mzwiessele]

* [newest patch updates, cleaned interfaces and mean_function addidtions] [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* Fix xrange. [Zhenwen Dai]

* Fix xrange. [Zhenwen Dai]

* Fixed bias+linear and bias+rbf with psi statistics. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Implement Gaussian quadrature psi-statistics for additive kernel. [Zhenwen Dai]

* Fixed MRD inducing point gradients. [Alan Saul]

* Rbf psi-statistics speedup. [Zhenwen Dai]

* Gently fall back if gpu psicomp fail. [Zhenwen Dai]

* Remove the automatic importing mpi4py. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Errobars_trainset -> plot_errorbars_trainset. [Ricardo]

* Bug fix for model_plots when specifying which_data_ycols. [Zhenwen Dai]

* More mocking. [Alan Saul]

* More mocking. [Alan Saul]

* More mocking, almost done. [Alan Saul]

* More mocking, almost done. [Alan Saul]

* More mocking. [Alan Saul]

* More mocking. [Alan Saul]

* More mocking. [Alan Saul]

* More mocking. [Alan Saul]

* Mocking blas. [Alan Saul]

* Mocking blas. [Alan Saul]

* Mocking blas. [Alan Saul]

* Mocking blas. [Alan Saul]

* Updated conf. [Alan Saul]

* [sphinx] new doc. [Max Zwiessele]

* [scipy] deleted from mocking. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* More mocking for scipy, impossible to check without committing :( [Alan Saul]

* More mocking for scipy, impossible to check without committing :( [Alan Saul]

* More mocking for scipy, impossible to check without committing :( [Alan Saul]

* More mocking. [Alan Saul]

* Removed directives rom docs. [Alan Saul]

* Mocked scipy for docs. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Add adadelta as an optimizer. [Zhenwen Dai]

* [yak shaving] [Max Zwiessele]

* [sparsegp] check for missing data. [mzwiessele]

* Missing numpy import. [Alan Saul]

* Merge pull request #246 from SheffieldML/travis2. [Max Zwiessele]

  Dapid's travis changes

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update rv_transformation_tests.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* Update base_plots.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* Update __init__.py. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Update .travis.yml. [Max Zwiessele]

* Dapid's travis changes. [Max Zwiessele]

  There was a conflict and I only had access to the web interface.

* Merge pull request #243 from SheffieldML/ep. [Max Zwiessele]

  Ep inference updates

* [#186] fixed distribution across files and added base class for reusability. [Max Zwiessele]

* [inference] changed gaussian variance to precision (which it really is) [Max Zwiessele]

* [ep] now calling exact inference instead of copying code. [Max Zwiessele]

* [setup] include headers in source dist. [Max Zwiessele]

* [classification] sparse gp classification and dtc update. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Bug fix for set_XY. [Zhenwen Dai]

* Add set_XY test. [Zhenwen Dai]


## v0.8.8 (2015-09-10)

### Other

* Bump version: 0.8.7 → 0.8.8. [Max Zwiessele]

* Bump version: 0.8.6 → 0.8.7. [Max Zwiessele]

* [py3] print statement. [Max Zwiessele]

* [pred_var] added predictive variable as property now. [Max Zwiessele]

* Bump version: 0.8.5 → 0.8.6. [Max Zwiessele]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Max Zwiessele]

* Apply bug fix for set_XY. [Zhenwen Dai]

* Converting .md to .rst automatically now. [Max Zwiessele]

* Bump version: 0.8.4 → 0.8.5. [Max Zwiessele]

* Merge pull request #240 from SheffieldML/devel. [Max Zwiessele]

  Devel

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Updated sampling and plots to be correct shape, and changed plotting of sampling to be posterior samples p(f*|f), like it used to be, and samples_y to plot samples of p(y*|y) [Alan Saul]

* Add ARD to MLP kernel. [Zhenwen Dai]

* Merge pull request #239 from mikecroucher/master. [mikecroucher]

  Cython fix

* Cython fix. [Mike Croucher]

* [setup] check if darwin. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

* [verbosity] option in tests. [Max Zwiessele]

* [plotting] no printing of warning unless you actually use plotting. [Max Zwiessele]

* [matplotlib] pylab -> pyplot. [Max Zwiessele]

* Bump version: 0.8.3 → 0.8.4. [Max Zwiessele]

* [merge] setup.py. [Max Zwiessele]

* Fixes for Python 3. [Mike Croucher]

* Fixed for Python3. [Mike Croucher]

* [README] updated. [Max Zwiessele]

* Bump version: 0.8.2 → 0.8.3. [Max Zwiessele]

* [requirements] added six as an requirement. [Max Zwiessele]

* Merge pull request #237 from SheffieldML/devel. [Max Zwiessele]

  Merge pull request #236 from SheffieldML/master

* Merge pull request #236 from SheffieldML/master. [Max Zwiessele]

  sync up master and devel

* Update setup.py. [Max Zwiessele]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Max Zwiessele]

* Added universal wheels. [Mike Croucher]

* Changed author to generic email address. [Mike Croucher]

* Bump version: 0.8.1 → 0.8.2. [Max Zwiessele]

* [setup] another patch for source dist. [Max Zwiessele]

* Bump version: 0.8.0 → 0.8.1. [Max Zwiessele]

* Merge pull request #232 from SheffieldML/versionfile. [Max Zwiessele]

  Versionfile

* [version] controlling the version from __version__.py using bumpversion and setup.cfg. [Max Zwiessele]

* [version] controlling the version from __version__.py using bumpversion and setup.cfg. [Max Zwiessele]

* [version] handling version in a file. [Max Zwiessele]

* Python 3 fixes. [Mike Croucher]

* Update README.md. [Max Zwiessele]

* [bumpversion] added bumpversion for release control. [Max Zwiessele]

* [linalg] testing suite update. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Update setup.py with new version number. [mikecroucher]

* Bug fix for the print function in samplers.py. [Zhenwen Dai]

* Bug fix for compilation on Mac. [Zhenwen Dai]

* Update README.md. [Neil Lawrence]

* Update AUTHORS.txt. [Neil Lawrence]

* Merge branch 'devel' [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Bug fix for compilation on Mac. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Removed annoying print. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Allowed gpyified var gauss. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Added full cov prediction. [Alan Saul]

* [merge] merge master into devel. [Max Zwiessele]

* [doc] build was pushed to github. [Max Zwiessele]

* Merge pull request #191 from mellorjc/master. [James Hensman]

  adding an exponential prior

* Added Exponential prior. Made some minor (and not exhaustive) formatting changes based on pep8. [mellorjc]

* Merge pull request #1 from SheffieldML/master. [mellorjc]

  merge updates from upstream

* Merge pull request #208 from strongh/tutorial-and-readme-updates. [James Hensman]

  Tutorial and readme updates

* Remove docstring for missing tensor param from add/prod. [Homer Strong]

  relevant to #162

* Update kernel tutorial. [Homer Strong]

* Fix links in readme. [Homer Strong]

  the ! is for adding images, but these are just links

* Merge pull request #211 from PredictiveScienceLab/master. [Zhenwen Dai]

  Fixes the PDF transformation bug by handpicking James Hensman's code from the devel branch

* Removed dir ib_tests. [Ilias Bilionis]

* PDF Transformation bug patched. [Ilias Bilionis]

* Handpicked James Hensman's code that ensures that fixes the PDF of transformed variables. Fixed minor plotting bug. [Ilias Bilionis]

* Fixed MCMC sampler. [Ilias Bilionis]

* Added compute of jacobian. [Ilias Bilionis]

* Beautification change. [Ilias Bilionis]

* Added proper comments to the test. [Ilias Bilionis]

* Fixed transformation plotting bug and added test that demonstrates the problem. [Ilias Bilionis]

* Merge pull request #184 from mikecroucher/master. [mikecroucher]

  Bug fix for issue #161

* Bug fix for issue #161. [Mike Croucher]

* [misc test] if there was no overflow, dont fail the test. [Max Zwiessele]

* [readthedocs] forcing readthedocs into not failing. [Max Zwiessele]

* [readthedocs] forcing readthedocs into not failing. [Max Zwiessele]

* [test] adding some more tests for coverage. [Max Zwiessele]

* [readthedocs] forcing readthedocs into not failing. [Max Zwiessele]

* [readthedocs] forcing readthedocs into not failing. [Max Zwiessele]

* [readthedocs] forcing readthedocs into not failing. [Max Zwiessele]

* Git Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Update README.md. [Max Zwiessele]

  updated installation instructions scipy 0.16

* [testing] [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Fixed for Python 3. [Mike Croucher]

* [doc] some changes to the doc, using mathjax some additions in math. [Max Zwiessele]

* Git pushMerge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Optimize util for mpi. [Zhenwen Dai]

* [licensing] replaced licensing with BSD, and erfcx. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Used scipy.log1p since it gives more consistent results cross-platform. [Mike Croucher]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Mike Croucher]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Changed gpu interface for mpi. [Zhenwen Dai]

* Switched to scipy.special.log1p@ [Mike Croucher]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Mike Croucher]

* Used scipy.log1p since it gives more consistent results cross-platform. [Mike Croucher]

* Fixed typos. [Mike Croucher]

* [more coverage] and predictive var fixes. [Max Zwiessele]

* Python 3 fixes. [Mike Croucher]

* Git pushMerge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Mike Croucher]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Speed tuning for mlp kernel and gauss qudrature for psi-statistics. [Zhenwen Dai]

* Python 3 fixes. [Mike Croucher]

* Fixes Cython compilation on Mac OS X. [Mike Croucher]

* Automatic fallback to Numpy if Cython modules not available. [Mike Croucher]

* [predict] using gp predict in sparse gp and predictive variable. [Max Zwiessele]

* [mean func] added in GPRegression. [Max Zwiessele]

* [coverage] tests for coverage increase. [Max Zwiessele]

* Fix linear kernel with NxMxM psi2. [Zhenwen Dai]

* New implementation for mlp kernel (speed improvemd) [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Merge pull request #227 from Dapid/clean_imports. [mikecroucher]

  Clean some imports

* Cleanup unused imports. [David Menéndez Hurtado]

* Unify the interface of psicomp, but the psi2n of linear kernel and gaussian qradrature still needs to be done. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* [MRD] fixed mrd for new structure. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* [empty space] literally : ) [Max Zwiessele]

* [pickling] pickle error. [Max Zwiessele]

* Revert "[core] visited as propery" [Max Zwiessele]

  This reverts commit a4ce1d473c13fa5cb577e4ff3dbdf76aa6a1a57f.

* Revert "[caching] different error" [Max Zwiessele]

  This reverts commit f7087ebc9003a6950b58533e0ce373c03c82a4a2.

* [caching] different error. [Max Zwiessele]

* [core] visited as propery. [Max Zwiessele]

* [setxy] always run the update after updating X and Y. [Max Zwiessele]

* Allow cache supporting boolean and integers. [Zhenwen Dai]

* [psi2n] Liner psi2 statistics now implemented for separate n. [James Hensman]

* Slightly improved computation for var_Gauss method. [James Hensman]

* [psi2n] added psi2n for static kernels. [Max Zwiessele]

* [spgp minibatch] linear calls the right psicomps and add kernel. [Max Zwiessele]

* [coverage] script to create coverage output html. [Max Zwiessele]

* [coverage] test predict sparse gp. [Max Zwiessele]

* [coverage] added normalizer tests. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* [psi2] NxMxM fixes for the add kernel. [Max Zwiessele]

* Fall back to the old psicomp interfacegit status! [Zhenwen Dai]

* Add psi-statistics test. [Zhenwen Dai]

* [mrd] fixes for updates on psi2. [Max Zwiessele]

* [spgp] minibatch testing. [Max Zwiessele]

* [merge] for spgp minibatch and psi NxMxM. [Max Zwiessele]

* [spgp minibatch] added new routine for psi NxMxM, much faster, little bigger mem footbprint. [Max Zwiessele]

* Small edits for linear kernel. [Alan Saul]

* Reindented, did some profiling which looks promising. [Alan Saul]

* Merge branch 'devel' into missing_opt. [Alan Saul]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Alan Saul]

* Corrected caching for psi derivatives. [Alan Saul]

* Try calculating dL_dpsi1*psi1 individually for each dimension as we go along. [Alan Saul]

* Passing psi statistics. [Alan Saul]

* Removed comment. [Alan Saul]

* Checking kerngrads full_value. [Alan Saul]

* Gradients for X seem to match. [Alan Saul]

* Made sparse gp work again. [Alan Saul]

* Tidied but broken. [Alan Saul]

* Optimizing missing data model, needs tidying but now much faster. [Alan Saul]

* Fixed array2string bug for N > 1000 default printing. [Alan Saul]

* Merge pull request #224 from Dapid/fix_print. [mikecroucher]

  Fix print statement

* FIX: missing compatibility Py2/3. [David Menéndez Hurtado]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* [add] renamed>sum. [Max Zwiessele]

* [magnification] static corrections. [Max Zwiessele]

* [magnification] added static kernel support and faster derivative computations. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* [magnification] plot_magnification expanded. [Max Zwiessele]

* Add util.parallel into import. [Zhenwen Dai]

* Errorbars fixed. [Ricardo]

* Gperrors color as parameter. [Ricardo]

* Shape of heteroscedastic variance corrected. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge pull request #219 from Dapid/fix_209. [mikecroucher]

  Now Scipy 0.16 is required, removing fixes for older versions.

* Fixing the blas arguments for DSYRK. [David Menéndez Hurtado]

* Fixing confussion between lapack and ctypes interfaces. [David Menéndez Hurtado]

* FIX: now Scipy 0.16 is required, removing fixes for older versions. Accessing blas through the scipy interface. [David Menéndez Hurtado]

* [magnification] mostly plotting and some model corrections for _predictive_variable. [Max Zwiessele]

* [hetnoise] import correction. [Max Zwiessele]

* [merge] [Max Zwiessele]

* Merge branch 'updates' into devel. [Max Zwiessele]

* [core] updating system, security branching. [Max Zwiessele]

* New function to plot just the errorbars of the training data. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Ensuring the shape of the mean vector at predict time fixes bug in EP prediction. [James Hensman]

* Note added to the docs. [Ricardo]

* Change in _diag_ufunc with @mzwiessele. [Ricardo]

* New likelihood: HeteroscedasticGaussian. [Ricardo]

* Model uses the new HeteroscedasticGaussian likelihood. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Add MRD for regression benchmark. [Zhenwen Dai]

* Remove the old housing benchmark. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Fixed random seed for kronecker tests. [James Hensman]

* Adding a white kernel to a sparseGP test for stability. [James Hensman]

* Add the regression benchmark. [Zhenwen Dai]

* Add missing file. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Gradients w.r.t. kappa corrected. [Ricardo Andrade]

  np.diag(dL_dK_small) needs to be copied

* Trying to make travis print warnings. [Alan Saul]

* Fixed laplace seed, added debugging for misc tests. [Alan Saul]

* Merge branch 'devel' of github.com:/sheffieldml/gpy into devel. [James Hensman]

* Removed installation conflict. [Alan Saul]

* Updated travis for 0.16 scipy. [Alan Saul]

* Sqrt(pi) term fix in Bernoulli. [James Hensman]

* Added variational expectation tests updates. [Alan Saul]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Alan Saul]

* Adding var_gauss.py. [James Hensman]

* Added fallback for non-identity link function for Gaussian variational expectations. [Alan Saul]

* Added testing of variational expectations, analytic vs numeric, and gradient checks. [Alan Saul]

* The Opper-Archambeau method is now implemented as an inference method in the GPy style. [James Hensman]

* Minor bugfixes in plotting: quantiles are now computed using predict_kw correctly. [James Hensman]

* Psi-statistics for any kernels via Gaussian quadrature. [Zhenwen Dai]

* Improve the stability of parallel inference code. [Zhenwen Dai]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Merge pull request #216 from Dapid/devel-cython_update. [James Hensman]

* ENH: various Cython enhancements, mostly releasing the GIL when not needed. [David Menéndez Hurtado]

* Tidied link_fn tests. [James Hensman]

* Tidied setup.py. [James Hensman]

* Merge branch 'Dapid-devel-cythonchol' into devel. [James Hensman]

* Running cython. [James Hensman]

* Reenabling gil. [James Hensman]

* Minor corrections :) [James Hensman]

* Merge branch 'devel' into Dapid-devel-cythonchol. [James Hensman]

* Adding new test for cholesky backprop. [James Hensman]

* Merge branch 'devel-cythonchol' of git://github.com/Dapid/GPy into Dapid-devel-cythonchol. [James Hensman]

* FIX: ensuring contiguity of the buffers for BLAS call and returning a Numpy array. [David Menéndez Hurtado]

* FIX: transforming the indexing to 2D. [David Menéndez Hurtado]

* ENH: implementing the Cholesky backpropagation through Scipy's BLAS. [David Menéndez Hurtado]

* ENH: fixed up BCGPLVM to work with new framework. [James Hensman]

* Another relative import (py3) bug. [James Hensman]

* Merge pull request #183 from AlexGrig/std_periodic_kernel. [James Hensman]

* TEST: Making test for the Standard Periodic Kernel similar to other kernel tests. [Alexander Grigorievskiy]

* TEST: Adding test for periodic kernel. [Alexander Grigorievskiy]

* ENH: Adding standard periodic kernel. [Alexander Grigorievskiy]

  The standard periodic kernel is the one which is mentioned
  in the Rasmussen and Williams book about Gaussian Processes.

* Fixing qualtile code for some likelhoods. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Change the default name of sparse_gp_mpi class. [Zhenwen Dai]

* Fixed a plotting bug. [James Hensman]

* Tidied model_tests. [James Hensman]

* Removed randomness from inference tests by setting np.seed. [James Hensman]

* Caught warnings in misc_tests. [James Hensman]

* Fixed relative import in spline. [James Hensman]

* Fixed conflict in plotting. [James Hensman]

* Merge branch 'tjhgit-devel' into devel. [James Hensman]

* Merging. conflict in relative import styles. [James Hensman]

* Merge remote-tracking branch 'upstream/devel' into devel. [tjhgit]

  Conflicts:
    GPy/kern/__init__.py

* Added spline kernel (from P. Hennig) to GPy. [tjhgit]

  Had to modify the base_plots and model_plots.py, since I had troubles
  installing GPy in anaconda on debian linux due to the dependency on
  Tango. Why is Tango needed to represent colors that can also be typed in
  Hex format thus eliminating further dependencies.

* Fixed strange bug. In python 3, numbers startin 0 are octal. [James Hensman]

* Merge pull request #212 from mikecroucher/devel. [mikecroucher]

  Python 3 fixes

* Python 3 fixes. [Mike Croucher]

* Fixed SVGP tests. [James Hensman]

* Rbf psi-statistics speed improvement. [Zhenwen Dai]

* Bug fix: the name of parameterable object is not removed when unlinking. [Zhenwen Dai]

* Allow Y to be uncertain. [Zhenwen Dai]

* The bug fix for the cblas.h problem in Mac os x. [Zhenwen Dai]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mzwiessele]

* Merge the changes. [Fariba]

* Fix DGPLVM prior. [Zhenwen Dai]

* Improve the documentation of infer_newX. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Change crescent data to optimize with .optimize() [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Commit to pull. [Neil Lawrence]

* Fixed model test. [Alan Saul]

* Changed quantile computation via sampling and added fallback for predictive mean and variance if conditional mean and variance are not implemented yet. [Alan Saul]

* Passing metadata. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Generalized the variatinoal Gaussian approximatino revisited code for any likelihood. [James Hensman]

* Cache some values. [Alan Saul]

* Changes. [Fariba]

* [bgplvm] technical new stuff. [mzwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mzwiessele]

* Some correction for ibp ssgplvm. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Python 3 fixes. [Mike Croucher]

* Passing on y_metadata. [Alan Saul]

* Allowing set initial noise variance for GPRegression. [javiergonzalezh]

* Sqrt pi term was missing in variational expectations. [James Hensman]

* Lots of work on cython choleskies. [James Hensman]

* Fiddling with cholesky backprop. [James Hensman]

* Parallelizing backprop of cholesky. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Jacobians should not be computed only for transforms. [James Hensman]

* Jacobians should not be computed only for transforms. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Updated plotting. [Alan Saul]

* [inference] minibatch inference needed polishing. [mzwiessele]

* [plotting] parameterized.values. [mzwiessele]

* [parameterized] merge in jacobian for priors? [mzwiessele]

* [verbose opt] verbose needed clear after finish push through. [mzwiessele]

* Merge branch 'devel' of github.com:/sheffieldml/gpy into devel. [James Hensman]

* Allow to set color for the skeleton visualization. [Zhenwen Dai]

* [ssgplvm] change the default gamma interval. [Zhenwen Dai]

* Jacobian bugfix. [James Hensman]

* Added log jacobian transofms for Exponent, Logexp. [James Hensman]

* Minor bugfix in raw_predict with full_cov for sparseGP. [James Hensman]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Alan Saul]

* [ssmrd] implement with IBP prior. [Zhenwen Dai]

* Reshaped log predictive density to have D outputs. [Alan Saul]

* Added option for plotting with SVGP. [Alan Saul]

* [ssgplvm] implement IBP prior. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* [param] added multilevel indexing note to doc. [Max Zwiessele]

* Merge branch 'devel' of github.com:/sheffieldml/gpy into devel. [James Hensman]

* Some cython improvments for stationary kerns. [James Hensman]

* Minor bugfix in mlp kernel. [James Hensman]

* Svgp bugfix. [James Hensman]

* [ssgplvm] group spike. [Zhenwen Dai]

* Start implement ICP for ssgplvm. [Zhenwen Dai]

* Prevent the predicted variance to be negative. [Zhenwen Dai]

* Small bug in cython tests. [James Hensman]

* Merge branch 'reorder_choleskies' into devel. [James Hensman]

* Svgp working with reordered chols. [James Hensman]

* Interim svgp commit. [James Hensman]

* Merge branch 'devel' into reorder_choleskies. [James Hensman]

* Svgp, more c-ordering. [James Hensman]

* Svgp tests are passing with re-ordered chols. [James Hensman]

* Preliminary reconfiguring or choleskies ordering. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Change the behavior the optimize_restarts to keep the original model parameters for the firt run. [Zhenwen Dai]

* Omp for gradients_X. [James Hensman]

* [heteroscedastic gauss] Implemented Heteroscedastic Guassian Lik with @ric70x7. [Max Zwiessele]

* [heteroscedastic gauss] Implemented Heteroscedastic Guassian Lik with @ric70x7. [Max Zwiessele]

* [opt messages] show DDdHHhMMmSSsMS. [Max Zwiessele]

* [opt messages] show dd/hh:mm:ss.ms. [Max Zwiessele]

* [cython import] error notices where it happens. [Max Zwiessele]

* [predict] documentation. [Max Zwiessele]

* [predict] documentation. [Max Zwiessele]

* Again with gradients. [Alan Saul]

* Fixing constant mapping gradients. [Alan Saul]

* Small name change. [Alan Saul]

* Addint constant mapping. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* [changepoint] cp is array. [mzwiessele]

* Fixed linalg for general matricies. [Alan Saul]

* Added another einsum operation. [Alan Saul]

* Added a constant mappiung. [James Hensman]

* Removing silly einsum. [James Hensman]

* Better cython compiler directives for choleskies. [James Hensman]

* Bugfix: confused output dim and num_latents in svgp. [James Hensman]

* Added faster einsums to linalg, with a couple of tests. [Alan Saul]

* Faster einsums in svgp. [James Hensman]

* Adding limited support for svg to have differnet number of latent functions to columns of Y. [James Hensman]

* Update .travis.yml. [James Hensman]

* Added cholesy backprop test. [James Hensman]

* Merge by running cython. [James Hensman]

* Added backprop of cholesky grads. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Update .travis.yml. [James Hensman]

* Update MANIFEST.in. [James Hensman]

* Merge branch 'cython2' into devel. [James Hensman]

* Changes to comments in compiled cython file. [James Hensman]

* Merge branch 'cython2' into devel. [James Hensman]

* All tests passing. [James Hensman]

* Fixed coregionalize cython. [James Hensman]

* Tests passing (cython) [James Hensman]

* Tidien up coregionlize w AS. [James Hensman]

* More edits to stationary to clean up for cython. [James Hensman]

* Modifiying stationary.py. [James Hensman]

* Some eidts to choleskies.pyx. [James Hensman]

* Help choleskies along a little. [James Hensman]

* Adding choleskies cythonized. [James Hensman]

* Start of cythoning coregionalize. [James Hensman]

* Added cython code for lengthscale gradients. [James Hensman]

* Initial cython commit. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* [basis funcs] memory efficient posterior inference. [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* [plotting] added predict_kw to plot function. [mzwiessele]

* [inference] exact Gaussian some constant jitter. [mzwiessele]

* [paramnonparam] adding cdf like. [mzwiessele]

* Fixed deg free gradient. [Alan Saul]

* Added log predictive density, student t degrees of freedom gradients and plotting functionality. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* [basisfunckern] gradients for non ard adjusted. [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* Merge with commit of dgplvm. [frb-yousefi]

* Prior with Lambda. [frb-yousefi]

* Descrimative BGPLVM prior with lambda added. [frb-yousefi]

* [basisfuncs] updated kernel to better reflect linear trends and added ARD support. [mzwiessele]

* [basis funcs] linear slope identifiability higher, symmetry plus true linear effect. [mzwiessele]

* [sparse gp] memory overflow with big data, iterating over dimensions now. [mzwiessele]

* Quadrature appeared to be out by a factor of 1/sqrt(pi) [Alan Saul]

* Reverted back. [Alan Saul]

* Updated svgp kernel gradients. [Alan Saul]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mzwiessele]

* Manual merging with AS. [James Hensman]

* Fixed log predictive density, added option for LOO to provide some intemediate variables. [Alan Saul]

* Removed jitter printing. [Alan Saul]

* Changed LOO implementation for Eq 30 instead of 37. [Alan Saul]

* Added LOO for laplace and exact inference on training data, Gaussian logpdf appeared to be wrong, now fixed. [Alan Saul]

* Added to init. [Alan Saul]

* Added option to plot the transformed link function (posterior once the link function has been applied) [Alan Saul]

* Added hessian and skew gradient checkers, some block functions. [Alan Saul]

* Fix for model gradients. [Alan Saul]

* Added some numerical stability to link functions with tests for link functions. [Alan Saul]

* Fix typo. [Alan Saul]

* Minor commenting changes. [Alan Saul]

* Added numerical clipping. [Alan Saul]

* Manual merging. [James Hensman]

* Merge pull request #189 from mikecroucher/devel. [James Hensman]

  Python 3 Compatibility

* Merge remote-tracking branch 'upstream/devel' into devel. [Mike Croucher]

* Fix printing error. [Mike Croucher]

* Added (SLOW) Pure Python implementations of flat_to_triang and triang_to_flat. [Mike Croucher]

* Working in Py2 but broken in Py3. [Mike Croucher]

* Fix merge conflicts. [Mike Croucher]

* Resolve merge conflicts. [Mike Croucher]

* Used 'six' to support Py3 and Py2 simultaneously. [Mike Croucher]

* Merge from upstream. [Mike Croucher]

* Fix README.md formatting. [Mike Croucher]

* Fix README.md formatting. [Mike Croucher]

* Xrange fixes for Python 3. [Mike Croucher]

* Updated README.md for recent Py3 work. [Mike Croucher]

* Kern fix. All tests now pass. [Mike Croucher]

* 2to3 itertools fixer. [Mike Croucher]

* Fixed leaky comprehension behaviour for Py3. [Mike Croucher]

* Various Py3 fixes. [Mike Croucher]

* Python 3 metaclass fix. [Mike Croucher]

* Various Python 3 fixes. [Mike Croucher]

* Types.TupleType -> tuple fix for python 3. [Mike Croucher]

* Im_self->__self__ fix for python 3. [Mike Croucher]

* Iterkeys fix for Python 3. [Mike Croucher]

* Removed debugger set up command. [Mike Croucher]

* Map fix for Python 3. [Mike Croucher]

* Changed refereences to iteritems() to items() for Py3 compat. [Mike Croucher]

* Updated README now that dict issues are fixed. [Mike Croucher]

* Fixed 'dict changed size' errors. [Mike Croucher]

* Import fixes for Py3. [Mike Croucher]

* Various Py3 related import fixes. [Mike Croucher]

* Changed refereences to iteritems() to items() for Py3 compat. [Mike Croucher]

* CPickle fix for Py3. [Mike Croucher]

* Changed refereences to iteritems() to items() for Py3 compat. [Mike Croucher]

* Updated README.m to refelect recent Py3 work. [Mike Croucher]

* Commented out weave functions for Py3 support. [Mike Croucher]

* Commented out weave functions for Py3 support. [Mike Croucher]

* Itertools fixes from 2to3. [Mike Croucher]

* Need to explicitly turn a range object to a list in for these tests. [Mike Croucher]

* Has_key has been removed from Python 3. [Mike Croucher]

* From functools import reduce for Py3 compat. [Mike Croucher]

* Ensure that object.__new__ never gets called with arguments. [Mike Croucher]

* Import reduce from functools for Py3 compat. [Mike Croucher]

* Ensure that object.__new__ never gets called with arguments. [Mike Croucher]

* Ensure that object.__new__ never gets called with arguments. [Mike Croucher]

* Fixed integer division for Python 3 compat. [Mike Croucher]

* Fixed string encoding for Python 3. [Mike Croucher]

* Reduce fix for Python 3. [Mike Croucher]

* Changed metaclass syntax to be Py3 compatible. This breaks Py2 compatibility. [Mike Croucher]

* Import reduce from functools for py3 compatibility. [Mike Croucher]

* Python3 compatbility fixes. [Mike Croucher]

* Print fixes for Python 3. [Mike Croucher]

* Print fixes for Python 3. [Mike Croucher]

* Fixed tab/space indentation issue. [Mike Croucher]

* Fix weave import for Py3. [Mike Croucher]

* More import fixes for Py3. [Mike Croucher]

* Import fixes for Py3. [Mike Croucher]

* Fixed inconsistent tab error. [Mike Croucher]

* Fixed inconsistent tab error. [Mike Croucher]

* More input fixes. [Mike Croucher]

* More input fixes. [Mike Croucher]

* Import fixes for Py3. [Mike Croucher]

* Import fix for Py3. [Mike Croucher]

* Changed <> to != for Py3 compatibility. [Mike Croucher]

* Updated Py3 work. [Mike Croucher]

* Removed import urllib2 since it wasn't being used. [Mike Croucher]

* Urllib2 fixes for Py3 compatibility. [Mike Croucher]

* Fixed cPickle import for Python 3. [Mike Croucher]

* Exception raising fix for Python 3. [Mike Croucher]

* Put weave import in a try block so it fails gracefullt in Py3. [Mike Croucher]

* Merge remote-tracking branch 'upstream/devel' into devel. [Mike Croucher]

* Fixed ConfigParser for Python 3 compat. [Mike Croucher]

* Commented out cholupdate since it uses weave and appears not to be used. [Mike Croucher]

* Added Python 3 progress to README.md. [Mike Croucher]

* Resolved merge conflict. [Mike Croucher]

* Exception fixes for Python 3 compat. [Mike Croucher]

* Exception fixes for Python 3 compat. [Mike Croucher]

* Exception fixes for Python 3 compat. [Mike Croucher]

* Exception fixes for Python 3 compat. [Mike Croucher]

* Exception fixes for Python 3 compat. [Mike Croucher]

* Exception fixes for Python 3 compat. [Mike Croucher]

* Typo. [Mike Croucher]

* Fixed Python 2 compatibility. [Mike Croucher]

* Convert print to function for Python 3 compatibility. [Mike Croucher]

* Convert print to function for Python 3 compatibility. [Mike Croucher]

* Convert print to function for Python 3 compatibility. This breaks compatibility for versions of Python < 2.6. [Mike Croucher]

* Convert print to function for Python 3 compatibility. [Mike Croucher]

* Convert print to function for Python 3 compatibility. This breaks compatibility for versions of Python < 2.6. [Mike Croucher]

* Convert print to function for Python 3 compatibility. [Mike Croucher]

* Merge remote-tracking branch 'upstream/devel' into devel. [Mike Croucher]

* Added details of Python 3 work. [Mike Croucher]

* Convert print to function for Python 3 compatibility. This breaks compatibility for versions of Python < 2.6. [Mike Croucher]

* More relative import fixes for Python 3 compatibility. [Mike Croucher]

* Relative import fixes for Python 3 compatibility. [Mike Croucher]

* Updated README.md to refer to GPy/testing for running the tests. [Mike Croucher]

* Speed ups for normal cdf. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Added Y_metadata to log_predictive_density. [Alan Saul]

* [minor edits] [mzwiessele]

* [basis func kernels] added support for simple basis function kernels, can be easily extended by implementing phi function in BasisFuncKern. [mzwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mzwiessele]

* Small bugfix in white kernel. [James Hensman]

* Tests to probe the mean-function functionality. [James Hensman]

* Merge branch 'mean_functions' into devel. [James Hensman]

* Merges. [James Hensman]

* Whitespace. [James Hensman]

* Mean functions now working for svgp. with tests. [James Hensman]

* Fixed up product kernel tests. [James Hensman]

* Mappings, including tests. [James Hensman]

* Working mean function examples. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into mean_functions. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into mean_functions. [James Hensman]

* Merged. ish. [James Hensman]

* Merge pull request #174 from beckdaniel/kernel_prod_bugfix. [James Hensman]

  Kernel product bugfix

* Test + code change in gradients_X. [Daniel Beck]

* A cleaner test. [Daniel Beck]

* Changed operator.mul to np.multiply for consistency. [Daniel Beck]

* Added decorator that changes numpy invalid op warning to exception. [Daniel Beck]

* First attempt. [Daniel Beck]

* Merge pull request #177 from mellorjc/master. [James Hensman]

  matplotlib interactive mode only in IPython

* Catch only a specific error. [mellorjc]

  catch only NameError, rather than everything.

* Matplotlib interactive mode only in IPython. [mellorjc]

  have interactive mode only in IPython so that running scripts that plot from python behave like normal.

* A temporal fix for the problem of sometimes the model not being updated. [Zhenwen Dai]

* Add mcmc into inference import. [Zhenwen Dai]

* Bug in linalg jitchol!!! [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Merge branch 'devel' [Max Zwiessele]

* Implement update_gradients_diag for MLP kernel. [Zhenwen Dai]

* Re-implemented warpedGP for new release of GPy. [Nicolo Fusi]

* Fixed minor bug in sparse gp minibatch. [Nicolo Fusi]

* Minimual edits to exact_inference. [James Hensman]

* Added mean function into the prediction. [James Hensman]

* Added parseing of mean func to gp.py. [James Hensman]

* Mean functions in place. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Optimize sslinear kernel. [Zhenwen Dai]

* Fallback the implementation of spike and slab prior. [Zhenwen Dai]

* Change the name of kernel DiffGenomeKern to DEtime. [Zhenwen Dai]

* More samples for predictive quantile. [Alan Saul]

* Merge branch 'saul_merge' into devel. [Alan Saul]

* Added safe_exp and tests. [Alan Saul]

* Small tidying up. [Alan Saul]

* Merge branch 'devel' into saul_merge. [Alan Saul]

* Merging with private repo, mostly fixed. [Alan Saul]

* Added block matrix dot product. [Alan Saul]

* Adding likelihoods and block matrices. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Relaxed inference test requirement. [Alan Saul]

* Adding a comment to clarify predictive_gradeints (Thanks AT) [James Hensman]

* Extra kernel stressing in benchmarks, bugfix in svgp. [James Hensman]

* Adding the beginnings of some benchmarks. [James Hensman]

* Shape changes for gradeitns of likelihood parameters in variational_expectations. [James Hensman]

* Added some clarifying comments with NDL. [James Hensman]

* Some tests for the svgp, and some changes to the likelihoods. [James Hensman]

* Derivatives of likelihood things now working for svgp. [James Hensman]

* Stupid bug. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Lots of changes to mappings. [James Hensman]

* [var plots] wrong return values. [mzwiessele]

* [variational] plot needed kwargs. [mzwiessele]

* [verbose opt] ipython notebook new version widget changes. [mzwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mzwiessele]

* A little work on mappings. [James Hensman]

* [verbose opt] ipython notebook new version widget changes. [mzwiessele]

* [optimization] html widget api changes in ipython notebook? [mzwiessele]

* [sparse gp] doc changes for missing data. [mzwiessele]

* [ploting init] minor. [mzwiessele]

* [optimization] model prints how many parameters there are to optimize. [mzwiessele]

* Remove printing. [Zhenwen Dai]

* Fix the param renaming problem. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* [optimize] max_f_eval -> max_iters. [mzwiessele]

* [optimize] added clear functionality for ipython notebook and kern input sensitivity error handling. [mzwiessele]

* Add trigger update to set_{X,Y,Z} [Zhenwen Dai]

* Add set_Z function. [Zhenwen Dai]

* Updated other likelihoods to give back logpdf and gradients for each link_f rather than summing on the inside. [Alan Saul]

* Added binomial likelihood. [James Hensman]

  Also some changes to pass through Y_metadata, where it had previously
  been (errorneously) omitted.

* Messy merge. [James Hensman]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [frb-yousefi]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mzwiessele]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Reconfigured svgp inference a little. [James Hensman]

* Added more stable expectations for Bernoulli. [James Hensman]

* [pickling errors] due to too little constant jitter, the gradient checks in pickle tests did not pass. [mzwiessele]

* [verbose optimization] added automatic detection of ipython notebook support, this is experimental. [mzwiessele]

* [sparse gp] prediction without missing data and uncertain inputs was bugged. [mzwiessele]

* [optimization] experimental auto detect of ipython notebook. [mzwiessele]

* [updateable] update field in observable. [mzwiessele]

* [var_dtc] constant jitter 1e-10. [mzwiessele]

* Add save param_array. [Zhenwen Dai]

* Minor error in corregionalization corrected. [javiergonzalezh]

* [sparse gp] prediction with uncertain inputs. [mzwiessele]

* Force set_XY to update the model. [Zhenwen Dai]

* DGPLVM. [frb-yousefi]

* Removed climin dependency unless actually needed. [James Hensman]

* Minor weave/numpy bug in coregionalize. [James Hensman]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Added logging for jitter so we know how much has been added and how many tries have been taken. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* [var dtc] added code for additional covariates, not affecting normal procedures. [Max Zwiessele]

* Bug in linalg jitchol!!! [Alan Saul]

* Fixed a plotting bug for sliced plots. [James Hensman]

* Update __init__.py. [Neil Lawrence]

  Add inference to __init__.py

* Update hmc.py. [Neil Lawrence]

* Update __init__.py. [Cristian Guarnizo]

  Added EQ_ODE2.

* Create eq_ode2.py. [Cristian Guarnizo]

  Added ODE2 kernel for latent force models.

* [opt] unified printing of status of optimization. [Max Zwiessele]

* [transformations] bugfix for pickling. [Max Zwiessele]

* [opt] bugfix. [Max Zwiessele]

* [verbose opt] messages would be overwritten when using ipython_notebooks, fixed. [Max Zwiessele]

* [kern] added Fixed kern to import list in GPY.kern. [Max Zwiessele]

* [objective grads] undid the change, as this would lead to dramatic problems in reloading old models using the pickle module. [Max Zwiessele]

* [objective grads] undid the change, as this would lead to dramatic problems in reloading old models using the pickle module. [Max Zwiessele]

* [progress] show progress of optimization using optimize(itpython_notebook=True) [Max Zwiessele]

* [progress] show progress of optimization using optimize(itpython_notebook=True) [Max Zwiessele]

* [updates] now handled in observable, should have from the begining :/ [Max Zwiessele]

* [optimization prints] unified printouts for optimizers, added ipython_notebook flag for use in ipython notebooks. [Max Zwiessele]

* [parent notification] is now priority -1000, instead of -inf. [Max Zwiessele]

* [model print] updates now shown in print out. [Max Zwiessele]

* [natural gradients] added natural gradients, usable but not analysed. [Max Zwiessele]

* Renamed opimizer methods to unobscure gradients and objective. [Max Zwiessele]

* [parameterized] print outs for ipython notebook. [Max Zwiessele]

* SVI now working with minibatches. [Alan Saul]

* SVI now implemented without natural natural gradients or batches. [Alan Saul]

* Multi-outputted the svgp inference (buggy, probably) [James Hensman]

* Added svgp in partially broken state ready for multiouputs. [Alan Saul]

* Fixed quadrature for bernoulli likelihood, started adding Gaussian likelihood derivatives for quadrature. [Alan Saul]

* Svgp inference added -- not working yet. [James Hensman]

* [html print] more table based corrections for html printing. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Implement update_gradients_diag for MLP kernel. [Zhenwen Dai]

* [html repr] included css styling for html print outs. [Max Zwiessele]

* [vardtc] sparse gplvm in bayesian gplvm minibatch. [Max Zwiessele]

* [vardtc] predict with uncertain inputs, the non principled way. [Max Zwiessele]

* [natgrad] taking the gradient in the old direction, without adjustment. [Max Zwiessele]

* [model] update messages a little nicer. [Max Zwiessele]

* [stationary] lengthscales will be scaled by variance now. [Max Zwiessele]

* [Updateable] deprecated updates. [Max Zwiessele]

* [setup] new version number, to avoid confusion. This will be the next minor update, including changes to README and bugfixes. [Max Zwiessele]

* Update README.md. [James Hensman]

* Update README.md. [James Hensman]

* Update README.md. [James Hensman]

* Huge merge correcting upstream master. [Alan Saul]

* Merge branch 'devel' [Max Zwiessele]

* Merge branch 'devel' of github.com:/sheffieldml/GPy into devel. [James Hensman]

* Another attempt by installing a mini version of anaconda, should be easier to maintain. [Alan Saul]

* Attempting to fix travis build. [Alan Saul]

* Removed more sgd. [James Hensman]

* [huge merge] the second. [Max Zwiessele]

* [huge merge] trying to merge old master and master. [Max Zwiessele]

* Merge pull request #148 from martinsch/normalization_fix. [James Hensman]

* Normalization: avoid division by zero for constant feature dimensions. [mschiegg]

* Merge pull request #160 from slinderman/master. [James Hensman]

  Removing set of numpy random seed

  Great spot! We're just about to relase the next iteratino of GPy, we'll make sure it happens there too.

  Thanks.

* Removing set of numpy random seed. [Scott Linderman]

* Removed unnecessary spaces from citation. [Nicolo Fusi]

* Added a way to cite. [Nicolo Fusi]

* Modified logexp transformation to prevent it returning zero when argument is under -36. [Neil Lawrence]

* Version change. [Max Zwiessele]

* Dim reduction examples clearer and init not as much black magic anymore. [Max Zwiessele]

* Plot_latent now shows selected inputs, even after switching dimensions. [Max Zwiessele]

* Added hapmap3 as dataset. [Max Zwiessele]

* Sparse gp stability improved. [Max Zwiessele]

* HapMap3 dataset added. [Max Zwiessele]

* Added hapmap download, need to put in data preprocessing for actual usability. [Max Zwiessele]

* Dim reduction examples. [mzwiessele]

* Dim reduction examples. [mzwiessele]

* Bgplvm steepest gradient map update. [mzwiessele]

* Bgplvm steepest gradient map update. [mzwiessele]

* Merge branch 'master' of github.com:SheffieldML/GPy. [mzwiessele]

* Update README.md with funding acknowledgements. [Neil Lawrence]

* Plotting bug for bgplvm fixed. [mzwiessele]

* Scg optimizer scale bounds back to 1e-15. [mzwiessele]

* Parameterized: added warning switch. [mzwiessele]

* Mrd corrections. [mzwiessele]

* Version change (early beta, do not change until everythin works. [Max Zwiessele]

* Fixed the SCG optimizer, thanks to Yarin Gal. [James Hensman]

* Version now 48. [Max Zwiessele]

* Version update. [Max Zwiessele]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Max Zwiessele]

* Pickling now allways binary as well as protocol -1. [Max Zwiessele]

* Pickling now allways binary as well as protocol -1. [Max Zwiessele]

* Using lbfgs algorithm from scipy.minimize, starting to convert all optimizers to minimize format. [Max Zwiessele]

* Windows -.- [Max Zwiessele]

* Versions update. [Max Zwiessele]

* Image is a PIL requirement and should only be imported when actually using it. [Max Zwiessele]

* Plot handling greatly improved for latent space visualizations. [Max Zwiessele]

* Version file added. [Max Zwiessele]

* Fixed Ctrl-C behaviour on Windows. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Fixed come path issues in sympykern. [James Hensman]

* Rename and redoc. [Alan Saul]

* Moving imports, attempting to update RTD. [Alan Saul]

* Changed initalise_latent to take lower of init=PCA and corrected import. [Alan Saul]

* Change order of imports for RTD. [Alan Saul]

* Merge branch 'devel' [Alan Saul]

* Merge branch 'master' into devel. [Alan Saul]

* Removed variational.py. [Alan Saul]

* Fixed plot_latent failure. [Alan Saul]

* Ignore example tests. [Alan Saul]

* Removed yes pipe for travis. [Alan Saul]

* Seems to handle without answering now. [Alan Saul]

* Reverse travis to see what it asks for. [Alan Saul]

* Fixed some tests. [Alan Saul]

* Merged and fixed conflict in ODE_UY.py. [Alan Saul]

* Merge completed. [Max Zwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mu]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Andreas]

* Small changes in svigp. [Andreas]

* ODE UY dkdtheta. [mu]

* UY dkdtheta. [mu]

* UY dkdtheta. [mu]

* BGPLVM with missing data. [Max Zwiessele]

* Pca adjustements to lvm models. [Max Zwiessele]

* Ppca added, ppca missing data not working yet. [Max Zwiessele]

* Diagonal operations. [Max Zwiessele]

* Subarray indexing. [Max Zwiessele]

* Documenting. [Max Zwiessele]

* Bug in ODE_UY fix. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Fixed the numerical quadrature, won't work with large f unless normalized. [Alan Saul]

* Fixed lots of breaking tests, reduced step size for checkgrad to 1e-4 in tests (perhaps this should be global), added some missing attributes to data_resources.json. [Alan Saul]

* Set warnings for truncated hessian, it has been noted that that by truncating we can have incorrect posteriors, though at convergence this should not be a problem, could be fixed by not using Cholsky as the decomposition as it cannot handle non-positive definite mats. [Alan Saul]

* Merge pull request #90 from jamesmcm/master. [Alan Saul]

  Fixing ReadTheDocs reading docstrings, adding data_resources.json

* Adding data_resources.json to setup data files. [James McMurray]

* Testing modification for ReadTheDocs to stop docstring errors. [James McMurray]

* Ensure_defaiult constraints in svigp. [James Hensman]

* Fixed Ctrl-C behaviour on Windows. [Nicolo Fusi]

* Removed print statements from config parser, commented out ODE kerns. [Nicolo Fusi]

* Merge branch 'devel' [Nicolo Fusi]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mu]

* Ode UY. [mu]

* Dk dparameter. [mu]

* ODE_UY. [mu]

* Merge branch 'devel' [Nicolo Fusi]

* Added some more error checking for downloading datasets. [Neil Lawrence]

* Fixed some bugs in mocap.py where errors weren't being raised when file format was incorrect and made datasets.py check for 404 errors which previously were occuring silently ... shhhhh. [Neil Lawrence]

* Fixed bugs in cmu_mocap loader where cmu_url was missing and loading in mocap data twice in same session led to incorrect url through copy error. [Neil Lawrence]

* Fixed examples tests, started changing datasets code which has a few bugs. [Alan Saul]

* Minor changes to naming of signitures. [Alan Saul]

* Changed more examples to accept optimize and plot. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Added comments for models module and adjusted setup. [Max Zwiessele]

* Merge branch 'naniny' into devel. [Max Zwiessele]

* Rename _models to models_modules to include in doc. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Some tidying in the regression examples. [James Hensman]

* Added optimize and plot for classification, non_gaussian and stochastic examples. [Alan Saul]

* Fixed gp_base and svigp for sampling (doesn't use it but needs the arguments) [Alan Saul]

* Added constant to Z_tilde, now log likelihoods are equal! [Alan Saul]

* Changed some parameters of the laplace, tidied up examples. [Alan Saul]

* Dimensionality reduction examples updated with optimize, plot and verbose. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Changing the seed seems to fix Alan's bug. [James Hensman]

* Fixed symmetry in checkgrad issue. [James Hensman]

* Reverted the brent optimisation in laplace. [James Hensman]

  (For the 1D linesearch using Brent)

* Improved detectino of sympy. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Fixed exception handling bug in GPy/util/linalg.py:29. [Teo de Campos]

* Argghdfklg. [James Hensman]

* Better warings for cathcing of blaslib detection. [James Hensman]

* Changeing models to _models in setup.py. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Fixed step size for likelihood tests and allowed randomizing of laplace. [Alan Saul]

* Fixed student_t approximation demo and changed convergence critera to difference of f. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Added cfg file to manfiest and package_data. [Alan Saul]

* Dimensionality reduction example (oil) updated. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Removed ipdb statement from kern, cleaned up some nasty whitespace. [James Hensman]

* More readme. [James Hensman]

* More readme stuff. [James Hensman]

* More readme edits. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Minor edits to the README. [James Hensman]

* Gradient checker comments and import updates. [Max Zwiessele]

* Gradientchecker added as a model. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Push minor fix to eq_sympy kernel test. [Neil Lawrence]

* Added some tips to the readme. [James Hensman]

* Fixed import errors in tests. [Max Zwiessele]

* Sympykern kern_tests now passing, code is inefficient but should be numerically stable. [Neil Lawrence]

* Modified to improve part of stability, gradient checks still passing. [Neil Lawrence]

* Added gradient of sympy kernel, seems to pass tests, but know it's not numerically stable. Checking in before making numerically stable. [Neil Lawrence]

* Fixed test in kern.py to request correct output dim for multioutput covariances. [Neil Lawrence]

* ODE_UY. [mu]

* Fixing up the blas detectino in linalg. [James Hensman]

* Removing ipdb statements. [James Hensman]

* Lots of medding with the likelihoods to get the tests working. the tests still don;t work. [James Hensman]

* Formatting docstring. [James Hensman]

* Changed nasty whitespace. [James Hensman]

* Eq_ode1 working but test failing? [Neil Lawrence]

* Added eq_ode1 to constructors.py. [Neil Lawrence]

* Fixed bug in sympy kernel and added sympolic.py back into utils __init__.py. [Neil Lawrence]

* Merge with James's changes. [Neil Lawrence]

* Merge. [James Hensman]

* Removed some sympy stuff. [James Hensman]

* Merge with James's changes. [Neil Lawrence]

* Committing change for master check out. [Neil Lawrence]

* Skipping crossterm tests instead of expected failure. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Added a path for the data resources. not all users will be working in the GPy directory. [James Hensman]

* Moved data resource information to a json file. [Neil Lawrence]

* Bug fix for single output sympy kernel. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mu]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Fixed problem in warping. [Nicolo Fusi]

* Constructor and init for ODE_UY. [mu]

* Working eq_ode1 in sympy now. [Neil Lawrence]

* Dim reduction imports. [Max Zwiessele]

* Testing imports update and expected failure for crossterms. [Max Zwiessele]

* Rename models to _models and import models in models.py. [Max Zwiessele]

* Psistattests update. [Max Zwiessele]

* Updated crossterms, rbf x any not working yet (derivatives) [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Weird merge. [James Hensman]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Part implementation of ode_eq functionality. Not yet numerically stable or efficient (some horrible use of cut and paste to get things working ...) [Neil Lawrence]

* A trial namespace renaming. [James Hensman]

* Better handling of missing config files. [James Hensman]

* Debugging the config paths. [James Hensman]

* Allowing the passing of 1D X to a GP. with warning of course. [James Hensman]

* More fiddling with the windows path for config. [James Hensman]

  Where is the windows guru? out playing beach volley?

* Changed how we search for config files on windows. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Fixed up symmetric kern. [James Hensman]

* Half way through crossterm objective. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Added block matrix utility. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Added **likelihood_params to predictive_values. [Ricardo]

* Changes in plot function: sampling vs numerical approximation. [Ricardo]

* Adding docstring for symmetric kern. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Bug fixed in numerical approx. to the predictive variance. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Numerical predictions fixed, sampling predictions are not working. [Ricardo]

* Predictive_mean and predictive_variance now use gp_var as a parameter, rather than gp_std. [Ricardo]

* Fixed product kern get and set state. [James Hensman]

* Added getstate/setstate for product kernel. [James Hensman]

* In the middle of crossterms. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* 2D plots fixed. [Ricardo]

* Passing **noise_args into predictive_values. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Missing term in the likelihood. [Ricardo]

* Reverted broken kern. [Max Zwiessele]

* Added variational distribution for latent space. [Max Zwiessele]

* BGPLVM test for crossterms. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Even more data plotting. [James Hensman]

* Fixed plotting isue with plot_f. [James Hensman]

* Fixed the dpotrs use.. [Alan Saul]

* Added dpotrs instead of cho_solve. [Alan Saul]

* Removed ipython dependency from kern. [Alan Saul]

* Sped up sampling a lot for student t, bernoulli and poisson, added sampling for gaussian and exponential (untested) [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' [Alan Saul]

* Ignoring examples tests again. [Alan Saul]

* Added sampling to student_t noise distribution, very slow and is possible to speed up. predictive mean analytical and variance need checking. [Alan Saul]

* Tidying up and fixed objective being vector. [Alan Saul]

* Added sampling for predictive quantiles and also mean and variance where necessary. [Alan Saul]

* Rederived gamma distribution. [Alan Saul]

* Added derivatives for poisson and a couple of examples, need to fix for EP. [Alan Saul]

* Merged with devel. [Alan Saul]

* Reimplemented gradients for exponential, seems to work for laplace now, needs a visual test though. [Alan Saul]

* Was a bug in the examples_tests.py, fixed and added brendan faces to ignore list. [Alan Saul]

* Minor clean up. [Alan Saul]

* Removed unnecessary laplace examples. [Alan Saul]

* Updated laplace example to use predictive density aswell as RMSE. [Alan Saul]

* Added log predictive density, ln p(y*|D) [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' [Alan Saul]

* Updated boston tests (more folds, allow a bias as the datasets are not normalized once split) and more folds. Tweaked some laplace line search parameters, added basis tests for ep. [Alan Saul]

* Fixed bug in gradient checker where it worked differently given a integer parameter to a float. [Alan Saul]

* Removed derivatives of variance wrt gp and derivatives of means with respect to gp from noise models. [Alan Saul]

* Tore out code no longer used from noise_distributions due to rewriting using quadrature. [Alan Saul]

* Added numerical mean and variance with quadrature, about to clean up. [Alan Saul]

* Merge branch 'master' into merge_branch. [Alan Saul]

* Changed the gradients (perhaps for the worse) [Alan Saul]

* A few typos. [Alan Saul]

* Gaussian likelihood errors, still not working. [Alan Saul]

* Added gaussian checker and gaussian likelihood, not checkgrading yet. [Alan Saul]

* Started adding gaussian sanity checker. [Alan Saul]

* Got rid of some overdoing the approximation. [Alan Saul]

* Started adding gaussian likelihood, changed round preloading old_a. [Alan Saul]

* Trying to debug kernel parameters learning (fails even when noise fixed) may be some instablility, seems like it can get it if it starts close. [Alan Saul]

* Fixed 2*variance plotting instead of 2*std plotting, tidied up. [Alan Saul]

* Changed incorrect naming. [Alan Saul]

* Reparameratised in terms of sigma2. [Alan Saul]

* Playing trying to find what makes it want to go so low. [Alan Saul]

* Fixed bug where B wasn't refering to current f location. [Alan Saul]

* Everything seems to be gradchecking again. [Alan Saul]

* Added minimizer for finding f, doesn't help. [Alan Saul]

* Now checkgrads a lot more of the time, but still fails in optimisation, seems also odd that when parameter is fixed kernel parameters go to infinity. [Alan Saul]

* Added another optimisation which doesn't use gradients. Seems like F is almost always found, but Y can be off, suggesting that Wi__Ki_W is wrong, maybe W? [Alan Saul]

* Trying to fix optimisation problem, fixed a few bugs but still fails at very low noise. [Alan Saul]

* Starting to fiddle with mode finding code. [Alan Saul]

* Fixed a sign wrong, now gradchecks weirdly only above certain points. [Alan Saul]

* Now gradchecks everytime but student_t fit is bad, noise is underestimated by a long way. [Alan Saul]

* Checkgrads with explicit and implicit components half the time. [Alan Saul]

* About to input new derivations for Z's... [Alan Saul]

* Took out all the asserts and using pure broadcasting method of diagonal now. [Alan Saul]

* Made it use the fact that W is diagonal and put assertions in to ensure that the results are the same. [Alan Saul]

* Broken it by getting rid of squeeze, but now working on making it faster using proper vector multiplciation for diagonals. [Alan Saul]

* Made more numerically stable in a hope that it will work and I will find a bug... [Alan Saul]

* Lots of name changing and went through all likelihood gradients again. [Alan Saul]

* Ripped out all things Laplace parameter estimation, starting again with new tactic. [Alan Saul]

* About to rip out old chain rule method of learning gradients. [Alan Saul]

* Trying to fix dL_dytil gradient. [Alan Saul]

* Changed name. [Alan Saul]

* FIXED DYTIL_DFHAT. [Alan Saul]

* Workong on doing explicit gradients. [Alan Saul]

* Gradients almost there for dytil_dfhat, diagonal terms are right. [Alan Saul]

* Merged with devel. [Alan Saul]

* Still getting closer to grads for likelihood. [Alan Saul]

* Almost have likelihood gradients working but kernels still way off. [Alan Saul]

* Working on putting callback to update laplace in callback. [Alan Saul]

* Seem to have gradients much closer now. [Alan Saul]

* Scale and switch KW+I. [Alan Saul]

* Merged with upstream. [Alan Saul]

* Added a debug examples. [Alan Saul]

* Merging. [Alan Saul]

* Merged with master. [Alan Saul]

* Plotting problematic kernel. [Alan Saul]

* Adding gradients, shapes starting to make sense. [Alan Saul]

* Attempted to introduce gradient methods, won't work yet I doubt. [Alan Saul]

* Merge remote-tracking branch 'upstream/devel' [Alan Saul]

* Merged likelihood functions. [Alan Saul]

* Should be working now, needed to change relative path names. [Alan Saul]

* Merge branch 'merge_trial' [Alan Saul]

* Merge branch 'old_repo' into merge_trial. [Alan Saul]

* Make directory structure match that of GPy. [Alan Saul]

* Tidy up comments. [Alan Saul]

* Merged in the GPy upstream. [Alan Saul]

* Merged in branch which had old_repo merged in. [Alan Saul]

* Adding weibull likelihood, requires 'extra_data' to be passed to likelihood, i.e. the censoring information. [Alan Saul]

* Fixed the z scalings. [Alan Saul]

* Fixed laplace approximation and made more numerically stable with cholesky decompositions, and commented. [Alan Saul]

* Stabalised most of the algorithm (apart from the end inversion which is impossible) [Alan Saul]

* Added timing and realised mdot can be faster as its almost always a diagonal matrix its multiplying with. [Alan Saul]

* Got the mode finding without computing Ki. [Alan Saul]

* Fixed broadcasting bug, rasm now appears to work. [Alan Saul]

* Still working on rasmussen, link function needs vectorizing I think. [Alan Saul]

* Tidying up. [Alan Saul]

* Added predicted values for student t, works well. [Alan Saul]

* Working laplace, just needs predictive values. [Alan Saul]

* Seemed to be working, now its not. [Alan Saul]

* Changing definitions again... [Alan Saul]

* Worked out in terms of W, needs gradients implementing. [Alan Saul]

* Just breaking some things... [Alan Saul]

* Following naming convention better, lots of inverses which should be able to get rid of one or two, unsure if it works. [Alan Saul]

* Got an idea of how to implement! written in docs. [Alan Saul]

* Trying to 'debug' [Alan Saul]

* Got most of laplace approximation working. [Alan Saul]

* Added some comments. [Alan Saul]

* Initial commit, setting up the laplace approximation for a student t. [Alan Saul]

* Initial commit. [James Hensman]

* Use bfgs for laplace instead. [Alan Saul]

* Moved transf_data to make data -1 or 1 from 0 or 1 for bernoulli with probit into the analytical moment match (but it 10% slower), needs removing from epmixednoise. [Alan Saul]

* Changed naming from old derivatives of likelihoods to new ones in noise distributions. [Alan Saul]

* Fixed breakage of dvar, tidied up to make more efficient. [Alan Saul]

* Doc stringing. [Alan Saul]

* Added quadrature numerical moment matching (but not predictive yet) [Alan Saul]

* Fixed a few laplace bits. [Alan Saul]

* Refactored gradients wrt parameters slightly, need to future proof against _get_param_names() disappearing. [Alan Saul]

* Added more options to generic tests (constraining link function values as bernoulli requies R^{0,1}) and implemented new gradients for bernoulli. [Alan Saul]

* Rename Binomial to Bernoulli (maybe generalise it with the constant later, but tilted distribution may change) [Alan Saul]

* Added pdf_link's for gaussian and student t, added third derivatives for transformations and tests for them. [Alan Saul]

* Renamed laplace_tests to likelihoods_tests. [Alan Saul]

* Tidying up laplace_tests.py. [Alan Saul]

* Fixed some bugs, added third derivative for log transformation, and did some doccing. [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' into merge_branch. [Alan Saul]

* Still tidying up, laplace now working again, gaussian and student_t likelihoods now done. [Alan Saul]

* Started on chaining, must remember to chain _laplace_gradients aswell! [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' into merge_branch. [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' into merge_branch. [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' into merge_branch. [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' into merge_branch. [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' into merge_branch. [Alan Saul]

* Finished tearing gaussian noise down, time for student t. [Alan Saul]

* Beginning to merge lik_functions and derivatives with richardos. [Alan Saul]

* Docs. [Alan Saul]

* Removed fit as it is unused. [Alan Saul]

* More doc strings. [Alan Saul]

* Doccing and testing for D dimensional input (not multiple dimensional Y yet) [Alan Saul]

* Tidying up a lot, works for 1D, need to check for more dimensions. [Alan Saul]

* Tidied up laplace. [Alan Saul]

* Integrated Laplace and merged Merge remote-tracking branch 'gpy_real/devel' into merge_branch. [Alan Saul]

  Conflicts:
    GPy/core/gp.py
    GPy/likelihoods/__init__.py
    GPy/likelihoods/likelihood_functions.py
    GPy/likelihoods/link_functions.py

* Fixed white variance. [Alan Saul]

* Boston housing works (apart from variance of student t is not valid below 2) [Alan Saul]

* Tests setup but not fitting properly yet. [Alan Saul]

* Changed the examples (started boston data) and increased tolerance of finding fhat. [Alan Saul]

* Added some stability and tidied up. [Alan Saul]

* Tidying up. [Alan Saul]

* Student t likelihood function checkgrads (summed gradients wrt to sigma2), maybe some numerical instability in laplace. [Alan Saul]

* Now checkgrads for gaussian, and ALMOST for student t. [Alan Saul]

* All gradients now gradcheck. [Alan Saul]

* Merge remote-tracking branch 'gpy_real/devel' into merge_branch. [Alan Saul]

* Refactored tests. [Alan Saul]

* Tidied up grad checking. [Alan Saul]

* Added tests and fixed some naming. [Alan Saul]

* Modified gradient_checker to allow for variable 'f' [Alan Saul]

* Renamed some things, made some small (incorrect) gradient changes, generalised the gp regression for any likelihood, and added a place holder link function waiting for Richardos changes. [Alan Saul]

* Removed unneeded dependency. [Alan Saul]

* Merged GP models. [Alan Saul]

* Dragged likelihood_function changes in. [Alan Saul]

* Checked out relavent files. [Alan Saul]

* Merged in real gpy. [Alan Saul]

* Empty branch. [Alan Saul]

* SPELLAFSDIUN. [Max Zwiessele]

* Fixed up plotting in sparse_gp also. [James Hensman]

* Fixed up the plotting. [James Hensman]

* Fixed up plot in GP_base. [James Hensman]

* Started changing the plotting in examples to remove plot_single_output. [James Hensman]

* General tidying in models. [James Hensman]

* Improved docstrings in svigp. [James Hensman]

* Some tidying in gp.py. [James Hensman]

* Docstrings and removal of duplicated plotting code in gp_base. [James Hensman]

* Turned omp off by default as discussed. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Added configuration file. [Nicolò Fusi]

  this was done to solve the OpenMP problem on Windows/mac, but I think it
  is useful in general. All unit tests pass except the sympy kern ones.

* Added xw_pen data. [Neil Lawrence]

* Added xw_pen data. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Basic sim code functional. [Neil Lawrence]

* Minor change in tutorial. [mu]

* Added link to user mailing list. [James Hensman]


## v0.6.0 (2014-11-21)

### Other

* Small random perturbations in kernel tests helps with the symmetry gradcheck bug. [James Hensman]

* More cooooooopyrighting. [James Hensman]

* More coooooopyrighting. [James Hensman]

* Removing old notes.py, issues are now all on github. [James Hensman]

* More cooooopyrighting. [James Hensman]

* More cooooopyrighting. [James Hensman]

* More coooopyrighting. [James Hensman]

* More cooopyrighting. [James Hensman]

* More coopyrighting. [James Hensman]

* More copyrighting. [James Hensman]

* More ]#copyrighting. [James Hensman]

* Branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* [kernel plots] updates on bar plots. [Max Zwiessele]

* Linear kernel speed up. [Zhenwen Dai]

* Performance improvement for sslinear kernel. [Zhenwen Dai]

* Copyrighting. [James Hensman]

* Working One vs All sparse gp classification wrapper. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Added lengthscales for a standard GPLVM with ARD. [Andreas]

* Small changes to 1vsAll. [Andreas]

* New file, sparse one vs all classification. [Ricardo]

* Missing file. [Ricardo]

* One vs all classification. [Ricardo]

* [inferenceX] with missing data. [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

  Conflicts:
    GPy/inference/latent_function_inference/inferenceX.py

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Bug fix for infer_newX. [Zhenwen Dai]

* Ssrbf bug fix. [Zhenwen Dai]

* Ssrbf bug fix. [Zhenwen Dai]

* [infer_newX] updated for missing data. [Max Zwiessele]

* [priors] pickling priors (not working for Discriminative prior) [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Spike and slab binary variable numerical enhancement. [Zhenwen Dai]

* Stabilization of the Bernouolli. [James Hensman]

* [priors] proprietary pickling of priors. [Max Zwiessele]

* [MRD] updates and nicer plotting. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Rename the save_params_H5 function to be a general function save which can potentially support other file format. [Zhenwen Dai]

* Add the function of saving all the parameters into a HDF5 file. [Zhenwen Dai]

* [MRD] init and sim nicer now. [Max Zwiessele]

* [updates] starting to extract out standalone modules. [Max Zwiessele]

* New ssrbf implementation. [Zhenwen Dai]

* Mergt push e branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Fixed or zero size models will now not raise an error when trying to optimize. [James Hensman]

* Fixed or zero size models will now not raise an error when trying to optimize. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Fixed lots of examples. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Fixing examples. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Documenting the core GP class. [Alan Saul]

* Add mpi support for sparsegpregression. [Zhenwen Dai]

* [updates] starting to extract out standalone modules. [Max Zwiessele]

* [linalg] ppca with missing data removed. [Max Zwiessele]

* [setup] tweaks for submission. [Max Zwiessele]

* [dim red] cmu_mocap normalize. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Redundant models deleted. [Ricardo]

* Update docstring for checkgrad. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Add test case for mpi. [Zhenwen Dai]

* [examples] dim red bgplvm with missing data. [Max Zwiessele]

* [testing] seed problems. [Max Zwiessele]

* [setup] updated and ready to ship. [Max Zwiessele]

* [mpi] deleted import. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

  for cleaning up of parallel

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Better handling of missing pods in examples. [James Hensman]

* Clean up parallel framework. [Zhenwen Dai]

* Model checkgrad enhancement. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Updated conf.py to work again rather than cause an infinite loop. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Added verbose checks for likelihood. [Alan Saul]

* Added kern _src files using api-doc -P ../doc ../ to include private modules. [Alan Saul]

* Remove dead hierachical code. [James Hensman]

* Work on kernel plotting. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Improved kernel plotting. [James Hensman]

* [renaming] now indexes names, instead of adding _ [Max Zwiessele]

* [mixed_noise] variance shape. [Max Zwiessele]

* [coreg regression] kernel name is coreg. [Max Zwiessele]

* [normalize] deleted in gplvm. [Max Zwiessele]

* Vardtc mixed noise. [Max Zwiessele]

* [examples] pods. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* A bug fix for set_XY. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Fixed prior error. [Alan Saul]

* Update the set_XY function. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Added homepage to main GPy project page. [Alan Saul]

* Extend inference X for all gp models. [Zhenwen Dai]

* [mixed noise] correction for mixed noise var dtc. still have to make a test. [Max Zwiessele]

* [tests] for issue ##146 and #147, fixing parameters inside __init__ [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Minor doc changes, fixed MPI dependency and 'stop' in var_dtc. [Alan Saul]

* Removed ordinal.py (to Symbolic). [Neil Lawrence]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Add test cases for inference new X for bayesian GPLVM. [Zhenwen Dai]

* Bug fix for inferenceX. [Zhenwen Dai]

* Add new inference X. [Zhenwen Dai]

* [tests] pickle tests updated to new structure. [Max Zwiessele]

* [MRD] running again, using missing_data classes, more details needed for missing data though. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Refine the docstring for hmc. [Zhenwen Dai]

* Removed pod dependency for pickle tests. [Alan Saul]

* Tidied up laplace warnings. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Add __init__.py to mcmc. [Zhenwen Dai]

* Add documentation for hmc. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Changed init for mcmc. [James Hensman]

* Removing dead bayesopt file. [James Hensman]

* Catch nosetests import error if not installed, now ignore GPy.tests() when nosetests GPy is called, but allows GPy.tests() to be called, and throws error if this is tried without nose being installed. [Alan Saul]

* Moving mcmc. [James Hensman]

* [parameterized] handle updates inside init. [Max Zwiessele]

* [minbatch var dtc] adjustments to bgplvm minibatch. [Max Zwiessele]

* [VarDTC] reverted SparseGP to previous state, updated BGPLVM accordingly. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Omp for dX. [James Hensman]

* [ParameterizedTests] added test functionality. [Max Zwiessele]

* [IndexOps] added new tests for index operations. [Max Zwiessele]

* [SparseGP] added self.full_values. [Max Zwiessele]

* [MRD] changes for new inference technique. [Max Zwiessele]

* [ObsAr] do not make a copy, if not needed. [Max Zwiessele]

* [var dtc missing] deleted code for missing data inference. [Max Zwiessele]

* [stochastics] updated some stuff on the stochastics. [Max Zwiessele]

* [sparse gp] prediction with posterior per dimension activated. [Max Zwiessele]

* [init] parameter updates now check if in init. [Max Zwiessele]

* [pickling] load added to gpy, allows for easy loading of pickled models. [Max Zwiessele]

* [latent plotting] some adjustments for nicer looking plots. [Max Zwiessele]

* [Laplace] sum now around argument, instead of call to array. [Max Zwiessele]

* Indexing bugfix in weave. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* [ObsAr] added .values. [Max Zwiessele]

* Merge. [James Hensman]

* [transformations] natural gradient all in testphase. [Max Zwiessele]

* [randomize] randomize now without init. [Max Zwiessele]

* [stochastics] added doc. [Max Zwiessele]

* [transformations] gradfactor change and Natural gradient transformations. not working fully, yet :( [Max Zwiessele]

* [sparse GP] fallback for other inference methods for missing_data. [Max Zwiessele]

* [kernel slicing] active_dims can be a single integer now. [Max Zwiessele]

* [classification] sparse gp inference for EPDTC. [Max Zwiessele]

* [pca] pca -> PCA. [Max Zwiessele]

* Weave ObsArray bugfix. [James Hensman]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Merge the discriminative prior to devel. [Fariba]

* The discriminative prior. [Fariba]

* Numerical stability in variational expectations. [James Hensman]

* Handles import of pods correctly. [James Hensman]

* Changed pylab for pyplot in classification examples. [James Hensman]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Additions to week2 MLAI. [Neil Lawrence]

* Added cloglog link fn. [James Hensman]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Remove symbolic import. [Neil Lawrence]

* Remove symbolic import. [Neil Lawrence]

* Remove symbolic import. [Neil Lawrence]

* Remove symbolic import. [Neil Lawrence]

* Remove symbolic import. [Neil Lawrence]

* More edits for variational expectations in likelihoods. [James Hensman]

* More variational quadtrature code. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* More removal of references to broken files. [James Hensman]

* [missingdata] file for missing data was missing O_o. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Fixing more issues cauesd by removal of symbolic.py. [James Hensman]

* Docstrings. [James Hensman]

* Commented out stochastics.py -- not added. [James Hensman]

* [stochastics] holds stochastic updates and rules. [Max Zwiessele]

* [vardtc] missing data handling and stochastic update in d. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Removed reference to symbolic.py, which NDL deleted. [James Hensman]

* Removed reference to symbolic.py, which NDL deleted. [James Hensman]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Attempt to align numbers to right. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Display of models and params for the notebook. [Neil Lawrence]

* Beginning of adding variational GH quadrature to the likelihood class. [James Hensman]

* Weaved some slow functions in the stationary class. We now fall back (and latch) to numpy if weave fails. [James Hensman]

* Merge branch 'devel' of github.com:sheffieldml/GPy into devel. [James Hensman]

* Stopped rounding to int in priors printing. [Alan Saul]

* Removed more more more imports. [James Hensman]

* Removed more more imports. [James Hensman]

* [whitespaces] [Max Zwiessele]

* [cacher] now taking over the attributes from cached functions, such as docstring. [Max Zwiessele]

* [datasets] deng et all, labels revisited. [Max Zwiessele]

* [kern psi2] added flag for returning psi2 in N, not used yet, see #139. [Max Zwiessele]

* [missing_data in sparse gp] can be extended towards missing_data handling in gp itself. Setting up gpy issue. [Max Zwiessele]

* [datasets] updated deng loading pandas bugs... [Max Zwiessele]

* [pca] missing data is now handled as mean. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* [missing data] general implementation for subsetting data. [Max Zwiessele]

* Minor edits to regression examples. [James Hensman]

* Minor bugfix on windows (thanks NC) [James Hensman]

* Whitespace. [James Hensman]

* [datasets] deng et al. single cell experiment prints for ipython notebook. [Max Zwiessele]

* [vardtc missing data] updated to new psi2 stuff. [Max Zwiessele]

* [param_to_array] deprecated and removed param_to_array from code, use param.values instead. [Max Zwiessele]

* Another bug fixed. [Ricardo]

* Bug fixed. [Ricardo]

* HalfT prior is working. [Ricardo]

* Half_t prior (Martin's contribution) [Ricardo]

* Edited coregionalize implementation. [James Hensman]

* Optionally unweaved the coregionalize kernel. [James Hensman]

  coregionalize shoudl now work without weave. Added kernel tests also.

* Bug fix for extending prod kernel. [Zhenwen Dai]

* Add set_X and set_Y interface to gp model. [Zhenwen Dai]

* Merge branch 'devel' for extending prod kernel. [Zhenwen Dai]

* Added notimplemented error to svigp. [James Hensman]

* Tidying soem comments. [James Hensman]

* Symmetrify now falls back gracefully to numpy if weave fails. [James Hensman]

* Now ising numpy for std_norm_cdf. [James Hensman]

* Removed fast_array_equal (no longer used) [James Hensman]

* Removed more imports. [James Hensman]

* Removed unnecessary imports in likelihoods when hunting down weave depencencies. [James Hensman]

* Extend prod kernel for handling more than 2 kernels. [Zhenwen Dai]

* Change BGPLVM to use sparsegp_mpi. [Zhenwen Dai]

* [#133] fix: chainging constraint in __init__ [Max Zwiessele]

* [updates] merged update structure. [Max Zwiessele]

* Some improvements to plotting 2d kernels. [James Hensman]

* Improved docsting for optimize. [James Hensman]

* Docstring for ExpQuad (thanks Mike O. ) [James Hensman]

* For loop speedup in grdients X. [James Hensman]

* NonContiguos tests fixed for Kdiag_dX. [James Hensman]

* Change setup.py accordingly. [Zhenwen Dai]

* Remove the dependency on matplotlib. [Zhenwen Dai]

* Finish the debug of sparsegp_mpi. [Zhenwen Dai]

* Bug fix: param object randomize. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Bug fixed in normalization. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Name can be modified. [Ricardo]

* Changes installtion instructions. [Zhenwen Dai]

* Add the Windows installation instructions for GPy. [Zhenwen Dai]

* Remove nose from install_requires. [Zhenwen Dai]

* A bug fix for VarDTC_minibatch. [Zhenwen Dai]

* Adapt the numerical stability strategy from VarDTC to VarDTC_minibatch. [Zhenwen Dai]

* Update sparse_gp_mpi for new interface. [Zhenwen Dai]

* [updates] updated update structure immensely. [Max Zwiessele]

* [link|unlink_parameter] renaming add_parameter to link_parameter. [Max Zwiessele]

* [documentation] updated big parts of the doc. [mzwiessele]

* [documentation] bits and pieces for interacting_with_models. [mzwiessele]

* [printing] warning when reconstraining now prints hierarchy names. [mzwiessele]

* [accassibility] GPy.constraints now accassible. [mzwiessele]

* [printing] added model details for printing. [mzwiessele]

* [parameter_core] empty space. [mzwiessele]

* [param] indexing routine simplified. [mzwiessele]

* Minor updates to the docs. [mzwiessele]

* [ard plotting] adjustments to the filtering. [mzwiessele]

* [updates] made updates a function, update_model(True|False|None) [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* Fix psicomp problem. [Zhenwen Dai]

* [updates] made updates a function. [mzwiessele]

* [testing] updated tests wrt normalization. [mzwiessele]

* [indexing] maybe? cannot tell, tests are broken. [mzwiessele]

* [indexing] maybe? cannot tell, tests are broken. [mzwiessele]

* [param] indexing fix, this can be memory intensive if millions of parameters. [Max Zwießele]

* Added missing comma. [James McMurray]

  Added missing comma in the hstack call in _get_params.

* Remove the print message in model.checkgrad. [Zhenwen Dai]

* Allow the default constraint of a Param object to be 'fixed' [Zhenwen Dai]

* More for debug. [Zhenwen Dai]

* Improve debug helper. [Zhenwen Dai]

* Some progress for parameter tie. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* [dim red plots] plotting big models. [mzwiessele]

  Conflicts:
    GPy/plotting/matplot_dep/dim_reduction_plots.py

* Add debug util module and try to debug sparsegp_mpi. [Zhenwen Dai]

* Adapt sparsegp_mpi for normalizer arguement. [Zhenwen Dai]

* Fix the bug: the randomize function cannot properly handle variables with prior. [Zhenwen Dai]

* [kernel ard plot] label adjustment. [mzwiessele]

* [normalizer] only mean, because variance could be not Gaussian... [mzwiessele]

* [normalize] [mzwiessele]

* [normalizer] first commit for normalizer in GPy. [mzwiessele]

  Conflicts:
    GPy/core/sparse_gp.py
    GPy/models/bayesian_gplvm.py

* Further bug fix for sparsegp_mpi. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Noise parameters built depending on Y_metadata. [Ricardo]

* Generalize the interface of mpi. [Zhenwen Dai]

* Recover the ss_gplvm.py. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* [ard] enhanced ard handling and plotting. [mzwiessele]

  Conflicts:
    GPy/kern/_src/linear.py
    GPy/models/ss_gplvm.py

* [linear] einsums. [mzwiessele]

* [pickling] wb as write parameter. [mzwiessele]

* Improve numerical stability of vardtc_parallel. [Zhenwen Dai]

* A bug fix for psi statistics related model pickle. [Zhenwen Dai]

* Update additive kernel for SSGPLVM. [Zhenwen Dai]

* Fix the pickle problem for models with psi statistics. [Zhenwen Dai]

* Some further performance improvement for linear kernel psi statistics. [Zhenwen Dai]

* Linear kernel psi statistics performance optimization. [Zhenwen Dai]

* Minor changes on SSGPLVM. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* [parameterization] Parameter adding more robust and better error handling. [mzwiessele]

* Added kronecker and variational gaussian approximation gp's, vargpapprox needs generalising to any factorizing likelihood. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Gradients of predictions for Trevor. [James Hensman]

* More bugfix. [James Hensman]

* Bugfix merge! [James Hensman]

* Strange bug in np.einsum fixed when using the _out_ argument (thanks T. Cohn) [James Hensman]

* Minor changes in ICM. [Ricardo]

* Fix the bug of caching w.r.t. ignore arguments. [Zhenwen Dai]

* Change for ssgplvm example. [Zhenwen Dai]

* Merge the current devel into psi2. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* [setup] no import of os, nanana alan :) [mzwiessele]

* [parameter core] offset for can be done without parameter slices. [mzwiessele]

* [gradcheck] some performance enhancement. [mzwiessele]

* [bgplvm] gradient settings. [mzwiessele]

* Merge branch 'hmc' into devel. [Zhenwen Dai]

  A HMC sampler for GP parameters

* Debug HMC shortcut. [Zhenwen Dai]

* Hmc shortcut. [Zhenwen Dai]

* Correct the initial distribution of p. [Zhenwen Dai]

* Remove prints. [Zhenwen Dai]

* Debug hmc code. [Zhenwen Dai]

* Add hmc.py. [Zhenwen Dai]

* Initial implementation of hmc. [Niu]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* [minor] minor changes. [mzwiessele]

* Coregionalized 2D plotting fixed. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Bug fixed. [Ricardo]

* Fizxed a numerical bug in stationary.py. [James Hensman]

* Returned setup.py read to old version. [Alan Saul]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Changes to datasets.py. [Neil Lawrence]

* Added forced extraction of eggs (as we have a fair few non-py files and use the directory structure) added some files to MANIFEST and setup.py's package_data so its included upon distributing. [Alan Saul]

* [inference] less constant jitter, and jitter adjustements. [mzwiessele]

  Conflicts:
    GPy/util/linalg.py

* Fixed a bug in optimize restarts: it now used optimizer_array. [James Hensman]

* [vardtc missing] logging. [maxz]

  Conflicts:
    GPy/inference/latent_function_inference/var_dtc.py

* [init] normalization bugfix. [mzwiessele]

* [whitespaces] [mzwiessele]

* [var dtc missing] [mzwiessele]

* [var dtc missing] performance increase with right iterations. [mzwiessele]

* Logging. [mzwiessele]

* [var dtc missing] no caching of indexed y anymore, no subarray indexing. [mzwiessele]

* [logging] more on logging. [mzwiessele]

* Var dtc missing] memory efficiency greatly improved. [mzwiessele]

* [parameterized] init greatly improved. [mzwiessele]

* Yak shaving and whitespaces. [mzwiessele]

* [brendan] netpbmfile. [mzwiessele]

* [optimize] bugfix. [mzwiessele]

* [parameterized] bugfix: downstream parameters did not get constraint update on add_parameter. [mzwiessele]

* [parameterized] adding parameters in hierarchy, did not update higher siblings. [mzwiessele]

* [cacher] removed logger. [mzwiessele]

* [var dtc missing] yak shaving. [mzwiessele]

* [logging] [mzwiessele]

* [gp] memory > only one copy. [mzwiessele]

* [subbarray] logging. [mzwiessele]

* [vardtc missing] performance fixes. [mzwiessele]

* [optimizer array] bugfix, when updating the model the optimizer array would not update. [mzwiessele]

* [optmimize] bugfix. [mzwiessele]

* [whitespaces] & [opt] minor fix of optimizer, when Optimizer is provided (set model to self) [mzwiessele]

* [model] optimizer can now be an optimizer instance, instead of a string. [mzwiessele]

* [parallel vardtc] minor adjustments to work with current implementation of psi stats. [mzwiessele]

* [randomize] adjusted parameters to go into random generator right. [mzwiessele]

* [scg] minor adjustements based on original publication. [mzwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* [linalg] fixed scipy 0.14 bugfix. sciy.linalg.lapack.dpotri was fixed to work right with lower=1, thus, the hack is gone now from GPy.util.linalg.dpotri, when using scipy 0.14 and higher. [Max Zwiessele]

* Re-doing the lee dataset. [James Hensman]

* Revert "Minor edits to reading Lee data in datasets.py" [James Hensman]

  This reverts commit 730e229238062fa22b726e8c30c891d0819b3c6e.

* Fixed unnecessary warnings when using periodic kernels. [durrande]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Datasets.py updates should have been committed before. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Reverting Maxs linalg changes. [James Hensman]

* Linalg had lowers missing for windows libraries to work correctly. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Merge branc( 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Editied whitespace. [James Hensman]

* [splitkern] support idx_p==0. [Zhenwen Dai]

* Minor changes. [Zhenwen Dai]

* DiffGenomeKern bug fix. [Zhenwen Dai]

* [splitkern] some more changes. [zhenwen]

* [splitkern] some additional implmentation. [Zhenwen Dai]

* [splitkern] bug fix. [Zhenwen Dai]

* [splitkern] buf fix. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* Merging. [James Hensman]

* Reverting the fixing behaviour. [James Hensman]

  two reasons: 1) the new behaviour is confusing for new users. Either
  something is fixed, or it's not. 2) the fixing didn't work! things that
  should have been fixed were passed to the optimizer for optimization.

  If we really want to save keystrokes, consider this:

  m.foo.fix()
  m.foo.unfix()
  m.foo.constrain_positive()

  is the same as

  m.foo.fix()
  m.foo.constrain_positive()

  but the latter throws a warning.

* Merge the bug of fixing function. [Zhenwen Dai]

* Developing split kernel. [Zhenwen Dai]

* Developing split kernel. [Zhenwen Dai]

* Minor edits to reading Lee data in datasets.py. [Neil Lawrence]

* Modified Spellman et al data load. [Neil Lawrence]

* Changes to configuration file set up: now uses defaults.cfg, installation.cfg and searches locally for .gpy_user.cfg in the users home directory. [Neil Lawrence]

* Added CIFAR-10 data to data sets. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Fixed an assertion, it was checking the dimensionality of the input data matrix, rather than that of the labels. [teodecampos]

* [bgplvm] do test latents updated for devel branch. [mzwiessele]

* [copy] is now fully functional, only hierarchy observers will be copied and pickled. [mzwiessele]

* [dim_reduce examples] updated robot_wireless. [mzwiessele]

* [dim_reduce examples] updated swiss roll. [mzwiessele]

* [dim_reduce examples] updated plotting of brendan and bgplvm_oil. [mzwiessele]

* [tests] added some unfix fix print and gradcheck tests, it basically just behaves as a user would do with a model. [mzwiessele]

* [caching] catching key error, when individuum is already gone. [mzwiessele]

* [optimizer&fixing] optimizer has only one optimizer copy and fixing remembers old constraint. [mzwiessele]

* [optimizer] one copy for the optimizer in optimizer_array, use this instead of _set|get_params_transformed. [mzwiessele]

* [reverts] some reverts, as one param etc does not work. [Max Zwiessele]

* [caching] first things first. [Max Zwiessele]

* [model] exactly two parameter copy in memory. [Max Zwiessele]

* [gp] output data is a copy now. [Max Zwiessele]

* [bgplvm&mrd] missing data greatly improved, still not there yet. [Max Zwiessele]

* New data sets. [Neil Lawrence]

* Added a numpy version of univariate gaussian, untested and is significantly slower but may be useful soon. [Alan Saul]

* Fix kern/__init__.py. [Zhenwen Dai]

* Some tidy up. [Zhenwen Dai]

* Implement the linear kernel with psi2 format. [Zhenwen Dai]

* Generalize the spike-and-slab prior with pi (N,Q) [Zhenwen Dai]

* Ssgplvm simulation example. [Zhenwen Dai]

* Fix the problem starting multiple process with limited number of GPUs. [Zhenwen Dai]

* Add plot_latent to sparse_gplvm. [Zhenwen Dai]

* Fix sparse_gplvm. [Zhenwen Dai]

* Simplify the interface of using mpi. [Zhenwen Dai]

* Fix add kernel and VarDTC_minibatch speed tuning. [Zhenwen Dai]

* Fix ss_mrd and fix white and bias kernel. [Zhenwen Dai]

* Ss_mrd with parameter tied. [Zhenwen Dai]

* Add ss_mrd model. [Zhenwen Dai]

* Fix pickle for ssgplvm and bgplvm with mpi. [Zhenwen Dai]

* Remove dependence of scikits.cuda from rbf kernel. [Zhenwen Dai]

* Remove dependence of cublas from rbf kernel. [Zhenwen Dai]

* Fix the gpu initialization for multiple cards. [Zhenwen Dai]

* Adapt gpu initialization multiple gpu cards. [Zhenwen Dai]

* Fix the SSGPLVM with MPI. [Zhenwen Dai]

* Fix the speed problem of the tie framework. [Zhenwen Dai]

* A little optimization of gpu code. [Zhenwen Dai]

* RBF for SSGPLVM gpu implemented. [Zhenwen Dai]

* Fix the problem of multiple ties on the same param array object. [Zhenwen Dai]

* Tie framework works roughly. [Zhenwen Dai]

* Merge ties branch into psi2. [Zhenwen Dai]

* Merge ties branch into psi2. [Zhenwen Dai]

* Var_dtc_parallel make YY.T speed up. [Zhenwen Dai]

* Fix pickle for RBF GPU kernel. [Zhenwen Dai]

* Fix the remaining problem of cache.py. [Zhenwen Dai]

* Rbf gpu usable. [Zhenwen Dai]

* Fix import issue on no-gpu machine. [Zhenwen Dai]

* Rbf kernel gpu implementation ready. [Zhenwen Dai]

* Rbf gpu psicomp pass gradcheck. [Zhenwen Dai]

* Rbf kernel gpu implementation in progress. [Zhenwen Dai]

* Add truncated linear kernel. [Zhenwen Dai]

* Rename sslinear_psi_comp.py. [Zhenwen Dai]

* Fix Linear kernel with SSGPLVM. [Zhenwen Dai]

* Support non-symmetric dL_dKmm for stationary kernel. [Zhenwen Dai]

* Basic vardtc working. [James Hensman]

* Reverting the fixing behaviour. [James Hensman]

  two reasons: 1) the new behaviour is confusing for new users. Either
  something is fixed, or it's not. 2) the fixing didn't work! things that
  should have been fixed were passed to the optimizer for optimization.

  If we really want to save keystrokes, consider this:

  m.foo.fix()
  m.foo.unfix()
  m.foo.constrain_positive()

  is the same as

  m.foo.fix()
  m.foo.constrain_positive()

  but the latter throws a warning.

* Restructure rbf kernel. [Zhenwen Dai]

* BayersianGPLVM mpi support. [Zhenwen Dai]

* Merge devel branch in. [Zhenwen Dai]

* [fixing] fixing now saves the old constraint. [Max Zwiessele]

* [index operations] added lookup for properties for a given index as dict <properties, subindex> for given index. [Max Zwiessele]

* [mrd] more control for init, some missing data adjustements, init greatly improved. [Max Zwiessele]

* [core updates] first try to switch updates off and on. Use m.updates = False to switch updates off, and vice-versa. [Max Zwiessele]

* [psi-stats] add kernel was missing a psi zero call. [Max Zwiessele]

* [stationary] input_sensitiviy is now 1/(l**2) [Max Zwiessele]

* [datasets] delete packed data in hapmap dataset. [Max Zwiessele]

* [vardtc missing data] can handle non broadcastable selections. [Max Zwiessele]

* [pca] colors now as iterator. [Max Zwiessele]

* [posteriot] adjusted for more then one covariance per output. [Max Zwiessele]

* [lmv_dimselect] we need to keep a pointer to the lvm_dimselect object, as the updates are weak references: dim_select = ... [Max Zwiessele]

* [mrd] missing data implemented, and plotting better. [Max Zwiessele]

* [mrd] plotting, init, inference etc. [Max Zwiessele]

* [ploting] dim reduction. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* [examples] stick demo. [mzwiessele]

* [psi2] implement RBF cpu. [Zhenwen Dai]

* [mpi] enable checkgrad. [Zhenwen Dai]

* [mpi] fix the bug of mpi. [Zhenwen Dai]

* Fix pickle. [Zhenwen Dai]

* Logistic transformation numerical robustness. [Zhenwen Dai]

* Bug fix for mpi SSGPLVM. [Zhenwen Dai]

* Merge chagnes from devel. [Zhenwen Dai]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Zhenwen Dai]

* EP is back. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* [latentfunctioninference] superclass LatentFunctionInference added, which contains a call just before and just after optimization. [Max Zwiessele]

* Flags added. [Ricardo]

* Minor changes. [Ricardo]

* Fix the bug in mocap demos. [Zhenwen Dai]

* A little merge. [Zhenwen Dai]

* [parameterized] restructered a lot and finalized some stuff. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* [caching] renaming of helper methods to make intention clear. [mzwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [mzwiessele]

* [indexing&memory] in memory view more efficient, catching some indexing errors. [mzwiessele]

* Add Drosophila data. [Neil Lawrence]

* Merge mu's changes into devel. [mu]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* [param] hierarchy traversal easier now. [mzwiessele]

* Fixing fruitfly_tomancak data load. [Neil Lawrence]

* Made openmp switch in only dependent on potion in rbf.py and linear.py. [Neil Lawrence]

* Sod1 Download. [Neil Lawrence]

* Add ordinal and attempt to fix downloads. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* [paramcore] fix for traversal. [mzwiessele]

* [pydot] build pydot with new observer list. [mzwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* [bgplvm] init lengthscale as 0./var. [mzwiessele]

* [param_array] doc. [mzwiessele]

* Add ordinal and attempt to fix downloads. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Made openmp switch in only dependent on potion in rbf.py and linear.py. [Neil Lawrence]

* Minor edit in scg, raise notimplemented dL_dX in hierarchical. [James Hensman]

* [stick] bgplvm example now working. [mzwiessele]

* [init] now returns normalized values. [mzwiessele]

* [variational] posterior object copies adjusted. [mzwiessele]

* [visualize] minor. [mzwiessele]

* [examples] stick man example corrected. [mzwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Working with OU kernel. [marahman]

* Changes to datasets.py. [Neil Lawrence]

* Changes to datasets.py. [Neil Lawrence]

* T. [mu]

* T. [mu]

* St. [mu]

* Uyc. [mu]

* [params] indexing with boolean arrays switched off, rases proper error now. [Max Zwiessele]

* [param] indexing now returns exactly like numpy arrays. [Max Zwiessele]

* [visualize] vector show again. [Max Zwiessele]

* [testing] minor. [Max Zwiessele]

* [visualize] some adjustments to vector_show. [Max Zwiessele]

* [kern] pow for kernels now in place again. [Max Zwiessele]

* [copy] handled hierarchy error for copying. [Max Zwiessele]

* [data] edit json file directly, removed datasets.py and data_resources_create. [Max Zwiessele]

* [data] data_resources edited, such that json file is edited directly. [Max Zwiessele]

* Whitespaces. [Max Zwiessele]

* [param concatenation] allows assignmend more robustly. [Max Zwiessele]

* [caching] when reset. [Max Zwiessele]

* [datasets] added singlecell dataset. [Max Zwiessele]

* [active_dims] all kernels now have int arrays as active_dims. [Max Zwiessele]

* [combination kernel] some fixing with error messages. [Max Zwiessele]

* [caching] id fix. [Max Zwiessele]

* [caching] done right. [Max Zwiessele]

* Merge branch 'devel' of github.com:/SheffieldML/GPy into devel. [James Hensman]

* Added citation to readme. [James Hensman]

* Sparse GPs can now accept kerns for predicting. [James Hensman]

* Added polynomial kernel. [James Hensman]

* [SSGPLVM] linear kernel cpu ready. [Zhenwen Dai]

* [ssgplvm] linear kernel. [Zhenwen Dai]

* Merge branch 'psi2' of github.com:SheffieldML/GPy into psi2. [Zhenwen Dai]

* Proper whitespace. [James Hensman]

* [mpi] add mpi into ssgplvm. [Zhenwen Dai]

* [GPU] varDTC_gpu ready. [Zhenwen Dai]

* [GPU] [Zhenwen Dai]

* Switch psi2 statistics design. [Zhenwen Dai]

* Some hacking on image_show in viaualize. [James Hensman]

* Removed another import of non-added file. [James Hensman]

* Removed import of non-added file (Mu) [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* St. [mu]

* Merge branch 'params' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Merge branch 'params' of https://github.com/SheffieldML/GPy into params. [Neil Lawrence]

* Pre-devel-move check in. [Neil Lawrence]

* Added the ability for GPs to predict with a different kernel. [James Hensman]

* Some work on the hierarchical kern. [James Hensman]

* Grabbed readme from old devel branch. [James Hensman]

* Bugfix: slicing checks needed to be suspended for combination kernels, checks are done in inner kernels now. [mzwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [mzwiessele]

* Merged. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/kern/_src/kern.py

* Missing file, import line commented. [Ricardo]

* Double quote deleted. [Ricardo]

* Some horrible hacking on hierarchical. [James Hensman]

* Bugfix: slicing. [mzwiessele]

* Bugfix: kern input_sens errir. [mzwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [mzwiessele]

* Merge kernel source. [Neil Lawrence]

* Removed imports of files the are not added to the repo. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Removing Neils mid-atlantic commit because he failed to add the relevant files to the repo. [James Hensman]

* Added some missing files. [Neil Lawrence]

* Slicing: slicing now thorughs the right error. [mzwiessele]

* HACK: plot_ARD is additive, should think of making it selectable through parameter handle. [mzwiessele]

* Bugfix: lineplot in visualize. [mzwiessele]

* Added documentation for parameterized objects, needs more detail and fleshing out with proper english. [mzwiessele]

* Part working on symbolics. Replacing data_resources.json with the correct full file (-hapmap). Don't know why we've gone for separate create file ... [Neil Lawrence]

* Need to fix missing data in likelihoods. [Neil Lawrence]

* Merge of kern/__init__.py. [Neil Lawrence]

* New test heteroscedastic noise model. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Bugfix: mixed up global and local index in unfixing. [mzwiessele]

* New file, special request. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/models/gp_classification.py
    GPy/models/sparse_gp_classification.py

* New file. [Ricardo]

* Minor change. [Ricardo]

* EPDTC added. [Ricardo]

* Minor change. [Ricardo]

* Changes according to new definitions. [Ricardo]

* Changes according to new definitions. [Ricardo]

* Pseudo_EM is not available for the moment. [Ricardo]

* Just had to do a check in from midlantic (showing off). [Neil Lawrence]

* Merge branch 'params' of https://github.com/SheffieldML/GPy into params. [Neil Lawrence]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* [datasets] merged hapmap dataset into params. [Max Zwiessele]

* Bugfix: slicing was still in stationary somehow. [Max Zwiessele]

* Removed random.seed from the base of kernel_tests.py (the tests still pass) [James Hensman]

* [tests] added test for fixing through regular expressions. [Max Zwiessele]

* Bugfix: fixing through regular expression matching. [Max Zwiessele]

* Bugfix: couldn't constrain single parameters, because of indexing of np. [Max Zwiessele]

* [Kern] added assertion for inputs X being matrix (ndim == 2) [Max Zwiessele]

* [Kern] added assertion for inputs X being matrix (ndim == 2) [Max Zwiessele]

* Bug fix: slicing was not checking dimensions. [Max Zwiessele]

* Bug fix: slicing can now be turned of by passing None as active_dims. [Max Zwiessele]

* Slicing .... maybe needs to be deleted. [Max Zwiessele]

* Enabled some more getting/setting parameters, such as regular expressions and params. [Max Zwiessele]

* Pickle test error fixed. [Max Zwiessele]

* First draft of base symbolic object, compiling with symbolic mapping. [Neil Lawrence]

* Copy and paste observable_array from repository to try and resolve bizzare merge request. [Neil Lawrence]

* Fix stick man example. [Zhenwen Dai]

* [SSGPLVM] add plotting class. [Zhenwen Dai]

* [GPU] add automatic batchsize estimation. [Zhenwen Dai]

* [GPU] GPU version of varDTC is ready. [Zhenwen Dai]

* Bug fix: caching.py w.r.t. ignore_args. [Zhenwen Dai]

* Made observers accessible and observers now only weak reference the observables. [mzwiessele]

* Making observables accessable. [mzwiessele]

* Not importable. [mzwiessele]

* [GPU] psi varDTC ready. [Zhenwen Dai]

* [GPU] caching not working. [Zhenwen Dai]

* [GPU] varDTC_gpu bug fix. [Zhenwen Dai]

* [GPU] varDTC_gpu almost done. [Zhenwen Dai]

* [GPU] varDTC_gpu minibatch. [Zhenwen Dai]

* [GPU] bug fix. [Zhenwen Dai]

* [GPU] bug fix. [Zhenwen Dai]

* [GPU] vardtc_likelihood 1. [Zhenwen Dai]

* [GPU] vardtc_likelihood. [Zhenwen Dai]

* [GPU] bug fix. [Zhenwen Dai]

* [GPU] gradient check ready. [Zhenwen Dai]

* [GPU] update gradients rest. [Zhenwen Dai]

* [GPU] bug fix. [Zhenwen Dai]

* [GPU] bug fix. [Zhenwen Dai]

* [GPU] bug fix. [Zhenwen Dai]

* [gpu] upate gradient. [Zhenwen Dai]

* [GPU] psi2 ssgplvm. [Zhenwen Dai]

* [GPU] psi1. [Zhenwen Dai]

* [GPU] psicommputation. [Zhenwen Dai]

* [GPU] psi1 after debug. [Zhenwen Dai]

* [GPU] add linalg_gpu ssrbf_gpucomp. [Zhenwen Dai]

* [GPU] GPU kernel. [Zhenwen Dai]

* [GPU] finish infere_likelihood. [Zhenwen Dai]

* [GPU] inference function part1. [Zhenwen Dai]

* [GPU] in progress. [Zhenwen Dai]

* [GPU] var_dtc_gpu in progress. [Zhenwen Dai]

* More changes to symbolic. [Neil Lawrence]

* Ongoing changes to symbolic. [Neil Lawrence]

* Partial changes to symbolic, including adding mapping covariance and beginning to unify code generation. [Neil Lawrence]

* Adapt likelihoods init to check for sympy. [Neil Lawrence]

* Renamed array_core to observable array. [mzwiessele]

* Renamed array_core to observable_array. [mzwiessele]

* Adding missing functions file. [Neil Lawrence]

* Added negative binomial likelihood based on symbolic: merge symbolic. [Neil Lawrence]

* Check for sympy. [Alan Saul]

* Added negative binomial likelihood based on symbolic. [Neil Lawrence]

* Added first draft of symbolic likelihood (working for a student-t example). [Neil Lawrence]

* Pcikle tests added. [Max Zwiessele]

* Minor changes bits and pieces. [Max Zwiessele]

* Parameterized tests deeper still. [Max Zwiessele]

* Current_slice is not a property. [Max Zwiessele]

* Index operations view delitem added. [Max Zwiessele]

* Gradient can be zero and two parameter cancellation is caught. [Max Zwiessele]

* Gradient can be zero and two parameter cancellation is caught. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Removed some dubuggnin. [James Hensman]

* Delete dangling fixed attribute in constraints. [Max Zwiessele]

* Copy had slight bug in id(_parent_index_) > ids for ints are shared globally. [Max Zwiessele]

* Student t noise now called t_scale2. [Max Zwiessele]

* Added kernel tests again. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Adding of symbolic likelihoods (not yet fully funcitonal). [Neil Lawrence]

* Update of symbolic likelihoods. [Neil Lawrence]

* Pickling and caching. [Max Zwiessele]

* Exact inference for N>D of Y. [Max Zwiessele]

* Kernel slicer now asserts X dimension on first seeing X. [Max Zwiessele]

* Assertion checks for all kernels. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Moved input_sensitivity to the gp class. [James Hensman]

* Slight adjustment to self.active_dims being a 0:n:1 slice. [Max Zwiessele]

* Independent output kernel now with single kernel/multiple kernels. [Max Zwiessele]

* Right active dims when adding kernels. [Max Zwiessele]

* New slicing done and first attempts at copy and pickling full models. [Max Zwiessele]

* Merged and updated slicing operations. [Max Zwiessele]

* Started copy implementation, have to get rid of _getstate_ and _setstate_ [mzwiessele]

* Slice operations now bound functions, not added after the fact. [mzwiessele]

* Array list now working with index. [mzwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Full Linear kernel added, inc testing. [James Hensman]

* Kern merge commencing. [Max Zwiessele]

* Mrd and bgplvm updates to conform new vardtc. [Max Zwiessele]

* Objective function seperate from calls for optimizer. [Max Zwiessele]

* Vardtc updates. [Max Zwiessele]

* Pca module for initialization. [Max Zwiessele]

* Bugfix for 3d and more dimensional _indices. [Max Zwiessele]

* Objective_function now standalone and only internal robust optimization loop. [Max Zwiessele]

* Deleted unused imports. [Max Zwiessele]

* GPclassification has to default inference method to EP. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Whoops! [Alan Saul]

* Fixed bug in product kernel gradients diag wrt to X. [Alan Saul]

* Core updates. [Max Zwiessele]

* Testing. [Max Zwiessele]

* Finally added pca package again. [Max Zwiessele]

* Variational returns now the right raveled indices. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Merge changes. [Zhenwen Dai]

* BayesOpt added. [javiergonzalezh]

* BayesOpt added. [javiergonzalezh]

* [GPU] partial implmented minibatch inference. [Zhenwen Dai]

* Caching functions now take two arguments: self and which, which is the argument which started the update. [Max Zwiessele]

* Caching now per instance, not at def time. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Added a couple of tests for model predictions. [Alan Saul]

* Fixing the logexp (with MZ) and some stability issue in the stationary class. [James Hensman]

* Correct predictions in Gaussian. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* ODE_UY gradient checks now kernel unit. [Max Zwiessele]

* Add kernel adding another add kernel. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [mzwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Missing docstrings. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/likelihoods/mixed_noise.py

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* 1D inducing inputs modified for coregionalized models. [Ricardo]

* Bug fixed. [Ricardo]

* Function predictive_quantiles added. [Ricardo]

* Lines not used deleted. [Ricardo]

* Default None for Y_metadata in predictive_quantiles. [Ricardo]

* Coregionalization examples fixed. [Ricardo]

* Param_array fixes. [mzwiessele]

* Param setting. [mzwiessele]

* Plotting, allot of plotting. [mzwiessele]

* Some work on the linear mapping. [James Hensman]

* Bugfix in setup.py. [James Hensman]

* Bugfix. [James Hensman]

* Adding a test for Mus code. [James Hensman]

* Mus code seems to work on params now. [James Hensman]

* Init.py for mus kernel. [James Hensman]

* Adding Mus kernel ODE_UY. [James Hensman]

* Changes to setup.py. [James Hensman]

* Metadata passing in fitc. [James Hensman]

* A simple test for fitc. [James Hensman]

* All the tests pass (though some are marked known-to-fail. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Known fail for EP tests in unit tests. [Max Zwiessele]

* Gaussian with identity link in tests. [James Hensman]

* All tests either notimplemented or known to fail. [Max Zwiessele]

* Adding kernels flattening and parameters already in hierarchy. [Max Zwiessele]

* Old_tests out of the way. [Max Zwiessele]

* Fixes in likelihoods. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Very weird merge conflict, including in files that I did not change. [James Hensman]

* Manual merging. [James Hensman]

* Tidying in likelihoods. [James Hensman]

* Work on likeluhoods and likelihoods tests. [James Hensman]

* All tests are now to check. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Merged. [James Hensman]

* Changed the way the Gaussian likelihood interfaces, to enable mixed_noise things. [James Hensman]

* Fixes now hierarchical, maybe need to be restructured as lookup from constraints. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

  Conflicts:
    GPy/likelihoods/gaussian.py

* Bug fix w.r.t. var_dtc.py. [Zhenwen Dai]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/inference/latent_function_inference/var_dtc.py

* New model SparseGPCoregionalizedRegression. [Ricardo]

* Minor changes. [Ricardo]

* Changes to allow mixed noise likelihoods. [Ricardo]

* New function added. [Ricardo]

* Changes to allow compatibility with mixed noise likelihoods. [Ricardo]

* Bug fixed. [Ricardo]

* Changes to allow heteroscedastic inference. [Ricardo]

* Minor changes. [Ricardo]

* Parameter missin in dL_dthetaL added. [Ricardo]

* ObservableArray -> ObsAr, because of pickling and ndarray printing. [Max Zwiessele]

* Merge conflict. [Max Zwiessele]

* DL_dthetaL in missing data vardtc. [James Hensman]

* Fix the bug regarding to the change of the name dL_dthetaL. [Zhenwen Dai]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Fixed Y_metadata bug. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Added jitter to fitc. [James Hensman]

* Added a hack fix as suggested by max, zeroing any negative values (should really be numerically negative values on diagonal) [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Stablised other quadrature (should speed things up also), added sampling ability to poisson. [Alan Saul]

* Stablised exp link_function and quadrature variances. [Alan Saul]

* Changes for compatiblity with changes in likelihood. [Ricardo]

* Re-definition of the week. [Ricardo]

* Re-definition of the week. [Ricardo]

* Changes in likelihood definition. [Ricardo]

* Y_metadata definition changed. [Ricardo]

* Changes in kernel parameters definition. [Ricardo]

* Pickling working for array-likes, but observers not yet connected back. [Max Zwiessele]

* Slicing finished with independent outputs. [Max Zwiessele]

* Slicing now returns the right shape, when computing derivative wrt X or Z. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Plotting now seems to work for Bernouilli. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Fixes to EP. [James Hensman]

* Mrd gradients. [Max Zwiessele]

* Independent output kernel gradients x. [Max Zwiessele]

* Prior domain check. [Max Zwiessele]

* Prior tests renewed. [Max Zwiessele]

* Fixes fixed and test updates. [Max Zwiessele]

* Fixes fixed and test updates. [Max Zwiessele]

* Changes due to tests in parameterization. [Max Zwiessele]

* Checkgrad is zero test. [Max Zwiessele]

* Active dim indices and slices. [Max Zwiessele]

* Testing. [Max Zwiessele]

* Static active dims. [Max Zwiessele]

* Kernel tests. [Max Zwiessele]

* Merge for new kernel slice handling. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Bugfix for grad_dict. [James Hensman]

* Alans change to checkgrad. [James Hensman]

* Independent output gradients. [James Hensman]

* Kernel slices allowed. [Max Zwiessele]

* Manual merge of tests. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Plotting fix. [James Hensman]

* Added test for independent kern. [Alan Saul]

* Kernel tests periodic. [Max Zwiessele]

* Active_dims as extra parameter for kernels, it tells which input dimensions to work on. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Merge branch 'params' of https://github.com/SheffieldML/GPy into params. [Neil Lawrence]

* Testing a bit cleaned periodic is turned off, bc it need different tests, discontinuous still needed. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Coregionalization example. [Ricardo]

* Fix needed. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Temporal fix. [Ricardo]

* Fixed mlp kern. [Max Zwiessele]

* Whitespaces. [Max Zwiessele]

* Caching now resets cache on error. [Max Zwiessele]

* We need to update all the tests: here discontinuous kernel testsee messing, mrd and bgplvm model tests not needed anymore. [Max Zwiessele]

* Periodic kernel gradients and parameterized updates. [Max Zwiessele]

* Constrain notifies observers. [Max Zwiessele]

* Object without args. [Max Zwiessele]

* Product kernel and combination kernel updates. [Max Zwiessele]

* Google trends and football data sets. [Neil Lawrence]

* Merge branch 'params' of https://github.com/SheffieldML/GPy into params. [Neil Lawrence]

* Merge branch 'params' of https://github.com/SheffieldML/GPy into params. [Neil Lawrence]

* Part written addition to datasets for loading in google trends. [Neil Lawrence]

* Fixed the posterior prediction for laplace. [James Hensman]

  The mis-match between the woodbury vector and KIf is still a bit of a
  mystery

* Various fixes in likelihoods, esp studentT and plotting. [James Hensman]

* Lots of fixes, including prediction being mean and variance only. [James Hensman]

* More chancges to laplace. [James Hensman]

* Chancges to where gradients are computed in laplace. [James Hensman]

* Import not relative in tests. [James Hensman]

* Ind ops. [James Hensman]

* Independent outputs kernel. [Max Zwiessele]

* Gp merge, grad dict is property of self + Y_metadata being passed through. [Max Zwiessele]

* Bugfixing. [James Hensman]

* Missing bracket. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Q Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Fixing fitc. [James Hensman]

* Fixing coreg kernel. [Ricardo]

* Fixing coreg kernel. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/core/gp.py
    GPy/plotting/matplot_dep/models_plots.py

* GPCoregionalizedRegresssion added. [Ricardo]

* Mixed_noise added. [Ricardo]

* Y_metadata is now **kwags. [Ricardo]

* Y_metadata added as parameter. [Ricardo]

* Changes for coregionalized models. [Ricardo]

* New functionality added. [Ricardo]

* New functionality added. [Ricardo]

* Add kernel has its own gradients update. [Max Zwiessele]

* Grad dict is property of self. [Max Zwiessele]

* Combination slices full now, independent output kernel. [Max Zwiessele]

* Whitespaces. [Max Zwiessele]

* Old way of tensor product. [Max Zwiessele]

* Copy. [Max Zwiessele]

* Kernel slicer. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into input_dims. [Max Zwiessele]

* Bugfix. [James Hensman]

* Added update_gradints_diag to the add and base kernels. [James Hensman]

* Gradient check and debug options. [Max Zwiessele]

* Uncertain_inputs_example plot changed. [Max Zwiessele]

* Diagonal add kmm. [Max Zwiessele]

* Psi_stat slices for kernels. [Max Zwiessele]

* Psi stat expectations with slices. [Max Zwiessele]

* Psi stat testing improvements, gradients not working yet. [Max Zwiessele]

* Plotting returns. [Max Zwiessele]

* Automatic slicing. [Max Zwiessele]

* Psi_stat_expectaions now working with new parameterized. [Max Zwiessele]

* Gradient check. [Max Zwiessele]

* Oh huge bug in checkgrad global. [Max Zwiessele]

* Empty spaces. [Max Zwiessele]

* Caching doc. [Max Zwiessele]

* Slicing tests and ipdb delete. [Max Zwiessele]

* Combination Kernel for add and prod. [Max Zwiessele]

* Merged params here. [Max Zwiessele]

* Merged params here. [Max Zwiessele]

* Whitespaces. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* All parameters in memory. [Max Zwiessele]

* Constant jitter. [Max Zwiessele]

* Likelihood test. [Max Zwiessele]

* Mrd for new parameterize. [Max Zwiessele]

* Slicing support for kernel input dimension. [Max Zwiessele]

* Add const_jitter back to varDTC. [Zhenwen Dai]

* [SSGPLVM] new plot variational posterior. [Zhenwen Dai]

* [SSGPLVM] support linear kernel with ARD off. [Zhenwen Dai]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Zhenwen Dai]

* Made sampling default for non-gaussian likelihoods as a quick fix to allow plotting again for likelihoods without predictive values. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Name added as a parameter of Prod. [Ricardo]

* Some missing .Ks. [Ricardo]

* Checkgrad divide by zero catches. [Max Zwiessele]

* Numerical global diff in gradcheck. [Max Zwiessele]

* Param concat fix. [Max Zwiessele]

* Gradcheck global diff. [Max Zwiessele]

* Printing for older numpy versions. [Max Zwiessele]

* Parameters once in memory. [Max Zwiessele]

* Dont call parameters_changed ever yourself anymore and parameters are now inplace once in memory. [Max Zwiessele]

* Indentation... [Max Zwiessele]

* [SSGPLVM] implemented linear kernel. [Zhenwen Dai]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Zhenwen Dai]

* Changed kernels in tests (lots still failing, but now mostly for good reason rather than silly naming problems) [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Fixed non_gaussian demo. [Alan Saul]

* Switch input_sensitivity function to model. [Zhenwen Dai]

* [SSGPLVM] numerical stability. [Zhenwen Dai]

* [SSGPLVM] add region constraint. [Zhenwen Dai]

* [SSGPLVM] fix plot_latent. [Zhenwen Dai]

* [SSGPLVM] Learn prior parameters. [Zhenwen Dai]

* [SSGPLVM] support non-ARD rbf. [Zhenwen Dai]

* Minor changes to sympy kernel (removing un-needed comments). [Neil Lawrence]

* Merge branch 'params' of https://github.com/SheffieldML/GPy into params. [Neil Lawrence]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Sparse gp with uncertain inputs. [Max Zwiessele]

* Plotting \o/ [Max Zwiessele]

* Logic edits for copy. [Max Zwiessele]

* Bit more testing of observable patter. [Max Zwiessele]

* Merge branch 'params' of https://github.com/SheffieldML/GPy into params. [Neil Lawrence]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Zhenwen Dai]

* Weaving a faster rbf. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Stability in stationary) [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Zhenwen Dai]

* Caching switched on. [Max Zwiessele]

* Global gradient test done and some parameterized fixes. [Max Zwiessele]

* Checkgrad. [Max Zwiessele]

* Hierarchy edits. adding removing parameters withing hierarchy. [Max Zwiessele]

* Einsumming in rbf for speed. [James Hensman]

* Einsumming in stationary. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Prediction code need updating, started with woodbury vector, but how to predict variance in sparse gp with uncertain inputs? [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Fixed caching bug with args having Nones. [Max Zwiessele]

* No longer caching denom in psi2_rbf. [James Hensman]

* Merged. [James Hensman]

* Caching in place again and working : ) [Max Zwiessele]

* Plotting with uncertain inputs. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Observer pattern has a handle to trigger only > min_priority observers. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Observer pattern now tested and fully operational. needed the good night rest : ) [Max Zwiessele]

* WARNING: switched caching off. [Max Zwiessele]

* Rbf. [Max Zwiessele]

* Non essential tidying in stationary. [James Hensman]

* Efficiencies in stationary. [James Hensman]

* Changes on rbf. [Zhenwen Dai]

* [SSGPLVM] update SSGPLVM with new inferface and merge ssrbf into rbf. [Zhenwen Dai]

* Messing with caching. [James Hensman]

* Linear fix. [James Hensman]

* Added some caching. [James Hensman]

* Caching can handle None. [Max Zwiessele]

* Caching can handle None. [Max Zwiessele]

* Parent observer now static and always last. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Fixes in the plotting and in the dot graphing. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Pydot graphing half done. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Moved plot functionality from add to kern. [James Hensman]

* Parent observer now static and always last. [Max Zwiessele]

* Parameters changed notify added. [Max Zwiessele]

* Observable pattern through and thorugh. [Max Zwiessele]

* Lengthscale fixes. [James Hensman]

* Sparse GP no longer accepts X_variance. [James Hensman]

* Maps import. [Max Zwiessele]

* Gradients. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Some work on ep, and some messing with wher ethe derivatives are computed (in the model, not the inference. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/kern/_src/constructors.py

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/kern/kern.py
    GPy/kern/parts/prod.py

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/kern/parts/prod.py

* Minor changes. [Ricardo]

* Rbf with new parameter structure. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* BayesianGPLVM init with paramschanged. [Max Zwiessele]

* Lots of hacking on RBF. [James Hensman]

* Plot latent updated. [James Hensman]

* Vdtc_missing data corrections. [Max Zwiessele]

* Merged variational posterior changes. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Mucho changes to linear.py. [James Hensman]

* Docstrings in kern.py. [James Hensman]

* Messing with kernels. [James Hensman]

* Parameter inheritance structure. [Max Zwiessele]

* LogexpNeg transformation. [Max Zwiessele]

* Sparse gp missing data. [Max Zwiessele]

* Ard plotting. [Max Zwiessele]

* Commit before switch to master. [Neil Lawrence]

* #2 merge SSGPLVM into params branch. [Zhenwen Dai]

* Merge SSGPLVM into params branch. [Zhenwen Dai]

* Removing testing code from kern.py (it's now in kern_tests.py. [James Hensman]

* More efficient computations in stationary. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Minor merges. [Neil Lawrence]

* Part working version of sympy covariance with new params version. [Neil Lawrence]

* Started sorting out some tests. [James Hensman]

* Renaming: posterior_variationa -> variational_posterior. [James Hensman]

* Kernel tests in working order (not all implemented though. [James Hensman]

* Hierarchical kern should be working. I'll let you know then the tests are up... [James Hensman]

* Tidying in kern. [James Hensman]

* [SSGPLVM] migrate SSGPLVM to params branch. [Zhenwen Dai]

* Sorting ouyt the variational posterior objects. [James Hensman]

* More bugfixin. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Fixed stationary again. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Added initialization. [Max Zwiessele]

* Bugfixin in bernioulli. [James Hensman]

* Stuf in rbf might be broken. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Merged static. [Max Zwiessele]

* Input_sensitivity and ard plotting. [Max Zwiessele]

* Revert "changed to 'update_gradients_q_variational'" [Max Zwiessele]

  This reverts commit f311bfdf17c78bc4f56f03514d4e28b26e2e5057.

* Fixed stationary. [James Hensman]

* Input senitivity in stationary. [James Hensman]

* Plotting conflict fixed. [James Hensman]

* Fixed likelihood tests. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Changed to 'update_gradients_q_variational' [Zhenwen Dai]

* 2d plotting. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Variational posterior and prior added, linear updated. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Got rid of debugging and failing ep tests. [Alan Saul]

* Adding and producting in stationary is no stationary. [James Hensman]

* Some work on periodics. [James Hensman]

* Ratquad working. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Some work pon EP (uninished) [James Hensman]

* Unfinished work on ratinoal quadratic kern. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Merge branch 'params' of https://github.com/SheffieldML/GPy into params. [Neil Lawrence]

* Changes to sympykern.py. [Neil Lawrence]

* Adding update_gradients to sympy.py. [Neil Lawrence]

* Using params class with sympy covariance. Adding conditional statements for presence of weave. [Neil Lawrence]

* Changes to sympy covariance. [Neil Lawrence]

* Minor fixes in kerns. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Zhenwen Dai]

* Parameterized now supports deleting of parameters. [Max Zwiessele]

* Adapt the new interface of the variational posterior distribution. [Zhenwen Dai]

* Bias now looks in shape. [James Hensman]

* Tidying. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Plot merge. [Max Zwiessele]

* Linear and rbf fix for variational gradients in Z. [Max Zwiessele]

* Working on coregionalize. [James Hensman]

* Removed materns. [James Hensman]

* Added Brownian motion. [James Hensman]

* Non-working grads in linear. [James Hensman]

* Manual merging. [James Hensman]

* Gradient operations and cachong. [Max Zwiessele]

* Workin gon linear kernel. [James Hensman]

* Linear without caching, derivatives done. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Spellings. [James Hensman]

* Rbf psi 2. [Max Zwiessele]

* Foo. [James Hensman]

* Weird Max related stuff is happening. [James Hensman]

* Empty init file. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Posterior with one covariance per dimension and param gradient fix. [Max Zwiessele]

* Merged in kern chancges. [James Hensman]

* Everything is broken. [James Hensman]

* Prod now seems to work for sparse. [James Hensman]

* Deleted kernpart, prod and add seem to work okay. [James Hensman]

* Init for src dir£ [James Hensman]

* Rbf and white seem to work. [James Hensman]

* Moved stuff. much breakage. Ow. [James Hensman]

* Gradients now lazy instantiated. [Max Zwiessele]

* Regexp now on all parameters. [Max Zwiessele]

* Updated naming to be consistent. [Max Zwiessele]

* Copy and missing data. [Max Zwiessele]

* Minor edits. [James Hensman]

* Added sparsegp with missing data. [Max Zwiessele]

* Removed sampling keyword (sampling is a silly thing to have as an option. [James Hensman]

* Bad git merge. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Rbf andl inear fixes. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Missing sys. [James Hensman]

* Beginning of bgplvm with missing data. [Max Zwiessele]

* Caching changes. [Max Zwiessele]

* Added index testing. [Max Zwiessele]

* Fixed Observable-weave clash in rbf. [James Hensman]

* Final prior computation issues killed. [Max Zwiessele]

* Some updates for params changes and likelihood fixes. [Max Zwiessele]

* Parameters changed more structured now, parameters changed goes from bottom to top, when calling _notify_parameters_changed() [Max Zwiessele]

* Roundtrip error fixed for likelihood tests. [Max Zwiessele]

* Oops index operations had an assignment error. [Max Zwiessele]

* Priors added. [Max Zwiessele]

* Merge branch 'params' into c_oredered. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Non verbose checkgrad adjusted to new system. [Max Zwiessele]

* Checkgrad    (╯°□°）╯︵ ┻━┻ [Max Zwiessele]

* Gradcheck fixes are not easy. [Max Zwiessele]

* Gradcheck now fully functional. [Max Zwiessele]

* Checkgrad was changing parameters. [Max Zwiessele]

* Parameter adding and removing now fully functional according to tests, including fixes. [Max Zwiessele]

* Fixing now works, removing parameters needs fixing. [Max Zwiessele]

* Lots of F-ordering nonsense. Seems to work though. [James Hensman]

* Variouschanges. [James Hensman]

* An ugly hack to work around the 'stickiness' of ObservableArray. TODO: remove this hack. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Fixed some examples and tests, and stated that Y metadata doesnt need to be the same size as Y. [Alan Saul]

* Fixed gradchecker and fixes for paramterized. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Fixed bernoulli likelihood divide by 0 and log of 0. [Alan Saul]

* Added metadata. [Alan Saul]

* Redid constraints. [Max Zwiessele]

* General bugfixing. [James Hensman]

* Fixed plotting bug. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Here's fitc. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Derivatives working in DTC. [James Hensman]

* Deleted list array. [Max Zwiessele]

* Merged array_core. [Max Zwiessele]

* Fixed merge conflict. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Changes to DTC. [James Hensman]

* Renaming dtc again. [James Hensman]

* Renaming dtc. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Fixed copy bug of observable array. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Minor reorganising. [Alan Saul]

* Deleted listarray. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* First draft of DTC. [James Hensman]

* Array core and bgplvm working > changes due to __i<op>__ will now be reported. [Max Zwiessele]

* Kernel adding now takes over constraints. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Dumb merge conflict in a comment. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Some messing with fitc. [James Hensman]

* Spelling. [James Hensman]

* Parameter handling with default constraints. [Max Zwiessele]

* Rename dK_dtheta > gradients_X. [Max Zwiessele]

* Psi stat and kernel tests new parameterization. [Max Zwiessele]

* Linear kern variational updates. [Max Zwiessele]

* Final touches to gradchecker. [Max Zwiessele]

* Gradchecker now with fixed inputs. [Max Zwiessele]

* Fixes added for gradchecking. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

  Conflicts:
    GPy/core/parameterization/param.py

* Fixed a couple of small params bugs. [Alan Saul]

* Merged parameterized fixing. [Alan Saul]

* Laplace now appears to be grad checking again. [Alan Saul]

* Moved fix parameter to constrainable. [Alan Saul]

* Checkgrad now global and callable from any parameter. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

  Conflicts:
    GPy/core/parameterization/param.py
    GPy/core/parameterization/parameter_core.py
    GPy/core/parameterization/parameterized.py

* Stupid error, needed to actually USE the gradients in student t... Looks like s2 of rasm's may have an extra -? dW_df == -d2logpdf_df not just d2logpdf_df? [Alan Saul]

* Have most of the likelihood testing working, laplace likelihood parameters need fixing, some of the signs are wrong I believe. [Alan Saul]

* Have most of the likelihood testing working, laplace likelihood parameters need fixing, some of the signs are wrong I believe. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Fixed likelihood tests for new parameters structure. [Alan Saul]

* Fixed parameter bugs. [Alan Saul]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Alan Saul]

* Fixed parameterized oddity where it was updating all constrained parameters as soon as any were constrained rather than after all are constrained@ @ [Alan Saul]

* _highest_parent_ now follows the tree, dK_dX > gradient_X, added update_grads_variational to linear, bgplvm for new framework. [Max Zwiessele]

* Adjusted periodic exponential to the new parameterization. [Max Zwiessele]

* Small changes to parameterization init. [Max Zwiessele]

* Fixed product kernel copy error. [Max Zwiessele]

* Added caching framework. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Highest parent fix. [Max Zwiessele]

* First crack at a caching object. [James Hensman]

* Assorted fixes. [James Hensman]

* Predictino working nicely for laplace. [James Hensman]

* Fiddling with plotting. [James Hensman]

* Merged simple conflict£ [James Hensman]

* Starting varDTC with uncertain inputs [not working] [Max Zwiessele]

* Add spike-and-slab gplvm kernel [unfinished].] [Zhenwen Dai]

* Not calling self.parameters_changed explicitly anymore -> not needed. [Max Zwiessele]

* An afternoon's work on the laplace approximation. [James Hensman]

* More owrk on the Laplace approx. [James Hensman]

* Some documenting, and fiddling with the laplace approx. [James Hensman]

* Sparse GP now working nicely. [James Hensman]

* Sparse GP now checkgrads, optimises sensibly. Predicitno still not working. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Adapter laplace inference into the param framework. [Zhenwen Dai]

* Some hacking on sparse_gp inference. [James Hensman]

* Tidied up sparse_gp_regression. [James Hensman]

* Some changes for coregionalize. [James Hensman]

* Wrapping docstrings. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* "Missing file?" [Ricardo]

* Changed gradient interface to gp and sparse GP. [James Hensman]

* Fixed syntax bug in sparse GP. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Pylab library not needed. [Ricardo]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Ricardo]

  Conflicts:
    GPy/core/sparse_gp.py

* Typo corrected. [Ricardo]

* New files. [Ricardo]

* Files relocated. [Ricardo]

* Changes according to files reloaction. [Ricardo]

* Lines that call matplotlib were commented. [Ricardo]

* Plotting functions modified. [Ricardo]

* Plotting functions modified. [Ricardo]

* Plotting functions modified. [Ricardo]

* Plotting functions modified. [Ricardo]

* Plotting functions modified. [Ricardo]

* Plotting functions modified. [Ricardo]

* Plotting functions modified. [Ricardo]

* Plotting functions modified. [Ricardo]

* Plotting functions modified. [Ricardo]

* Relocated. [Ricardo]

* Relocated. [Ricardo]

* Relocated. [Ricardo]

* Relocated. [Ricardo]

* Relocated. [Ricardo]

* Relocated. [Ricardo]

* Relocated. [Ricardo]

* Relocated and renamed. [Ricardo]

* Relocated and renamed. [Ricardo]

* Relocated. [Ricardo]

* Util/plot moved to plotting directory. [Ricardo]

* Noddling. [James Hensman]

* Removed marginal and derivative from posterior object. [James Hensman]

* Tidying in sparse gp. [James Hensman]

* Removed some superfluous things from the model class. [James Hensman]

* Fixing scg on this branch. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Plot_latent: added param_to_array to model.X and model.Z for matplotlib plotting. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Some gplvm related fixes. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Beginnings of gplvm. [James Hensman]

* Changed priority of observable array to 0. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Foo. [James Hensman]

* Added all authors to GP Regression copyright. [James Hensman]

* Bug in setting _highest_parent_ fixed. [Max Zwiessele]

* Getstate > _getstate and setstate > _setstate. [Max Zwiessele]

* New gradient handling way nicer. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Gradient field added to param. [Max Zwiessele]

* More gradient based tomfoolery. [James Hensman]

* Changes to rbf and white to allow new parameter gradient structure. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [James Hensman]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Merged posterior changes. [Max Zwiessele]

* IMPORTS. [Max Zwiessele]

* Typo. [Max Zwiessele]

* Minor. [Max Zwiessele]

* Added new gradient functinoality to rbf. [James Hensman]

* Beginnings of neat gradient framework. [James Hensman]

* Tidying in kern.py. [James Hensman]

* Exact gaussian inference now accepts a kern and X, not K. [James Hensman]

* Basic GP regression now working again. [James Hensman]

* Removed some X_normalising things. [James Hensman]

* Added PCA to linalg. [James Hensman]

* Removed fitc_classification modle. [James Hensman]

* Made svigp depend on GP instead of gp_base. [James Hensman]

* Removed sinc. [James Hensman]

* Adding back param_to_array. [James Hensman]

* Removed shapefile dep. [James Hensman]

* Adding gpy_congif from devel. [James Hensman]

* Merged (hard) the util from devel. [James Hensman]

* Hard-merging in the examples and testing dirs from master. [James Hensman]

  This is probably a dumb way to do it, but I don;t know better.

* Noodling. [James Hensman]

* Lots of messing with the sparse inference method. [James Hensman]

* Removed a lot of unnecessary code in sparse GP. [James Hensman]

* Removed the gp_base abstraction class. [James Hensman]

* Just general tidying. [James Hensman]

* Fixed up the Gaussian likelihood a little. [James Hensman]

* All parameterization stuff now in seperate module -> GPy.core.parameterization. [Max Zwiessele]

* Fixed unsized param bug. [Max Zwiessele]

* Some tifying in the models classes. [James Hensman]

* Some minor edit to Bernoulli. [James Hensman]

* Moving fitc. [James Hensman]

* Adding a copright notice. [James Hensman]

* Some work in the Gaussian likelihood. [James Hensman]

* Mostly docstring noodling. [James Hensman]

* Some noodling around in the likelihoods. [James Hensman]

* Fixed bug introduced my merge. [James Hensman]

* Merged in params. [James Hensman]

* Array handling in plotting and weave. [Max Zwiessele]

* Naming and pil changes. [Max Zwiessele]

* Modified docstrings. [James Hensman]

* Adding empyy init file. [James Hensman]

* More work on the posterior class. [James Hensman]

* Added the structure to posterior.py  to enable... [James Hensman]

  to enable computation from the posterior mean and variance, instead of
  the woodbury componnents

  This iss the first step in being able to use this structre for EP and
  the laplace approximation.

* More massive and destructive changes everywhere. [James Hensman]

* Many dramatic cahnges. at least it import without error. [James Hensman]

* Moved functional part of sparseGP to inference/dtcvar. [James Hensman]

* Tidied up gp_base and gp. [James Hensman]

* Lots of tidying in the inference section. [James Hensman]

* Adding exact gaussian inference. [James Hensman]

* Added some docstrings and the posterior class structure. [James Hensman]

* Messing wih he inference directory now. [James Hensman]

* Some more messing with the likelihood directory. [James Hensman]

* First commit in new structure. [James Hensman]

* Changes to symbolic. [Neil Lawrence]

* Latest updates for ties, still bery buggy, considering restructuring... [Max Zwiessele]

* Simple tieing now working, still buggy though, progress with care. [Max Zwiessele]

* Biiig changes in tieing, and printing -> hirarchy now always shown. [Max Zwiessele]

* Minor edits, like spacing, spelling. [James Hensman]

* Added variational. [Max Zwiessele]

* Bgplvm integrated. [Max Zwiessele]

* Added BGPLVM in parameterized. [Max Zwiessele]

* Added gplbm and sparse gp to new parameterized structure. [Max Zwiessele]

* Merge branch 'params' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Adjusted parameters to report their changes. [Max Zwiessele]

* Updated scg messages. [Max Zwiessele]

* Tutorial update and X observer. [Max Zwiessele]

* X caching is not yet done, parameter caching working fine. X cache must be adjusted to update at the right times. [Max Zwiessele]

* Changing all parameterized objects to be compatible with the new parameterization. [Max Zwiessele]

* Adjusted gaussian likelihood to new parameterization. [Max Zwiessele]

* Gp_base newly paramterized. [Max Zwiessele]

* Updated white, bias and rbf to new parameter handling. [Max Zwiessele]

* GPRegression working, gradients still todo. [Max Zwiessele]

* First adjustments to the model and gps -> updates and gradient transforms. [Max Zwiessele]

* Parameters have a update hirarchy, in which updates to parameters are hirarchically pursuit. [Max Zwiessele]

* Likelihood is now parameterized. [Max Zwiessele]

* Kern is now parameterized. [Max Zwiessele]

* Kern is now parameterized. [Max Zwiessele]

* Parameters now work efficiently, tieing is iwth observer pattern. [Max Zwiessele]

* Still todo: untie, gradients, priors, print ties. [Max Zwiessele]

* Parameter_testing has to be written new. [Max Zwiessele]

* Starting to sort out likelihoods WARNING not working. [Max Zwiessele]

* Make sure _init_ is not overriden. [Max Zwiessele]

* WARNING: half way through commit, this is a non working middle thing! everything should be in place now, figure tieing and printing with broadcasting. [Max Zwiessele]

* Parameter indexing now linear in number of printed values. [Max Zwiessele]

* Index ops now with own dict. [Max Zwiessele]

* Documentation added. [Max Zwiessele]

* More testing. [Max Zwiessele]

* Warning messages optional for re-constraining. [Max Zwiessele]

* Indexing fixed, some equality testing. [Max Zwiessele]

* Parameter object done, printing fixed. [Max Zwiessele]

* Merged dimen reduction. [Max Zwiessele]

* Docstrinfs in kern.py. [James Hensman]

* Updated sympy code, multioutput grad checks pass apart from wrt X. Similar problems with prediction as to sinc covariance, needs investigation. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Seems to work on windows now. [Nicolò Fusi]

  not everything works yet, but I've identified the main issues. Still

* Added olivetti faces data set. It required adding netpbmfile.py a bsd licensed pgm file reader from Christoph Gohlke, which doesn't seem to have a spearate installer. Also modified image_show to assume by default that array ordering is python instead of fortran. Modified brendan_faces demo to explicilty force fortran ordering. Notified Teo of change. [Neil Lawrence]

* Domain and trtansformations namespace prettyfying. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into params. [Max Zwiessele]

* Some fixes and changes to the sympykern. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Nparam changes to num_params. [James Hensman]

* Changes Nparts for num_parts in kern. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Andreas]

* Minor changes. [Andreas]

* Normalize Y given as an argument to constructor. [Andreas]

* Fixed stick datasets bug ... but sympykern is currently in a rewrite so will be broken. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Plots tidied up. [Ricardo]

* Modifications to allow noise_model related parameters. [Ricardo]

* Coregionalization examples fixed. [Ricardo]

* Sampling function added. [Ricardo]

* Added first draft of functionality for multiple output sympy kernels. [Neil Lawrence]

* Tests updated. [Max Zwiessele]

* Dim reduction examples Q= > input_dim= [Max Zwiessele]

* Numpy non hashable AHHHHHH. [Max Zwiessele]

* Dont print brackets in transformations. [Max Zwiessele]

* Parameterized first beta test. [Max Zwiessele]

* Merge branch 'devel' into params. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [mu]

* Testing ODE. [mu]

* Added capability for sinc covariance via sympy kernel. [Neil Lawrence]

* Minor changes to della_gatta example (multiple optima). [Neil Lawrence]

* Replaced check for sympy in constructors.py. [Neil Lawrence]

* Change to criterion on positive definite check (epsilon*10 instead of epsilon). [Neil Lawrence]

* Remove coregionalization test as it's causing a core dump! Need to chase this up. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Some tidying in the EP likelihood. [James Hensman]

  Changes self.N to self.num_data for consistency with everywhere else
  added the factor of 2pi to Z.

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Allowed passing of factr to bfgs algorithm. [James Hensman]

* Updates to sympykern including bug fixes and ability to name covariance. Include test for rbf_sympy in kernel tests. Remove coregionalization test as it's causing a core dump! Need to chase this up. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Fixing W_columns and num_outputs nomenclature. [Ricardo]

* Added missing files. [Neil Lawrence]

* Added missing file. [Neil Lawrence]

* Missing file. [Neil Lawrence]

* Merging changed files. [Neil Lawrence]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Merge pull request #82 from jamesmcm/devel. [James McMurray]

  Devel

* Fixed more errors in docs 2. [James McMurray]

* Fixed more errors in docs. [James McMurray]

* Adding extra tests for bgplvm. [James Hensman]

* Merging changed files. [Neil Lawrence]

* Add eq_ode1 covariance. [Neil Lawrence]

* More samples for higher sampling accuracy. [Max Zwiessele]

* Sparse_gp_multioutput test added. [Ricardo]

* Removing unnecessary stuff... [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

  Conflicts:
    GPy/examples/classification.py

* Fixed a bug in Neil's otherwise tidy hetero kernel. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Merge pull request #80 from jamesmcm/devel. [James McMurray]

  Devel

* Merge branch 'devel' of git://github.com/SheffieldML/GPy into devel. [James McMurray]

* Fixed docstring warnings - could still be mistakes. [James McMurray]

* Started fixing docs. [James McMurray]

* Fixed three tests by being _slightly_ less stringeent about poositive-definiteness. [James Hensman]

* Small change in crescent demo. [Ricardo]

* Please stop breaking this module. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Do_test_latents appears to be working now. [James Hensman]

* Added hetero back to the init. [James Hensman]

* Epsilon and power_ep now are parameters of update_likelihood. [Ricardo]

* Crescent data example is better organized. [Ricardo]

* Cross term testing switched on. [Max Zwiessele]

* Comment linear x linear for testing purposes. [Max Zwiessele]

* Parameter_testing. [Max Zwiessele]

* Index operations now work on flattened indices. [Max Zwiessele]

* Unoptimized parameter, still slower than current implementation. [Max Zwiessele]

* Ndarray subclass cleaned up. [Max Zwiessele]

* Subclassing ndarray almost functional. [Max Zwiessele]

* Parameters ndarray, stuck at using strides for transformations. [Max Zwiessele]

* Transformations are singletons now, weak refs for memory managment. [Max Zwiessele]

* Almost there with array inheriting. [Max Zwiessele]

* With subclassing ndarray, current_slice problems not solved... [Max Zwiessele]

* Without inheriting from numpy.ndarray. ndarray functionality missing. [Max Zwiessele]

* NegativeLogexp Pep8ted. [Max Zwiessele]

* Index operations finalized. [Max Zwiessele]

* Added index_operations and deleted them from paramter. [Max Zwießele]

* Added parameter files - Alan. [Max Zwiessele]

* Transformations singleton. [Max Zwiessele]

* Merge branch 'devel' into params. [Max Zwiessele]

  Conflicts:
    GPy/core/transformations.py
    GPy/kern/parts/kernpart.py

* Merge pull request #77 from jamesmcm/devel. [James McMurray]

  Devel

* Merge branch 'devel' of git://github.com/SheffieldML/GPy into devel. [James McMurray]

* Bug in prod-coreg kernels fixed, not in the most elegant way though. [Ricardo]

* Disp=False in scipy optimizations. [Ricardo]

* Function grep_model added, works like print model, but accepts regexp. [Ricardo]

* Reverting error. [Ricardo]

* Rebuilt documentation. [James McMurray]

* Changes to fix autodoc - need to test with ReadTheDocs still. [James McMurray]

* Fixed readme. [James McMurray]

* Updated readme with instructions for compiling documentation, running unit tests. [James McMurray]

* Normalize_Y is passed to core function. [Ricardo]

* Redundant code commented. [Ricardo]

* Samples in plot_f fixed for sparse_models. [Ricardo]

* Comments added to assertions. [Ricardo]

* Coreg_spars fixed. [Ricardo]

* Heaviside instead of heavyside. [Ricardo]

* 1 docstring line completed. [Ricardo]

* Random 2 bug neutralized... not fixed. [Ricardo]

* Heaviside transformation fixed. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Likelihoods are now Parameterized objects. [Ricardo]

* Change in gaussian noise to avoid confusion with Gaussian likelihood. [Ricardo]

* AddedHeraviside functionality to EP. [James Hensman]

* Fixed bug in rat_quad for RW. [James Hensman]

* Fixed Zsolts bug in prod.py. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

  Conflicts:
    GPy/examples/regression.py
    GPy/kern/constructors.py
    GPy/testing/kernel_tests.py
    GPy/util/multioutput.py

* Other local changes. [Neil Lawrence]

* Actually changing coregionalise to coregionalize. [James Hensman]

* Other local changes. [Neil Lawrence]

* Added covariance for input dependent noise levels (hetero.py). [Neil Lawrence]

* Renamed coregionalise to coregionalize to make it consistent with other spellings (optimize etc.) [Neil Lawrence]

* Merged conflict of model.py. [Neil Lawrence]

* Changed kern.py so that X2 is correctly passed as None to the kern parts for dK_dX. Modified several kern parts so that dK_dX is correctly computed (factors of 2 missing). Removed spurious factors of 2 from gplvm, bcgplvm, sparse_gp and fitc code. [Neil Lawrence]

* Print_all function removed, print m works as before. [Ricardo]

* Coregionalisation changed to coregionalization. [Ricardo]

* LinK2_functions2 merged. [Ricardo]

* Merge branch 'linK_functions2' into devel. [Ricardo]

  Conflicts:
    GPy/core/gp.py
    GPy/core/gp_base.py
    GPy/core/sparse_gp.py
    GPy/examples/regression.py
    GPy/kern/constructors.py
    GPy/kern/parts/coregionalise.py
    GPy/models/__init__.py
    GPy/models/sparse_gp_classification.py
    GPy/util/__init__.py

* Useless files deleted. [Ricardo]

* Moved to (sparse_)gp_multioutput_regression. [Ricardo]

* Doesn't matter I'll erase this file. [Ricardo]

* Duplicated line erased. [Ricardo]

* Sparse multiple outputs model with Gaussian noise. [Ricardo]

* Multiple outputs model with Gaussian noise. [Ricardo]

* Likelihood gradients for heteroscedastic noise. [Ricardo]

* Changes in plot functions, to allow 1D multiple outputs visualization. [Ricardo]

* Docstrings in new functions. [Ricardo]

* Works well now. [Ricardo]

* Redundant variable eliminated. [Ricardo]

* R paramter renamed as W_columns, and Nout renamed as num_outputs. [Ricardo]

* New files added. [Ricardo]

* Build_cor_kernel is now called build_lcm. [Ricardo]

* Docstring completed. [Ricardo]

* Changes according to new likelihoods definition. [Ricardo]

* Errors fixed. [Ricardo]

* Docstrings completed. [Ricardo]

* Docstrings completed. [Ricardo]

* Bug fixed. [Ricardo]

* Step function modified, now the output is either 1 or -1. [Ricardo]

* Step transformation added. [Ricardo]

* New Gaussian likelihood for multiple outputs. [Ricardo]

* Error in plot corrected. [Ricardo]

* Multioutput models added. [Ricardo]

* Changes to allow multiple output plotting. [Ricardo]

* Plots for multiple outputs. [Ricardo]

* Multioutput is working. [Ricardo]

* Gamma noise added. [Ricardo]

* Random changes. [Ricardo]

* The next step is to optimize the noise models' parameters. [Ricardo]

* Files re-organized. [Ricardo]

* Confidence interval fixed. [Ricardo]

* Massive changes. [Ricardo]

* More files. [Ricardo]

* Further corrections. [Ricardo]

* Massive changes. [Ricardo]

* Predictive mean done. [Ricardo]

* More changes. [Ricardo]

* Predictive values, new method. [Ricardo]

* Some cool stuff for EP. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Fixed args to bfgs. [James Hensman]

* Fixed Alans checkgrad bug. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Working on the Poisson likelihood. [Ricardo]

* Parameterization changes take a while. [Max Zwiessele]

* Correcterd minor errors (imports etc) [Max Zwiessele]

* Pep8'ed transformations module. [Max Zwiessele]

* Mrd_sim stable and deprecated. [Max Zwiessele]

* Expectation test slicing. [Max Zwiessele]

* Only compare Z cache once. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Added gibbs.py, although test is still failing. [Neil Lawrence]

* Added slicing to kern.py. [Max Zwiessele]

* Cleaned up imports. [Max Zwiessele]

* Removed white kernel, bc of likelihood variance. [Max Zwiessele]

* Plotting adjusted. [Max Zwiessele]

* Added print m and print m.all differentiation. [Max Zwiessele]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Merge branch 'devel' of git@github.com:SheffieldML/GPy.git into devel. [Max Zwiessele]

* Merge dim reduction. [Max Zwiessele]

* Fixed gplvm unittest which was sampling an output matrix of size input_dim (and corresponding error in bcgplvm unittest which was based on it). [Neil Lawrence]

* Added mlp mapping (with possibility of having multiple layers). [Neil Lawrence]

* Added simple back constrained GPLVM model. [Neil Lawrence]

* Added unit tests for mapping functions. [Neil Lawrence]

* Implemented Mapping framework and associated linear and kernel mappings. This is needed for mean functions, back constrained GPLVM and the non-stationary Gibbs kernel. [Neil Lawrence]

* Implemented MLP gradients with respect to X. [Neil Lawrence]

* Polynomial kernel gradients wrt X working. [Neil Lawrence]

* Changed default values of W and kappa for coregionalisation kernel. Changed names of keyword arguments from Nout and R to output_dim and rank. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Fixed target slicing bug in prod kernel. [James Hensman]

* Added gpx dataset. [Neil Lawrence]

* Added missing poly.py file. [Neil Lawrence]

* Added models for testing kernel gradients in unit tests. [Neil Lawrence]

* Various doc string edits and consistency changes in comments. [Neil Lawrence]

* Minor fixes. [Neil Lawrence]

* Added notes file for issues raised while looking through code, some are things I need to raise on github, others need some informal discussion, but for the moment thought to put them informally here, given flakiness of internet connection. [Neil Lawrence]

* Minor fixes to classification to allow kernel choice, change of oil example to use full test set and full training set. [Neil Lawrence]

* Minor fixes to regression example with robot_wireless. [Neil Lawrence]

* Added robot_wireless data set and examples. [Neil Lawrence]

* Added ability to load in cmu motion capture data bases in the new data base loading format. [Neil Lawrence]

* Finished first draft of system for downloading data sets. [Neil Lawrence]

* Added first draft of polynomial kernel. [Neil Lawrence]

* Added automatic computation of dKdiag_dtheta in kernpart.py using dK_dtheta. [Neil Lawrence]

* Added mlp covariance (x gradient not working) [Neil Lawrence]

* Modifications to transformations ... not sure which tests to run to make sure I haven't messed things up. New code avoids exponentiating values greater than -log(eps) or less than log(eps). Also changed negative code to call the positive code (I think they should inherit the positive code ... but maybe not. [Neil Lawrence]

* Mocking matplotlib pyplot as readthedocs is failing to import it (again...) [Alan Saul]

* Prodpart bugfix. [Max Zwiessele]

* Linear kernel old implementation. [Max Zwiessele]

* Better bound extimate for plot_latent background. [Max Zwiessele]

* Visual? [Max Zwiessele]

* Changed psi0 of white to be zero. [Max Zwiessele]

* Constant jitter to Kmm, deleted some white kernels in models and examples. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Added connections.txt. [Neil Lawrence]

* Implemented smart pickling for svigp. [Andreas]

* Added plot of magnification factor. [Alessandra Tosi]

* Added jacobian and magnification factor. [Alessandra Tosi]

* Oops, that was silly bug. Don't code tired, kids. [James Hensman]

  problem summing psi2 in sparse_GP

* Some profilng and debugging in sparse_GP. [James Hensman]

* Input_sensitivity right way. [Max Zwiessele]

* Import handling... [Max Zwiessele]

* Import handling... [Max Zwiessele]

* Added imshow controller and label controller. [Max Zwiessele]

* Minor/pep8. [Max Zwiessele]

* Minor adjustment to fast_array_equal. [Max Zwiessele]

* Mrd plot_scales improved. [Max Zwiessele]

* Gradient checker more robust against name changes. [Max Zwiessele]

* Experimental plotting of gradient in latent space (and addition of steepest gradient dimensions) [Max Zwiessele]

* Pca initializer more robust to lower dimensions. [Max Zwiessele]

* Ard and latent plotting improved. [Max Zwiessele]

* Merge of rbf_inv failed, corrected with AD. [Max Zwiessele]

* Gradient checker implemented. [Max Zwiessele]

* Gradient checker implemented. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Changes to psi2 in linear. [James Hensman]

* Minor fix. [Andreas]

* Gradient checker implemented. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Fixed asarray in example toy_ARD. [Andreas]

* Rbf kern chaching bug fixed. [Max Zwiessele]

* Merge rbf_inv changes. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Small changes to rbf and rbf_inv. [James Hensman]

* Added gradient checker model. [Max Zwiessele]

* Caching bugfix for psi2 computations. [Max Zwiessele]

* Reverted stupid merge error. [Max Zwiessele]

* Dimensionality reduction demo. [Max Zwiessele]

* Ard plotting enhanced. [Max Zwiessele]

* Rewritten most parts wrt inv_lengthscale (still missing dK_dtheta and dPsi2_dtheta can be written better) [Andreas]

* Input sensitivity for rbf_inv. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Andreas]

* Plot_ARD greatly improved, crossterm plotting enabled. [Max Zwiessele]

* Plot_ARD greatly improved, crossterm plotting enabled. [Max Zwiessele]

* Git pushMerge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Fast_array_equals now handles 3d matrices. [Nicolo Fusi]

* Dimensionality reduction merge. [Max Zwiessele]

* BGPLVM example correction. [Max Zwiessele]

* Stick_bgplvm to original version with rbf. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Fixed logic for fast_array_equal. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* (much) faster comparison between arrays. Useful for kernel caching. [Nicolo Fusi]

* Changes to regression and svigp. [Andreas]

* Bayesian gplvm plots inducing inputs automatically. [Max Zwiessele]

* BGPLVM example testing with rbf_inv. [Max Zwiessele]

* Plot_ARD greatly improved, plotting of ARD paramters for multiple ARD kernels now supported. [Max Zwiessele]

* Minor pep8 stuff. [Max Zwiessele]

* BGPLVM example testing with rbf_inv. [Max Zwiessele]

* Added unittests for sparse gp with uncertain inputs. [Max Zwiessele]

* BGPLVM pca init now able to cope with all dimension issues. [Max Zwiessele]

* Plot_ARD greatly improved, plotting of ARD paramters for multiple ARD kernels now supported. [Max Zwiessele]

* Bgplvm_stick demo with rbf_inv. [Andreas]

* Rbf inv now working. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Andreas]

* Removed abomnibal matrix class. [James Hensman]

* Minor. [Andreas]

* Added rbf_inv.py. [Andreas]

* Added rbf_inv.py kernel which is parametrised with the variances. [Andreas]

* Added toy_ARD_sparse.py. [Andreas]

* Added sparseGPLVM_oil example. [Andreas]

* Added sparseGPLVM in the init of models. [Andreas]

* Small mod in toy_ARD. [Andreas]

* Removed depricated max_f_eval. [Andreas]

* Removed depricated max_f_eval from bgplvm_simulation. [Andreas]

* Fixed max_iters argument for scg. [Andreas]

* Added ARD demo. [Andreas]

* Added the 2 pi term to the likelihood of the gp. [James Hensman]

  Why the hell was this missing?

* Improved docstring for MRD. [Max Zwiessele]

* GPLVM get/setstate added. [Max Zwiessele]

* SCG optimizer now greatly improved in printing. [Max Zwiessele]

* SCG optimizer now greatly improved in printing. [Max Zwiessele]

* Changes to the hierarchical kernpart. [James Hensman]

  Looks to work now.

* Added constructor for hierarchical kernel. [Nicolo Fusi]

* Merged. [Nicolo Fusi]

* Created a hierarchical kernel. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Fixed a bug in constructor of periodic_matern52. [James Hensman]

* Changes to multiout constructor. [Nicolo Fusi]

* Minor adjustments. [Max Zwiessele]

* Minor printing improvements. [Max Zwiessele]

* Added value restoring if unpickling objects. [Max Zwiessele]

* Plot ard ticks improved. [Max Zwiessele]

* Optional plotting of inducing inputs added. [Max Zwiessele]

* Plot_latent enhancements. [Max Zwiessele]

* Massively improved printing. [Max Zwiessele]

* More robust gradient clippinggit stat. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Fixed bug in constructors. [Nicolas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* New constrain_negative negative_logexp (selected by default) [Nicolas]

* Plot_latent bug-fix of creating no figure. [Max Zwiessele]

* Plot_latent bug-fix in mrd. [Max Zwiessele]

* Added docstring to domains. [Max Zwiessele]

* _get/set_params into parameterized. [Max Zwiessele]

* _get/set_params into parameterized. [Max Zwiessele]

* Added tutorial for creating models. [Max Zwiessele]

* Params left in __get/setitem__ [Max Zwiessele]

* Added robust pickling, switches to old behaviour, if get/setstate not implemented. [Max Zwiessele]

* Pickling unified with __getstate__ and __setstate__ [Max Zwiessele]

* Merge branch 'devel' into pickle. [Max Zwiessele]

* Fixed an import. [Teo de Campos]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Added anopther simple subplotting function. [James Hensman]

* Removed sympy import. [Nicolo Fusi]

* Added missing import in util.linalg. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge remote-tracking branch 'origin/devel' into devel. [Ricardo]

* Bug fix in the confusion matrix. [Ricardo]

* Removed sympy helpers from init. [Nicolo Fusi]

* Added init. [Nicolo Fusi]

* Removed unnecessary gitignore line. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Kernels are now consistent with pep8 and common reason. [Nicolo Fusi]

* Added an include. [Teo de Campos]

* Merging by hand... [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo]

* New version number. [Ricardo]

* Ensure_default_constraints is on by default. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Bug fix in the confusion matrix. [Ricardo]

* Began to merege the SVIGP code into GPy. [James Hensman]

  Example is implemented, but the step length is a bit crazy!

* Changed manifest from docs to doc. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Corrected minor bug in Brownian kernel. [Nicolas]

* Comment import visual in visualize due to issues in Windows/OSX. [Neil Lawrence]

* Mods to visualize and dimensionality to make stick demos work for summer school. [Neil Lawrence]

* Changes to the efficiency of the sparse GP when there are many outputs. [James Hensman]

* Fixed fixed kernel (aha!) [Nicolo Fusi]

* Pcikling? [Max Zwiessele]

* Pypi release update. [Max Zwiessele]

* Merge branch 'master' into devel. [Max Zwiessele]

* Removed sympy dependency, incremented version. [James Hensman]

* Incremented version number. [James Hensman]

* Incremented version. [James Hensman]

* Merge branch 'devel' [James Hensman]

* Merge branch 'devel' [Alan Saul]

* Changed version. [Alan Saul]

* Model robustness greatly improved and avisualize adjustments for plot_latent. [Max Zwiessele]

* Robust failure handling in model objective and gradient. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Fixed blaslib bug, I hope. [James Hensman]

* Catch linalg errors inside model and more sopihisticated non-pd checks. [Max Zwiessele]

* Adjusted doc to new pep8 format. [Max Zwiessele]

* Fixed a transpose bug in the sparse GP prediction. [James Hensman]

* Merged with master. [Alan Saul]

* Removed slices from GP regression wrapper. [James Hensman]

  Xslices no longer exists in the GPy framework.

* Commented out a buggy ax.set_aspect line. [Teo de Campos]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Removed the unnecessary test skip. [Nicolas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Un-commented out visualize.py. [Teo de Campos]

* Resurrected visualize.py. [Teo de Campos]

* Fixed mrd init call. [Alan Saul]

* Removed overwriting_b in lapack. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Fixed lapack importing for old scipy version. [James Hensman]

* Ignoring example tests again. [Alan Saul]

* Changed deprecation supression to 12. [Alan Saul]

* Merged. [Alan Saul]

* Merge branch 'psi1_transpose_fix' into devel. [James Hensman]

* Psi1 is now the right way around. [James Hensman]

* Fixing lapacks. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Removing unused link_functions. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Modified lengthscale gradients - demo works now. [James Hensman]

* Removed visualize dependencies and added dKdiag_dX for linear. [Nicolas]

* Factored out lapack into utils so we can check version and give deprecation warnings. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Making sure GPy imports. [Neil Lawrence]

* Part changes to datasets.py and mocap.py to download data resources for examples. Not working currently! [Neil Lawrence]

* Placed back in examples for motion capture! Added spheres to visualization of figure. [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Added visualization for motion capture data using python visual module. [Neil Lawrence]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* _Xmean is now Xoffset and _Xstd is now _Xscale. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Removed examples with non public datasets. [Nicolas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Removed examples with non public datasets. [Nicolas]

* FITC example: bound for lengthscale. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Fixed tuto example. [Nicolas]

* Non_Gaussian exampless deleted. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Fitc and generalized_fitc models deleted. [Ricardo]

* Reduced number of iterations for a couple of things. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Poisson likelihood implementations needs to be thought carefully. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Dimensionality reduction mrd example less interations. [Max Zwiessele]

* Link_function class renamed as LinkFunction. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Minor changes. [Ricardo]

* Pseudo_EM modified. [Ricardo]

* Examples corrected. [Ricardo]

* MRD fix. [Max Zwiessele]

* Nice plot handling in tutorials. [Nicolas]

* Example multiple optima now returns a model. [Nicolas]

* Refactored example tests. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Kern constructors now have input_dim instead of D. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merged conflict in tutorial's tests (again) [Nicolas]

* Merged conflict in tutorial's tests. [Nicolas]

* Bugs fixed in tutorial's tests. [Nicolas]

* Dim reduction adaption. [Max Zwiessele]

* Num_data refactoring. [Max Zwiessele]

* Fixed merge conflict in examples_tests. [Alan Saul]

* Git pushMerge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

  Conflicts:
    GPy/core/fitc.py

* FITC example added. [Ricardo]

* FITC is back. [Ricardo]

* Psi stat tests adapted (psi1 still not tested, bc of transpose confusion) [Max Zwiessele]

* Merged num_data conflicts. [Max Zwiessele]

* Merge kern conflicts in examples. [Max Zwiessele]

* Kern params adapted: Nparams > num_params and fixes of input_dim. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Nparams > num_params and Nparam_tranformed > num_params_transformed. [Max Zwiessele]

* Made examples possible to run all examples and throw out a dictionary of problems at the end (and whilst it's running, tried to ignore deprecation warnings. [Alan Saul]

* Lots of bugfixes after refactoring. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* New file. [Ricardo]

* Fixed merge conflicts, M now num_inducing. [Alan Saul]

* Refactoring: self.D > self.input_dim in kernels. [Max Zwiessele]

* Output_dim instead of input_dim. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* An assortment of fixes. [James Hensman]

* Change input for output. [Ricardo]

* Seems to run now. [Ricardo]

* Change input for output. [Ricardo]

* New file. [Ricardo]

* Changed all M's for num_inducing. [Alan Saul]

* Fixed naming to standardized PEP8. [Ricardo]

* Merged and fixed conflicts, names still need changing accordingly. [Ricardo]

* Refactoring files added. [Max Zwiessele]

* Merged regression example, corrected refactoring. [Max Zwiessele]

* Preferred optimiser is now scg bydefault. [James Hensman]

* Fixed the multiple optima demo. [James Hensman]

* Getting examples running. [Max Zwiessele]

* REFACTORING: model names, lowercase, classes uppercase. [Max Zwiessele]

* New FITC model and other stuff. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* How did that happen? [James Hensman]

* Merged an emty line... [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Dim reduc plotting. [Max Zwiessele]

* Merged plotting and input_dim. [Max Zwiessele]

* Dimensionalityreduction plotting adjusted to new syntax. [Max Zwiessele]

* Constrain fixed now unconstrains first. [Nicolas]

* Changed the behaviour of checkgrad. [James Hensman]

  checkgrad usd to check the passed string (for name matching) against the
  list of _get_param_names(). Then it would index along
  _get_param_names_transformed()!

  this led to inconsistensies when fixed or tied variables were used,
  which screwed up the ordering of the variable names.

  We now match against _get_param_names_transformed().

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Replaced Q by input_dim. [Alan Saul]

* Tutorials updated to comply with changes throught the code. [Nicolas]

* Modified the regression demos r. [James Hensman]

  they all seem to work now.

* Made input_ubncertainty plotting work, modified example a little. [James Hensman]

* Minor changes. [Ricardo]

* Correction to some tests. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Plotting in the model behaves better. [James Hensman]

* Plotting behaviour adapted for BGPLVM. [Max Zwiessele]

* Plotting behaviour adapted for BGPLVM. [Max Zwiessele]

* Plotting behaviour adapted for kern and mrd. [Max Zwiessele]

* Making examples working. [Nicolas]

* New model. [Ricardo]

* New model. [Ricardo]

* New models included. [Ricardo]

* Examples corrected. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Fixed printing, some example. [James Hensman]

* Added priors behaviour as intended and issue #38 closed and fixed. [Max Zwiessele]

* New GP_classification model. [Ricardo]

* FITC test not skipped any more. [Ricardo]

* Unit_tests corrected. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Re-merged. only RA's errors (probit?) remain. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* MERGE. [Max Zwiessele]

* Domains added and class names in priors capitalized. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

  Conflicts:
    GPy/models/GPLVM.py

* Remove copies (they are now in visualize code ...). [Neil Lawrence]

* Neil's flailing attempts to update the flailing stick man. [Neil Lawrence]

* Match_moments function passes transformed values. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Added domain matching in set_prior. [Max Zwiessele]

* Added domains to priors. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Examples changed to use new link_functions. [Ricardo]

* Merge branch 'devel' into link_functions. [Ricardo]

* Link functions defined. [Ricardo]

* Added domains in transformatins and priors. [Max Zwiessele]

* Tests are now passing. [James Hensman]

* Fixing some examples. [James Hensman]

* Fixed Alan's dependency nightmare. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Scg optimizer enhancments and mrd demo fix. [Max Zwiessele]

* Removed NL's notes, they are now integrated in the issue tracker. [Nicolo Fusi]

* Git branchMerge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Mrd and bgplvm simulation examples working. [Max Zwiessele]

* SCG optimizer scale down to 1e-30. [Max Zwiessele]

* Added parameters max_nb_eval_optim in regression examples. [Nicolas]

* Added ability to handle likelihood with zero variance. [Nicolas]

* Regular expressions now match rather than search. [James Hensman]

* Broken whilst unlinking GP and sparse_GP, kern not being imported. [Alan Saul]

* Fixing travis multiprocessing issue. [Alan Saul]

* Cleaning up setup.py. [Alan Saul]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Test fixed. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Alan Saul]

* Added oil test and validation. [Neil Lawrence]

* More merge conflicts. [James Hensman]

* REVERT a53690ab7f5990f51a50afe25fe56edcc25816cc, flapack back substitued in. [Max Zwiessele]

* SCG: printing corrected, dont return before end of method. [Max Zwiessele]

* Merge fixed. [James Hensman]

* Bugfix: sparseGP.likelihood.Z not added to log_ll. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* DeprecationWarning: Substituded all (\!) flapack occ. with lapack (scipy said so) [Max Zwiessele]

* Modified EP code, should be more stable I hope. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Fixed symmetrify for when C/F compiler doesn't work. [Andreas]

* Used scipy.weave to improve the speed of rbf grads. [James Hensman]

  for large number of input dimensions with ARD, wqe have approx tenfold
  speedup.

* Sparse_GP now has a separate predict function. [James Hensman]

  GP and sparse_GP used t share a predict fumction. Since we'd like to
  propagate uncertainty in predictions, sparse_GP.predict needs to accept
  X_new_variance.

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Added max to authors. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Handling printouts without messages. [Max Zwiessele]

* Added empty directoy for cmu mocap data. [James Hensman]

* Mocap dataset automatic download. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Logexp_clipped adjust & mrd error messages. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Bug fix (kernel copy) in mrd. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merged GPLVM, used Andreas changes. [Max Zwiessele]

* Minor changes, dimensionality reduction tests. [Max Zwiessele]

* Logexp_clipped bounding behaviour. [Max Zwiessele]

* Changed likelihood and kernel handling. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Fixed bug in BGPLVM plot. [Teo de Campos]

* Implemented plot_latents as an external function in util. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Teo de Campos]

* Minor changes to make the demo run faster. [Teo de Campos]

* Mew argument in plotting routine. [Max Zwiessele]

* New file. [Ricardo]

* Classification added. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

  Conflicts:
    GPy/models/GP.py

* Plot function got broken with last commit, this fixes it. [Ricardo]

* Think the MANIFEST.in file is required to publish to PyPi. [Alan Saul]

* Completed the automatic mocap dataset fetch from url. [Andreas]

* Pickling for Bayesian_GPLVM simplified. [Max Zwiessele]

* Nosetests do not test expextation of psi_statistics. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Removed the useless print y line. [Teo de Campos]

* Fixed bug in sparse GP plotting. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Sgd. [Nicolo Fusi]

* Modified mrd with MZ. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Changes to MRD test. [Ricardo]

* Sparse_GP can now predict (variationally) from an uncertain input. [James Hensman]

* Bayesian GPLVM can now take either a likelihood or data matrix as first argument. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Allowing EP in BGPLVM and MRD. [Ricardo]

* Added do_test_latents to BGPLVM. [James Hensman]

* Psi_stat tests renamed. [Max Zwiessele]

* Plotting labels are now in order as passed in and marker can be passed with as many markers as there are labels. [Max Zwiessele]

* Catching precision infinity exceptions. [Max Zwiessele]

* Removed logexp_clipped for now. [Max Zwiessele]

* Structural changes for printing. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Teo de Campos]

* Changed optimization constraints in GPy/examples/dimensionality_reduction.py. [Teo de Campos]

* Update to fetch_dataset. [Andreas]

* Removed fisrt prints if display is off. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Automatically fetch datasets and first init. attempt for mocap. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Last stability changes. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Some likelihood.flatten somewhere. [Ricardo]

* Implemented inverse Gamma prior. [James Hensman]

* More tidying in sparse_GP. [James Hensman]

* Tidied the computation of A in sparse_GP. [James Hensman]

* Enabled DSYR on feeble machines. [James Hensman]

  i.e. where numpy is compiled without proper blas linkage

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Cross-terms. [Nicolo Fusi]

* Test. [Nicolo Fusi]

* Minor SGD changes. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Changes in SGD. [Nicolo Fusi]

* SCG printing prettyfied. [Max Zwiessele]

* Merge branch 'mrd' into devel. [Max Zwiessele]

* Last changes on BGPLVM stable version. [Max Zwiessele]

* Newline only on display. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Andreas]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Teo de Campos]

* Readded float variance. [Max Zwiessele]

* Max_bound adjust. [Max Zwiessele]

* Swiss_roll example changes. [Max Zwiessele]

* Catch beta > 0 not needed anymore. [Max Zwiessele]

* Swiss_roll adjustments. [Max Zwiessele]

* Stability enhancing clipping in logexp_clipped and reverse of stability clipping of parameters. [Max Zwiessele]

* Added propper clipping, before overflow happens. [Max Zwiessele]

* Added proper linebeak for SCG printing. [Max Zwiessele]

* Merge branch 'mrd' into devel. [Max Zwiessele]

* Swiss_roll example added, BGPLVM_oil now working. [Max Zwiessele]

* Clipping bounds released to 1e6. [Max Zwiessele]

* Clipping now upper bound adjustable. [Max Zwiessele]

* Added .orig and noseid files to be ignored. [Max Zwiessele]

* Minor formatting changes. [Max Zwiessele]

* WARNING: added parameter clipping for catching infinity problems [p in (-1e300, 1e300)] [Max Zwiessele]

* Printing improved. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Swiss_roll example. [Max Zwiessele]

* Added swissroll example. [Max Zwiessele]

* Making clipping adjustable. [Max Zwiessele]

* Modified:   GPy/models/GPLVM.py Using the following kernel by default: kernel = kern.rbf(Q, ARD=Q>1) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2)) [Teo de Campos]

* Modified:   GPy/util/visualize.py Added the functionality of showing a mosaic of NxN reconstructed images when the size of the number of elements in the feature vector greater than dimensions[0]*dimensions[1]. [Teo de Campos]

* Modified:   GPy/examples/dimensionality_reduction.py brendan_faces(): normalizing the feature vectors w.r.t. the global mean and standard deviation. Changed optimisation constraints because it was never converging. [Teo de Campos]

* Need to commit to resolve a conflict ... [Neil Lawrence]

* Overwrite my changes with James's. [Neil Lawrence]

* Merge changes. [Neil Lawrence]

* Separated out untransform_params, enabling flexibility downstream. [James Hensman]

* Docstrings in SCG. [Andreas]

* Prediction returns Nx1 vectors. [Ricardo]

* Chol_inv added somewhere. [Ricardo]

* Chol_inv added somewhere. [Ricardo]

* Convenient but not important changes. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Changed prod_orthogonal in tests. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Allowed GP models to plot multiple outputs (in 1D at least) [James Hensman]

* Test corrected, by Nicolas. [Ricardo]

* EP case is working fine. [Ricardo]

* Some changes to speed up... just a little. [Ricardo]

* Explanation added to DSYR. [Ricardo]

* Some changes. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Teo de Campos]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Speeding up FITC. [Ricardo]

* GPy/util/visualize.py: fixed conflict. [Teo de Campos]

* Ricardo told me to do this. [Teo de Campos]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Removed RA's profile deco. [James Hensman]

* Generalised backsub_both_sides. [James Hensman]

* Remove profile. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Moved linalg function to GPy.linalg. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolò Fusi]

* Trying to follow changes in likelihood. [Nicolò Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolò Fusi]

* Small changes. [Nicolò Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolò Fusi]

* Better f_inv. [Nicolò Fusi]

* Small change to GPLVM param indexing. [Nicolò Fusi]

* Minor changes to the symmetrify function. [James Hensman]

* DSYR is being used now. [Ricardo]

* Gradients are working now. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Added DSYR for ricardo. [James Hensman]

* Gradietns check :) [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Using weaved symmetrify in pdinv now. [James Hensman]

* Fixed the bug in stick. [James Hensman]

* New functions for EP-matching moments. [Ricardo]

* Minor change. [Ricardo]

* Change in gradients computation. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolas]

* Fixed transformations (Sorry Andreas) [James Hensman]

* Merge conflict in transformations. [James Hensman]

* Minor modifications to the constraint behaviour. [James Hensman]

* Unified framework for addition and product of kernels, with a tensor flag (boolean) instead of   and. [Nicolas]

* MRD updates merge. [Max Zwiessele]

  - termination rule changes for SCG
  - oil flow updates

* Example update to run oil dataset. [Max Zwiessele]

* MRD updates and minor changes. [Max Zwiessele]

* Termination rule update. [Max Zwiessele]

* More changes. [Ricardo]

* Adding FITC to the list. [Ricardo]

* Some changes. [Ricardo]

* Changes to FITC. [Ricardo]

* New FITC... again :( [Ricardo]

* Reverted linalg lapack.flapack for the poor NIPS deadline people. [Alan Saul]

* Image_show() is now able to use a palette to map quantized images to their original color. This uses PIL (import Image). I also enabled the image to be normalised from 0 to 255 in a more robust way (before this, it was assuming images ranged from 0 to 1). [Teo de Campos]

* Merged. [Teo de Campos]

* Merge MRD stability and scg termination rule into devel. [Max Zwiessele]

* Last changes for mrd stability. [Max Zwiessele]

* Scale Factor removed and moved V=Y*beta into likelihoods. [Max Zwiessele]

* BGPLVM MRD Examples and plotting adjustments. [Max Zwiessele]

* Conjugate gradient optimizer without callback (no c.join) [Max Zwiessele]

* SCG convergence tolerance increased, will now converge more easily. [Max Zwiessele]

* Added logexp_clipped transformation. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* New termination rule for scg. [Max Zwiessele]

* Fixed deprecated warning and sense_axes bug. [Teo de Campos]

* Small changes to Brownian. [James Hensman]

* Merging. [James Hensman]

* Some changes according to  the changes in sparse_GP. [Ricardo]

* Broken file fixed. [Ricardo]

* Tried to eliminate the regexp overflow error for large models. [James Hensman]

* Weaved linear kern. [James Hensman]

* Much tidy9ing in sparse_GP. [James Hensman]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

  Conflicts:
    GPy/examples/dimensionality_reduction.py

* Minor changes. [Neil Lawrence]

* Removed unnecessary computaiotn of P in sparse GP. [James Hensman]

* Reverted EP procedure (removed cholupdate) [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* BGPLVM example MATLAB compare. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Last opt updates and tests. [Max Zwiessele]

* Some minor example modifications and cgd adjustments. [Max Zwiessele]

* Various stability working on sparse GP (with MZ) [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Added @testing.deepTest property for skipping tests for deep scan only. [Max Zwiessele]

* Testing updates. [Max Zwiessele]

* Linear psi2 statistics done, all gradients working. [Max Zwiessele]

* Added absolute difference check to gradcheck. [Max Zwiessele]

* Mu to go. [Max Zwiessele]

* Mu to go. [Max Zwiessele]

* Merge devel into mrd > transformations added. [Max Zwiessele]

* Error bars fixed. [Ricardo]

* Auto_scale option for heteroscedastic noise. [Ricardo]

* Correcting linearCF, mu to go. [Max Zwiessele]

* LinearCF Psi Stat not working yet, strange bug in psi computations. [Max Zwiessele]

* Restructuring and merge with devel. [Max Zwiessele]

* Added sample dataset for BGPLVM Matlab comparison. [Max Zwiessele]

* Async optimize working. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Added conjugate gradient descent asunc. [Max Zwiessele]

* Cholesky update for RA. [James Hensman]

* Fixed a bug in all_constrained_indices. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* Typo corrected for negative constrains. [Nicolas]

* Fixed a tie-bug for ND. [James Hensman]

* Added file:transformations. [James Hensman]

* Eigenvalue decomposition of psi2. [James Hensman]

* Merge branch 'devel' into new_constraints. [James Hensman]

  Conflicts:
    GPy/core/model.py
    GPy/core/parameterised.py

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* Minor tidy up of names in visualize (replace histogram with bar chart in lvm_dimselect). [Neil Lawrence]

* Merge branch 'devel' of https://github.com/SheffieldML/GPy into devel. [Neil Lawrence]

* Trying to upgrade numpy version to 1.7.1 as there was an error introduced for weave on 1.7.0 causing tests to fail. [Alan Saul]

* Unification of the visualize object hierarchy and standardization of the click and move behaviour of lvm and lvm_dimselect. Set colours of input sensitivity histogram to red for left (port) and green for right (starboard). [Neil Lawrence]

* Readded parameterized changes. [Max Zwiessele]

* Model re compilation added. [Max Zwiessele]

* Whitespace. [James Hensman]

* Fixed bug in constrain_fixed where soem values weren't deleted. [James Hensman]

* Fixed has_uncertain_inputs weirdness. [James Hensman]

* Improved stability of sparse GP for certain-input case. [James Hensman]

* First attemot at the new constraint framework. [James Hensman]

* Yak shaving. [James Hensman]

* More stabilisation of sparse GP. [James Hensman]

* Removed uncollapsed sparse GP. superceeded by the forthcoming svigp package. [James Hensman]

* Reimplemented caching in prod_orthogonal... [James Hensman]

* Weaved coregionalise. much performance gained. [James Hensman]

* Remo0ved slices from models. [James Hensman]

  slices are now handles by special indexing kern parts, such as
  coregionalisation, independent_outputs. The old slicing functionality
  has been removed simply to clean up the code a little.

  Now that input_slices still exist (and will continue to be useful) in
  kern.py. They do need a little work though, for the psi-statistics

* Cmu_mocap() example mostly working except some fiddling with axes for visualization. Also changes to naming of scaling and offset parameters in GP.py and deal with the case where the scale parameter is zero. [Neil Lawrence]

* Added CMU 35 motion capture data. [Neil Lawrence]

* Fixed two bugs in to_xyz, checked on a test version of MATLAB code. [Neil Lawrence]

* Added first draft of acclaim mocap functionality. [Neil Lawrence]

* James and Nicolos massive Yak shaving session. [James Hensman]

* Manual merging. [James Hensman]

* Tests ignored my nosetests (__test__ = False) [Max Zwiessele]

* Commented out kern tests. [Max Zwiessele]

* Changes pull from devel. [Max Zwiessele]

* Kern psi statistic tests. [Max Zwiessele]

* Added a tdot function (thanks Iain) [James Hensman]

* More minor simplifications. [James Hensman]

* Minor simplifications in dLdK. [James Hensman]

* Old amatplotlib. [Max Zwiessele]

* BGPLVM updates and debug helper. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into mrd. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* One more instance of dpotrs instead of dot in sparse GP. [James Hensman]

* Baysian gplvm and example changes. [Max Zwiessele]

* Rewritten dim_reduction demo to match new style of getters and setters. [Max Zwiessele]

* New getters and setters for self.params, added m['var'] getter and setter. [Max Zwiessele]

* Merge devel into mrd. [Max Zwiessele]

* Pull branch 'devel' of github.com:SheffieldML/GPy into devel. [Max Zwiessele]

* More re-enstating of some preiov commits. [James Hensman]

* Re-enabled a previous bugfix which was lost in a merge. [James Hensman]

* Re-added indepenent_output kern. [James Hensman]

* Added m['ard'] gives all parameters matching 'ard', as well as setting m['ard'] = x to set all mrd parameters. [Max Zwiessele]

* Stupid kern stash merge. [Max Zwiessele]

* Kern conflict. [Max Zwiessele]

* Kern stash conflict. [Max Zwiessele]

* Psi_stat_test stash. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Bugfix: cross term psi1   bias + linear. [Max Zwiessele]

* Psi1 not working (strange transposes) [Max Zwiessele]

* Moved *2. of psi2 statistics into kern and corrected bias+linear cross term. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Psi stat tests done and failing gracefully. [Max Zwiessele]

* Revert "merge devel mrd" [Max Zwiessele]

  This reverts commit 3f625a9347fde47625f14898c0a3a6ed4f49b55a, reversing
  changes made to dc6faeb30355bf9c6f0f3694e8546bcdf26372a8.

* Merge devel mrd. [Max Zwiessele]

* Fixed a weird regular expression bug in ensure_def_constraints. [James Hensman]

* More minor bugs. [James Hensman]

* Fixing small bug in independent outputs kern. [James Hensman]

* Prod_orthogonal now caches the K matrices. [James Hensman]

* Added a kernel for independent outputs. [James Hensman]

* Nparam_transformed work better now. [James Hensman]

  Before, counted the number of fixes, which failed when a fix fixed more
  than one parameter...

* GPy now fails silently if sympy is not present. [James Hensman]

* Made the basic GP class use dtrtrs where possible. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [James Hensman]

* A litle more stability in svigp. [James Hensman]

  Another instance of dpotrs instead of dot

* Merge branch 'mrd' of github.com:SheffieldML/GPy into mrd. [Max Zwiessele]

* Xticklabels improved. [Max Zwiessele]

* Psi stat tests. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Pdinv passes extra args to jitchol. [andreas]

* Demo changed, bgplvm still broken. [Max Zwiessele]

* BGPLVM still failing, doesn't seem to be numerical : ( [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Small changes. [Nicolo Fusi]

* Merge branch 'mrd' into devel. [Nicolo Fusi]

* Removed useless _set_params() [Nicolo Fusi]

* Merge branch 'mrd' into devel. [Nicolo Fusi]

* Small changes. [Nicolo Fusi]

* Fixed psi0 psi1 renaming error. [Nicolo Fusi]

* Added bgplvm_simulation on same simulation. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Merge branch 'mrd' into devel. [Max Zwiessele]

* Comments only. [James Hensman]

* Improved stability of sparse GP computations. [James Hensman]

  Specifically in computing dL_dKmm

* Rbf computation of psi2 now works if there's only one datum. [James Hensman]

* Reverting last change. [Ricardo]

* More manual merging. [James Hensman]

* Manual merge in plot_latent. [James Hensman]

* Auto_scale. [Ricardo Andrade]

* More fun with vizualize. [James Hensman]

* More work on visualize. [James Hensman]

* Some simplification of the psi2_statistics in rbf. [James Hensman]

* Fixed Browninan motion kern (doesnt seem to have a unittest?) [James Hensman]

* Simulation data changes. [Max Zwiessele]

* Adjusted plotting behaviour in X1d. [Max Zwiessele]

* Changed printing behaviour in cholesky to kill last line. [Max Zwiessele]

* New gradcheck for more stability in mrd_tests. [Max Zwiessele]

* Example change mrd. [Max Zwiessele]

* Plotting debug. [Max Zwiessele]

* New functions mrd init_X update. [Max Zwiessele]

* Merge branch 'devel' into mrd. [Max Zwiessele]

* Fixed merge conflict on BPGLVM. [Nicolo Fusi]

* Minor changes. [Nicolo Fusi]

* Added automatic scale_factor to sparse GPs. [Nicolo Fusi]

* --march=native was causing problems on the stupid compiler on MacOS. [Nicolo Fusi]

* Now returning the ax for plot_latent in BGPLVM. [Nicolo Fusi]

* Bounds added. [Max Zwiessele]

* Added debug plot. [Max Zwiessele]

* Readded mrd plotting changes. [Max Zwiessele]

* Finished mrd and added plotting functions. [Max Zwiessele]

* Plot_latent added for mrd. [Max Zwiessele]

* Mrd example added. [Max Zwiessele]

* Some minor improvements in visualize. [James Hensman]

* Merged master back into devel (to sync bugfixes) [Nicolo Fusi]

* Merge branch 'master' into devel. [Nicolo Fusi]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Nicolo Fusi]

* Pdinv now uses dpotri instead of dtrtri and dot. [James Hensman]

* OMP for psi2 computations in RBF. [Nicolo Fusi]

* Improved weaving. [James Hensman]

* Weaved some rbf code. [James Hensman]

* Kern plotting with axisa. [Max Zwiessele]

* GPLVM readded. [Max Zwiessele]

* GPLVM merge?? [Max Zwiessele]

* Merged local branch. [Nicolo Fusi]

* Merge branch 'new_warping' [Nicolo Fusi]

* Merge branch 'master' into new_warping. [Nicolo Fusi]

* Changed prediction code. [Nicolo Fusi]

* Merge branch 'devel' into new_warping. [Nicolo Fusi]

* Merge branch 'devel' into new_warping. [Nicolo Fusi]

* Merge branch 'master' into new_warping. [Nicolo Fusi]

* Changed version. [Nicolo Fusi]

* Testing priors in the demo. [Nicolo Fusi]

* Moved randomize() in a more proper place. [Nicolo Fusi]

* Rebased from master in older to get all the goodies. [Nicolo Fusi]

* Fixed _get_param_names. [Nicolò Fusi]

* Merged master. [Nicolò Fusi]

* Minor. [Nicolo Fusi]

* Merge branch 'master' into new_warping. [Nicolo Fusi]

* Added a term to warping function. [Nicolo Fusi]

* Big merge. [Nicolo Fusi]

* Changed version number in setup.py. [James Hensman]

* Changed version number in setup.py. [James Hensman]

* Merge branch 'devel' [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Added (optional) iter param dump. [Nicolo Fusi]

* Fixed small bug in SGD. [Nicolo Fusi]

* Made BGPLVM oil flow demo work, added ARD weights plot. [Nicolo Fusi]

* Added BGPLVM oil flow demo and changed default X_variance init. [Nicolo Fusi]

* Changes in GPLVM plotting. [Nicolò Fusi]

* Made parallel optimize_restart responsive to ctrl+c. [Nicolo Fusi]

* Mrd touches. [Max Zwiessele]

* Merge remote-tracking branch 'origin' into mrd. [Max Zwiessele]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Bug fixed in periodic kernels: Warning were not handled properly. [Nicolas]

* Added testing to modules. [Alan Saul]

* First trivial model touches. [Max Zwiessele]

* Stability improvements in sparse_GP. [James Hensman]

* Changed X_uncertainty for X_variance (in the code) for consistency with actual naming (in the printing) [James Hensman]

* Various work on BGPLVM oil demo. [James Hensman]

* Added simple BGPLVM_oil demo. [James Hensman]

* Merge branch 'devel' of github.com:SheffieldML/GPy into devel. [Ricardo Andrade]

* Yak shaving. [James Hensman]

* Added the rbfcos kernel. [James Hensman]

  ARD seems to work

  dK_dX still todo

* Edit to linalg.py PCA function to stop it changing data matrix. [Neil Lawrence]

* Minor modifications to visualization routines and examples. [Neil Lawrence]

* Further edits on visualization code for faces example. [Neil Lawrence]

* Added base implementation of data visualization framework for use with GP-LVM. [Neil Lawrence]

* Added mocap.py for loading in motion capture data. [Neil Lawrence]

* Changes in FITC approximation computation. [Ricardo Andrade]

* Merge branch 'fitc' into devel. [Ricardo Andrade]

* Not needed. [Ricardo Andrade]

* Small changes. [Ricardo Andrade]

* Merge branch 'em_fix' into fitc. [Ricardo Andrade]

* Print iteration number. [Ricardo Andrade]

* Minor changes. [Ricardo Andrade]

* Merge branch 'master' into fitc. [Ricardo Andrade]

* Generalized fitc + examples. [Ricardo Andrade]

* Generalized FITC is back. [Ricardo Andrade]

* Small efficiency changes in rbf. [James Hensman]

* Rbf now works in a more memory friendly fashion. [James Hensman]

* Merge branch 'master' into devel. [James Hensman]

* Insignificant but annoying bug corrected. [Ricardo Andrade]

* Pseudo EM algorithm for EP and maybe Laplace. [Ricardo Andrade]

* Merge branch 'master' into devel. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* More fixing of the predictive variance (correct for full_cov now) [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Corrected the predictive variance for Gaussian likelihoods. [James Hensman]

* Linear is now by default non-ARD. [Nicolo Fusi]

* Added fixed effect kernel. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Merge branch 'debug' [James Hensman]

* Increased stability of _compuations in sparse_GP. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* The warnings are now handeled properly in the periodic kernels. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Errors handled in Mat32. [Nicolas]

* Changed version. [Nicolo Fusi]

* Decorator documentation. [Nicolo Fusi]

* Now it actually works. [Nicolo Fusi]

* Added decorator to silence errors. [Nicolo Fusi]

* Ensure_default_constraints() now also works with the BGPLVM. [Nicolo Fusi]

* Added CI status. [Nicolò Fusi]

* Better GPLVM oil flow demo. [Nicolo Fusi]

* Made SCG work nicely with the optimization framework. [Nicolo Fusi]

* Fixed trace_dot to be a litle faster... [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Non working integratino of SCG into GPy. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Fixed bug in dK_dX for the quadratic kernel. [Nicolas]

* Added SCG code. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Updated list of implemented kernels in the documentation. [Nicolas]

* Updated list of implemented kernels in the documentation. [Nicolas]

* Typo in comments. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* More messing with the linear algebra in sparse_GP. [James Hensman]

* Some messing with the linear algebra in sparse_GP. This should be more efficient... let's hope nothing breaks. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Added trace_sum for efficiency. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

* Merge branch 'master' of github.com:SheffieldML/GPy. [andreas]

* Tie_param changed to tie_params in tutorials. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Few bugs fixed in the documentation. [Nicolas]

* Fixed plots for BGPLVM. [andreas]

* Tutorial finished. [Ricardo Andrade]

* Deactivated test_models() [Nicolo Fusi]

* T push :qMerge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Fixed checkgrad test to randomize before checking. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Got rid of foo. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Trying to shuffle. [Alan Saul]

* Should now test all (although upon error it stops trying to generate any more) [Alan Saul]

* Add example test generation. [Alan Saul]

* Added test generator (not quite finished yet) [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Changed example tests. [Alan Saul]

* Fixed bug in RBF, added inducing inputs to BGPLVM plots. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

* Example fixed. [Ricardo Andrade]

* Update in the rational quadratic kernel and new the tutorial on writting kernels. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* New rational quadratic kernel. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Small changes. [Nicolas]

* Added plot_latent to GPLVM. [James Hensman]

* Changed the filename from BGPLVM to Bayesian_GPLVM to tidy the namespace a little. [James Hensman]

* Changes tie_param to tie_params. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

* Setup.py requires nose now. [James Hensman]

* Attempted to make sparse models more stable through ordered multiplication. [James Hensman]

* Temporarily removed a test (linear X bias) [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Implemented psi2 'cross terms' for rbfXbias. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* A small demo for model tutorial. [James Hensman]

* Examples are working. [Ricardo Andrade]

* Examples working. [Ricardo Andrade]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

  Conflicts:
    GPy/examples/__init__.py

* Working on psi cross terms. [Nicolo Fusi]

* Skipping tests. [Alan Saul]

* Adding testing file for examples. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Example files for tutorials are now in Neil's format. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Added init. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Removed unused partial1. [Alan Saul]

* Update in the documentation on kernel implementation. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Now running nosetest doesn't run unittests twice. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Added GPy.tests(), removed some useless tests. [Nicolo Fusi]

* Added in documentation the current status of kernel implementation. [Nicolas]

* Americanized spellings. [James Hensman]

* Fixed merge conflicts. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Added the outline of a tutorial on 'interacting with models' [James Hensman]

* Skipping a test known to fail (linear sparse) [James Hensman]

* Added sparse_gplvm_tests -- they fail. [James Hensman]

* Added simple gplvm_tests. [James Hensman]

* FIxed a transpose bug in sparse_GPLVM. [James Hensman]

* Deprecated flapack, namespace changed to lapack.flapack. [Alan Saul]

* Removed keyname partial. [Alan Saul]

* Examples directory organized. [Ricardo Andrade]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

* Draft of documentation for implemented kernels. [Nicolas]

* Draft of documentation for implemented kernels. [Nicolas]

* Draft of documentation for implemented kernels. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Sometidying of the psi statistic  cross terms. [James Hensman]

* Removed log_likelihood_gradients_transformed, now everything is done in the objective functions. [Nicolo Fusi]

* Added tutorial in examples. [Nicolas]

* All the product_orthogonal have been changed to prod_orthogonal for consistency. [Nicolas]

* Merge branch 'fixEP' [Ricardo Andrade]

* Generalized_FITC removed. [Ricardo Andrade]

* Removed generalized_FITC.py. [Ricardo Andrade]

* Irrelevant changes. [Ricardo Andrade]

* Test for EP_DTC added. [Ricardo Andrade]

* Plotting functions for sparse_GP added. [Ricardo Andrade]

* Plot function moved to GP model. [Ricardo Andrade]

* Some small changes. [Ricardo Andrade]

* Merge branch 'master' of github.com:SheffieldML/GPy into genFITC. [Ricardo Andrade]

* Merge branch 'master' of github.com:SheffieldML/GPy into genFITC. [Ricardo Andrade]

* Sparse GP with EP is working now. [Ricardo Andrade]

* Coregionalisation seems to be a go-go. [James Hensman]

* Some changes to product_orthogonal. [James Hensman]

  dKdiag_dX is now implemented, some of the cod eis a little tidier

* Debugging the coregionalisation kern. [James Hensman]

* More messing around with coregionalize. [James Hensman]

* JH bugfix for slices. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Using setuptools instead of distutils. [Max Zwiessele]

* First attempt at making coregionalise work with the sparse model. [James Hensman]

  Gradients are failing! have implemented prod_othogonal.dKdiag_dtheta

* Yak shaving. [James Hensman]

* Added unit test for coregionalisation. [James Hensman]

* Coregionalisation. [James Hensman]

* Added symmtrical covariance functions. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Added dKdiag_dtheta for linear. [James Hensman]

* Indentation for dKdiag_dtheta fixed. [Nicolas]

* Added dKdiag_dtheta for the periodic kernels. [Nicolas]

* Minor changes for het. noise and uncertin inuputs. [James Hensman]

* Re-enstated compute_kernel_matrices. [James Hensman]

* Removed unnecessary computation of psi2. [James Hensman]

* Fixed levels in GP.plot. [James Hensman]

* Added optional number of contour levels to the 2D plotting in GP.plot. [James Hensman]

* Some commentary on Neil's notes.txt. [James Hensman]

* Effiiency improvements in sparse_GP. [James Hensman]

  the recasting of derivatives through psi2 into psi1 is now only done in
  one place

* Start of psi2 crossterms. [James Hensman]

* Moved randomize() in a more proper place. [Nicolo Fusi]

* Optimize_restarts() is now parallel (load-balanced). It also mantains compatibility with the verbose and robust options. [Nicolò Fusi]

* Added unit test for param tie at the kernel level. [Nicolò Fusi]

* Sparse-GPLVM now seems to work beautifully with product kernels. [Nicolas]

* Some bugfixes that have affected GPLVM/sparseGPLVM since the hetero noise change. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Tutorial improved (and finished) [Nicolas]

* Tutorial improved (and finished) [Nicolas]

* Tutorial improved. [Nicolas]

* Tutorial improved. [Nicolas]

* Tutorial improved. [Nicolas]

* Tutorial improved. [Nicolas]

* Tutorial update due to some changes in GPy. [Nicolas]

* Tutorial update due to some changes in GPy. [Nicolas]

* The product by default is now on the same space. [Nicolas]

* Added the product of kernels defined on the same space + a few bugs fixed in the prod_orthogonal. [Nicolas]

* New features in the product_orthogonal of kernels. [Nicolas]

* The 2D plot can now handle  *args, **kwargs. [Nicolas]

* Small changes to the lengthscales such that the periodic kernels can be called as the non-periodic ones. [Nicolas]

* Fixed the bug where we couldn't tie parameters at the kern level. [James Hensman]

* Added target_param option to checkgrad(), removed unused function parameter. [Nicolò Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolò Fusi]

* Made the name of the Gaussian noise variance noise_variance, for consistency. [James Hensman]

* Changes to the uncollapsed GP. [James Hensman]

* Bugfixin' [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Added a set_data method to the Gaussian likelihood. [James Hensman]

* Pretty-printing of objective function. [Nicolò Fusi]

* Added contribution from the prior to marginal LL printed in the model __str__ [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Fixed a bug in sparse GP relating to the psi representation. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Reinsert the plot function for kernel that diserpered at one point... [Nicolas]

* Re-enables uncollapsed GP. [James Hensman]

* Fixed inconsistent naming of parameters in LVM models. [Nicolo Fusi]

* Addedd BGPLVM unit tests. [Nicolo Fusi]

* All kernels working fine with the psi statistics now. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Merge branch 'debug_bias' [James Hensman]

* Fixed bug with the bias kernel. [James Hensman]

* Debugging BGPLVM. [Nicolo Fusi]

* Some minor changes in SGD. [Nicolo Fusi]

* Efficient handling of Y and YYT. [Nicolo Fusi]

* Fixed import error. [Nicolo Fusi]

* Added KL term to BGPLVM. [Nicolo Fusi]

* Working on SGD merge. [Nicolo Fusi]

* Trying to fix the likelihood.Y madness. [Nicolo Fusi]

* Merged with master. [Nicolo Fusi]

* Added a default kernel option in BGPLVM. [James Hensman]

* Merged. [Nicolo Fusi]

* Added ANOVA kernel print output. [Alan Saul]

* Removed ipython code from tuto. [Alan Saul]

* Testing ipython on rtd. [Alan Saul]

* Cleaning up. [Alan Saul]

* More debugging. [Alan Saul]

* Debugging finding matplotlib... [Alan Saul]

* Cant install with pip. [Alan Saul]

* Try installing with pip? eek... [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Few changes to tutorial bis. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Few changes to tutorial. [Nicolas]

* New tutorial draft called 'A kernel overview' [Nicolas]

* Trying to get plotting working. [Alan Saul]

* About to exchange sphinxext. [Alan Saul]

* Added matplotlib test, probably won't work. [Alan Saul]

* Fixed typo. [Alan Saul]

* Typo. [Alan Saul]

* Okay definietely no paths adding... lets see what is required for ipython. [Alan Saul]

* Added ipython test code back and extensions loading. [Alan Saul]

* Added a path back. [Alan Saul]

* Added ipython to setup again and went back to numpy.distribute. [Alan Saul]

* No path inset or append. [Alan Saul]

* Removed api make from makefile (although maybe it belongs there? [Alan Saul]

* Reverted model.py. [Alan Saul]

* Fixed the fecking subpackage... [Alan Saul]

* Try changing the setup... [Alan Saul]

* More api doc hacking. [Alan Saul]

* Now hacking makefile..: [Alan Saul]

* With insert. [Alan Saul]

* Changed path back, think we're closer when its in. [Alan Saul]

* Appended path. [Alan Saul]

* Removed all paths adding. [Alan Saul]

* Changed paths and model.py. [Alan Saul]

* Added paths back. [Alan Saul]

* Changed path down a level. [Alan Saul]

* Added debuggin, and interestingly likelihood rst.. shouldnt make a difference: [Alan Saul]

* More path debugging. [Alan Saul]

* Added normpath... and debugging. [Alan Saul]

* Revert. [Alan Saul]

* Again playing with relative to absolute paths, test. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

* Broken file removed until new notice. [Ricardo Andrade]

* Make paths more relative. [Alan Saul]

* Back to original. [Alan Saul]

* Just testing an absolute import. [Alan Saul]

* Adding more paths...: [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Ricardo Andrade]

* Bug found and fixed in plots for normalized X. [Ricardo Andrade]

* Moved GPy import. [Alan Saul]

* Removed path again. [Alan Saul]

* Added GPy module to path. [Alan Saul]

* Changed path to append it rather than inset it. [Alan Saul]

* Added root path back. [Alan Saul]

* Removed sphinx from setup. [Alan Saul]

* Reverting setup.py. [Alan Saul]

* Debugging in make.bat. [Alan Saul]

* Removed import of builddoc. [Alan Saul]

* Got rid of ipython example for a sec. [Alan Saul]

* Silly me copying and pasting. [Alan Saul]

* Force up to date version of sphinx. [Alan Saul]

* Force up to date version of sphinx. [Alan Saul]

* Trying to rid buildsphinx error. [Alan Saul]

* Removing GPy import from conf (may put back in) [Alan Saul]

* Added ordering of methods. [Alan Saul]

* Fixed some broken imports (likelihoods has moved), remember to tell everyone to delete their pyc file. [Alan Saul]

* Removed version check. [Alan Saul]

* Adding requirements removing mock.py. [Alan Saul]

* More docs. [Alan Saul]

* More docs testing. [Alan Saul]

* Adding ipython requirements (temporary) and removing unnecessary mock requirement. [Alan Saul]

* Adding ipython extensions. [Alan Saul]

* Taking another bash at executable docs... in vain. [Alan Saul]

* Small changes in tutorial. [Nicolas]

* Solving merge conflicts. [Nicolas]

* Probit likelihood modified for plotting. [Ricardo Andrade]

* Change in plot() y-limits. [Ricardo Andrade]

* Example is working. [Ricardo Andrade]

* Modifications made to tutorial due to changes in GPy. [Nicolas]

* Added new plotting function for kernels. [Nicolas]

* Fixed bug in the product of kernels with tied parameters. [Nicolas]

* Few more fix to the plotings and predictions. [Nicolas]

* Small fixes to ploting. [Nicolas]

* Many modifications in GP plots to make it work. [Nicolas]

* Tidying up after wide reaching changes: removed sparse_GP_old.py. [James Hensman]

* Various merge conflicts from the newGP branch. [James Hensman]

* Merge branch 'newGP' [James Hensman]

  Conflicts:
    GPy/models/GP_regression.py

* Simplified the checkgrad logic somewhat. [James Hensman]

* Proper propagation of variance through the Gaussian likelihood. [James Hensman]

* Var[:,None] added in full_cov = false, sparse_GP. [Ricardo Andrade]

* Fixed bug in my schoolboy mathematics. [James Hensman]

* Partial derivatives for the new likelihood framework. [James Hensman]

* Merge branch 'newGP' of github.com:SheffieldML/GPy into newGP. [Ricardo Andrade]

* Made the BGPLVM work in the new world order. [James Hensman]

* Changes in plotting functions. [Ricardo Andrade]

* Merge branch 'newGP' of github.com:SheffieldML/GPy into newGP. [James Hensman]

* Classification examples corrected (2/3) [Ricardo Andrade]

* Merge branch 'newGP' of github.com:SheffieldML/GPy into newGP. [Ricardo Andrade]

* EPEM is running. [Ricardo Andrade]

* Assorted work on combining the EP and sparse methods. [James Hensman]

* Merge branch 'newGP' of github.com:SheffieldML/GPy into newGP. [James Hensman]

  Conflicts:
    GPy/likelihoods/EP.py

* James' debugging of the EP/GP interface. [Ricardo Andrade]

  It seems that the GP-EP algorithm works now.

* Merged changes in likelihood_functions (James) [Ricardo Andrade]

* Merge branch 'newGP' of github.com:SheffieldML/GPy into newGP. [Ricardo Andrade]

  Conflicts:
    GPy/likelihoods/EP.py
    GPy/likelihoods/likelihood_functions.py

* So many changes. [Ricardo Andrade]

* Merge branch 'newGP' of github.com:SheffieldML/GPy into newGP. [Ricardo Andrade]

* Predictive_values implemented in EP. [Ricardo Andrade]

* Beginning of work to make sparse GP ork with RA's EP methods. [James Hensman]

* Added a likelihood atom class. [James Hensman]

  and also some import tidying in the EP.py file

* Some tidying in the likelihood classes. [James Hensman]

* Merged conflict. [James Hensman]

* Re-indented the plot function. [James Hensman]

* Predictive_mean changed to predictive_values. [Ricardo Andrade]

* Very basic functionality is now working. [James Hensman]

* Added a Gaussian likelihood class. [James Hensman]

* Merge branch 'newGP' of github.com:SheffieldML/GPy into newGP. [James Hensman]

* Changed docs back for newGP. [Alan Saul]

* Merge branch 'newGP' of github.com:SheffieldML/GPy into newGP. [Alan Saul]

* Trying to fix docs, might break them. [Alan Saul]

* Massive restructuting to make the EP likelihoods work consistently. [James Hensman]

* Much tidying and breakage in the GP class. [James Hensman]

* Merged conflicts after merging in master to newGP branch. [James Hensman]

* Merge remote-tracking branch 'Falkor/newGP' into newGP. [Ricardo Andrade]

* Working for regression, still some bugs for EP. [Ricardo Andrade]

* EP plots samples now for the phi transformation. [Ricardo Andrade]

* Log-likelihood,predictions and plotting are working. [Ricardo]

* More changes. [Ricardo Andrade]

* Merge remote-tracking branch 'Falkor/newGP' into newGP. [Ricardo Andrade]

* Minor changes. [Ricardo Andrade]

* _compute_GP_variables. [Ricardo Andrade]

* Assertions included. [Ricardo Andrade]

* Now it works. [Ricardo Andrade]

* Merge remote-tracking branch 'Falkor/newGP' into newGP. [Ricardo Andrade]

* Minor change in EM explanation. [Ricardo Andrade]

* Merge branch 'newGP' [Ricardo Andrade]

* Merge remote-tracking branch 'Falkor/newGP' into newGP. [Ricardo Andrade]

* Other change. [Ricardo Andrade]

* Test file. [Ricardo Andrade]

* GP model works now. [Ricardo Andrade]

* EM algorithm for EP. [Ricardo Andrade]

* EM algorithm. [Ricardo Andrade]

* Other changes. [Ricardo Andrade]

* Merge remote-tracking branch 'Falkor/newGP' into newGP. [Ricardo Andrade]

* Sparse EP. [Ricardo]

* Merge branch 'master' into newGP. [Ricardo Andrade]

* No more GP_EP stuff. [Ricardo Andrade]

* Fixing EP and merging it with GP_regression. [Ricardo Andrade]

* Fixing GP_EP. [Ricardo]

* Fixed bug in the product of kernels. [Nicolas]

* Fixed small bug in m.plot() when samples are shown. [Nicolas]

* Latex in doc is now beautiful. [Nicolas]

* Trying to fix doc 2. [Nicolas]

* Trying to fix doc. [Nicolas]

* Small fixes in the kernel documentation. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Small changes in the way covariance functions handle lengthscale as input. [Nicolas]

* Working on linear kernel. [Nicolò Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Reverted back to working docs. [Alan Saul]

* Insert to beginning of path. [Alan Saul]

* Moved mock below extension loading. [Alan Saul]

* Changed docs. [Alan Saul]

* Changed to matplotlib sphinxext. [Alan Saul]

* Added sphinxext package to ipython_directive. [Alan Saul]

* Still working? [Alan Saul]

* Still working? [Alan Saul]

* Checking before any extensions. [Alan Saul]

* Added other ipython directive extension. [Alan Saul]

* Trying ipython extension instead. [Alan Saul]

* More debugging. [Alan Saul]

* Added some debugging statements. [Alan Saul]

* Removed soem more extensions. [Alan Saul]

* Removed some mocks. [Alan Saul]

* Added plot_directive and mathmpl extensions. [Alan Saul]

* Got rid of some extensions we're not sure we're using. [Alan Saul]

* Added extensions for inline doc plotting. [Alan Saul]

* Asf. [Alan Saul]

* Added sympy.parsing. [Alan Saul]

* Added sympy.core. [Alan Saul]

* Added sympy.core.cache mock...... [Alan Saul]

* Sympy.utilities.codegen. [Alan Saul]

* Mocked sympy.utilities. [Alan Saul]

* Mocked sympy aswell... [Alan Saul]

* Changed mock. [Alan Saul]

* Changed mock back. [Alan Saul]

* Changed default ARD setting in linear. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Added path higher. [Alan Saul]

* Moved import a bit. [Alan Saul]

* Moved mock into docs. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Added mock file. [Alan Saul]

* Added mock to dependencies for docs. [Alan Saul]

* "fixed" Tango imports. [Nicolo Fusi]

* Changed travis conf. [Nicolo Fusi]

* Useless commit to get travis-ci started. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* More. [Alan Saul]

* Testing differen mock. [Alan Saul]

* Testing differen mock. [Alan Saul]

* Added matplotlib color. [Alan Saul]

* Moved mock to top. [Alan Saul]

* Added matplotlib to mock. [Alan Saul]

* Add just pylab mock back. [Alan Saul]

* Back to the beginning? [Alan Saul]

* More. [Alan Saul]

* More. [Alan Saul]

* Conf edit. [Alan Saul]

* Adding requirements file? [Alan Saul]

* Remove matplotlib mock? [Alan Saul]

* More... [Alan Saul]

* More. [Alan Saul]

* Importing mock better. [Alan Saul]

* More fixing... [Alan Saul]

* More attempts at mocking. [Alan Saul]

* Added some more mocks. [Alan Saul]

* Removed matplotlib mock. [Alan Saul]

* Adding extra mock... hopefully this won't carry on. [Alan Saul]

* Forgot exceptions import. [Alan Saul]

* Above again.... [Alan Saul]

* Same again. [Alan Saul]

* Adding pylab mock module. [Alan Saul]

* Attempting to fix docs but may break them. [Alan Saul]

* Making travis-ci work again. [Nicolo Fusi]

* Trying to fix bugs in kerns. [Nicolo Fusi]

* Just some rearranging. [Nicolo Fusi]

* Added centering and fixed serious bug. [Nicolo Fusi]

* Psi statistics working for linear ARD kernel. [Nicolò Fusi]

* Precomputations for linear psi statistics. [Nicolo Fusi]

* Psi statistics for the linear kernel. [Nicolo Fusi]

* Added support for sparse matrices. [Nicolo Fusi]

* Pretty much the version running on ec2. [Nicolo Fusi]

* Sometimes a step with missing data can be a bit unstable. [Nicolo Fusi]

* Made it clear that we are working with -f(x) [Nicolo Fusi]

* Fixed bug in SGD. [Nicolo Fusi]

* Minor changes. [Nicolo Fusi]

* Merge branch 'master' into SGD. [Nicolo Fusi]

* Convenience change in linear.py. [Nicolo Fusi]

* Made SGD work with new get/set param. [Nicolo Fusi]

* Merge branch 'master' into SGD. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Robustified the assertion re lengthscales in rbf. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Fixed bug in rbf.py, removed the ARD moniker from the name variable. [James Hensman]

* Images for tutorials. [Nicolas]

* Rst files from documentation. [Nicolas]

* Improved tutorial for GP_regression. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Added missing dataset from mlprojects. [James Hensman]

* Missing file product_orthogonal from previous commit. [Nicolas]

* Solved merge conflict. [Nicolas]

* Added datasets (from GPY_assembla) [James Hensman]

  ... and removed a nasty hard link in the examples file

* New operator: the kernels can be multiplied directly with the '*' character. [Nicolas]

* Some more documentation documentation on the index page. [Nicolas]

* Doc style change. [Nicolo Fusi]

* Merged master. [Nicolo Fusi]

* First broken port of the psi stats to the linear kernel. [Nicolo Fusi]

* RBF (both ARD and non-ARD) kernels working nicely with psi statistics. [Nicolo Fusi]

* Added BGPLVM demo (not working yet) [Nicolo Fusi]

* Now skipping FITC test. [Nicolo Fusi]

* Integrated sparse GP regression and BGPLVM classes. [Nicolo Fusi]

* Removed imports from __init__.py. [Nicolo Fusi]

* Massive merge of the debug branch. [Nicolo Fusi]

* Trying to get psi2 cross terms to work. [Nicolo Fusi]

* Added links to readme. [Nicolo Fusi]

* Added links to readme. [Nicolo Fusi]

* Linear kernel now has an ARD flag. [Nicolas]

* Added unit tests for priors. [Nicolò Fusi]

* Some love for the priors class. [Nicolò Fusi]

* Untabified priors.py. [Nicolò Fusi]

* Merge branch 'periodic' [Nicolas]

* Few bugs fixed in periodic kernels. [Nicolas]

* Added some more files for periodic kernels. [Nicolas]

* Added periodic kernels. [Nicolas]

* Missing scale and location arguments. [Ricardo]

* Golden serach and Simpson's rule explained. [Ricardo]

* Test. [Ricardo Andrade]

* Deleted line. [Ricardo Andrade]

* TODO added. [Ricardo Andrade]

* Merge remote-tracking branch 'rick70x7/master' [Ricardo Andrade]

* Poisson and Gaussian likelihood. [Ricardo Andrade]

* Poisson likelihood. [Ricardo Andrade]

* __init__.py for Examples directory (see comments in code). [Neil Lawrence]

* __init__.py for Examples directory (see comments in code). [Neil Lawrence]

* Added some documentation and example files. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Added path for RTD. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Updated index.rst. [Alan Saul]

* Restored examples folder. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Merged trivial conflict. [James Hensman]

* Fixed optimize_restarts. [James Hensman]

* Delete unnecessary rbf_ARD.py. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Added sympy dependency and scipy version dependency again. [Alan Saul]

* Trying to give pylab dependency. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Added unsupervised.py examples file and datasets.py@ [Neil Lawrence]

* Added pylab to requirements. [Alan Saul]

* The below again. [Alan Saul]

* Tried fixing build call. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Tried changing the location of the apidoc compilation. [Alan Saul]

* Bug fixed in example (in regression.py) [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Adding old command I read on the internet for own build. [Alan Saul]

* Adding fake kernel to test docs. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Testing docs more. [Alan Saul]

* More tests in unit_tests. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Trying to get ReadTheDocs to recompile for us. [Alan Saul]

* Last ARD flag changes to kernels. [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* Mods to regression.py now that we have get to get parameters. Moved Youter to YYT. [Neil Lawrence]

* More ARD flags in exponential and Matern32. [Nicolas]

* Added ARD flag to exponential. [Nicolas]

* Change in unit_test to take into account the ARD changes in rbf. [Nicolas]

* Solved conflicts for rbf kernel. [Nicolas]

* Modified log_like_gradients to make it _log_like_gradients and moved extract_gradient to _log_like_gradients_transformed. [Neil Lawrence]

* Fix error introduced into GP_regression when doing name changes. [Neil Lawrence]

* Merge branch 'master' of https://github.com/SheffieldML/GPy. [Neil Lawrence]

* Removed version specification of scipy. [Alan Saul]

* Removed latent force model fortran code from setup.py from github code. [Alan Saul]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Alan Saul]

* Added apt-get scipy installation for travis, need to ensure version. [Alan Saul]

* Expand_param and extract_param replaced with set_params_transformed and get_params_transformed. [Neil Lawrence]

* Merge branch 'master' of https://github.com/SheffieldML/GPy. [Neil Lawrence]

* Fixed version setting for numpy scipy installatioN. [Alan Saul]

* Changing travis installation. [Alan Saul]

* Adding travis.yml file for Travis continuous intregration service, may not work yet. [Alan Saul]

* Changed get_param and set_param to _get_params and _set_params. [Neil Lawrence]

* Rbf kernel now has an ARD flag. [Nicolas]

* Fixed up dK_dX in the exponential and Matern kerns. [James Hensman]

* Minor changes. [Nicolo Fusi]

* Merge branch 'bgplvm' into SGD. [Nicolo Fusi]

* Working on cross terms. [Nicolo Fusi]

* Broken commit, working on cross terms for psi statistics. [Nicolo Fusi]

* New shape for psi2. [Nicolo Fusi]

* BGPLVM working. [Nicolo Fusi]

* BGPLVM working with rbf+white. [Nicolo Fusi]

* Decent gradients for most parameters. [Nicolo Fusi]

* Minor changes to the apsre regression demo. [James Hensman]

* Various hackday stuff, including scale factor in sparse GP. [James Hensman]

* Fixed optimize_restarts. [James Hensman]

* Scale factor added to sparse_GP_regression. [James Hensman]

  and sparse_GP_demo ammended to be less annoying (m1)

* General changes to bebugging code. [James Hensman]

* Fixed up dK_dX in the exponential and Matern kerns. [James Hensman]

* Simplified the debug classes. [James Hensman]

* Changed the colous of plotting in the grid_parameters debug script. [James Hensman]

* Plotting changes. [James Hensman]

* Allowed the gradchecker to return the gradient ratio. [James Hensman]

  Just to help with debugging.

* Some code to debug the sprase GP gradients with. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy into debug. [James Hensman]

* Parameters gridding with checkgrad to aid debugging. [James Hensman]

* Fixed index. [Nicolo Fusi]

* Merge branch 'master' into SGD. [Nicolo Fusi]

* Pretty printing of gradchecks. [Nicolo Fusi]

* Removed ticks and checkmarks from checkgrad() output, coloring param name instead. [Nicolo Fusi]

* Removed unused posix import. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Fixed a NF induced bug in the guts of GPy. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Sphinx configuratino for readthedocs.org. [James Hensman]

* Removed some redundant looping in kern.py. [James Hensman]

* GP_regression and sparse_GP_regression now only return the full posterior covariance matrix when requested. [James Hensman]

* Forced simplification of sympy expressions before converting to c++ [James Hensman]

* Changed the behaviour of checkgrad. [James Hensman]

  verbose now works as (I) expected. discussion welcome

* Added an ARD option to the sympy RBF kern. [James Hensman]

* Reduced the memory requirements of the sparse GP. [James Hensman]

  by a factor of M!

* Added a constructor for a generic sympy kernel. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Made sympykern truly work in place. [James Hensman]

* Removed dL_dZ from sympykern. [James Hensman]

  (it's not needed, we can always use dK_dX)

* Added demo for uncollapsed GP. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Added Alan's bugfix to this version of GPy: [James Hensman]

  sympykern is now forced to recompile if the function changes.

  Also re-enabled openmp loops, since I only diabled them for bugfinding

* Added iterate.dat to gitignore. [James Hensman]

* Tidied upt he kwargs in sympykern. [James Hensman]

* Merge branch 'sympykern' [James Hensman]

* Added sympykern as a 'kernpart' object. [James Hensman]

  now we can add sympykerns to any other kern

* Some gradient tidying and a small correction in the natural gradients. [James Hensman]

* Some simplification of the gradient expressions. [James Hensman]

* Some natural gradients of the uncollapsed GP implemented. [James Hensman]

* Gradients now working in uncollapsed GP. [James Hensman]

* T of the gradients are now working in the uncollapsed sparse GP: one term still to do. [James Hensman]

* DL_dbeta now works in the uncollapsed sparse GP. [James Hensman]

* Chenged a little of the notation in the sparse GP. [James Hensman]

  This should allow easier implementation of het. noise

* Small tweak to the gradients in sparse GP. [James Hensman]

* Minor perfomance tweak for GP regression. [James Hensman]

* More skeletal work on the uncollapsed GP. [James Hensman]

  None of the gradients work, but lots more things are in place

* Added the raw_predict function in the uncollapsed sparse GP. [James Hensman]

* Added get and set attributes to the mode class. [James Hensman]

  ... so that we can deal with the parameters in a Neil friendly way.

* Fixed interface change in optimization.py. [Nicolo Fusi]

* Added autodection of Rasmussen's minimize. [Nicolo Fusi]

* Merge branch 'master' into SGD. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Added notes on issues found. [Neil Lawrence]

* Implemented default constraints. [James Hensman]

  via m.ensure_default_constraints()

* Changed the name of GP_EP (from simple) in the unit test, added a messages option for full EP. [James Hensman]

* Changes the naming of kern components. [James Hensman]

  Kern components now only get a number if their name is duplicated

* Fixed small bugs in linalg, setup.py. [James Hensman]

* Changed the behaviour of pdinv: now returns L and Li as well as the inverse. [James Hensman]

* More tidying in EP, removed examples from _module_ ( and opened discussion on github. [James Hensman]

* Trivial merge resolution. [James Hensman]

* Some tidying in the EP code. [James Hensman]

* Removed uncertain gp regression from the model __init__, since it's now just a switch in the sparse GP. [James Hensman]

* Fixed SGD to work with new interface. [Nicolo Fusi]

* Fixed merge. [Nicolo Fusi]

* Models are now pickleable. [Nicolo Fusi]

* Working implementation of SGD. [Nicolo Fusi]

* GPLVM accepts an initial value for X (in case you don't want to use the default random/PCA init) [Nicolo Fusi]

* Now passing a reference of the model to the optimizer (used in SGD) [Nicolo Fusi]

* Fixed bug in rbf_ARD kernel (dK_dX) [Nicolo Fusi]

* Fixed import error for examples and import error for uncertain inputs GP. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Merge changes for model.py and optimization.py on comments. [Neil Lawrence]

* Minor edits. [James Hensman]

* Models can now specify a preferred optimser (defaults to tnc) [James Hensman]

* Some tidying in the uncollapsed GP. [James Hensman]

* Made uncertain inputs a simple swith in the sparse GP class.  This simplifies the inherritance structure. [James Hensman]

* Merge branch 'master' of github.com:SheffieldML/GPy. [James Hensman]

* Rbf_ARD now in the updated format for the computation of the derivatives (included for the psi-statistics, but not tested) [Nicolas]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolas]

* GPy: Some rewriting for the exponential and Matern kernels. They now pass the unit test. [Nicolas]

* Added a skeleton of the uncollapsed sparse GP. [James Hensman]

* Minor commenting. [James Hensman]

* General tidying in sparse_GP_regression. [James Hensman]

  Have also made a small ampount of headway in
  enabling heteroscedatic noise.

* Saved a little computation by exploiting the symmetry of a matrix. [James Hensman]

* I think the gradients bug in the sparse GP model is due to Kmm being unstable to invert. REducing M in some of the examples really helps. [James Hensman]

* Added datasets.py back in and minor changes. [Neil Lawrence]

* Fixed bug in GP_regression. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Derivatives of the exponential kernel in the right format. [Nicolas]

* Added precomputation of linear kernel, changed the logic a bit. [Nicolo Fusi]

* Fixed bug in linear_ARD. [Nicolo Fusi]

* Merge branch 'master' of github.com:SheffieldML/GPy. [Nicolo Fusi]

* Added vim's swp files to gitignore. [James Hensman]

* Demo for GP regressio with uncertain inputs. [James Hensman]

* Bias kernel psi stats implemented. [James Hensman]

* New file:   uncertain_input_GP_regression.py. [James Hensman]

* Tidied up some commented code from sparse_GP_regression. [James Hensman]

* Added support for partial derivatives to ARD kern. [Nicolo Fusi]

* SparseGPLVM demo now working. [James Hensman]

* Sparse GP regression now working on this branch. [James Hensman]

* Fixed bias kern for dk_dx. [James Hensman]

* Fixed some slicing in kern.py. [James Hensman]

* Refactored the kernpart base class. [James Hensman]

* GP_regression demo working with new style gradients for rbf, linear, white, bias. [James Hensman]

* GPLVM demo working. [James Hensman]

* Very basic GP_regression demo is working. [James Hensman]

* Docstringing dK_dtheta. [James Hensman]

* Removed DelayedDecorator.py: no longer required. [James Hensman]

* Removed gradient transforming ability from kern.py. [James Hensman]

* Made GP_regression wwork with partial-passed gradients. [James Hensman]

* Added copyright notice and license at the top. [Nicolo Fusi]

* Models. [Nicolo Fusi]

* Kerns. [Nicolo Fusi]

* Inference. [Nicolo Fusi]

* Tests. [Nicolo Fusi]

* Examples. [Nicolo Fusi]

* Utils. [Nicolo Fusi]

* Core file. [Nicolo Fusi]

* Initial commit. [Nicolò Fusi]
