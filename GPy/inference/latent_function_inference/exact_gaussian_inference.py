# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

def exact_gaussian_inference(K, likelihood, Y, Y_metadata=None):


    Wi, LW, LWi, W_logdet = pdinv(K + likelhood.covariance(Y, Y_metadata))

    alpha, _ = dpotrs(LW, YYT_factor, lower=1)
    dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

    log_marginal =  (-0.5 * Y.size * np.log(2.*np.pi) -
        0.5 * Y.shape[1] * W_logdet + np.sum(np.square(alpha))
