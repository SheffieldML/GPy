# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from .parameterization.priorizable import Priorizable
from paramz import Model as ParamzModel
from paramz import transformations
import numpy as np
import warnings
import timeit
from ..util.linalg import pdinv

class Model(ParamzModel, Priorizable):

    def __init__(self, name):
        super(Model, self).__init__(name)  # Parameterized.__init__(self)

    def log_likelihood(self):
        raise NotImplementedError("this needs to be implemented to use the model class")

    def _log_likelihood_gradients(self):
        return self.gradient#.copy()

    def objective_function(self):
        """
        The objective function for the given algorithm.

        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the objective function here.

        For probabilistic models this is the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your objective to minimize here!
        """
        return -float(self.log_likelihood()) - self.log_prior()

    def objective_function_gradients(self):
        """
        The gradients for the objective function for the given algorithm.
        The gradients are w.r.t. the *negative* objective function, as
        this framework works with *negative* log-likelihoods as a default.

        You can find the gradient for the parameters in self.gradient at all times.
        This is the place, where gradients get stored for parameters.

        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the gradient here.

        For probabilistic models this is the gradient of the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your *negative* gradient here!
        """
        return -(self._log_likelihood_gradients() + self._log_prior_gradients())

    def CCD(self, step_size=1., auto_scaling=False):
        """
        Code is based on implementation within GPStuff, INLA and the original Sanchez and Sanchez paper (2005)
        Original Author: Mike Smith
        Changes: Akash
        """
        modal_params = self.optimizer_array[:].copy()
        # num_free_params = modal_params.shape[0]
        num_free_params = self.num_params

        # Calculate the numerical hessian for *every* parameter
        H = self.numerical_parameter_hessian()

        try:
            curv = pdinv(H)[0]
        except np.linalg.linalg.LinAlgError:
            print("Hessian is not positive definite, ensure parameters are at their MAP solution by optimizing, if that doesn't work try modifying the step length parameter")
            return
        # CCD points are calculated for unit circle, so we take the principle components and scale them such that we have a unit sphere in our parameter space, which we can then map back to parameter space
        [w, V] = np.linalg.eig(curv)
        z = (V*np.sqrt(w[None,:])).T
        # step-size will be 1 by default
        z = z*step_size

        # Calculate points, we will do just CCD for now
        # First we build the hadamarx matrix from the paper that encodes *all* the points possible in the num_free_params dimensional space.
        grow = np.sum(num_free_params >= np.array([1, 2, 3, 4, 6, 7, 9, 12, 18, 22, 30, 39, 53, 70, 93]))

        H0 = 1

        #Build the design matrix that holds *every* corner on the num_free_params dimensional hyper-cube (the hadamard matrix), H in the paper
        # actually this might be computationally expensive, check if there could be a better alternative for this code.
        for i in range(grow):
            H0 = np.vstack([np.hstack([H0, H0]), 
                            np.hstack([H0, -H0])])
        
        # For each additional parameter our dimensional space increases, as does the number of CCD points we require to do the approximate integration. CCD is a method of deciding which points should be included and which can be ignored without a significant effect. It would be ideal but too computationally expensive - when we have many parameters - to use all the corners of the hypercube in our integration
        # See Sanchez and Sanchez (2005) for details. on how the points are chosen
        # @TODO: could this line be moved to utils..
        walsh_inds = np.array([1, 2, 4, 8, 15, 16, 32, 51, 64, 85, 106, 128, 150, 171, 219, 237, 247, 256, 279, 297, 455, 512, 537, 557, 597, 643, 803, 863, 898, 1024, 1051, 1070, 1112, 1169, 1333, 1345, 1620, 1866, 2048, 2076, 2085, 2158, 2372, 2456, 2618, 2800, 2873, 3127, 3284, 3483, 3557, 3763, 4096, 4125, 4135, 4176, 4435, 4459, 4469, 4497, 4752, 5255, 5732, 5801, 5915, 6100, 6369, 6907, 7069, 8192, 8263, 8351, 8422, 8458, 8571, 8750, 8858, 9124, 9314, 9500, 10026, 10455, 10556, 11778, 11885, 11984, 13548, 14007, 14514, 14965, 15125, 15554, 16384, 16457, 16517, 16609, 16771, 16853, 17022, 17453, 17891, 18073, 18562, 18980, 19030, 19932, 20075, 20745, 21544, 22633, 23200, 24167, 25700, 26360, 26591, 26776, 28443, 28905, 29577, 32705])

        used_walsh_inds = walsh_inds[:num_free_params]

        # For every additional parameters, we get an additional number of points (which as we decide to drop points using Sanchezes rules, won't necessarily grow at the same rate)
        # in practise actually it does not work well beyond 15-20 - and so some technique like PSIS(Importance sampling) has to be used.
        ccd_points = H0[:, used_walsh_inds]
        
        # ccd only gives corner points, so lets add an additional one for the centre point
        ccd_points = np.vstack([np.zeros((1,num_free_params)), ccd_points])

        # rotate= True for sphere CCD, False for face-centred central-composite design.
        rotate = True

        # Now we add the points on the unit hyper-sphere that arent on the corners also known as star points.
        for i in range(num_free_params):
            top_edge = np.zeros(num_free_params)
            if rotate == True:
                top_edge[i] = np.sqrt(num_free_params)
            else:
                top_edge[i] = 1.
            bottom_edge = np.zeros(num_free_params)
            if rotate == True:
                bottom_edge[i] = -np.sqrt(num_free_params)
            else:
                bottom_edge[i] = 1.
            ccd_points = np.vstack([ccd_points, top_edge, bottom_edge])

        # Find the appropriate scaling such that the edges lie on a boundary of equal density
        # First find the density at the mode
        log_marginal = lambda p: -self._objective(p)
        mode_density = log_marginal(modal_params)  # FIXME: We really want to evaluate the log_marginal not the negative objective as it makes more sense, but they are equivalent in this case

        # Treating the posterior over hyperparameters as a standard normal, moving z*sqrt(2) should make the likelihood drop by 1, as z should be acting as a unit vector along the principle components of the posterior. 
        num_ccd_points = ccd_points.shape[0]
        scalings = np.ones_like(ccd_points)
        # This is naive scaling, assuming that it is well approximated by a standard normal, in practice you might want to scale in different directions seperately (split normal approximation)
        j = np.arange(num_free_params*2)
        temp = np.zeros((1, num_free_params))
        direction = np.ones_like(j)
        direction[::2] = -1
        # # each direction has a numeric index.
        dir_ind = np.arange(num_free_params)
        #  # make a matrix where each column is a contour point.
        modal_params = modal_params[:,np.newaxis]
        modal_params = modal_params.T
        # print modal_params.shape
        modal_params_mat = np.repeat(modal_params, [num_free_params*2], axis=0)
        temp = np.eye(num_free_params)
        temp = np.repeat(temp, [2], axis=0)
        dir_vec = np.asarray([1, -1])
        dir_vec = dir_vec[:,np.newaxis]
        dir_vec = np.tile(dir_vec, (num_free_params, num_free_params))
        temp1 = np.multiply(temp, dir_vec)
        # contour points is a matrix where each column is a contour point - num_free
        contour_points = modal_params_mat + np.sqrt(2)*temp1.dot(z)

        # calculate marginal likelihood for each contour pint here ..
        point_density_mat = np.apply_along_axis(log_marginal, 1, contour_points)
        # point_density_mat = log_marginal(contour_points)
        diff_density = mode_density - point_density_mat
        new_mode_ind = diff_density <= 0.0
        new_mode_ind = np.asarray(new_mode_ind)
        if np.sum(new_mode_ind) > 0:
            warnings.warn("Contour point has higher density than mode density, must be multi-modal or saddle point, Please check the optimisation again.")
        non_mode_ind = diff_density > 0
        scalings_bi = np.ones((1, 2*num_free_params))
        scalings_bi[0,non_mode_ind] = np.sqrt(1.0 / diff_density[non_mode_ind])

        # this part also has to be matricised, just not though out yet by me.
        for i in range(num_free_params*2):
            dir_ind = int(np.floor(i / 2))
            direction = 1
            if i % 2:
                direction = -1
            scalings[ccd_points[:, dir_ind]*direction > 0, dir_ind] = np.maximum(np.minimum(scalings_bi[0,i], 10.0), 1 / 10.0)

        # Make the points *slightly* further out. Aki Vehtari has shown this to be more accurate
        over_scale = 1.1
        ccd_points = over_scale*scalings*ccd_points
        # We have the scaled ccd_points, now we need to map them into parameter space by projecting along principle components and shifting to mode
        param_points = ccd_points.dot(z) + modal_params
        num_param_points = param_points.shape[0]
        point_densities = np.ones(num_ccd_points)*np.nan
        point_densities = np.apply_along_axis(log_marginal, 1, param_points)
        # for point_ind, param_point in enumerate(param_points):
        #     point_densities[point_ind] = log_marginal(param_point)

        # Remove nan densities
        non_nan_densities = np.isfinite(point_densities).flatten()
        num_nan_densities = num_ccd_points - non_nan_densities.size
        if num_nan_densities > 0:
            warnings.warn("{} out of {} design points have nan as density".format(num_nan_densities, num_ccd_points))

        point_densities = point_densities[non_nan_densities]
        param_points = param_points[non_nan_densities, :]
        #
        # exponentiating to transform log marginal to marginal densities after scaling down by max value.
        point_densities = point_densities - np.max(point_densities)
        point_densities = np.exp(point_densities)

        if num_free_params > 0:
            # Each section that the ccd point represents needs an associated weight. Since every point is on the same contour, all weights are the same, except the centre weight
            point_weights = 1.0/((2.0*np.pi)**(-num_free_params*0.5) * np.exp(-0.5*num_free_params*over_scale**2) * (param_points.shape[0]-1)*over_scale**2)
            centre_weight = (2*np.pi)**(num_free_params*0.5) * (1 - 1.0/over_scale**2)
            # normalize
            point_weights = point_weights / centre_weight
            centre_weight = 1.0
            point_densities[1:] *= point_weights
        # Normalize density
        point_densities /= point_densities.sum()

        # Remove small density points having less than 1% probability mass
        non_small_densities = (point_densities > 0.01/point_densities.size).flatten()
        point_densities = point_densities[non_small_densities]
        param_points = param_points[non_small_densities, :]
        point_densities /= point_densities.sum()

        # check for too many design points.
        if point_densities.size > 200:
            warnings.warn("Number of design points with finite mass density is too high.")

        # @TODO: implement PSIS: Gelman, Vehtari 2014, when the number of design points is too high.
        transformed_points = param_points.copy()
        # alan's original code to transform those parameters, to the true space of parameters again
        #mike's change: some parameters have no transform, and thus won't be called by any of the
        #iterations through m2.constraints.items(). To handle these we keep track of those parameters
        #not included, and then add them (untransformed) at the end.
        f = np.ones(self.size).astype(bool)
        f[self.constraints[transformations.__fixed__]] = transformations.FIXED
        #TODO Check: Presumably only one constraint applies to each parameter?
        new_t_points = [] 
        todo = range(0,sum(f))
        new_t_points = np.zeros_like(transformed_points)
        for c, ind in self.constraints.items():
            if c != transformations.__fixed__:
                for i in ind[f[ind]]:
                    z[:,i] = c.f(z[:,i])
                    new_t_points[:,i] = (c.f(transformed_points[:, i]))
                    todo.remove(i)

        for i in todo:
            new_t_points[:,i] = transformed_points[:, i]

        return new_t_points, point_densities, scalings, z

    def numerical_parameter_hessian(self, step_length=1e-3):
        """
        Calculate the numerical hessian for the posterior of the parameters

        Often this will want to be done at the modal values of the parameters so the optimizer should be called first
        
        H = [d^2L_dp1dp1, d^2L_dp1dp2, d^2L_dp1dp3, ...;
             d^2L_dp2dp1, d^2L_dp2dp2, d^2L_dp3dp3, ...;
            ...
             ]
        Where the vector p is all of the parameters in param_array, not just parameters of this model class itself
        """
        # We use optimizer array as we want to ignore any parameters with fixed values
        num_child_params = self.optimizer_array.shape[0]
        
        H = np.eye(num_child_params)

        initial_params = self.optimizer_array[:].copy()
        # Calculate gradient of marginal likelihood moving one at a time
        for i in range(num_child_params):
            # Pick out the e vector with one in the parameter that we wish to move
            e = H[:,i]
            # Get parameters required to evaluate by nudging left and right
            left = initial_params - step_length*e
            right = initial_params + step_length*e

            left_grad = self._grads(left)
            right_grad = self._grads(right)

            finite_diff = (right_grad - left_grad) / (2*step_length)

            #Put in the correct row of the hessian
            H[:,i] = finite_diff

        return H