import numpy as np
import scipy as sp
import scipy.sparse
from optimization import Optimizer
from scipy import linalg, optimize
import pylab as plt
import copy, sys, pickle

class opt_SGD(Optimizer):
    """
    Optimize using stochastic gradient descent.

    *** Parameters ***
    Model: reference to the Model object
    iterations: number of iterations
    learning_rate: learning rate
    momentum: momentum

    """

    def __init__(self, start, iterations = 10, learning_rate = 1e-4, momentum = 0.9, model = None, messages = False, batch_size = 1, self_paced = False, center = True, iteration_file = None, learning_rate_adaptation=None, actual_iter=None, schedule=None, **kwargs):
        self.opt_name = "Stochastic Gradient Descent"

        self.Model = model
        self.iterations = iterations
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.x_opt = None
        self.f_opt = None
        self.messages = messages
        self.batch_size = batch_size
        self.self_paced = self_paced
        self.center = center
        self.param_traces = [('noise',[])]
        self.iteration_file = iteration_file
        self.learning_rate_adaptation = learning_rate_adaptation
        self.actual_iter = actual_iter
        if self.learning_rate_adaptation != None:
            if self.learning_rate_adaptation == 'annealing':
                self.learning_rate_0 = self.learning_rate
            else:
                self.learning_rate_0 = self.learning_rate.mean()

        self.schedule = schedule
        # if len([p for p in self.model.kern.parts if p.name == 'bias']) == 1:
        #     self.param_traces.append(('bias',[]))
        # if len([p for p in self.model.kern.parts if p.name == 'linear']) == 1:
        #     self.param_traces.append(('linear',[]))
        # if len([p for p in self.model.kern.parts if p.name == 'rbf']) == 1:
        #     self.param_traces.append(('rbf_var',[]))

        self.param_traces = dict(self.param_traces)
        self.fopt_trace = []

        num_params = len(self.Model._get_params())
        if isinstance(self.learning_rate, float):
            self.learning_rate = np.ones((num_params,)) * self.learning_rate

        assert (len(self.learning_rate) == num_params), "there must be one learning rate per parameter"

    def __str__(self):
        status = "\nOptimizer: \t\t\t %s\n" % self.opt_name
        status += "f(x_opt): \t\t\t %.4f\n" % self.f_opt
        status += "Number of iterations: \t\t %d\n" % self.iterations
        status += "Learning rate: \t\t\t max %.3f, min %.3f\n" % (self.learning_rate.max(), self.learning_rate.min())
        status += "Momentum: \t\t\t %.3f\n" % self.momentum
        status += "Batch size: \t\t\t %d\n" % self.batch_size
        status += "Time elapsed: \t\t\t %s\n" % self.time
        return status

    def plot_traces(self):
        plt.figure()
        plt.subplot(211)
        plt.title('Parameters')
        for k in self.param_traces.keys():
            plt.plot(self.param_traces[k], label=k)
        plt.legend(loc=0)
        plt.subplot(212)
        plt.title('Objective function')
        plt.plot(self.fopt_trace)


    def non_null_samples(self, data):
        return (np.isnan(data).sum(axis=1) == 0)

    def check_for_missing(self, data):
        if sp.sparse.issparse(self.Model.likelihood.Y):
            return True
        else:
            return np.isnan(data).sum() > 0

    def subset_parameter_vector(self, x, samples, param_shapes):
        subset = np.array([], dtype = int)
        x = np.arange(0, len(x))
        i = 0

        for s in param_shapes:
            N, input_dim = s
            X = x[i:i+N*input_dim].reshape(N, input_dim)
            X = X[samples]
            subset = np.append(subset, X.flatten())
            i += N*input_dim

        subset = np.append(subset, x[i:])

        return subset

    def shift_constraints(self, j):

        constrained_indices = copy.deepcopy(self.Model.constrained_indices)

        for c, constraint in enumerate(constrained_indices):
            mask = (np.ones_like(constrained_indices[c]) == 1)
            for i in range(len(constrained_indices[c])):
                pos = np.where(j == constrained_indices[c][i])[0]
                if len(pos) == 1:
                    self.Model.constrained_indices[c][i] = pos
                else:
                    mask[i] = False

            self.Model.constrained_indices[c] = self.Model.constrained_indices[c][mask]
        return constrained_indices
        # back them up
        # bounded_i = copy.deepcopy(self.Model.constrained_bounded_indices)
        # bounded_l = copy.deepcopy(self.Model.constrained_bounded_lowers)
        # bounded_u = copy.deepcopy(self.Model.constrained_bounded_uppers)

        # for b in range(len(bounded_i)): # for each group of constraints
        #     for bc in range(len(bounded_i[b])):
        #         pos = np.where(j == bounded_i[b][bc])[0]
        #         if len(pos) == 1:
        #             pos2 = np.where(self.Model.constrained_bounded_indices[b] == bounded_i[b][bc])[0][0]
        #             self.Model.constrained_bounded_indices[b][pos2] = pos[0]
        #         else:
        #             if len(self.Model.constrained_bounded_indices[b]) == 1:
        #                 # if it's the last index to be removed
        #                 # the logic here is just a mess. If we remove the last one, then all the
        #                 # b-indices change and we have to iterate through everything to find our
        #                 # current index. Can't deal with this right now.
        #                 raise NotImplementedError

        #             else: # just remove it from the indices
        #                 mask = self.Model.constrained_bounded_indices[b] != bc
        #                 self.Model.constrained_bounded_indices[b] = self.Model.constrained_bounded_indices[b][mask]


        # # here we shif the positive constraints. We cycle through each positive
        # # constraint
        # positive = self.Model.constrained_positive_indices.copy()
        # mask = (np.ones_like(positive) == 1)
        # for p in range(len(positive)):
        #     # we now check whether the constrained index appears in the j vector
        #     # (the vector of the "active" indices)
        #     pos = np.where(j == self.Model.constrained_positive_indices[p])[0]
        #     if len(pos) == 1:
        #         self.Model.constrained_positive_indices[p] = pos
        #     else:
        #         mask[p] = False
        # self.Model.constrained_positive_indices = self.Model.constrained_positive_indices[mask]

        # return (bounded_i, bounded_l, bounded_u), positive

    def restore_constraints(self, c):#b, p):
        # self.Model.constrained_bounded_indices = b[0]
        # self.Model.constrained_bounded_lowers = b[1]
        # self.Model.constrained_bounded_uppers = b[2]
        # self.Model.constrained_positive_indices = p
        self.Model.constrained_indices = c

    def get_param_shapes(self, N = None, input_dim = None):
        model_name = self.Model.__class__.__name__
        if model_name == 'GPLVM':
            return [(N, input_dim)]
        if model_name == 'Bayesian_GPLVM':
            return [(N, input_dim), (N, input_dim)]
        else:
            raise NotImplementedError

    def step_with_missing_data(self, f_fp, X, step, shapes):
        N, input_dim = X.shape

        if not sp.sparse.issparse(self.Model.likelihood.Y):
            Y = self.Model.likelihood.Y
            samples = self.non_null_samples(self.Model.likelihood.Y)
            self.Model.N = samples.sum()
            Y = Y[samples]
        else:
            samples = self.Model.likelihood.Y.nonzero()[0]
            self.Model.N = len(samples)
            Y = np.asarray(self.Model.likelihood.Y[samples].todense(), dtype = np.float64)

        if self.Model.N == 0 or Y.std() == 0.0:
            return 0, step, self.Model.N

        self.Model.likelihood._offset = Y.mean()
        self.Model.likelihood._scale = Y.std()
        self.Model.likelihood.set_data(Y)
        # self.Model.likelihood.V = self.Model.likelihood.Y*self.Model.likelihood.precision

        sigma = self.Model.likelihood._variance
        self.Model.likelihood._variance = None # invalidate cache
        self.Model.likelihood._set_params(sigma)


        j = self.subset_parameter_vector(self.x_opt, samples, shapes)
        self.Model.X = X[samples]

        model_name = self.Model.__class__.__name__

        if model_name == 'Bayesian_GPLVM':
            self.Model.likelihood.YYT = np.dot(self.Model.likelihood.Y, self.Model.likelihood.Y.T)
            self.Model.likelihood.trYYT = np.trace(self.Model.likelihood.YYT)

        ci = self.shift_constraints(j)
        f, fp = f_fp(self.x_opt[j])

        step[j] = self.momentum * step[j] + self.learning_rate[j] * fp
        self.x_opt[j] -= step[j]
        self.restore_constraints(ci)

        self.Model.grads[j] = fp
        # restore likelihood _offset and _scale, otherwise when we call set_data(y) on
        # the next feature, it will get normalized with the mean and std of this one.
        self.Model.likelihood._offset = 0
        self.Model.likelihood._scale = 1

        return f, step, self.Model.N

    def adapt_learning_rate(self, t, D):
        if self.learning_rate_adaptation == 'adagrad':
            if t > 0:
                g_k = self.Model.grads
                self.s_k += np.square(g_k)
                t0 = 100.0
                self.learning_rate = 0.1/(t0 + np.sqrt(self.s_k))

                import pdb; pdb.set_trace()
            else:
                self.learning_rate = np.zeros_like(self.learning_rate)
                self.s_k = np.zeros_like(self.x_opt)

        elif self.learning_rate_adaptation == 'annealing':
            #self.learning_rate = self.learning_rate_0/(1+float(t+1)/10)
            self.learning_rate = np.ones_like(self.learning_rate) * self.schedule[t]


        elif self.learning_rate_adaptation == 'semi_pesky':
            if self.Model.__class__.__name__ == 'Bayesian_GPLVM':
                g_t = self.Model.grads
                if t == 0:
                    self.hbar_t = 0.0
                    self.tau_t = 100.0
                    self.gbar_t = 0.0

                self.gbar_t = (1-1/self.tau_t)*self.gbar_t + 1/self.tau_t * g_t
                self.hbar_t = (1-1/self.tau_t)*self.hbar_t + 1/self.tau_t * np.dot(g_t.T, g_t)
                self.learning_rate = np.ones_like(self.learning_rate)*(np.dot(self.gbar_t.T, self.gbar_t) / self.hbar_t)
                tau_t = self.tau_t*(1-self.learning_rate) + 1


    def opt(self, f_fp=None, f=None, fp=None):
        self.x_opt = self.Model._get_params_transformed()
        self.grads = []

        X, Y = self.Model.X.copy(), self.Model.likelihood.Y.copy()

        self.Model.likelihood.YYT = 0
        self.Model.likelihood.trYYT = 0
        self.Model.likelihood._offset = 0.0
        self.Model.likelihood._scale = 1.0

        N, input_dim = self.Model.X.shape
        D = self.Model.likelihood.Y.shape[1]
        num_params = self.Model._get_params()
        self.trace = []
        missing_data = self.check_for_missing(self.Model.likelihood.Y)

        step = np.zeros_like(num_params)
        for it in range(self.iterations):
            if self.actual_iter != None:
                it = self.actual_iter

            self.Model.grads = np.zeros_like(self.x_opt) # TODO this is ugly

            if it == 0 or self.self_paced is False:
                features = np.random.permutation(Y.shape[1])
            else:
                features = np.argsort(NLL)

            b = len(features)/self.batch_size
            features = [features[i::b] for i in range(b)]
            NLL = []
            import pylab as plt
            for count, j in enumerate(features):
                self.Model.input_dim = len(j)
                self.Model.likelihood.input_dim = len(j)
                self.Model.likelihood.set_data(Y[:, j])
                # self.Model.likelihood.V = self.Model.likelihood.Y*self.Model.likelihood.precision

                sigma = self.Model.likelihood._variance
                self.Model.likelihood._variance = None # invalidate cache
                self.Model.likelihood._set_params(sigma)

                if missing_data:
                    shapes = self.get_param_shapes(N, input_dim)
                    f, step, Nj = self.step_with_missing_data(f_fp, X, step, shapes)
                else:
                    self.Model.likelihood.YYT = np.dot(self.Model.likelihood.Y, self.Model.likelihood.Y.T)
                    self.Model.likelihood.trYYT = np.trace(self.Model.likelihood.YYT)
                    Nj = N
                    f, fp = f_fp(self.x_opt)
                    self.Model.grads = fp.copy()
                    step = self.momentum * step + self.learning_rate * fp
                    self.x_opt -= step

                if self.messages == 2:
                    noise = self.Model.likelihood._variance
                    status = "evaluating {feature: 5d}/{tot: 5d} \t f: {f: 2.3f} \t non-missing: {nm: 4d}\t noise: {noise: 2.4f}\r".format(feature = count, tot = len(features), f = f, nm = Nj, noise = noise)
                    sys.stdout.write(status)
                    sys.stdout.flush()
                    self.param_traces['noise'].append(noise)

                self.adapt_learning_rate(it+count, D)
                NLL.append(f)
                self.fopt_trace.append(NLL[-1])
                # fig = plt.figure('traces')
                # plt.clf()
                # plt.plot(self.param_traces['noise'])

                # for k in self.param_traces.keys():
                #     self.param_traces[k].append(self.Model.get(k)[0])
            self.grads.append(self.Model.grads.tolist())
            # should really be a sum(), but earlier samples in the iteration will have a very crappy ll
            self.f_opt = np.mean(NLL)
            self.Model.N = N
            self.Model.X = X
            self.Model.input_dim = D
            self.Model.likelihood.N = N
            self.Model.likelihood.input_dim = D
            self.Model.likelihood.Y = Y
            sigma = self.Model.likelihood._variance
            self.Model.likelihood._variance = None # invalidate cache
            self.Model.likelihood._set_params(sigma)

            self.trace.append(self.f_opt)
            if self.iteration_file is not None:
                f = open(self.iteration_file + "iteration%d.pickle" % it, 'w')
                data = [self.x_opt, self.fopt_trace, self.param_traces]
                pickle.dump(data, f)
                f.close()

            if self.messages != 0:
                sys.stdout.write('\r' + ' '*len(status)*2 + '  \r')
                status = "SGD Iteration: {0: 3d}/{1: 3d}  f: {2: 2.3f}   max eta: {3: 1.5f}\n".format(it+1, self.iterations, self.f_opt, self.learning_rate.max())
                sys.stdout.write(status)
                sys.stdout.flush()
