# -*- coding: utf-8 -*-
"""
Copyright (c) 2021 Showa Denko Materials co., Ltd. All rights reserved.

This software is for non-profit use only.

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN THIS SOFTWARE.
"""

import copy
import numpy as np
import GPy
from GPyOpt.models.base import BOModel


class MultiObjGPModel(BOModel):
    """
    Class for handling multiple Gaussian Process models in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, n_obj, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000, optimize_restarts=5, sparse = False, num_inducing = 10,  verbose=True, ARD=False):
        self.n_obj = n_obj
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse = sparse
        self.num_inducing = num_inducing
        self.model = None
        self.ARD = ARD

    @staticmethod
    def fromConfig(config):
        return MultiObjGPModel(**config)
    
    def normalizeY(self, Y):
        Y = np.array(Y)
        self.Ymean = Y.mean(axis=0)
        self.Ystd = Y.std(axis=0)
        Y = Y - self.Ymean
        Y = Y/self.Ystd
        return Y
    
    def re_normalizeY(self, Y):
        Y = np.array(Y)
        Y = Y - self.Ymean
        Y = Y/self.Ystd        
        return Y

    def re_scaleY(self, Y):
        Y = np.array(Y)
        Y = Y/self.Ystd        
        return Y

    def reverse_scaleY(self, Y):
        Y = np.array(Y)
        Y = Y*self.Ystd        
        return Y

    def _create_model(self, X, Y):
        """
        Creates the models given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD)
        else:
            kern = self.kernel
            self.kernel = None
        kerns = [copy.deepcopy(kern) for _ in range(self.n_obj)]

        # --- define model
        noise_var = [y.var()*0.01 if self.noise_var is None else self.noise_var for y in Y.T]

        if not self.sparse:
            self.model = [GPy.models.GPRegression(X, y.reshape(-1,1), kernel=k, noise_var=noise) for y, noise, k in zip(Y.T, noise_var, kerns)]
        else:
            self.model = [GPy.models.SparseGPRegression(X, y.reshape(-1,1), kernel=kern, num_inducing=self.num_inducing) for y in Y.T]

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval == True:
            for m in self.model:
                m.Gaussian_noise.constrain_fixed(1e-6, warning=False)          
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            for m in self.model:
                m.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the models with new observations.
        """
        
        Y_all = self.normalizeY(Y_all)
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            for m, y in zip(self.model, Y_all.T):
                m.set_XY(X_all, y.reshape(-1,1))

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts==1:
                for m in self.model:
                    m.optimize(optimizer=self.optimizer, 
                               max_iters = self.max_iters,
                               messages=False, 
                               ipython_notebook=False)
            else:
                for m in self.model:
                    m.optimize_restarts(num_restarts=self.optimize_restarts,
                                        optimizer=self.optimizer,
                                        max_iters = self.max_iters,
                                        verbose=self.verbose)
        self.fmin_multi = self._get_fmin_multi()

    def _predict(self, X, full_cov, include_likelihood):
        if X.ndim == 1:
            X = X[None,:]
        m = []
        v = []
        for md in self.model:
            m0, v0 = md.predict(X, full_cov=full_cov, include_likelihood=include_likelihood)
            m.append(m0)
            v.append(np.clip(v0, 1e-10, np.inf))
        return m, v

    def predict(self, X, with_noise=True):
        """
        Predictions with the models. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        m, v = self._predict(X, False, with_noise)
        return m, [np.sqrt(v0) for v0 in v]

    def predict_covariance(self, X, with_noise=True):
        raise NotImplementedError("predict_covariance is not modified for multi-objective")


    def get_fmin(self):
        raise NotImplementedError("get_fmin is not modified for multi-objective")

    
    def _get_fmin_multi(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        X = self.model[0].X
        mean, sig = self.predict(X)
        idxes = [mu.argmin() for mu in mean]
        return [mu[idx] for mu, idx in zip(mean, idxes)], [si[idx] for si, idx in zip(sig, idxes)]

    def get_fmin_multi(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        return self.fmin_multi

    def is_achived(self, goals):
        X = self.model[0].X
        mean, sig = self.predict(X)
        mean = np.concatenate(mean, axis=1)
        goal_bool = ((mean-goals)<=0).all(axis=1)
        return True in goal_bool

    def get_y(self):
        return np.concatenate([model.Y for model in self.model], axis=1) 

    def get_x(self):
        return self.model[0].X


    def predict_withGradients(self, X):
        raise NotImplementedError("predict_withGradients is not modified for multi-objective")

    def copy(self):
        raise NotImplementedError("copy is not modified for multi-objective")

    def get_model_parameters(self):
        return None

    def get_model_parameters_names(self):
        raise NotImplementedError("get_model_parameters_names is not modified for multi-objective")

    def get_covariance_between_points(self, x1, x2):
        raise NotImplementedError("get_covariance_between_points is not modified for multi-objective")