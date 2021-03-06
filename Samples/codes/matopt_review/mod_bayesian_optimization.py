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

import numpy as np
from GPyOpt.methods import ModularBayesianOptimization

class MultiObjectiveBayesianOptimization(ModularBayesianOptimization):

    """
    ModularBayesianOptimization in GPyOpt was modified to handle multiple objectives.
    Note that only a method, _compute_results, has been modified at this point.

    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: instantiated de_duplication GPyOpt class.
    """
    def __init__(self, model, space, objective, acquisition, evaluator,
                 X_init, Y_init=None, cost = None, normalize_Y = True,
                 model_update_interval = 1, de_duplication=False):

        self.initial_iter = True
        self.modular_optimization = True

        super(MultiObjectiveBayesianOptimization ,self).__init__(  model                  = model,
                                                            space                  = space,
                                                            objective              = objective,
                                                            acquisition            = acquisition,
                                                            evaluator              = evaluator,
                                                            X_init                 = X_init,
                                                            Y_init                 = Y_init,
                                                            cost                   = cost,
                                                            normalize_Y            = normalize_Y,
                                                            model_update_interval  = model_update_interval,
                                                            de_duplication         = de_duplication)
    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        self.Y_best = [y.min() for y in self.Y.T]
        self.x_opt = [self.X[np.argmin(y),:] for y in self.Y.T]
        self.fx_opt = self.Y_best