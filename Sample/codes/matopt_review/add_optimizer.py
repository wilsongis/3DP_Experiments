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

import GPyOpt
from GPyOpt.optimization.optimizer import OptLbfgs, OptDirect, OptCma, apply_optimizer
from GPyOpt.optimization.anchor_points_generator import ObjectiveAnchorPointsGenerator, ThompsonSamplingAnchorPointsGenerator

max_objective_anchor_points_logic = "max_objective"
thompson_sampling_anchor_points_logic = "thompsom_sampling"
sobol_design_type = "sobol"
random_design_type = "random"

class InvalidArgumentError(Exception):
    pass


class AcquisitionOptimizer(GPyOpt.optimization.AcquisitionOptimizer):
    """
    AcquisitionOptimizer of GPyOpt was modified to control some parameters including, max_AcOpt_iter and num_anchor_points.
    Note that the default paramaters of GPyOpt were used in the study of the goal-oriented Bayesian optimization. 

    :param space: design space class from GPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    :param max_AcOpt_iter: maximun number of optimization steps.
    :param num_anchor_points: number of initial search points.
    """
    def __init__(self, space, optimizer='lbfgs', max_AcOpt_iter=1000, num_anchor_points=1000, **kwargs):
        super(AcquisitionOptimizer, self).__init__(space, optimizer, **kwargs)
        self.max_AcOpt_iter = max_AcOpt_iter
        self.num_anchor_points = num_anchor_points
        
    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df

        ## --- Update the optimizer, in case context has beee passed.
        self.optimizer = self.choose_optimizermod(self.optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f, num_samples=self.num_anchor_points)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)

        ## -- Select the anchor points (with context)
        anchor_points = anchor_points_generator.get(num_anchor=5, duplicate_manager=duplicate_manager, context_manager=self.context_manager)

        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)

        optimized_points = [apply_optimizer(self.optimizer, a, f=f, df=df, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points]
        x_min, fx_min = min(optimized_points, key=lambda t:t[1])

        return x_min, fx_min
    
    
    def choose_optimizermod(self, optimizer_name, bounds):
        """
        Selects the type of local optimizer
        """
        if optimizer_name == 'lbfgs':
            optimizer = OptLbfgs(bounds, self.max_AcOpt_iter)   
            
        elif optimizer_name == 'DIRECT':
            optimizer = OptDirect(bounds, self.max_AcOpt_iter)

        elif optimizer_name == 'CMA':
            optimizer = OptCma(bounds, self.max_AcOpt_iter)
        else:
            print(optimizer_name)
            raise InvalidArgumentError('Invalid optimizer selected.')

        return optimizer
