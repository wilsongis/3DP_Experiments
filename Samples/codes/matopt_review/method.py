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
from . import add_acquisition
from . import add_optimizer
from . import add_models
from . import add_objective
from . import mod_bayesian_optimization
import GPyOpt

class boBASE():
    """
    Newly implemented class to handle settings for each Bayesian optimization method.

    param f: function to optimize.
    param space: GPyOpt class of domain. 
    param dim: dimension of design space.


    """
    def __init__(self, f, space, dim):        

        self.f = f
        self.space = space
        self.dim = dim
        
    def suggest_next_locations(self):
        self.set_bo()
        return self.bo_org.suggest_next_locations() 
        
    def _set_initialX(self):
        if hasattr(self, "X_init"):
            return self.X_init
        else:
            return GPyOpt.experiment_design.initial_design('random', self.space, self.N_init)        
        
    def _set_evaluator(self, model, acquisition_optimizer, acquisition):
        assert self.batch_size == 1, "Batch BO is not supported."
        if self.evaluator_type == 'sequential':
            return GPyOpt.core.evaluators.Sequential(acquisition)
        else:
            print(self.evaluator_type, "is not supported. The sequential is used instead.")
            return GPyOpt.core.evaluators.Sequential(acquisition)
               
    def _set_model(self, MCMC=False, exact_feval=False):
        if MCMC == True:
            model = GPyOpt.models.GPModel_MCMC(verbose=False, kernel=self.kern, exact_feval=exact_feval)
            if hasattr(self, "custom_model"):
                print("Custom model cannot be used with MCMC.")
                print("Custom model is ignored and GP model is used.")
        else:
            if hasattr(self, "custom_model"):
                model = self.custom_model
            else:
                model = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False, kernel=self.kern, exact_feval=exact_feval)
        return model

    def _set_multi_model(self, n_obj, exact_feval):
        if hasattr(self, "custom_model"):
            model = self.custom_model
        else:
            model = add_models.MultiObjGPModel(n_obj=n_obj, optimize_restarts=5,verbose=False, kernel=self.kern, exact_feval=exact_feval)
        return model
        

    def set_GPyOpt(self, algorism="EI", MCMC=False):
        objective = GPyOpt.core.task.SingleObjective(self.f)
        acquisition_optimizer = add_optimizer.AcquisitionOptimizer(self.space, optimizer=self.acquisition_optimizer_type, max_AcOpt_iter=self.max_AcOpt_iter, num_anchor_points=self.num_anchor_points)
        initial_design = self._set_initialX()
        if MCMC == False:
            model = self._set_model(MCMC, exact_feval=self.exact_feval)
            if algorism == "UCB":
                acquisition = GPyOpt.acquisitions.AcquisitionLCB(model, self.space, optimizer=acquisition_optimizer, exploration_weight=self.exploration_weight)
            elif algorism == "EI":
                acquisition = GPyOpt.acquisitions.AcquisitionEI(model, self.space, optimizer=acquisition_optimizer)
                
        if MCMC == True:
            model = self._set_model(MCMC, exact_feval=self.exact_feval)
            if algorism == "UCB":
                acquisition = GPyOpt.acquisitions.AcquisitionLCB_MCMC(model, self.space, optimizer=acquisition_optimizer, exploration_weight=self.exploration_weight)
            elif algorism == "EI":
                acquisition = GPyOpt.acquisitions.AcquisitionEI_MCMC(model, self.space, optimizer=acquisition_optimizer)
        evaluator = self._set_evaluator(model, acquisition_optimizer, acquisition)
        if hasattr(self, "Y_init"):
            self.bo_org = GPyOpt.methods.ModularBayesianOptimization\
                (model, self.space, objective, acquisition, evaluator, initial_design, Y_init=self.Y_init)
        else:
                        self.bo_org = GPyOpt.methods.ModularBayesianOptimization\
                (model, self.space, objective, acquisition, evaluator, initial_design)

    def set_GPyOptRecsat(self):
        objective = add_objective.MultiObjective(self.f, len(self.goals))
        acquisition_optimizer = add_optimizer.AcquisitionOptimizer(self.space, optimizer=self.acquisition_optimizer_type, max_AcOpt_iter=self.max_AcOpt_iter, num_anchor_points=self.num_anchor_points)
        model = self._set_multi_model(len(self.goals), exact_feval=self.exact_feval)
        initial_design = self._set_initialX()
        acquisition = add_acquisition.AcquisitionRecsat(model, self.space, goals=self.goals, optimizer=acquisition_optimizer)
        evaluator = self._set_evaluator(model, acquisition_optimizer, acquisition)
        if hasattr(self, "Y_init"):
            self.bo_org = mod_bayesian_optimization.MultiObjectiveBayesianOptimization\
                (model, self.space, objective, acquisition, evaluator, initial_design, Y_init=self.Y_init, normalize_Y=False)
        else:
                        self.bo_org = mod_bayesian_optimization.MultiObjectiveBayesianOptimization\
                (model, self.space, objective, acquisition, evaluator, initial_design, normalize_Y=False)
       

class boEI(boBASE):
    def run_optimization(self, max_iter):
        self.bo_org.run_optimization(max_iter, eps=-1)
    
    def set_bo(self):
        self.set_GPyOpt(algorism="EI")


class boRecsat(boBASE):
    def run_optimization(self, max_iter):
        self.bo_org.run_optimization(max_iter, eps=-1)
    
    def set_bo(self):
        self.set_GPyOptRecsat()
        
class boVanillaUCB(boBASE):                 
    def run_optimization(self, max_iter):
        self.bo_org.run_optimization(max_iter, eps=-1)
         
    def set_bo(self):
        self.set_GPyOpt(algorism="UCB")