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
from . import method
import numpy as np
from matplotlib import pyplot as plt
from matopt import util

class InvalidArgumentError(Exception):
    pass

class BOtest:
    """
    Newly implemented class to handle benchmark task using GPyOpt and benchmark problems.

    param funcname: name of benchmark function.
    param flexible_dim: user degined dimension of of the benchmark function (Not used in the study of the goal-oriented Bayesian optimization) 
    param ASF: name of the scalarization function.
    param ASFconfig: setting of the scalarization function.
    param optimization: bool.


    """
    def __init__(self, funcname, flexible_dim=10, ASF=None, ASFconfig=None, optimization=True):
        self.ASF = ASF
        self.ASFconfig = ASFconfig
        self.dim = flexible_dim
        self.results = []
        self.simplex01 = False
        self.__set_testfunc(funcname, optimization)
        
    def _apply_ASF(self, y, optimization=True):
        if optimization:
            if not hasattr(self.bowrap.bo_org, "y_mult"):
                self.bowrap.bo_org.y_mult = []
            self.bowrap.bo_org.y_mult.append(y)
        
        if self.ASF == None:
            return y

        elif self.ASF == "weightsum":
            assert len(y.T) == len(self.ASFconfig["weight"]), "Number of objectives in y and weights in ASFconfig are different."
            return (y * np.array(self.ASFconfig["weight"])).sum(axis=1).reshape(-1,1)

        elif self.ASF == "achievement":
            goals = np.array(self.bowrap.goals).flatten()
            rho = 0.05
            assert len(y.T) == len(goals), "Number of objectives in y and goals are different."
            y_all = np.concatenate(self.bowrap.bo_org.y_mult, axis=0)
            pareto = util.get_non_dominated_solutions(y_all)
            maxi = pareto.max(axis=0)
            mini = pareto.min(axis=0)                
            scale = maxi-mini
            # Scalling is not used when there is zero in scale.
            if 0 in scale:
                scale = np.ones(len(scale))
            weight = 1/(scale)
            aug_term = (y_all*weight).sum(axis=1).reshape(-1,1)*rho
            achieve_term = ((y_all-goals)*weight).max(axis=1).reshape(-1,1)
            scalar = aug_term + achieve_term
            self.bowrap.bo_org.Y = scalar[:-len(y)]
            return scalar[-len(y):]
        else:
            raise InvalidArgumentError("Select ASF from None, weightsum or achievement")
            
   
    def __makeMulFonseca(self, optimization=True):

        self.dim = 2
        def func(x):
            obj_0 = 1 - np.exp( - np.sum((x - 1. / np.sqrt(len(x.T)))**2, axis=1))
            obj_0 = obj_0.reshape(-1,1)
            obj_1 = 1 - np.exp( - np.sum((x + 1. / np.sqrt(len(x.T)))**2, axis=1))
            obj_1 = obj_1.reshape(-1,1)
            y = np.concatenate([obj_0, obj_1], axis=1)
            return self._apply_ASF(y, optimization)
        space = GPyOpt.Design_space\
        (space =[{"name": "var"+str(idx), "type": 'continuous', 'domain': (-4,4)} for idx in range(self.dim)])
        return func, space    

    def __makeMulKursawe(self, optimization=True):

        self.dim = 3
        a = 0.8
        b = 3        
        def func(x):
            x0 = x[:,0].reshape(-1,1)
            x1 = x[:,1].reshape(-1,1)
            x2 = x[:,2].reshape(-1,1)
            obj_0 = -10 * np.exp(-0.2*np.sqrt(x0**2 + x1**2))\
                    -10 * np.exp(-0.2*np.sqrt(x1**2 + x2**2))
            obj_1 = np.absolute(x)**a + 5*(np.sin(x))**b
            obj_1 = obj_1.sum(axis=1).reshape(-1,1)
            
            y = np.concatenate([obj_0, obj_1], axis=1) 
            return self._apply_ASF(y, optimization)
        space = GPyOpt.Design_space\
        (space =[{"name": "var"+str(idx), "type": 'continuous', 'domain': (-5,5)} for idx in range(self.dim)])
        return func, space  

    def __makeMulViennet(self, optimization=True):

        self.dim = 2
        def func(x):
            x0 = x[:,0].reshape(-1,1)
            x1 = x[:,1].reshape(-1,1)
            x2y2 = (x**2).sum(axis=1).reshape(-1,1)
            obj_0 = 0.5*x2y2+np.sin(x2y2)
            obj_1 = ((3*x0 - 2*x1 + 4)**2)/8\
                   +((1*x0 - 1*x1 + 1)**2)/27\
                   + 15
            obj_2 = 1/(x2y2 + 1)\
                   - 1.1*np.e**(-x2y2)
            y = np.concatenate([obj_0, obj_1, obj_2], axis=1) 
            return self._apply_ASF(y, optimization)
        space = GPyOpt.Design_space\
        (space =[{"name": "var"+str(idx), "type": 'continuous', 'domain': (-30,30)} for idx in range(self.dim)])
        return func, space   
    
    def __makeMulZDT1(self, optimization=True):

        self.dim = 30
        n = self.dim
        def func(x):
            obj_0 = x[:,0].reshape(-1,1)
            g = 1 + 9/(n-1)*x[:,1:].sum(axis=1).reshape(-1,1)
            obj_1 = g * (1 - np.sqrt(obj_0/g))
            y = np.concatenate([obj_0, obj_1], axis=1)
            return self._apply_ASF(y, optimization)
        space = GPyOpt.Design_space\
        (space =[{"name": "var"+str(idx), "type": 'continuous', 'domain': (0,1)} for idx in range(self.dim)])
        return func, space   
    
    def __makeMulZDT2(self, optimization=True):

        self.dim = 30
        n = self.dim
        def func(x):
            obj_0 = x[:,0].reshape(-1,1)
            g = 1 + 9/(n-1)*x[:,1:].sum(axis=1).reshape(-1,1)
            obj_1 = g * (1 - (obj_0/g)**2)
            y = np.concatenate([obj_0, obj_1], axis=1)
            return self._apply_ASF(y, optimization)
        space = GPyOpt.Design_space\
        (space =[{"name": "var"+str(idx), "type": 'continuous', 'domain': (0,1)} for idx in range(self.dim)])
        return func, space
    
    def __makeMulZDT3(self, optimization=True):

        self.dim = 30
        n = self.dim
        def func(x):
            obj_0 = x[:,0].reshape(-1,1)
            g = 1 + 9/(n-1)*x[:,1:].sum(axis=1).reshape(-1,1)
            obj_1 = g * (1 - np.sqrt(obj_0/g) - (obj_0/g)*np.sin(10*np.pi*obj_0))
            y = np.concatenate([obj_0, obj_1], axis=1)
            return self._apply_ASF(y, optimization)
        space = GPyOpt.Design_space\
        (space =[{"name": "var"+str(idx), "type": 'continuous', 'domain': (0,1)} for idx in range(self.dim)])
        return func, space  
    
    def __makeOilSorbent(self, optimization=True):
        print("Maximization problems are converted to minimization problems by multiplying minus 1.")
        self.dim = 7
        def func(x):
            V1 = x[:,0].reshape(-1,1)
            V2 = x[:,1].reshape(-1,1)
            V3 = x[:,2].reshape(-1,1)
            V4 = x[:,3].reshape(-1,1)
            V5 = x[:,4].reshape(-1,1)
            V6 = x[:,5].reshape(-1,1)
            V7 = x[:,6].reshape(-1,1)

            WCA = -197.0928 - 78.3309*V1 + 98.6355*V2 + 300.0701*V3 + 89.8360*V4\
                + 208.2343*V5 + 332.9341*V6 + 135.6621*V7 - 11.0715*V1*V2\
                + 201.8934*V1*V3 + 17.1270*V1*V4 + 2.5198*V1*V5\
                - 109.3922*V1*V6 + 30.1607*V1*V7 - 46.1790*V2*V3\
                + 19.2888*V2*V4 - 102.9493*V2*V5 - 19.1245*V2*V6\
                + 53.6297*V2*V7 - 73.0649*V3*V4 - 37.7181*V3*V5\
                - 219.1268*V3*V6 - 55.3704*V3*V7 + 3.8778*V4*V5 - 6.9252*V4*V6\
                - 105.1650*V4*V7 - 34.3181*V5*V6 - 36.3892*V5*V7\
                - 82.3222*V6*V7 - 16.7536*V1*V1 - 45.6507*V2*V2 - 91.4134*V3*V3\
                - 76.8701*V5*V5
            WCA = np.array(WCA).reshape(-1,1)
    
            Q = -212.8531 + 245.7998*V1 - 127.3395*V2 + 305.8461*V3 + 638.1605*V4\
                + 301.2118*V5 - 451.3796*V6 - 115.5485*V7 + 42.8351*V1*V2\
                + 262.3775*V1*V3 - 103.5274*V1*V4 - 196.1568*V1*V5\
                - 394.7975*V1*V6 - 176.3341*V1*V7 + 74.8291*V2*V3\
                + 4.1557*V2*V4 - 133.8683*V2*V5 + 65.8711*V2*V6\
                - 42.6911*V2*V7 - 323.9363*V3*V4 - 107.3983*V3*V5\
                - 323.2353*V3*V6 + 46.9172*V3*V7 - 144.4199*V4*V5\
                + 272.3729*V4*V6 + 49.0799*V4*V7 + 318.4706*V5*V6\
                - 236.2498*V5*V7 + 252.4848*V6*V7 - 286.0182*V4*V4\
                + 393.5992*V6*V6
            Q = np.array(Q).reshape(-1,1)
        
            sigma = 7.7696 + 15.4344*V1 - 10.6190*V2 - 17.9367*V3 + 17.1385*V4 + 2.5026*V5\
                - 24.3010*V6 + 10.6058*V7 - 1.2041*V1*V2 - 37.2207*V1*V3\
                - 3.2265*V1*V4 + 7.3121*V1*V5 + 52.3994*V1*V6 + 9.7485*V1*V7\
                - 15.9371*V2*V3 - 1.1706*V2*V4 - 2.6297*V2*V5\
                + 7.0225*V2*V6 - 1.4938*V2*V7 + 30.2786*V3*V4 + 14.5061*V3*V5\
                + 48.5021*V3*V6 - 11.4857*V3*V7 - 3.1381*V4*V5\
                - 14.9747*V4*V6 + 4.5204*V4*V7 - 17.6907*V5*V6 - 19.2489*V5*V7\
                - 9.8219*V6*V7 - 18.7356*V1*V1 + 12.1928*V2*V2 - 17.5460*V4*V4\
                + 5.4997*V5*V5 - 26.2718*V6*V6
            sigma = np.array(sigma).reshape(-1,1)
    
            y = np.concatenate([WCA, Q, sigma], axis=1)*(-1)#non-scaled
            return self._apply_ASF(y, optimization)
        domains = [{"name":"PS:PAN ratio", 'domain':(0.3/0.7, 0.7/0.7), "type": 'continuous'},
                 {"name":"Feed rate(mL/h)", 'domain':(0.7/2, 2/2), "type": 'continuous'},
                 {"name":"Distance(cm)", 'domain':(12/24, 24/24), "type": 'continuous'},
                 {"name":"Mass fraction of solute", 'domain':(12/18,18/18), "type": 'continuous'},
                 {"name":"Mass fraction of SiO2 in solute ", 'domain':(0/20,20/20), "type": 'continuous'},
                 {"name":"Applied voltage(kV)", 'domain':(16/28,28/28), "type": 'continuous'},
                 {"name":"Inner diameter(mm)", 'domain':(0.41/1.32, 1.32/1.32), "type": 'continuous'}
                ]
        space = GPyOpt.Design_space(space=domains)
        return func, space  


     

    func_dict = {"MulFonseca":__makeMulFonseca, "MulKursawe":__makeMulKursawe,
                 "MulViennet":__makeMulViennet, "MulZDT1":__makeMulZDT1,
                 "MulZDT2":__makeMulZDT2, "MulZDT3":__makeMulZDT3,
                 "OilSorbent":__makeOilSorbent
                 }
    setting_dict = {"EI":method.boEI,
                    "Recsat":method.boRecsat,
                    "UCB":method.boVanillaUCB,
                    }

    def __set_testfunc(self, funcname, optimization=True):
        if not funcname in self.func_dict.keys():
            print(funcname, "does not exist in", self.func_dict.keys())
        else:
            self.funcname = funcname
            if optimization:
                self.f, self.space = self.func_dict[funcname](self)
            else:
                self.f, self.space = self.func_dict[funcname](self, optimization=False)
            
    def __set_setting(self, settingname):
        if not settingname in self.setting_dict.keys():
            print(settingname, "does not exist in", self.setting_dict.keys())
        else:
            self.settingname = settingname
            
    def set_special_params(self, paramdict):
        
        for param in paramdict.keys():
            validparam = ["kern", "delta", "N_init", 
            "custom_model", "acquisition_optimizer_type",\
            "max_AcOpt_iter", "evaluator_type", "batch_size", \
            "n_sample", "sampler_type", "X_init", "f", "space", "eps", \
            "constraint", "const_model", "const_func", "Y_init",\
            "constr_val", "constr_model", "constr_func", "goals",\
            "exploration_weight",\
            "exact_feval",
            "num_anchor_points"]
            if not param in validparam:
                raise InvalidArgumentError(param, "does not exist in", validparam)
        if "kern" in paramdict.keys():
            self.bowrap.kern = paramdict["kern"]
            self.bowrap.comment = self.bowrap.comment + ":"+"kern"+self.bowrap.kern.name
        if "delta" in paramdict.keys():
            self.bowrap.delta = paramdict["delta"]
            self.bowrap.comment = self.bowrap.comment + ":"+"delta"+str(self.bowrap.delta)
        if "N_init" in paramdict.keys():
            self.bowrap.N_init = paramdict["N_init"]
            self.bowrap.comment = self.bowrap.comment + ":"+"N_init"+str(self.bowrap.N_init)                  
        if "custom_model" in paramdict.keys():
            self.bowrap.custom_model = paramdict["custom_model"]
            self.bowrap.comment = self.bowrap.comment + ":"+"custom_model"
        if "acquisition_optimizer_type" in paramdict.keys():
            self.bowrap.acquisition_optimizer_type = paramdict["acquisition_optimizer_type"]
            self.bowrap.comment = self.bowrap.comment + ":"+"acq_opt_type"+str(self.bowrap.acquisition_optimizer_type)  
        if "max_AcOpt_iter" in paramdict.keys():
            self.bowrap.max_AcOpt_iter = paramdict["max_AcOpt_iter"]
            self.bowrap.comment = self.bowrap.comment + ":"+"max_AcOpt_iter"+str(self.bowrap.max_AcOpt_iter) 
        if "evaluator_type" in paramdict.keys():
            self.bowrap.evaluator_type = paramdict["evaluator_type"]
            self.bowrap.comment = self.bowrap.comment + ":"+"evaluator_type"+str(self.bowrap.evaluator_type) 
        if "batch_size" in paramdict.keys():
            self.bowrap.batch_size = paramdict["batch_size"]
            self.bowrap.comment = self.bowrap.comment + ":"+"batch_size"+str(self.bowrap.batch_size) 
        if "n_sample" in paramdict.keys():
            self.bowrap.n_sample = paramdict["n_sample"]
            self.bowrap.comment = self.bowrap.comment + ":"+"n_sample"+str(self.bowrap.n_sample) 
        if "sampler_type" in paramdict.keys():
            self.bowrap.sampler_type = paramdict["sampler_type"]
            self.bowrap.comment = self.bowrap.comment + ":"+"sampler_type"+str(self.bowrap.sampler_type) 
        if "X_init" in paramdict.keys():
            self.bowrap.X_init = paramdict["X_init"]
            self.bowrap.comment = self.bowrap.comment + ":"+"X_init_given"
        if "Y_init" in paramdict.keys():
            self.bowrap.Y_init = paramdict["Y_init"]
            self.bowrap.comment = self.bowrap.comment + ":"+"Y_init_given"
        if "f" in paramdict.keys():
            self.bowrap.f = paramdict["f"]
            self.bowrap.comment = self.bowrap.comment + ":"+"f_given"
        if "space" in paramdict.keys():
            self.bowrap.space = paramdict["space"]
            self.bowrap.comment = self.bowrap.comment + ":"+"space_given"
        if "eps" in paramdict.keys():
            self.bowrap.eps = paramdict["eps"]
            self.bowrap.comment = self.bowrap.comment + ":"+"eps"+str(self.bowrap.eps)         
        if "constr_val" in paramdict.keys():
            self.bowrap.constr_val = paramdict["constr_val"]
            self.bowrap.comment = self.bowrap.comment + ":"+"constr_val_given"   
        if "constr_model" in paramdict.keys():
            self.bowrap.constr_model = paramdict["constr_model"]
            self.bowrap.comment = self.bowrap.comment + ":"+"constr_model_given"
        if "constr_func" in paramdict.keys():
            self.bowrap.constr_func = paramdict["constr_func"]
            self.bowrap.comment = self.bowrap.comment + ":"+"constr_func_given"            
        if "goals" in paramdict.keys():
            self.bowrap.goals = paramdict["goals"]
            self.bowrap.comment = self.bowrap.comment + ":"+"goals" 
        if "exploration_weight" in paramdict.keys():
            self.bowrap.exploration_weight = paramdict["exploration_weight"]
            self.bowrap.comment = self.bowrap.comment + ":"+"exploration_weight"+ str(self.bowrap.exploration_weight)
        if "exact_feval" in paramdict.keys():
            self.bowrap.exact_feval = paramdict["exact_feval"]
            self.bowrap.comment = self.bowrap.comment + ":"+"exact_feval"
        if "num_anchor_points" in paramdict.keys():
            self.bowrap.num_anchor_points = paramdict["num_anchor_points"]        
        
        if not hasattr(self.bowrap, "kern"):
            self.bowrap.kern = None
        if not hasattr(self.bowrap, "N_init"):
            self.bowrap.N_init = self.dim+1
        if not hasattr(self.bowrap, "delta"):
            self.bowrap.delta = 0.1
        if not hasattr(self.bowrap, "max_AcOpt_iter"):
            self.bowrap.max_AcOpt_iter = 1000 
        if not hasattr(self.bowrap, "evaluator_type"):
            self.bowrap.evaluator_type = "sequential"
        if not hasattr(self.bowrap, "batch_size"):
            self.bowrap.batch_size = 1
        if not hasattr(self.bowrap, "n_sample"):
            self.bowrap.n_sample = 10000
        if not hasattr(self.bowrap, "sampler_type"):
            self.bowrap.sampler_type = "RSM"       
        if not hasattr(self.bowrap, "acquisition_optimizer_type"):
            self.bowrap.acquisition_optimizer_type = "lbfgs"  
        if not hasattr(self.bowrap, "eps"):
            self.bowrap.eps = 1e-8         
        if not hasattr(self.bowrap, "Y_init"):
            self.bowrap.Y_init = None  
        if not hasattr(self.bowrap, "goals"):
            self.bowrap.goals = []
        if not hasattr(self.bowrap, "exploration_weight"):
            self.bowrap.exploration_weight = 2   
        if not hasattr(self.bowrap, "exact_feval"):
            self.bowrap.exact_feval = False  
        if not hasattr(self.bowrap, "num_anchor_points"):
            self.bowrap.num_anchor_points = 1000         

        
    def run_BO(self, settingname, max_iter, sampling=1, comment="" ,restart=False , **paramdict):
        self.__set_setting(settingname)
        current_result = []
        for samp in range(sampling):
            if restart == True and sampling==1 and len(self.results)>0:
                self.bowrap = self.results[-1][-1]
            else:
                if restart == True:
                    restart=False
                    print("Restart is only valid with sampling = 1. and there must be previous run.")
                self.bowrap = self.setting_dict[self.settingname](self.f, self.space, self.dim)
                self.bowrap.comment=comment
                self.bowrap.settingname=self.settingname
                self.bowrap.funcname=self.funcname
                self.set_special_params(paramdict)
                self.bowrap.set_bo()  
            self.bowrap.run_optimization(max_iter=max_iter) # modification for non GP model that does not have model params
            #self.bowrap.run_optimization(max_iter=max_iter, save_models_parameters= False)
            current_result.append(self.bowrap)
        if restart == False:
            self.results.append(current_result)

    def get_nextexperiment(self, settingname, comment="", **paramdict):
        self.__set_setting(settingname)
        self.bowrap = self.setting_dict[self.settingname](self.f, self.space, self.dim)
        self.bowrap.comment=comment
        self.bowrap.settingname=self.settingname
        self.set_special_params(paramdict)
        self.bowrap.set_bo()
        return self.bowrap.suggest_next_locations()
    
    def get_acquisition_values(self, x):
        return self.bowrap.bo_org.acquisition.acquisition_function(x)

    def plot_average_goal_distance(self, short_traj=None):
        for current_result in self.results:
            label = str(current_result[0].settingname) + " " + str(current_result[0].funcname)\
            + " " + str(current_result[0].comment)
            if short_traj == None: 
                avecurve = np.array([[(np.absolute(np.concatenate(bowrap.bo_org.y_mult[:t], axis=0)\
                    -np.array(bowrap.goals).reshape(1,-1))).sum(axis=1).min()\
                        for t in range(1, len(bowrap.bo_org.y_mult)+1)]\
                        for bowrap in current_result ]).mean(axis=0)
                stdcurve = np.array([[(np.absolute(np.concatenate(bowrap.bo_org.y_mult[:t], axis=0)\
                    -np.array(bowrap.goals).reshape(1,-1))).sum(axis=1).min()\
                        for t in range(1, len(bowrap.bo_org.y_mult)+1)]\
                        for bowrap in current_result ]).std(axis=0)
            elif short_traj == "ignore":
                maxstep = np.array([len(bowrap.bo_org.Y) for bowrap in current_result ]).max()
                avecurve = np.array([[(np.absolute(np.concatenate(bowrap.bo_org.y_mult[:t], axis=0)\
                    -np.array(bowrap.goals).reshape(1,-1))).sum(axis=1).min()\
                        for t in range(1, len(bowrap.bo_org.y_mult)+1)]\
                        for bowrap in current_result if len(bowrap.bo_org.Y) == maxstep]).mean(axis=0)
                stdcurve = np.array([[(np.absolute(np.concatenate(bowrap.bo_org.y_mult[:t], axis=0)\
                    -np.array(bowrap.goals).reshape(1,-1))).sum(axis=1).min()\
                        for t in range(1, len(bowrap.bo_org.y_mult)+1)]\
                        for bowrap in current_result if len(bowrap.bo_org.Y) == maxstep]).std(axis=0)                
            else:
                print("Possible values of short_traj are None and ignore.")
            plt.plot(avecurve, label=label)
            plt.fill_between(range(len(avecurve)),
                             avecurve - stdcurve,
                             avecurve + stdcurve,
                             alpha=0.5,)
        plt.legend()

    def get_problem(self, settingname="UCB", **paramdict): 
        # coded to allow external access for problems given by botest.
        self.__set_setting(settingname)
        self.bowrap = self.setting_dict[self.settingname](self.f, self.space, self.dim)
        self.bowrap.settingname=self.settingname
        self.bowrap.funcname=self.funcname
        self.set_special_params(paramdict)
        self.bowrap.set_bo()
        return self.f, self.space.get_bounds()
    
    def sampling_problem(self, n_sampling, settingname="UCB", **paramdict):
        f, bounds = self.get_problem(settingname=settingname, **paramdict)
        if not self.simplex01:
            X = [(np.random.rand(n_sampling)*(bound[1]-bound[0])+bound[0]).reshape(-1,1) for bound in bounds]
            X = np.concatenate(X, axis=1)
        else:
            X = np.random.dirichlet([1]*(self.dim+1), n_sampling)
            X = X[:,:-1]
        Y = f(X)
        return X, Y
        
        