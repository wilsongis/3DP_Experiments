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

from GPyOpt.acquisitions.base import AcquisitionBase
import numpy as np
from scipy.stats import norm

    
class AcquisitionRecsat(AcquisitionBase):
    """
    Probability of achievement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :goals: Predifined goals
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function(not used)

    .. Note:: does not allow to be used with cost

    """

    #analytical_gradient_prediction = True
    analytical_gradient_prediction = False

    def __init__(self, model, space, goals=None, optimizer=None, cost_withGradients=None):
        self.optimizer = optimizer
        super(AcquisitionRecsat, self).__init__(model, space, optimizer)
        assert goals != None, "No goal! Plaese set a goal."
        self.goals = goals

        if cost_withGradients is not None:
            print('The set cost function is ignored! Recsat acquisition does not make sense with cost.')  

    def _compute_acq(self, x):
        """
        Computes the log of probability of achievement
        """

        goals = self.model.re_normalizeY(np.array(self.goals).reshape(1,-1))
        goals = goals.flatten()
        m, s = self.model.predict(x) 
        d = [(mu-g)/si for mu, g, si in zip(m, goals, s)]            
        logs = [norm.logsf(dis).reshape(-1,1) for dis in d]
        f_acqu = np.concatenate(logs, axis=1).sum(axis=1).reshape(-1,1)
        return f_acqu