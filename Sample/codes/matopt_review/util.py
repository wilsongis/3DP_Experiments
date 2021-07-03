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
def get_non_dominated_solutions(X):
    non_dominated = []
    X = np.array(X)
    for x in X:
        diff = np.array(X) - x
        bool_dominated1 = (diff <= 0).all(axis=1)
        bool_dominated2 = (diff < 0).any(axis=1)
        bool_dominated = np.array([b1 and b2 for b1, b2 in zip(bool_dominated1, bool_dominated2)])
        if not True in bool_dominated:
            non_dominated.append(x) 
    return np.array(non_dominated)