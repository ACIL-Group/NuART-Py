"""
   Copyright 2018 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
from numpy import linalg as la

__author__ = 'Islam Elnabarawy'


def max_norm(x):
    # noinspection PyTypeChecker
    return la.norm(x, ord=1)


def fuzzy_and(x, y):
    return np.min(np.array([x, y]), 0)


def fuzzy_or(x, y):
    return np.max(np.array([x, y]), 0)
