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
# References:
# [1] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch II, "Dual Vigilance
#     Fuzzy Adaptive Resonance Theory," Neural networks, vol. 109, pp. 1–5, 2019.
# [2] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch II, "Dual Vigilance
#     Fuzzy ART," 2018. [Online]. Available: https://github.com/ACIL-Group/DVFA
# [3] G. C. Anagnostopoulos and M. Georgiopoulos, "Ellipsoid ART and ARTMAP
#     for incremental clustering and classification," in Proceedings of the
#     International Joint Conference on Neural Networks (IJCNN '01), 2001,
#     vol. 2, pp. 1221-1226.

import random

import numpy as np
from numpy.linalg import norm as l2_norm
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClusterMixin

__author__ = 'Islam Elnabarawy'


class DualVigilanceHypersphereART(BaseEstimator, ClusterMixin):
    def __init__(self, rho_ub, rho_lb, r_bar, alpha=0.001, beta=1.0, r_max=None, max_epochs=np.inf, shuffle=True,
                 random_seed=None):
        self.rho_ub = rho_ub
        self.rho_lb = rho_lb
        self.r_bar = r_bar
        self.alpha = alpha
        self.beta = beta
        self.r_max = r_max

        self.max_epochs = max_epochs
        self.shuffle = shuffle
        self.random_seed = random_seed

        self.w = None
        self.num_clusters = None
        self.num_categories = None
        self.num_features = None
        self.cluster_map = None
        self.labels = None
        self.category_clusters = None
        self.iterations = 0

    def fit(self, inputs, labels=None):

        self.num_features = inputs.shape[1]

        self.w = np.empty((0, self.num_features + 1))
        self.cluster_map = {}
        self.num_categories = 0
        self.num_clusters = 0

        if self.r_max is None:
            self.r_max = pdist(inputs).max() / 2

        assert self.r_bar >= self.r_max

        # initialize variables
        self.iterations = 0
        w_old = None

        if self.shuffle and self.random_seed is not None:
            random.seed(self.random_seed)

        # repeat the learning until either convergence or max_epochs
        while not np.array_equal(self.w, w_old) and self.iterations < self.max_epochs:
            w_old = self.w
            self.labels = np.zeros(inputs.shape[0])

            indices = list(range(inputs.shape[0]))

            if self.shuffle:
                random.shuffle(indices)

            # present the input patters to the ART module
            for ix in indices:
                self.labels[ix] = self.train_pattern(inputs[ix, :])

            self.iterations += 1

        # return results
        return self.labels

    def predict(self, inputs):
        return np.array(list(map(self.eval_pattern, inputs)), dtype=np.int32)

    def train_pattern(self, pattern):
        # evaluate the pattern to get the winning category
        winner, cluster = self.eval_pattern(pattern)

        # check if the uncommitted node was the winner
        if (winner + 1) > self.num_categories:
            self.num_categories += 1
            self.w = np.concatenate((self.w, np.zeros((1, self.w.shape[1]))))
            self.w[-1, 1:] = pattern
        else:
            # update the weight of the winning neuron
            self.w[winner, :] = self.weight_update(pattern, self.w[winner, :], self.beta)

        # check if a new cluster was created
        if (cluster + 1) > self.num_clusters:
            self.num_clusters += 1

        # update the cluster mapping of the winning neuron
        self.cluster_map[winner] = cluster

        return cluster

    def eval_pattern(self, pattern):
        # calculate the category match values
        matches = np.array([self.category_choice(pattern, category, self.alpha, self.r_bar) for category in self.w])
        # pick the winning category using winner-take-all selection
        for ix in matches.argsort()[::-1]:
            if self.vigilance_check(pattern, self.w[ix, :], self.rho_ub, self.r_bar):
                # the winning category passed the upper bound vigilance test
                return ix, self.cluster_map[ix]
            if self.vigilance_check(pattern, self.w[ix, :], self.rho_lb, self.r_bar):
                # the winning category passed the lower bound vigilance test
                # create a new category but map it to the same label as winning category
                return self.num_categories, self.cluster_map[ix]
        # if none of the existing categories resonate, create a new category and a new cluster
        return self.num_categories, self.num_clusters

    @staticmethod
    def category_choice(pattern, category_w, alpha, r_bar):
        r_cat, m_cat = category_w[0], category_w[1:]
        return (r_bar - max(r_cat, l2_norm(pattern - m_cat))) / (r_bar - r_cat + alpha)

    @staticmethod
    def vigilance_check(pattern, category_w, rho, r_bar):
        r_cat, m_cat = category_w[0], category_w[1:]
        return (1 - (max(r_cat, l2_norm(pattern - m_cat)) / r_bar)) >= rho

    @staticmethod
    def weight_update(pattern, category_w, beta):
        r_old, m_old = category_w[0], category_w[1:]
        dist_old = pattern - m_old
        dist_norm = l2_norm(dist_old)
        r_new = r_old + 0.5 * beta * (max(r_old, dist_norm) - r_old)
        m_new = (m_old + 0.5 * beta * (1 - (min(r_old, dist_norm) / dist_norm)) * dist_old) if dist_norm else m_old
        return np.concatenate(([r_new], m_new))
