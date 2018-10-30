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
"""
    References:
    [1] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch II, "Dual Vigilance 
        Fuzzy Adaptive Resonance Theory," Neural networks, vol. 109, pp. 1–5, 2019.
    [2] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch II, "Dual Vigilance 
        Fuzzy ART," 2018. [Online]. Available: https://github.com/ACIL-Group/DVFA
"""

import random

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from nuart.common.linear_algebra import max_norm, fuzzy_and

__author__ = 'Islam Elnabarawy'


class DualVigilanceART(BaseEstimator, ClusterMixin):
    def __init__(self, rho_ub, rho_lb, alpha, beta, max_epochs=np.inf, shuffle=True, random_seed=None, w_init=None,
                 cluster_map_init=None, distance_fn=None, match_fn=None, vigilance_fn=None, update_fn=None):
        self.rho_ub = rho_ub
        self.rho_lb = rho_lb
        self.alpha = alpha
        self.beta = beta

        self.max_epochs = max_epochs
        self.shuffle = shuffle
        self.random_seed = random_seed

        if w_init is not None and cluster_map_init is None:
            raise ValueError("Must specify cluster_map_init if w_init is specified.")
        if cluster_map_init is not None and w_init is None:
            raise ValueError("Must specify w_init if cluster_map_init is specified.")
        self.w_init = w_init
        self.cluster_map_init = cluster_map_init

        self.distance_fn = distance_fn
        self.match_fn = match_fn
        self.vigilance_fn = vigilance_fn
        self.update_fn = update_fn

        self.w = None
        self.num_clusters = None
        self.num_categories = None
        self.num_features = None
        self.cluster_map = None
        self.labels = None
        self.category_clusters = None
        self.iterations = 0

    def set_params(self, **params):
        keys = ['rho_ub', 'rho_lb', 'alpha', 'beta', 'w_init', 'max_epochs', 'shuffle', 'random_seed']

        for p in params.keys():
            if p not in keys:
                raise KeyError('Invalid parameter specified')
            self.__setattr__(p, params[p])

        return self

    def get_params(self, deep=True):
        return {
            'rho_ub': self.rho_ub, 'rho_lb': self.rho_lb, 'alpha': self.alpha, 'beta': self.beta,
            'w_init': self.w_init, 'max_epochs': self.max_epochs,
            'shuffle': self.shuffle, 'random_seed': self.random_seed
        }

    def fit(self, inputs, labels=None):

        self.num_features = inputs.shape[1]

        if self.w_init is not None:
            assert self.w_init.shape[1] == (self.num_features * 2)

        self.w = self.w_init if self.w_init is not None else np.ones((1, self.num_features * 2))
        self.cluster_map = self.cluster_map_init if self.cluster_map_init is not None else {0: 0}
        self.num_categories = self.w.shape[0] - 1
        self.num_clusters = len(set(self.cluster_map.values()))

        # complement-code the data
        dataset = np.concatenate((inputs, 1 - inputs), axis=1)

        # initialize variables
        self.labels = np.zeros(dataset.shape[0])
        self.iterations = 0
        w_old = None

        if self.shuffle and self.random_seed is not None:
            random.seed(self.random_seed)

        # repeat the learning until either convergence or max_epochs
        while not np.array_equal(self.w, w_old) and self.iterations < self.max_epochs:
            w_old = self.w
            self.labels = np.zeros(dataset.shape[0])

            indices = list(range(dataset.shape[0]))

            if self.shuffle:
                random.shuffle(indices)

            # present the input patters to the ART module
            for ix in indices:
                self.labels[ix] = self.train_pattern(dataset[ix, :])

            self.iterations += 1

        # return results
        return self.labels

    def predict(self, inputs):
        dataset = np.concatenate((inputs, 1 - inputs), axis=1)
        labels = np.array(list(map(self.eval_pattern, dataset)), dtype=np.int32)
        return labels

    def train_pattern(self, pattern):
        # evaluate the pattern to get the winning category
        winner, cluster = self.eval_pattern(pattern)

        # update the weight of the winning neuron
        self.w[winner, :] = self.weight_update(pattern, self.w[winner, :], self.beta)

        # update the cluster mapping of the winning neuron
        self.cluster_map[winner] = cluster

        # check if the uncommitted node was the winner
        if (winner + 1) > self.num_categories:
            self.num_categories += 1
            self.w = np.concatenate((self.w, np.ones((1, self.w.shape[1]))))

        # check if a new cluster was created
        if (cluster + 1) > self.num_clusters:
            self.num_clusters += 1

        return cluster

    def eval_pattern(self, pattern):
        # initialize variables
        matches = np.zeros(self.w.shape[0])
        # calculate the category match values
        for jx in range(self.w.shape[0]):
            matches[jx] = self.category_choice(pattern, self.w[jx, :], self.alpha)
        # pick the winning category
        match_attempts = 0
        while match_attempts < len(matches):
            # winner-take-all selection
            winner = np.argmax(matches)
            # vigilance test
            if self.vigilance_check(pattern, self.w[winner, :], self.rho_ub):
                # the winning category passed the upper bound vigilance test
                return winner, self.cluster_map[winner]
            elif self.vigilance_check(pattern, self.w[winner, :], self.rho_lb):
                # the winning category passed the lower bound vigilance test
                # create a new category but map it to the same label as winning category
                return len(matches) - 1, self.cluster_map[winner]
            else:
                # shut off this category from further testing
                matches[winner] = 0
                match_attempts += 1
        return len(matches) - 1, self.num_clusters  # create a new category and a new cluster

    def pattern_compare(self, pattern, category_w):
        if self.distance_fn is not None:
            return self.distance_fn(pattern, category_w)
        return fuzzy_and(pattern, category_w)

    def category_choice(self, pattern, category_w, alpha):
        if self.match_fn is not None:
            return self.match_fn(pattern, category_w, alpha)
        return max_norm(self.pattern_compare(pattern, category_w)) / (alpha + max_norm(category_w))

    def vigilance_check(self, pattern, category_w, rho):
        if self.vigilance_fn is not None:
            return self.vigilance_fn(pattern, category_w, rho)
        return max_norm(self.pattern_compare(pattern, category_w)) >= rho * max_norm(pattern)

    def weight_update(self, pattern, category_w, beta):
        if self.update_fn is not None:
            return self.update_fn(pattern, category_w, beta)
        return beta * self.pattern_compare(pattern, category_w) + (1 - beta) * category_w
