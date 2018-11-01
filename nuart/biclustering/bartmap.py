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
# [1] R. Xu and D. C. Wunsch II, "BARTMAP: A viable structure for biclustering,"
#     Neural Networks, vol. 24, no. 7, pp. 709-716, 2011.
# [2] I. Elnabarawy, D. C. Wunsch II, and A. M. Abdelbar, "Biclustering ARTMAP
#     Collaborative Filtering Recommender System," in Proceedings of the 2016 International
#     Joint Conference on Neural Networks (IJCNN ’16), 2016, pp. 2986-2991.

import multiprocessing
import random

import numpy as np
from sklearn import preprocessing

from nuart.common.linear_algebra import fuzzy_and, max_norm

__author__ = 'Islam Elnabarawy'


class FuzzyARTModule(object):
    def __init__(self, rho, alpha, beta, num_features):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        self.num_clusters = 0
        self.num_features = num_features
        self.w = np.ones((self.num_clusters, self.num_features * 2))

    def train_dataset(self, dataset, max_epochs=np.inf, shuffle=False, random_seed=None):
        # complement-code the data
        dataset = np.concatenate((dataset, 1 - dataset), axis=1)

        # initialize variables
        labels = np.zeros(dataset.shape[0])
        iterations = 0
        w_old = None

        indices = list(range(dataset.shape[0]))

        if shuffle:
            if random_seed is not None:
                random.seed(random_seed)
            random.shuffle(indices)

        while not np.array_equal(self.w, w_old) and iterations < max_epochs:
            w_old = self.w
            for ix in indices:
                labels[ix] = self.train_pattern(dataset[ix, :])
            iterations += 1

        return labels, iterations

    def train_pattern(self, pattern):
        # evaluate the pattern to get the winning category
        winner = self.eval_pattern(pattern)

        # commit the pattern to the winning category
        self.commit_pattern(pattern, winner)

        return winner

    def commit_pattern(self, pattern, category):
        # check if the uncommitted node was the winner
        if (category + 1) > self.num_clusters:
            self.num_clusters += 1
            self.w = np.concatenate((self.w, np.ones((1, self.w.shape[1]))))

        # update the weight of the winning neuron
        self.w[category, :] = self.weight_update(pattern, self.w[category, :], self.beta)

    def eval_pattern(self, pattern):
        # initialize variables
        matches = np.zeros(self.num_clusters)

        # calculate the category match values
        for jx in range(self.num_clusters):
            matches[jx] = self.category_choice(pattern, self.w[jx, :], self.alpha)

        # pick the winning category
        match_attempts = 0

        while match_attempts < self.num_clusters:
            # winner-take-all selection
            winner = np.argmax(matches)

            # vigilance test
            if self.vigilance_check(pattern, self.w[winner, :], self.rho):
                # the winning category passed the vigilance test
                return winner
            else:
                # shut off this category from further testing
                matches[winner] = 0
                match_attempts += 1

        return self.num_clusters

    @staticmethod
    def category_choice(pattern, category_w, alpha):
        return max_norm(fuzzy_and(pattern, category_w)) / (alpha + max_norm(category_w))

    @staticmethod
    def vigilance_check(pattern, category_w, rho):
        return max_norm(fuzzy_and(pattern, category_w)) >= rho * max_norm(pattern)

    @staticmethod
    def weight_update(pattern, category_w, beta):
        return beta * fuzzy_and(pattern, category_w) + (1 - beta) * category_w


class BARTMAP(object):
    def __init__(self, arta_settings, artb_settings, corr_thresh, step_size):
        """
        Create a Biclustering ARTMAP object

        :param artb_settings: A 3-tuple containing the rho, alpha, and beta parameters of ARTa
        :param arta_settings: A 3-tuple containing the rho, alpha, and beta parameters of ARTb
        :param corr_thresh: A float specifying the correlation threshold to use for BARTMAP's inter-ART module
        :param step_size: The step size parameter for BARTMAP's inter-ART module
        """
        super(BARTMAP, self).__init__()

        self.arta_settings = arta_settings
        self.artb_settings = artb_settings
        self.corr_thresh = corr_thresh
        self.step_size = step_size

        self.num_samples = None
        self.num_features = None

        self.ARTa = None
        self.ARTb = None

        self.sample_labels = None
        self.num_sample_labels = 0

        self.feature_labels = None
        self.num_feature_labels = 0

        self.map = map

    def train(self, data):
        sample_data = preprocessing.MinMaxScaler().fit_transform(data)
        feature_data = preprocessing.MinMaxScaler().fit_transform(data.transpose())

        return self.train_preprocessed(sample_data, feature_data)

    def train_preprocessed(self, sample_data, feature_data):

        pool = multiprocessing.Pool()
        self.map = pool.map

        self.num_samples, self.num_features = sample_data.shape
        self.ARTa = FuzzyARTModule(*self.arta_settings, self.num_features)
        self.ARTb = FuzzyARTModule(*self.artb_settings, self.num_samples)

        self.feature_labels, _ = self.ARTb.train_dataset(feature_data)
        self.num_feature_labels = self.ARTb.num_clusters

        self.sample_labels = np.zeros(self.num_samples, dtype=np.int32)
        self.num_sample_labels = 0

        for ix in range(self.num_samples):
            # re-initialize the ARTa vigilance parameter for each sample
            self.ARTa.rho = self.arta_settings[0]

            sample = np.concatenate([sample_data[ix, :], 1 - sample_data[ix, :]], axis=0)

            while True:
                sample_category = self.ARTa.eval_pattern(sample)

                if sample_category == self.ARTa.num_clusters:
                    # new cluster created; always allow new clusters
                    self.ARTa.commit_pattern(sample, sample_category)
                    self.sample_labels[ix] = sample_category
                    self.num_sample_labels += 1
                    break
                else:
                    # the sample was assigned to an existing cluster; check correlation threshold
                    sample_cluster = sample_data[np.nonzero(self.sample_labels[:ix] == sample_category)]
                    correlations = np.array([
                        self.get_bicluster_correlations(jx, sample, sample_cluster)
                        for jx in range(self.num_feature_labels)
                    ])

                    # check the correlations against the threshold
                    if np.any(correlations > self.corr_thresh) or self.ARTa.rho >= 1:
                        # allow this sample to be committed into the bicluster
                        self.ARTa.commit_pattern(sample, sample_category)
                        self.sample_labels[ix] = sample_category
                        break
                    else:
                        # increase the ARTa vigilance threshold and try again
                        self.ARTa.rho += self.step_size
                        if self.ARTa.rho > 1:
                            self.ARTa.rho = 1

        pool.close()
        self.map = map

    def get_bicluster_correlations(self, jx, sample, sample_cluster):
        feature_ix = np.nonzero(self.feature_labels == jx)
        return self.get_average_correlation(sample_cluster[:, feature_ix], sample[feature_ix])

    def get_average_correlation(self, bicluster, sample):
        # compute the average of the correlation between each pair of samples
        return np.array(list(self.map(BARTMAP.get_correlation_args, [(row, sample) for row in bicluster]))).mean()

    @staticmethod
    def get_correlation_args(args):
        return BARTMAP.get_correlation(*args)

    @staticmethod
    def get_correlation(x, y):
        # compute the terms for all the item values
        terms1 = x - np.mean(x)
        terms2 = y - np.mean(y)
        # compute the sums to find the pairwise correlation
        numerator = np.sum(np.multiply(terms1, terms2))
        root1 = np.sqrt(np.sum(np.multiply(terms1, terms1)))
        root2 = np.sqrt(np.sum(np.multiply(terms2, terms2)))
        return numerator / (root1 * root2) if root1 != 0 and root2 != 0 else 0
