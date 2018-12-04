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
import itertools
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial.distance import pdist

from nuart.clustering import HypersphereART
from nuart.clustering.hypersphere_art import ha_cluster
from nuart.preprocessing import vat

__author__ = "Islam Elnabarawy"

if __name__ == '__main__':
    # load the data
    data = np.loadtxt('data/csv/Target.csv', delimiter=',')
    inputs = data[:, :-1]
    targets = data[:, -1]

    # visualize the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap='Set1')
    fig.show()

    # apply vat ordering
    vat_ix = vat(inputs)
    inputs = inputs[vat_ix, :]
    targets = targets[vat_ix]

    # run a parameter study
    r_max = pdist(inputs).max() / 2
    rho_step, r_bar_step = 0.05, 0.05
    rho_values = np.arange(0, 1, rho_step)
    r_bar_values = np.arange(r_max, 2 * r_max, r_bar_step)
    args_comb = list(itertools.product(rho_values, r_bar_values))
    with mp.Pool() as pool:
        ari_values = list(pool.map(ha_cluster, [(rho, r_bar, inputs, targets) for rho, r_bar in args_comb]))

    # find the best performance parameters
    best_ix = np.argmax(ari_values)
    best_rho, best_r_bar = args_comb[best_ix]
    best_ari = ari_values[best_ix]

    # show ARI scores
    plt.figure()
    plt.plot(ari_values)
    plt.title('Best rho: {:.2}, r_bar: {:.4} (ARI: {:.2})'.format(best_rho, best_r_bar, best_ari))
    plt.show()

    # visualize ARI scores on a rho/r_bar matrix
    ari_matrix = np.zeros((rho_values.size, r_bar_values.size)) - 1
    for ix, ari in enumerate(ari_values):
        i, j = (int(round((y - args_comb[0][x]) / (rho_step, r_bar_step)[x])) for x, y in enumerate(args_comb[ix]))
        ari_matrix[i, j] = ari

    plt.matshow(ari_matrix, aspect='auto', extent=[r_bar_values[0], r_bar_values[-1], rho_values[0], rho_values[-1]])
    plt.xlabel('r_bar')
    plt.ylabel('rho')
    plt.colorbar()
    plt.show()

    # rerun the best settings to get categories
    ha = HypersphereART(best_rho, best_r_bar, shuffle=False, max_epochs=1)
    labels = ha.fit(inputs)

    # visualize clusters
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap='Set1')
    for category in ha.w:
        r, (x, y) = category[0], category[1:]
        ax.add_patch(Circle((x, y), r, fill=False))
    plt.show()
