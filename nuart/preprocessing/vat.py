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
# [1] J. C. Bezdek and R. J. Hathaway, "VAT: a tool for visual assessment of (cluster) tendency," in Proceedings
#     of the 2002 International Joint Conference on Neural Networks. IJCNN'02 (Cat. No.02CH37290), pp. 2225–2230.

import numpy as np
from scipy.spatial.distance import pdist, squareform

__author__ = "Islam Elnabarawy"


def vat(data):
    num_samples = data.shape[0]
    pairwise_dist = squareform(pdist(data))
    indicies = []
    remaining = list(range(num_samples))

    _, jx = np.unravel_index(pairwise_dist.argmax(), pairwise_dist.shape)
    indicies.append(jx)
    remaining.pop(jx)

    while remaining:
        sub_matrix = pairwise_dist[indicies][:, remaining]
        _, jx = np.unravel_index(sub_matrix.argmin(), sub_matrix.shape)
        indicies.append(remaining[jx])
        remaining.pop(jx)

    return indicies
