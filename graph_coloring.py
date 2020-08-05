# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx
from dqmclient import solve
import numpy as np

# Graph coloring with DQM solver

# input: number of colors in the graph
num_colors = 4

# Make Networkx graph of a hexagon
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)])
n_edges = len(G)

# initial setup of linear list
linear = [(i, j, 0) for i in range(n_edges) for j in range(num_colors)]

# introduce ability to refer to variables by name, and make the third
# variable float
linear = np.asarray(linear, dtype=[('v', np.intc), ('case', np.intc),
                    ('bias', np.float)])

# the gradient will be used as the bias. Start at zero and go up to the
# number of colors, in steps. This will favor lowest-numbered colors and
# penalize higher-numbered colors. We're assuming a linear relationship
# as a first guess.
gradient = np.linspace(0, num_colors - 1, num_colors)
linear['bias'] = np.tile(gradient, n_edges)

# define the connectivity dict as the graph
connectivity = np.asarray(G.edges, dtype=[('u', np.intc), ('v', np.intc)])

# initial value of Lagrange parameter
lagrange = max(gradient)

# We use an identity matrix to help set up the quadratic. Whenever
# (node, color) is the same - the on-diagonal terms - we penalize it with
# strength 'lagrange'.
# All other terms are zero.
coloring_objective = np.eye(num_colors).reshape([-1]) * lagrange
quadratic = np.tile(coloring_objective, len(connectivity))

# DQM solver parameters
num_reads = 10
num_sweeps = 10

# use dqmclient's solve to get to the solver
sampleset = solve(linear, connectivity, quadratic,
                  num_reads=num_reads, num_sweeps=num_sweeps,
                  profile='dqm_prod', connection_close=True)

# get the first solution
sample = sampleset.first.sample
energy = sampleset.first.energy

# check that colors are different
valid = True
for edge in G.edges:
    i, j = edge
    if sample[i] == sample[j]:
        valid = False
        break
print("Graph coloring solution: ", sample)
print("Graph coloring solution energy: ", energy)
print("Graph coloring solution validity: ", valid)
