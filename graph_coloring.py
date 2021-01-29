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
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

# Graph coloring with DQM solver

# input: number of colors in the graph
# the four-color theorem indicates that four colors suffice for any planar
# graph
num_colors = 4
colors = range(num_colors)

# Initialize the DQM object
dqm = DiscreteQuadraticModel()

# Make Networkx graph of a hexagon
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)])
n_edges = len(G.edges)

# initial value of Lagrange parameter
lagrange = max(colors)

# Load the DQM. Define the variables, and then set biases and weights.
# We set the linear biases to favor lower-numbered colors; this will
# have the effect of minimizing the number of colors used.
# We penalize edge connections by the Lagrange parameter, to encourage
# connected nodes to have different colors.
for p in G.nodes:
    dqm.add_variable(num_colors, label=p)
for p in G.nodes:
    dqm.set_linear(p, colors)
for p0, p1 in G.edges:
    dqm.set_quadratic(p0, p1, {(c, c): lagrange for c in colors})

# Initialize the DQM solver
sampler = LeapHybridDQMSampler()

# Solve the problem using the DQM solver
sampleset = sampler.sample_dqm(dqm, label='Example - Graph Coloring')

# get the first solution, and print it
sample = sampleset.first.sample
energy = sampleset.first.energy

# check that colors are different
valid = True
for edge in G.edges:
    i, j = edge
    if sample[i] == sample[j]:
        valid = False
        break
print("Solution: ", sample)
print("Solution energy: ", energy)
print("Solution validity: ", valid)
