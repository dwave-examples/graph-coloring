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
import matplotlib

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

# Graph coloring with DQM solver

# input: number of colors in the graph
# the four-color theorem indicates that four colors suffice for any planar
# graph

print("\nSetting up graph...")
num_colors = 7
colors = range(num_colors)

# Make networkx graph
G = nx.powerlaw_cluster_graph(50, 3, 0.4)
pos = nx.spring_layout(G)
nx.draw(G, pos=pos, node_size=50, edgecolors='k', cmap='hsv')
plt.savefig("original_graph.png")
plt.clf()

# initial value of Lagrange parameter
lagrange = max(colors)

# Load the DQM. Define the variables, and then set biases and weights.
# We set the linear biases to favor lower-numbered colors; this will
# have the effect of minimizing the number of colors used.
# We penalize edge connections by the Lagrange parameter, to encourage
# connected nodes to have different colors.
print("\nBuilding discrete model...")

# Initialize the DQM object
dqm = DiscreteQuadraticModel()

# Add the variables
for p in G.nodes:
    dqm.add_variable(num_colors, label=p)

# Add the biases
for v in G.nodes:
    dqm.set_linear(v, colors)
for u, v in G.edges:
    dqm.set_quadratic(u, v, {(c, c): lagrange for c in colors})

# Initialize the DQM solver
print("\nRunning model on DQM sampler...")
sampler = LeapHybridDQMSampler()

# Solve the problem using the DQM solver
sampleset = sampler.sample_dqm(dqm, label='Example - Graph Coloring')

# get the first solution, and print it
sample = sampleset.first.sample
node_colors = [sample[i] for i in G.nodes()]
nx.draw(G, pos=pos, node_color=node_colors, node_size=50, edgecolors='k', cmap='hsv')
plt.savefig('graph_result.png')

# check that colors are different
valid = True
for edge in G.edges:
    i, j = edge
    if sample[i] == sample[j]:
        valid = False
        break
print("\nSolution validity: ", valid)

colors_used = max(sample.values())+1
print("\nColors required:", colors_used, "\n")
