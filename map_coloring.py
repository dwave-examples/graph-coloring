# Copyright 2021 D-Wave Systems Inc.
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

from collections import defaultdict

import geopandas
import matplotlib.pyplot as plt
import networkx as nx
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

# Data file from https://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/bound-limit-2011-eng.cfm

def build_map(shp_file):

    print("\nBuilding Canada map...")

    states = geopandas.read_file(shp_file)

    state_neighbors = defaultdict(list)
    for _, state in states.iterrows():   

        # get 'not disjoint' states
        neighbors = states[~states.geometry.disjoint(state.geometry)].PRNAME.tolist()

        # remove own name of the state from the list
        neighbors = [ name for name in neighbors if state.PRNAME != name ]
        
        # add entry to dictionary
        state_neighbors[state['PRNAME']] = neighbors

    return states, state_neighbors

def draw_solid_map(states):

    states.plot()
    plt.show()

    return

def build_graph(state_neighbors):

    print("\nBuilding graph from map...")

    G = nx.Graph()
    for key, val in state_neighbors.items():
        for nbr in val:
            G.add_edge(key, nbr)

    return G

def build_dqm(G, num_colors):

    print("\nBuilding discrete quadratic model...")

    colors = range(num_colors)

    # Initialize the DQM object
    dqm = DiscreteQuadraticModel()

    # n_edges = len(G.edges)

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

    return dqm

def run_hybrid_solver(dqm):

    print("\nRunning hybrid sampler...")

    # Initialize the DQM solver
    sampler = LeapHybridDQMSampler()

    # Solve the problem using the DQM solver
    sampleset = sampler.sample_dqm(dqm, label='Example - CN Map Coloring')

    return sampleset

def draw_sample(sample, states):

    print("\nProcessing sample...")

    color_column = [0] * states.shape[0]

    states['COLOR'] = color_column

    for key, val in sample.items():
        row = states.index[states['PRNAME'] == key]
        states.at[row, 'COLOR'] = val

    states.plot(column='COLOR')
    plt.axis('off')
    plt.savefig("result_canada.png")

    return

# ------- Main program -------
if __name__ == "__main__":

    input_shp_file = 'shp_files/gpr_000b11a_e/gpr_000b11a_e.shp'

    states, state_neighbors = build_map(input_shp_file)

    G = build_graph(state_neighbors)

    num_colors = 7

    dqm = build_dqm(G, num_colors)

    sampleset = run_hybrid_solver(dqm)

    # get the first solution, and print it
    sample = sampleset.first.sample

    colors_used = max(sample.values())+1

    draw_sample(sample, states)

    print("\nColors required:", colors_used, "\n")
