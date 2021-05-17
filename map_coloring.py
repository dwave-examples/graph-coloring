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
import argparse
import sys

import geopandas
import matplotlib
import networkx as nx
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

# Data file from https://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/bound-limit-2011-eng.cfm

def read_in_args(args):
    """ Read in user specified parameters."""

    # Set up user-specified optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country", default='usa', choices=['usa', 'canada'], help='Color either USA or Canada map (default: %(default)s)')

    return parser.parse_args(args)

def build_map(shp_file):
    """ Builds country map from shp file into geopandas dataframe."""

    print("\nBuilding map...")

    states = geopandas.read_file(shp_file)

    state_neighbors = defaultdict(list)
    for _, state in states.iterrows():   
        
        # add entry to dictionary
        neighbors = state['NEIGHBORS']

        try:
            neighbors = neighbors.split(",")
        except:
            neighbors = []

        state_neighbors[state['NAME']] = neighbors

    return states, state_neighbors

def draw_solid_map(states):
    """ Draw map in one color."""

    states.plot()
    plt.show()

    return

def build_graph(state_neighbors):
    """ Build graph corresponding to neighbor relation."""

    print("\nBuilding graph from map...")

    G = nx.Graph()
    for key, val in state_neighbors.items():
        for nbr in val:
            G.add_edge(key, nbr)

    return G

def build_dqm(G, num_colors):
    """ Build DQM model."""

    print("\nBuilding discrete quadratic model...")

    colors = range(num_colors)

    # Initialize the DQM object
    dqm = DiscreteQuadraticModel()

    # initial value of Lagrange parameter
    lagrange = max(colors)

    # Load the DQM. Define the variables, and then set biases and weights.
    # We set the linear biases to favor lower-numbered colors; this will
    # have the effect of minimizing the number of colors used.
    # We penalize edge connections by the Lagrange parameter, to encourage
    # connected nodes to have different colors.
    for v in G.nodes:
        dqm.add_variable(num_colors, label=v)
    for v in G.nodes:
        dqm.set_linear(v, colors)
    for u, v in G.edges:
        dqm.set_quadratic(u, v, {(c, c): lagrange for c in colors})

    return dqm

def run_hybrid_solver(dqm):
    """ Solve DQM using hybrid solver through the cloud."""

    print("\nRunning hybrid sampler...")

    # Initialize the DQM solver
    sampler = LeapHybridDQMSampler()

    # Solve the problem using the DQM solver
    sampleset = sampler.sample_dqm(dqm, label='Example - Map Coloring')

    return sampleset

def draw_sample(sample, states):
    """ Draw best result found."""

    print("\nProcessing sample...")

    states['COLOR'] = [0] * states.shape[0]

    for key, val in sample.items():
        row = states.index[states['NAME'] == key]
        states.at[row, 'COLOR'] = val

    states.plot(column='COLOR')
    plt.axis('off')
    plt.savefig("map_result.png")

    return

# ------- Main program -------
if __name__ == "__main__":

    args = read_in_args(sys.argv[1:])

    if args.country == 'canada':
        print("\nCanada map coloring demo.")
        input_shp_file = 'shp_files/canada/canada.shp'
    else:
        print("\nUSA map coloring demo.")
        input_shp_file = 'shp_files/usa/usa.shp'

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
