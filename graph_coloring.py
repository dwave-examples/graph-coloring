# Copyright 2022 D-Wave Systems Inc.
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

import matplotlib
import networkx as nx
from dimod import CQM, BinaryQuadraticModel
from dwave.system import LeapHybridCQMSampler

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def build_graph(num_nodes):
    """Build graph."""

    print("\nBuilding graph...")

    G = nx.powerlaw_cluster_graph(num_nodes, 3, 0.4)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_size=50, edgecolors='k', cmap='hsv')
    plt.savefig("original_graph.png")
    plt.clf()

    return G, pos

def build_cqm(G, num_colors):
    """Build CQM model."""

    print("\nBuilding constrained quadratic model...")

    # Initialize the CQM object
    cqm = CQM()

    # Build the objective: sum all colors values
    obj = BinaryQuadraticModel('BINARY')
    for n in G.nodes():
        for i in range(num_colors):
            obj.set_linear((n,i), i)
    cqm.set_objective(obj)

    # Add constraint to make variables discrete
    for n in G.nodes():
        cqm.add_discrete([(n,i) for i in range(num_colors)])
  
    # Build the constraints: edges have different color end points
    for u, v in G.edges:
        for i in range(num_colors):
            c = BinaryQuadraticModel('BINARY')
            c.set_quadratic((u,i),(v,i),1)
            cqm.add_constraint(c == 0)

    return cqm

def run_hybrid_solver(cqm):
    """Solve CQM using hybrid solver."""

    print("\nRunning hybrid sampler...")

    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler()

    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(cqm, label='Example - Graph Coloring')
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    ss = feasible_sampleset.first.sample

    soln = {key[0]: key[1] for key, val in ss.items() if val == 1.0}

    return soln

def plot_soln(sample, pos):
    """Plot results and save file.
    
    Args:
        sample (dict):
            Sample containing a solution. Each key is a node and each value 
            is an int representing the node's color.

        pos (dict):
            Plotting information for graph so that same graph shape is used.
    """

    print("\nProcessing sample...")

    node_colors = [sample[i] for i in G.nodes()]
    nx.draw(G, pos=pos, node_color=node_colors, node_size=50, edgecolors='k', cmap='hsv')
    plt.savefig('graph_result.png')

# ------- Main program -------
if __name__ == "__main__":

    num_colors = 7
    num_nodes = 50

    G, pos = build_graph(num_nodes)

    cqm = build_cqm(G, num_colors)

    sample = run_hybrid_solver(cqm)

    plot_soln(sample, pos)

    colors_used = max(sample.values())+1
    print("\nColors required:", colors_used, "\n")
