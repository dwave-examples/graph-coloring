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

from collections import defaultdict
import argparse
import sys

from descartes import PolygonPatch
import shapefile
import matplotlib
import networkx as nx
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel
from dwave.system import LeapHybridCQMSampler

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def read_in_args(args):
    """Read in user specified parameters."""

    # Set up user-specified optional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country", default='usa', choices=['usa', 'canada'], help='Color either USA or Canada map (default: %(default)s)')
    return parser.parse_args(args)

def get_state_info(shp_file):
    """Reads shp_file and returns state records (includes state info and 
    geometry) and each state's corresponding neighbors"""

    print("\nReading shp file...")

    sf = shapefile.Reader(shp_file, encoding='CP1252')

    state_neighbors = defaultdict(list)
    for state in sf.records():
        neighbors = state['NEIGHBORS']
        try:
            neighbors = neighbors.split(",")
        except:
            neighbors = []

        state_neighbors[state['NAME']] = neighbors

    return sf.shapeRecords(), state_neighbors

def build_graph(state_neighbors):
    """Build graph corresponding to neighbor relation."""

    print("\nBuilding graph from map...")

    G = nx.Graph()
    for key, val in state_neighbors.items():
        for nbr in val:
            G.add_edge(key, nbr)

    return G

def build_cqm(G, num_colors):
    """Build CQM model."""

    print("\nBuilding constrained quadratic model...")

    # Initialize the CQM object
    cqm = ConstrainedQuadraticModel()

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
    """Solve CQM using hybrid solver through the cloud."""

    print("\nRunning hybrid sampler...")

    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler()

    # Solve the problem using the CQM solver
    sampleset = sampler.sample_cqm(cqm, label='Example - Map Coloring')
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    ss = feasible_sampleset.first.sample

    soln = {key[0]: key[1] for key, val in ss.items() if val == 1.0}

    return soln

def plot_map(sample, state_records, colors):
    """Plot results and save map file.
    
    Args:
        sample (dict):
            Sample containing a solution. Each key is a state and each value 
            is an int representing the state's color.

        state_records (shapefile.ShapeRecords):
            Records retrieved from the problem shp file.

        colors (list):
            List of colors to use when plotting.
    """

    print("\nProcessing sample...")

    fig = plt.figure()
    ax = fig.gca()

    for record in state_records:
        state_name = record.record['NAME']
        color = colors[sample[state_name]]
        poly_geo = record.shape.__geo_interface__
        ax.add_patch(PolygonPatch(poly_geo, fc=color, alpha=0.8, lw=0))

    ax.axis('scaled')
    plt.axis('off')

    fname = "map_result.png"
    print("\nSaving results in {}...".format(fname))
    plt.savefig(fname, bbox_inches='tight', dpi=300)

# ------- Main program -------
if __name__ == "__main__":

    args = read_in_args(sys.argv[1:])

    if args.country == 'canada':
        print("\nCanada map coloring demo.")
        input_shp_file = 'shp_files/canada/canada.shp'
    else:
        print("\nUSA map coloring demo.")
        input_shp_file = 'shp_files/usa/usa.shp'

    state_records, state_neighbors = get_state_info(input_shp_file)

    G = build_graph(state_neighbors)

    colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red', 'grey']
    num_colors = len(colors)

    cqm = build_cqm(G, num_colors)

    sample = run_hybrid_solver(cqm)

    plot_map(sample, state_records, colors)

    colors_used = max(sample.values())+1
    print("\nColors required:", colors_used, "\n")
