import dimod
import numpy as np
from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

# Set up the map coloring example - the provinces and borders
provinces = ["AB", "BC", "ON", "MB", "NB", "NL", "NS", "NT", "NU", "PE", "QU", "SK", "YT"]
borders = [("BC", "AB"), ("BC", "NT"), ("BC", "YT"), ("AB", "SK"),
           ("AB", "NT"), ("SK", "MB"), ("SK", "NT"), ("MB", "ON"),
           ("MB", "NU"), ("ON", "QU"), ("QU", "NB"), ("QU", "NL"),
           ("NB", "NS"), ("YT", "NT"), ("NT", "NU")]

# input: number of colors in the graph
n_colors = 4
colors = np.linspace(0, n_colors - 1, n_colors)

# Initialize the DQM object
dqm = dimod.DiscreteQuadraticModel()

# Load the DQM. Define the variables, and then set quadratic weights.
# No biases necessary because we don't care what the colors are, as long as
# they are different at the borders.
for p in provinces:
    dqm.add_variable(4, label=p)
for p0, p1 in borders:
    dqm.set_quadratic(p0, p1, {(c, c): 1 for c in colors})

# Initialize the DQM solver
sampler = LeapHybridDQMSampler()

# Solve the problem using the DQM solver
sampleset = sampler.sample_dqm(dqm, label='Example - Map Coloring')

# get the first solution, and print it
sample = sampleset.first.sample
energy = sampleset.first.energy
print(sample, energy)
