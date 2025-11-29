import math
import numpy as np
from scipy.stats import invgamma

# Load the data and put it in a dictionary.
all_data = {}
with open('data.txt', 'r') as data:
    for line in data:
        vals = [float(x) for x in line.split()]
        all_data[int(vals[0])] = (vals[1], vals[2])

# Parameters on the prior for m.
mu_zero_m = 5.0
sigma_zero_m = 10.0

# Parameters on the prior for c.
mu_zero_c = 50.0
sigma_zero_c = 100.0

# Parameters on the prior for sigma^2.
alpha = 10.0
beta = 1.0

# Initial estimates for the three model parameters.
m = 20.0
c = 50.0
sigma = 200.0


# Write this for 1a).
def sample_sigma():
    """Placeholder for sampling sigma."""
    return 0.0


for _ in range(10):
    sample_sigma()


# Write this for 1b).
def sample_c():
    """Placeholder for sampling c."""
    return 0.0


for _ in range(10):
    sample_c()


# Write this for 1c).
def sample_m():
    """Placeholder for sampling m."""
    return 0.0


for _ in range(10):
    sample_m()


def get_error():
    """This computes the error of the current model."""
    error = 0.0
    count = 0
    for x in all_data:
        y = all_data[x]
        residual = c + y[0] * m - y[1]
        error += residual ** 2
        count += 1
    return error / count


# For part 2, you run 1000 iterations of a Gibbs sampler.
for _ in range(1000):
    get_error()
    sigma = sample_sigma()
    m = sample_m()
    c = sample_c()
