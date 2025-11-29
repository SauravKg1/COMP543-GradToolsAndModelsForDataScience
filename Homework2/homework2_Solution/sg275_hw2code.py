import math
import random
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


random.seed(42)
np.random.seed(42)

# Write this for 1a).
def sample_sigma():
    """Sample sigma (standard deviation) instead of sigma squared."""
    global m, c, all_data, alpha, beta
    n = len(all_data)
    sum_squared_error = 0.0
    
    for y in all_data.values():
        height = y[0]
        weight = y[1]
        predicted = m * height + c
        error = weight - predicted
        sum_squared_error += error ** 2

    alpha_post = alpha + n / 2
    beta_post = beta + sum_squared_error / 2
    sigma_squared = invgamma.rvs(a=alpha_post, scale=beta_post)
    sigma = math.sqrt(sigma_squared)
    return sigma


print("Task 1a: sigma samples")
for i in range(10):
    val = sample_sigma()
    print(f"Iteration {i + 1}: {val}")



# Write this for 1b).
def sample_c():
    global m, sigma, all_data, mu_zero_c, sigma_zero_c

    n = len(all_data)
    x_vals = [y[1] - m * y[0] for y in all_data.values()]  # x_i = weight_i - m * height_i
    sum_x = sum(x_vals)

    precision_prior = 1 / (sigma_zero_c ** 2)
    precision_likelihood = n / (sigma ** 2)

    posterior_variance = 1 / (precision_prior + precision_likelihood)
    posterior_mean = posterior_variance * (
        (mu_zero_c / (sigma_zero_c ** 2)) + (sum_x / (sigma ** 2))
    )

    return np.random.normal(posterior_mean, math.sqrt(posterior_variance))

print("Task 1b: sample c")
for i in range(10):
    val = sample_c()
    print(f"Iteration {i + 1}: {val}")


# Write this for 1c).
def sample_m():
    global c, sigma, all_data, mu_zero_m, sigma_zero_m

    sum_weighted_x = 0.0
    sum_weighted_alpha_sq = 0.0

    for y in all_data.values():
        height = y[0]
        weight = y[1]
        alpha_i = height
        x_i = (weight - c) / height
        weight_factor = (alpha_i ** 2) / (sigma ** 2)

        sum_weighted_x += x_i * weight_factor
        sum_weighted_alpha_sq += weight_factor

    prior_precision = 1 / (sigma_zero_m ** 2)
    posterior_variance = 1 / (prior_precision + sum_weighted_alpha_sq)
    posterior_mean = posterior_variance * (
        (mu_zero_m / (sigma_zero_m ** 2)) + sum_weighted_x
    )

    return np.random.normal(posterior_mean, math.sqrt(posterior_variance))

    

sample_ms = []
print("Task 1c: sample m")
for i in range(10):
    val = sample_m()
    print(f"Iteration {i + 1}: {val}")


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
errors = []
for _ in range(1000):
    err = get_error()
    errors.append(err)
    sigma = sample_sigma()
    m = sample_m()
    c = sample_c()

print("\nTask 2: Gibbs Sampling")
print("First 5 errors:", errors[:5])
print("Last 5 errors:", errors[-5:])
print(f"Final m: {m:.4f}")
print(f"Final c: {c:.4f}")
print(f"Final sigma: {sigma:.4f}")
