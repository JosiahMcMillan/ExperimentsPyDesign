"""
This script is intended for users to determine the power of their designs
The hope is that users will use their intended design matrix and model and then determine the power 

I also want to do this with non-bayesian methods.
"""

import pymc3 as pm
import arviz as az

# Define your factors and levels
variables = ['factor1', 'factor2']
levels = [[0, 1], [0, 1, 2]]

# Generate design
X_df = generate_taguchi_design(variables, levels)

# Convert DataFrame to numpy array for computations
X = X_df.values

# Now the shape of X has changed, we need to adjust P accordingly
P = X.shape[1]  # number of factors

# Assume some arbitrary beta for the purpose of data generation
true_beta = np.random.normal(0, 1, P) 

# Generate your outcome variable based on the design matrix and true beta
y = linear_regression(X, true_beta) + np.random.normal(0, 1, X.shape[0]) # added noise

# Bayesian modeling
with pm.Model() as model:
    # Priors
    beta = pm.Normal("beta", mu=0, sd=1, shape=P)

    # Likelihood
    sigma = pm.HalfNormal("sigma", sd=1)
    mu = pm.math.dot(X, beta)  # modified to use pymc3 dot product
    y_obs = pm.Normal("y_obs", mu=mu, sd=sigma, observed=y)

    # Markov Chain Monte Carlo (MCMC) sampling
    trace = pm.sample(2000, tune=1000)

# Print summary statistics and plots
az.summary(trace, hdi_prob=0.95)  # modified to use arviz summary with 95% highest density interval (HDI)
az.plot_trace(trace)

# Check if the 95% credible interval contains the true_beta values and zero
summary = az.summary(trace, var_names=["beta"], hdi_prob=0.95)
for i, beta_val in enumerate(true_beta):
    lower, upper = summary.loc[f'beta[{i}]', ['hdi_2.5%', 'hdi_97.5%']]
    print(f"For beta[{i}], true value: {beta_val}, 95% CI: ({lower}, {upper})")
    if lower <= beta_val <= upper:
        print(f"True beta[{i}] is within the 95% credible interval.")
    else:
        print(f"True beta[{i}] is NOT within the 95% credible interval.")
    if lower <= 0 <= upper:
        print(f"Zero is within the 95% credible interval for beta[{i}].")
    else:
        print(f"Zero is NOT within the 95% credible interval for beta[{i}].")
