import numpy as np

# -----------------------------------
# Portfolio Risk/Return Model
# -----------------------------------

# Expected annual returns of 3 assets
returns = np.array([0.10, 0.14, 0.08])

# Covariance matrix
cov_matrix = np.array([
    [0.005, 0.001, 0.002],
    [0.001, 0.006, 0.0015],
    [0.002, 0.0015, 0.004]
])

# Portfolio weights
weights = np.array([0.4, 0.35, 0.25])

# Check that weights add up to 1
print("Sum of weights:", np.sum(weights))

# Portfolio expected return
portfolio_return = np.dot(weights, returns)

# Portfolio risk (standard deviation)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
portfolio_risk = np.sqrt(portfolio_variance)

print(f"Expected portfolio return: {portfolio_return:.4f}")
print(f"Portfolio risk (standard deviation): {portfolio_risk:.4f}")