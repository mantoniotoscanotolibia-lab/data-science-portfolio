import numpy as np
import matplotlib.pyplot as plt

# Initial stock price
initial_price = 100

# Expected annual return
mu = 0.08

# Annual volatility
sigma = 0.20

# Time horizon in years
T = 1

# Trading days in one year
steps = 252

# Number of simulations
num_simulations = 50

# Time step
dt = T / steps

# Create matrix to store simulated prices
prices = np.zeros((steps, num_simulations))

# Set initial price for all simulations
prices[0] = initial_price

# Generate price paths
for t in range(1, steps):
    random_values = np.random.standard_normal(num_simulations)
    prices[t] = prices[t - 1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_values
    )

# Plot all simulations
plt.figure(figsize=(10, 6))
plt.plot(prices)
plt.title("Monte Carlo Simulation of Stock Prices")
plt.xlabel("Trading Days")
plt.ylabel("Simulated Stock Price")
plt.grid(True)
plt.show()

# Show average final price
average_final_price = np.mean(prices[-1])
print(f"Average final simulated price after 1 year: {average_final_price:.2f}")
