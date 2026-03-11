import numpy as np
from scipy.stats import norm

# ----------------------------------
# Black-Scholes Call Pricing
# ----------------------------------

def black_scholes_call(S, K, T, r, sigma):
    """
    S = current stock price
    K = strike price
    T = time to maturity in years
    r = risk-free interest rate
    sigma = volatility
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Example values
S = 100      # current stock price
K = 100      # strike price
T = 1        # 1 year
r = 0.05     # 5% risk-free rate
sigma = 0.20 # 20% volatility

price = black_scholes_call(S, K, T, r, sigma)

print(f"Black-Scholes Call Option Price: {price:.2f}")
