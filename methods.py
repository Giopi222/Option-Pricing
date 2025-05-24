import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

'''Parameters'''
S = 105              
K = 105                 
T = 1                   
r = 0.05                
sigma = 0.2            
N = 1000 


'''Black-Scholes'''
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
def greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + (sigma**2 / 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Delta = norm.cdf(d1)
    Gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    Theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    Vega = S * norm.pdf(d1) * np.sqrt(T)
    Rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return Delta, Gamma, Theta, Vega, Rho

call_price_bs = black_scholes(S, K, T, r, sigma, option_type="call")
put_price_bs = black_scholes(S, K, T, r, sigma, option_type="put")
Delta, Gamma, Theta, Vega, Rho = greeks(S, K, T, r, sigma, "call")


'''Monte Carlo'''
def montecarlo_pricing(S, K, T, r, sigma, N, option_type="call"):
    dt = T  #N(0,1)
    Z = np.random.randn(N)
    ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff) 

call_price_mc = montecarlo_pricing(S, K, T, r, sigma, N, option_type="call")
put_price_mc = montecarlo_pricing(S, K, T, r, sigma, N, option_type="put")

'''Binomial tree'''
def binomial_tree(S, K, T, r, sigma, N, option_type="call", american = False):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r * dt) - d) / (u - d)
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    option_tree = np.zeros((N + 1, N + 1))

    if option_type == "call":
        option_tree[:, N] = np.maximum(stock_tree[:, N] - K, 0)
    elif option_type == "put":
        option_tree[:, N] = np.maximum(K - stock_tree[:, N], 0)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if american:
                if option_type == "call":
                    exercise_value = np.maximum(stock_tree[j, i] - K, 0)
                else:
                    exercise_value = np.maximum(K - stock_tree[j, i], 0)
                option_tree[j, i] = np.maximum(continuation_value, exercise_value)
            else:
                option_tree[j, i] = continuation_value

    return option_tree[0, 0]

call_price_bt = binomial_tree(S, K, T, r, sigma, N, option_type="call", american = False)
put_price_bt = binomial_tree(S, K, T, r, sigma, N, option_type="put", american = False)


'''Results'''
S_values = np.linspace(50, 150, 50)
sigma_values = np.linspace(0.1, 0.5, 50)
S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

price_grid_bs = np.array([[black_scholes(S, K, T, r, sigma, "call") for S in S_values] for sigma in sigma_values])
price_grid_mc = np.array([[montecarlo_pricing(S, K, T, r, sigma, N, "call") for S in S_values] for sigma in sigma_values])
price_grid_bt = np.array([[binomial_tree(S, K, T, r, sigma, N=100, option_type="call", american=False) for S in S_values] for sigma in sigma_values])


fig = plt.figure(figsize=(15, 10))


# Black-Scholes 
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(S_grid, sigma_grid, price_grid_bs, cmap="viridis")
ax1.set_title("Black-Scholes")
ax1.set_xlabel("S")
ax1.set_ylabel("$\sigma$")
ax1.set_zlabel("Call Price")

# Monte Carlo 
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(S_grid, sigma_grid, price_grid_mc, cmap="plasma")
ax2.set_title("Monte Carlo")
ax2.set_xlabel("S")
ax2.set_ylabel("$\sigma$")
ax2.set_zlabel("Call Price")

# Binomial Tree 
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(S_grid, sigma_grid, price_grid_bt, cmap="coolwarm")
ax3.set_title("Binomial Tree")
ax3.set_xlabel("S")
ax3.set_ylabel("$\sigma$")
ax3.set_zlabel("Call Price")


plt.tight_layout()
plt.show()