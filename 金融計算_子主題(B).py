import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

font_size = 16
plt.rcParams.update({"font.size": font_size})
plt.close("all")

# Black-Scholes 理論
def Black_Scholes(S0, K, r, T, sigma, option_type="call"):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        option_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return option_price
#%%
# Black-Scholes 理論 Delta
def Black_Scholes_delta(S0, K, r, T, sigma, option_type="call"):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        Delta = norm.cdf(d1)
    elif option_type == "put":
        Delta = -norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return Delta
#%%
# Numerical Delta
def Numerical_delta_forward(S0, K, r, T, sigma, h, option_type="call"):
    price_up = Black_Scholes(S0 + h, K, r, T, sigma, option_type)
    price_now = Black_Scholes(S0, K, r, T, sigma, option_type)

    delta_num = (price_up - price_now) / h
    return delta_num
#%%
# Monte Carlo 
def MC_call(S0, K, r, T, sigma, Z):
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff, axis=0)

    return option_price
#%%
# Monte Carlo Delta：Forward Difference
def MC_call_delta(S0, K, r, T, sigma, Z, h):
    price_up = MC_call(S0 + h, K, r, T, sigma, Z)
    price_now = MC_call(S0, K, r, T, sigma, Z)

    delta = (price_up - price_now) / h
    return delta
#%%
# Moment Matching
def moment_matching(Z):
    Z_mm = (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0)
    return Z_mm
#%%
# 基本參數設定
S0 = 100
K = 100
r = 0.02
T = 0.5
sigma = 0.2
option_type = "call"

m_paths = 10000
N = 100
seed = 42

sigma_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# Numerical Delta 與 h 誤差分析
delta_true = Black_Scholes_delta(S0, K, r, T, sigma, option_type)

hs = np.logspace(-16, 1, 100)

delta_numerical = np.zeros(len(hs))
relative_error = np.zeros(len(hs))

for i in range(len(hs)):
    h_temp = hs[i]

    delta_numerical[i] = Numerical_delta_forward(
        S0, K, r, T, sigma, h_temp, option_type)

    relative_error[i] = np.abs((delta_numerical[i] - delta_true) / delta_true)


plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(hs, delta_numerical, linestyle="--", linewidth=2,
         label="Forward Difference Delta")
plt.axhline(y=delta_true, linewidth=2, label="True Delta")
plt.xlabel("h")
plt.ylabel("Numerical Delta")
plt.title("Numerical Delta vs Step Size h")
plt.xscale("log")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(hs, relative_error, linestyle="--", linewidth=2,
         label="Forward Difference Error")
plt.axhline(y=0, linewidth=2)
plt.xlabel("h")
plt.ylabel("Absolute Relative Error")
plt.title("Forward Difference Error vs Step Size h")
plt.xscale("log")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

best_index = np.argmin(relative_error)
best_h = hs[best_index]

print("True Delta =", delta_true)
print("Best h =", best_h)
print("Numerical Delta at best h =", delta_numerical[best_index])
print("Minimum Relative Error =", relative_error[best_index])
#%%
# Monte Carlo Delta 與 Black-Scholes Delta 比較(不同 volatility 下)
mc_delta_mean = []
bs_delta_list = []

np.random.seed(seed)
Z_base = np.random.standard_normal((m_paths, N))

for sigma_temp in sigma_list:
    delta_mc = MC_call_delta(S0, K, r, T, sigma_temp, Z_base, best_h)
    delta_bs = Black_Scholes_delta(S0, K, r, T, sigma_temp, option_type)

    mc_delta_mean.append(np.mean(delta_mc))
    bs_delta_list.append(delta_bs)

plt.figure(figsize=(8, 5))
plt.plot(sigma_list, mc_delta_mean, marker="o", label="Monte Carlo Delta Mean")
plt.plot(sigma_list, bs_delta_list, marker="s", label="Black-Scholes Delta")
plt.xlabel("Volatility sigma")
plt.ylabel("Delta")
plt.title("Volatility vs European Call Delta")
plt.legend()
plt.grid(True)
plt.show()
#%%
# (a) 固定 Seed 與不固定 Seed 比較
def seed_comparison(S0, K, r, T, sigma_list, m_paths, N, h, seed):
    fixed_mean = []
    unfixed_mean = []
    fixed_std = []
    unfixed_std = []

    for sigma_temp in sigma_list:
        # 固定 seed
        np.random.seed(seed)
        Z_fixed = np.random.standard_normal((m_paths, N))
        delta_fixed = MC_call_delta(S0, K, r, T, sigma_temp, Z_fixed, h)

        # 不固定 seed
        np.random.seed(None)
        Z_unfixed = np.random.standard_normal((m_paths, N))
        delta_unfixed = MC_call_delta(S0, K, r, T, sigma_temp, Z_unfixed, h)

        fixed_mean.append(np.mean(delta_fixed))
        unfixed_mean.append(np.mean(delta_unfixed))
        fixed_std.append(np.std(delta_fixed))
        unfixed_std.append(np.std(delta_unfixed))

    return (np.array(fixed_mean),
            np.array(unfixed_mean),
            np.array(fixed_std),
            np.array(unfixed_std))

fixed_mean, unfixed_mean, fixed_std, unfixed_std = seed_comparison(S0, K, r, T, sigma_list, m_paths, N, best_h, seed)

plt.figure(figsize=(8, 5))
plt.plot(sigma_list, fixed_mean, marker="o", label="Fixed seed")
plt.plot(sigma_list, unfixed_mean, marker="s", label="Unfixed seed")
plt.xlabel("Volatility sigma")
plt.ylabel("Average simulated Delta")
plt.title("Fixed Seed vs Unfixed Seed: Average Delta")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(sigma_list, fixed_std, marker="o", label="Fixed seed")
plt.plot(sigma_list, unfixed_std, marker="s", label="Unfixed seed")
plt.xlabel("Volatility sigma")
plt.ylabel("Standard deviation of simulated Delta")
plt.title("Fixed Seed vs Unfixed Seed: Delta Variability")
plt.legend()
plt.grid(True)
plt.show()
#%%
# (b) Moment Matching 比較
def moment_matching_comparison(S0, K, r, T, sigma_list, m_paths, N, h, seed):
    no_mm_mean = []
    mm_mean = []
    no_mm_std = []
    mm_std = []

    np.random.seed(seed)

    for sigma_temp in sigma_list:
        Z = np.random.standard_normal((m_paths, N))

        delta_no_mm = MC_call_delta(S0, K, r, T, sigma_temp, Z, h)

        Z_mm = moment_matching(Z)
        delta_mm = MC_call_delta(S0, K, r, T, sigma_temp, Z_mm, h)

        no_mm_mean.append(np.mean(delta_no_mm))
        mm_mean.append(np.mean(delta_mm))
        no_mm_std.append(np.std(delta_no_mm))
        mm_std.append(np.std(delta_mm))

    return (np.array(no_mm_mean),
            np.array(mm_mean),
            np.array(no_mm_std),
            np.array(mm_std))

no_mm_mean, mm_mean, no_mm_std, mm_std = moment_matching_comparison(S0, K, r, T, sigma_list, m_paths, N, best_h, seed)

plt.figure(figsize=(8, 5))
plt.plot(sigma_list, no_mm_mean, marker="o",
         label="Without moment matching")
plt.plot(sigma_list, mm_mean, marker="s",
         label="With moment matching")
plt.plot(sigma_list, bs_delta_list, marker="^",
         label="Black-Scholes Delta")
plt.xlabel("Volatility sigma")
plt.ylabel("Average simulated Delta")
plt.title("Moment Matching Effect: Average Delta")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(sigma_list, no_mm_std, marker="o",
         label="Without moment matching")
plt.plot(sigma_list, mm_std, marker="s",
         label="With moment matching")
plt.xlabel("Volatility sigma")
plt.ylabel("Standard deviation of simulated Delta")
plt.title("Moment Matching Effect: Delta Variability")
plt.legend()
plt.grid(True)
plt.show()