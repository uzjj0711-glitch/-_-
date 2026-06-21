import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def Black_Scholes_delta(S0, K, r, T, sigma, option_type="call"):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        Delta = norm.cdf(d1)
    elif option_type == "put":
        Delta = -norm.cdf(-d1)
    return Delta
#%%
# Monte Carlo 
def MC_call(S0, K, r, T, sigma, Z):
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff, axis=0)
#%%
# Monte Carlo Delta
def MC_call_delta(S0, K, r, T, sigma, Z, h):
    price_up = MC_call(S0 + h, K, r, T, sigma, Z)
    price_now = MC_call(S0, K, r, T, sigma, Z)

    return (price_up - price_now) / h
#%%
# Moment Matching
def moment_matching(Z):
    return (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0)
#%%
# Streamlit

st.title("子組題(B) European Option Delta Simulation")
st.sidebar.markdown('<h1 style="color:green; font-size:20px;">參數設置</h1>',
                     unsafe_allow_html=True)
st.write(" Monte Carlo Delta 、 (a) Seed比較 與 (b) Moment Matching 比較")

# Sidebar
st.sidebar.header("參數設定")

S0 = st.sidebar.slider("Initial Price (S0)",
                        min_value=10.0,
                        max_value=1000.0,
                        value=100.0,
                        step=10.0)

K = st.sidebar.slider("Strike (K)",
                       min_value=10.0,
                       max_value=1000.0,
                       value=100.0,
                       step=10.0)
    
r = st.sidebar.slider("Risk-free Rate (r)",
                       min_value=0.0,
                       max_value=0.20,
                       value=0.02,
                       step=0.01)

T = st.sidebar.selectbox( "Time to Maturity (T)",
                           options=[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                           index=2)
   

sigma = st.sidebar.slider("Volatility (σ)",
                           min_value=0.01,
                           max_value=1.00,
                           value=0.20,
                           step=0.01)

h = st.sidebar.slider("Delta Step Size (h)",
                       min_value=0.01,
                       max_value=5.00,
                       value=1.00,
                       step=0.01)

m_paths = st.sidebar.selectbox( "Simulation Paths",
                                 options=[1000, 5000, 10000, 50000, 100000],
                                 index=2)

N = st.sidebar.selectbox("Repeated Experiments",
                          options=[10, 50, 100, 200],
                          index=2)

seed = st.sidebar.number_input("Seed",
                                value=42,
                                step=1)
use_fixed_seed = st.sidebar.checkbox("Fixed Seed",
                                      value=True)

use_moment_matching = st.sidebar.checkbox("Moment Matching",
                                           value=False)
option_type = "call"
sigma_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

#%%
# 單次模擬結果
st.header("1.單次模擬結果")

if use_fixed_seed:
    np.random.seed(int(seed))

Z = np.random.standard_normal((m_paths, N))

if use_moment_matching:
    Z = moment_matching(Z)
    
bs_delta = Black_Scholes_delta(S0, K, r, T, sigma)
mc_delta = MC_call_delta(S0, K, r, T, sigma, Z, h)
mc_delta_mean = np.mean(mc_delta)
simulation_error = abs(mc_delta_mean - bs_delta)

col1, col2, col3 = st.columns(3)
col1.metric("Black-Scholes Delta", f"{bs_delta:.4f}")
col2.metric("Monte Carlo Delta Mean", f"{mc_delta_mean:.4f}")
col3.metric("Simulation Error", f"{simulation_error:.4f}")

# Volatility vs Delta
st.header("2.不同 Volatility 下的 Delta 比較")

mc_delta_mean_list = []
bs_delta_list = []

np.random.seed(int(seed))
Z_base = np.random.standard_normal((m_paths, N))

for sig in sigma_list:
    mc = MC_call_delta(S0, K, r, T, sig, Z_base, h)
    bs = Black_Scholes_delta(S0, K, r, T, sig)
    mc_delta_mean_list.append(np.mean(mc))
    bs_delta_list.append(bs)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma_list, mc_delta_mean_list, marker="o", label="Monte Carlo Delta Mean")
ax.plot(sigma_list, bs_delta_list, marker="s", label="Black-Scholes Delta")
ax.set_xlabel("Volatility sigma")
ax.set_ylabel("Delta")
ax.set_title("Volatility vs European Call Delta")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# (a) 固定 Seed /不固定 Seed 比較
st.header("(a) 固定 Seed /不固定 Seed 比較")

fixed_mean, unfixed_mean = [], []
fixed_std, unfixed_std = [], []

for sig in sigma_list:
    #固定Seed
    np.random.seed(int(seed))
    Z_fixed = np.random.standard_normal((m_paths, N))
    delta_fixed = MC_call_delta(S0, K, r, T, sig, Z_fixed, h)
    #不固定Seed
    np.random.seed(None)
    Z_unfixed = np.random.standard_normal((m_paths, N))
    delta_unfixed = MC_call_delta(S0, K, r, T, sig, Z_unfixed, h)

    fixed_mean.append(np.mean(delta_fixed))
    unfixed_mean.append(np.mean(delta_unfixed))
    fixed_std.append(np.std(delta_fixed))
    unfixed_std.append(np.std(delta_unfixed))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma_list, fixed_mean, marker="o", label="Fixed Seed")
ax.plot(sigma_list, unfixed_mean, marker="s", label="Unfixed Seed")
ax.set_xlabel("Volatility sigma")
ax.set_ylabel("Average Delta")
ax.set_title("Fixed Seed vs Unfixed Seed: Average Delta")
ax.legend()
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma_list, fixed_std, marker="o", label="Fixed Seed")
ax.plot(sigma_list, unfixed_std, marker="s", label="Unfixed Seed")
ax.set_xlabel("Volatility sigma")
ax.set_ylabel("Standard Deviation of Delta")
ax.set_title("Fixed Seed vs Unfixed Seed: Delta Variability")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# (b) Moment Matching 比較
st.header("(b) Moment Matching 比較")

no_mm_mean, mm_mean = [], []
no_mm_std, mm_std = [], []

np.random.seed(int(seed))

for sig in sigma_list:
    Z = np.random.standard_normal((m_paths, N))

    delta_no_mm = MC_call_delta(S0, K, r, T, sig, Z, h)

    Z_mm = moment_matching(Z)
    delta_mm = MC_call_delta(S0, K, r, T, sig, Z_mm, h)

    no_mm_mean.append(np.mean(delta_no_mm))
    mm_mean.append(np.mean(delta_mm))
    no_mm_std.append(np.std(delta_no_mm))
    mm_std.append(np.std(delta_mm))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma_list, no_mm_mean, marker="o", label="Without Moment Matching")
ax.plot(sigma_list, mm_mean, marker="s", label="With Moment Matching")
ax.plot(sigma_list, bs_delta_list, marker="^", label="Black-Scholes Delta")
ax.set_xlabel("Volatility sigma")
ax.set_ylabel("Average Delta")
ax.set_title("Moment Matching Effect: Average Delta")
ax.legend()
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma_list, no_mm_std, marker="o", label="Without Moment Matching")
ax.plot(sigma_list, mm_std, marker="s", label="With Moment Matching")
ax.set_xlabel("Volatility sigma")
ax.set_ylabel("Standard Deviation of Delta")
ax.set_title("Moment Matching Effect: Delta Variability")
ax.legend()
ax.grid(True)
st.pyplot(fig)

#  Delta 分布圖
st.header("Monte Carlo Delta 分布")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(mc_delta, bins=30, alpha=0.7)
ax.axvline(bs_delta, linewidth=2, label="Black-Scholes Delta")
ax.axvline(mc_delta_mean, linestyle="--", linewidth=2, label="Monte Carlo Delta Mean")
ax.set_xlabel("Simulated Delta")
ax.set_ylabel("Frequency")
ax.set_title("Monte Carlo Delta Distribution")
ax.legend()
ax.grid(True)
st.pyplot(fig)
