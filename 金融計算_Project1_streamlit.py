import numpy as np
from scipy import stats
import streamlit as st
import matplotlib.pyplot as plt

font_size = 12
plt.rcParams.update({"font.size": font_size})

def Black_Scholes(S0, K, r, T, sigma, option_type="call"):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        option_price = S0*stats.norm.cdf(d1)-K*np.exp(-r*T)*stats.norm.cdf(d2)
    elif option_type == "put":
        option_price = K*np.exp(-r*T)*stats.norm.cdf(-d2)-S0*stats.norm.cdf(-d1)

    return option_price
    
#%%
def European_option_simulation(S0, K, r, T, sigma, Z, moment_matching=False, option_type="call"):
    if moment_matching:
        Z = (Z-np.mean(Z, axis=0))/np.std(Z, axis=0)

    ST = S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*Z)

    if option_type == "call":
        payoff = np.maximum(ST-K, 0)
    if option_type == "put":
        payoff = np.maximum(K-ST, 0)

    option_prices = np.exp(-r*T)*np.mean(payoff, axis=0)

    return option_prices

#%%
# Streamlit 
st.title("子主題(A)European Call Option Simulation")
st.sidebar.markdown('<h1 style="color:green; font-size:20px;">參數設置</h1>', unsafe_allow_html=True)

# Sidebar (側邊攔)參數
st.sidebar.header("參數設定")

S0 = st.sidebar.slider("Initial Price (S0)", 
                       min_value = 10.0, 
                       max_value =1000.0, 
                       value = 100.0,
                       step=10.0)
K = st.sidebar.slider("Strike (K)",
                       min_value = 10.0, 
                       max_value = 1000.0, 
                       value =100.0,
                       step = 10.0)
r = st.sidebar.slider("Risk-free rate (r)", 
                      min_value = 0.0, 
                      max_value =0.2, 
                      value = 0.02,
                      step=0.01)
T = st.sidebar.selectbox("Time to maturity (T)", 
                          options=[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                          index=2)
sigma = st.sidebar.slider("Volatility (σ)", 
                          min_value=0.01,
                          max_value=1.0,
                          value=0.2,
                          step=0.01)

moment_matching = st.sidebar.checkbox("Use Moment Matching", value=True)

option_type = st.sidebar.radio("option type",
                                options=["call", "put"],
                                index=0)

option_price_true = Black_Scholes(S0, K, r, T, sigma, option_type=option_type)

#%%
m_paths = st.sidebar.slider("Simulation Paths", 
                            min_value=100,
                            max_value=100000,
                            value=10000,
                            step=100)
N = st.sidebar.number_input("Repeated experiments N",
                            min_value=500,
                            max_value=10000,
                            value=1000,
                            step=500) 
seed = st.sidebar.selectbox("Seed", 
                            options=[42, 123457],
                            index=0)
use_seed = st.sidebar.checkbox("Use fixed random seed", value=True)

random_type = st.sidebar.selectbox("random_type",
                                    options=["pseudo", "np.random.randn" , "np.random.normal"],
                                    index=0)

# Single simulation (MC vs Black-Scholes)
st.header(" Single simulation (MC vs Black-Scholes)")

if use_seed:
    np.random.seed(seed)

if random_type == "pseudo":
    u = np.random.rand(m_paths, N)
    z = stats.norm.ppf(u)
    
if random_type == "np.random.randn":
    z = np.random.randn(m_paths, N)

if random_type =="np.random.normal":
    z = np.random.normal(loc=0.0, scale=1.0, size=(m_paths, N))

option_prices = European_option_simulation(S0, K, r, T, sigma, z, moment_matching = moment_matching)  
 
bs_price = Black_Scholes(S0, K, r, T, sigma)
#%%
st.subheader("1.模擬結果")

col1, col2, col3 = st.columns(3)
col1.metric("Black-Scholes Price", f"{bs_price:.4f}")
col2.metric("Monte Carlo Mean",f"{np.mean(option_prices):.4f}")
col3.metric("Simulation Error",f"{np.mean(option_prices) - bs_price:.4f}")

st.subheader("2.Monte Carlo模擬價格分布")
fig = plt.figure()
bins = st.sidebar.slider("直方圖分割區間數(bins)", 
                          min_value = 10,
                          max_value = 100,
                          value = 50,
                          step = 10 )
alpha = st.sidebar.slider("直方圖顏色深淺度(alpha)",
                           min_value = 0.0,
                           max_value = 1.0,
                           value = 0.3,
                           step = 0.1 )
plt.hist(option_prices,
        bins = bins,
        density = True,
        alpha = alpha,
        ec = "black")
plt.axvline(x = bs_price, color = "red",)
plt.xlabel("simulate option price")
plt.ylabel("Density")
plt.title(f"Random type = {random_type}")
st.pyplot(fig)

st.subheader("3.不同volatility下的價格比較")

sigma_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

mc_mean_list = []
bs_list = []

for sigma in sigma_list:
    
    if use_seed:
        np.random.seed(seed)
    if random_type == "pseudo":
        u_temp = np.random.rand(m_paths, N)
        z_temp = stats.norm.ppf(u_temp)
    if random_type == "np.random.randn":
        z_temp = np.random.randn(m_paths, N)
    if random_type == "np.random.normal":
        z_temp = np.random.normal(loc = 0.0, scale=1.0, size=(m_paths, N))
        
    prices_temp = European_option_simulation( S0, K, r, T, sigma, z_temp,moment_matching)
        
    mc_mean_list.append(np.mean(prices_temp))
    bs_list.append(Black_Scholes(S0, K, r, T, sigma))
        
fig2 = plt.figure()
plt.plot(sigma_list,
         mc_mean_list,
         marker="o",
         label="Monte Carlo mean")
plt.plot(sigma_list,
          bs_list,
          marker="s",
          label="Black-Scholes")
plt.xlabel("Volatility sigma")
plt.ylabel("European option price")
plt.title("Volatility vs European Call Option Price")
plt.grid(True)

st.pyplot(fig2)
