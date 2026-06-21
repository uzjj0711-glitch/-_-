import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

font_size = 14
plt.rcParams.update({"font.size": font_size})
plt.close("all")


#%%
# Black-Scholes 理論模擬
def Black_Scholes(S0, K, r, T, sigma, option_type="call"):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        option_price = S0*stats.norm.cdf(d1)-K*np.exp(-r*T)*stats.norm.cdf(d2)
    elif option_type == "put":
        option_price = K*np.exp(-r*T)*stats.norm.cdf(-d2)-S0*stats.norm.cdf(-d1)

    return option_price
    
#%%
# Monte Carlo 模擬
def MC_call(S0, K, r, T, sigma, Z):
    ST = S0*np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    option_price = np.exp(-r*T) * np.mean(payoff)
    return option_price

#%%
# Moment Matching
def moment_matching(Z):
    Z_mm = (Z-np.mean(Z, axis=0))/np.std(Z, axis=0)
    return Z_mm

#%%
# 1. 基本參數
S0 = 100
K = 110
r = 0.02
T = 0.5
m_paths = 10000
N = 100
seed = 42

sigma_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

#%% 
# 2. volatility 與 option price關係

def volatility_price_comparison(S0, K, r, T, sigma_list, m_paths, seed):
    mc_prices = []
    bs_prices = []
   
    
    np.random.seed(seed)
    Z = np.random.standard_normal(m_paths)
    
    for sigma in sigma_list:
      mc_price = MC_call(S0, K, r, T, sigma, Z)
      bs_price = Black_Scholes(S0, K, r, T, sigma, option_type = "call")
    
      mc_prices.append(mc_price)
      bs_prices.append(bs_price)
         
    return np.array(mc_prices), np.array(bs_prices)

mc_prices, bs_prices = volatility_price_comparison(S0, K, r, T, sigma_list, m_paths, seed)


#畫圖: volatility (σ) vs option price

plt.figure(figsize=(8,5))
plt.plot(sigma_list, mc_prices, marker="o", label="Monte Carlo")
plt.plot(sigma_list, bs_prices, marker="s", label="Black-Scholes")
plt.xlabel("Volatility sigma")
plt.ylabel("European Call Option Price")
plt.title("Volatility vs European Call Option Price")
plt.legend()
plt.grid(True)
plt.show()

#%%
#(a)不同波動(sigma)下，固定/不固定 seed 比較

def seed_comparison(S0, K, r, T, sigma_list, m_paths, N, seed):
    fixed_mean = []
    unfixed_mean = []
    fixed_std = []
    unfixed_std = []
    
    for sigma in sigma_list:
        
        prices_fixed = []
        prices_unfixed = []

        for i in range(N):

            # 固定seed
            np.random.seed(42)
            Z_fixed = np.random.standard_normal(m_paths)
            price_fixed = MC_call(S0, K, r, T, sigma, Z_fixed)
            prices_fixed.append(price_fixed)
    
            # 不固定seed
            Z_unfixed = np.random.standard_normal(m_paths)
            price_unfixed = MC_call(S0, K, r, T, sigma, Z_unfixed)
            prices_unfixed.append(price_unfixed)

        fixed_mean.append(np.mean(prices_fixed))
        unfixed_mean.append(np.mean(prices_unfixed))
        fixed_std.append(np.std(prices_fixed))
        unfixed_std.append(np.std(prices_unfixed))
    return(np.array(fixed_mean),
           np.array(unfixed_mean),
           np.array(fixed_std),
           np.array(unfixed_std))

fixed_mean, unfixed_mean, fixed_std, unfixed_std = seed_comparison(S0, K, r, T, sigma_list, m_paths, N, seed)

#畫圖 1: 固定/ 不固定 seed 的平均選擇權比較
plt.figure(figsize=(8,5))
plt.plot(sigma_list, fixed_mean, marker="o", label="Fixed seed")
plt.plot(sigma_list, unfixed_mean, marker="s", label="Unfixed seed")
plt.xlabel("Volatility sigma")
plt.ylabel("Average simulated call price")
plt.title("Fixed Seed vs Unfixed Seed: Average Price")
plt.legend()
plt.grid(True)
plt.show()

#畫圖 2: 固定/ 不固定 seed 的模擬波動程度比較
plt.figure(figsize=(8,5))
plt.plot(sigma_list, fixed_std, marker="o", label="Fixed seed")
plt.plot(sigma_list, unfixed_std, marker="s", label="Unfixed seed")
plt.xlabel("Volatility sigma")
plt.ylabel("Stsndard deviation of simulated prices")
plt.title("Fixed Seed vs Unfixed Seed: Simulation Variability")
plt.legend()
plt.grid(True)
plt.show()

#%%
#(b)Moment Matching

def moment_matching_comparison(S0, K, r, T, sigma_list, m_paths, N, seed):
    no_mm_mean = []
    mm_mean = []
    no_mm_std = []
    mm_std = []
    bs_price_list = []

    np.random.seed(42)

    for sigma in sigma_list:
        
        prices_no_mm = []
        prices_mm = []
        
        for i in range(N):   
            Z = np.random.standard_normal(m_paths)
            
            price_no_mm = MC_call(S0, K, r, T, sigma, Z)
            prices_no_mm.append(price_no_mm)

            Z_mm = moment_matching(Z)
            price_mm = MC_call(S0, K, r, T, sigma, Z_mm)
            prices_mm.append(price_mm)
            
        no_mm_mean.append(np.mean(prices_no_mm))   
        mm_mean.append(np.mean(prices_mm))
        no_mm_std.append(np.std(prices_no_mm))
        mm_std.append(np.std(prices_mm))
        bs_price_list.append(Black_Scholes(S0, K, r, T, sigma, option_type="call"))
        
    return(np.array(no_mm_mean),
           np.array(mm_mean),
           np.array(no_mm_std),
           np.array(mm_std),
           np.array(bs_price_list))
no_mm_mean, mm_mean, no_mm_std, mm_std, bs_price_list = moment_matching_comparison(S0, K, r, T, sigma_list, m_paths, N, seed)

#畫圖 1: 有無Moment Matching 的平均價格
plt.figure(figsize=(8,5))
plt.plot(sigma_list, no_mm_mean, marker="o", label="Without moment matching")
plt.plot(sigma_list, mm_mean, marker="s", label="With moment matching")
plt.plot(sigma_list,bs_price_list , marker="^", label="Black-Scholes")
plt.xlabel("Volatility sigma")
plt.ylabel("Average simulated call price")
plt.title("Moment Matching Effect:Average Price")
plt.legend()
plt.grid(True)
plt.show()

#畫圖 2:有無Moment Matching的模擬波動程度比較
plt.figure(figsize=(8,5))
plt.plot(sigma_list, no_mm_std, marker="o", label="Without moment matching")
plt.plot(sigma_list, mm_std, marker="s", label="With moment matching")
plt.xlabel("Volatility sigma")
plt.ylabel("Standard deviation of simulated prices")
plt.title("Moment Matching Effect:Simulation Variability")
plt.legend()
plt.grid(True)
plt.show()

