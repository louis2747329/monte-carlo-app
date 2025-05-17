''''
------------------
This code aims at valuing derivatives using Monte Carlo Simulation
------------------
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf
from scipy.stats import norm
import os
import time
from scipy import stats

start = time.time()

# 1) Call and Put Premium computation

def Option(S, K, T, r, q, vol, otype):
    global T_chooser
    if otype.lower() == 'chooser':
        T_prime = T - T_chooser
        d1 = ( np.log(S/K) + T_prime*( r-q+ (vol**2)/2 ) ) / ( vol*np.sqrt(T_prime) )
        d2 = d1 - vol*np.sqrt(T_prime)
        Call_Premium = S*np.exp(-q*T_prime)*norm.cdf(d1) - K*np.exp(-r*T_prime)*norm.cdf(d2)
        Put_Premium = K*np.exp(-r*T_prime)*norm.cdf(-d2) - S*np.exp(-q*T_prime)*norm.cdf(-d1)
        Chooser = np.maximum(Call_Premium, Put_Premium)*np.exp(-r*T_chooser)
    else:
        d1 = ( np.log(S/K) + T*( r-q+ (vol**2)/2 ) ) / ( vol*np.sqrt(T) )
        d2 = d1 - vol*np.sqrt(T)
        Call_Premium = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        Put_Premium = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    if otype.lower() == 'call':
        return Call_Premium
    elif otype.lower() == 'put':
        return Put_Premium
    elif otype.lower() == 'chooser':
        return Chooser

# 2) Initial derivative parameters

with st.sidebar:
    st.header('⚙️ Parameters')

    logo_size = 24  # Change this value to adjust size

    st.markdown(
        f"""
        <a href="https://www.linkedin.com/in/louisracaud" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="{logo_size}" style="vertical-align:middle; margin-right:10px;">
            <span style="font-weight:bold; font-size:16px;">Louis Racaud</span>
        </a>
        """,
        unsafe_allow_html=True
    )
    st.write(" ")
    st.write(" ")
    st.subheader('Option Parameters')
    otype = st.selectbox("Option Type", options=["Call", "Put", "Chooser"], index=0)
    S = st.number_input("Spot Price (S)", value=62.0, min_value=0.0001)
    K = st.number_input("Strike Price (K)", value=60.0, min_value=0.0001)
    vol = st.number_input("Volatility (vol)", value=0.2, min_value=0.0001)
    q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0)
    r = st.number_input("Risk-free Rate (r)", value=0.1, min_value=0.0)
    T = st.number_input("Time to Maturity (T)", value=5/12, min_value=0.0001)
    if otype.lower() == 'chooser':
        T_chooser = st.number_input("Chooser date", value=3/12, min_value=0.0001, max_value=T-0.0001)
    st.write("---")

    st.subheader('Simulation Parameters')
    N = st.number_input("Number of Time Steps (N)", min_value=1, max_value=10000, step=1, value = 10)
    M = st.number_input("Number of Simulations (M)", min_value=1, max_value=10000, value=500, step=1)
    st.write("---")

    st.subheader('Animation Parameters')
    fps_input = st.number_input("FPS", min_value=1, max_value=10000, step=1, value = 15)
    dpi_value = st.number_input("DPI", min_value=1, max_value=10000, step=1, value = 200)
    file_name = st.text_input("File name", value="animated-plot")
    popular_matplotlib_styles = [
    "default",               # Modern Matplotlib default
    "classic",               # Original Matplotlib style
    ]
    #style_name = st.selectbox("Color theme: ", options=popular_matplotlib_styles, index=0)
    style_name = 'classic'
    

# Number of time steps = N
# Number of simulation = M

# 3) Precompute constants

if otype.lower() == 'chooser':
    dt = T_chooser/N                        # Time steps
else: 
    dt = T/N   
nudt = (r - 0.5*vol**2)*dt      # The drift term
volsdt = vol*np.sqrt(dt)        # vol time sqrt(Time steps)
lnS = np.log(S)                 # np.log is the natural log by default

# 4) Standard error Placeholders

sum_CT = 0
sum_CT2 = 0

# 5) Monte Carlo Method

all_paths = np.zeros((M, N+1))      # Animated plot

for i in range(M):
    lnSt = lnS                      # Initialisation of lnST
    path = [S]                      # Initialize path with S before the loop (plot)
    for _ in range(N):
        lnSt = lnSt + nudt + volsdt * np.random.normal()  # We take a step through the time step
        path.append(np.exp(lnSt))   # Animated plot
    all_paths[i] = path             # Animated plot

    ST = path[-1]                   # Get it back into the stock price at maturity (ST)
    # path is a list storing the simulated prices of the underlying asset at each time step from t = 0 to t = T.
	# path[-1] accesses the final element in the list, i.e., the simulated asset price at maturity.
	# So this line gives you the terminal stock price used to compute the payoff at maturity.

    # Payoff computation
    if otype.lower() == 'call':
        CT = max(0, ST - K)             
    elif otype.lower() == 'put':
        CT = max(0, K - ST)
    elif otype.lower() == 'chooser':
        CT = Option(ST, K, T, r, q, vol, otype)

    sum_CT = sum_CT + CT           # Sum of all CTs
    sum_CT2 = sum_CT2 + CT*CT      # Sum of the CTs squared

# 6) Animated plot

fig, ax = plt.subplots()
plt.style.use(style_name)
lines = [ax.plot([], [], lw=0.8)[0] for _ in range(M)]

ax.set_xlim(0, N)
ax.set_ylim(0, np.max(all_paths) * 1.05)

writer = PillowWriter(fps=fps_input)

with writer.saving(fig, f"{file_name}.gif", dpi=dpi_value):
    for t in range(N+1):
        for i, line in enumerate(lines):
            line.set_data(range(t+1), all_paths[i, :t+1])  # Update lines at each step
        writer.grab_frame()


# 7) Compute the expectation and SE

# Discounted average payoff at maturity
if otype.lower() == 'chooser':
    C0 = np.exp(-r*T_chooser) * sum_CT / M
    sigma = np.sqrt((sum_CT2 - (sum_CT**2)/M) * np.exp(-2*r*T_chooser) / (M - 1))
else:
    C0 = np.exp(-r*T) * sum_CT / M
    sigma = np.sqrt((sum_CT2 - (sum_CT**2)/M) * np.exp(-2*r*T) / (M - 1))

SE = sigma / np.sqrt(M)           # To obtain the Standard Error we normalise by sqrt(M)

theoretical_price = Option(S, K, T, r, q, vol, otype.lower())

# Visualisation of convergence

fig_convergence, ax = plt.subplots()

x1 = np.linspace(C0-3*SE, C0-1*SE, 100)
x2 = np.linspace(C0-1*SE, C0+1*SE, 100)
x3 = np.linspace(C0+1*SE, C0+3*SE, 100)

s1 = stats.norm.pdf(x1, C0, SE)
s2 = stats.norm.pdf(x2, C0, SE)
s3 = stats.norm.pdf(x3, C0, SE)

plt.fill_between(x1, s1, color='tab:blue', label='> StDev')
plt.fill_between(x2, s2, color='cornflowerblue', label='1 StDev')
plt.fill_between(x3, s3, color='tab:blue')

plt.plot([C0, C0], [max(s2)*1.1, 0], 'k',
         label = 'Monte Carlo Value')
plt.plot([theoretical_price, theoretical_price],[max(s2)*1.1, 0], 'r',
         label = 'B&S Value')

plt.ylabel('Probability')
plt.xlabel('Option Price')
plt.legend()

# Show animated GIF in Streamlit

end = time.time()

st.header('Monte-Carlo Simulation')

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"Estimated {otype} Value (Monte-Carlo): ")
    st.success(f"**${round(C0, 4)}** ± **{round(SE, 2)}** (Standard Error)")
with col2:
    st.markdown(f"Theoretical {otype} Price (Black-Scholes): ")
    st.info(f" **${round(theoretical_price, 4)}**")
st.write(f'Code execution time: {round(end - start, 4)} seconds.')

st.subheader(f'Underlying price path from t=0 to t={round(T_chooser,2) if otype.lower() == 'chooser' else round(T,2)}')
if os.path.exists(f"{file_name}.gif"):
    st.image(f"{file_name}.gif", use_container_width=True)

st.subheader('Convergence graph')
st.pyplot(fig_convergence)

# ''' 
# -------------------------------------------
# More explanation of Monte Carlo Simulation:
# - Valuation by simulation
# - Risk-neutral pricing: value of an option = risk-neutral expectation of its discounted payoff
# - Expectation = average of a large number of discounted payoffs
# - Formula: C(T,i): 
#     - Payoff at time T for a particular simulation i
#     - exp(-r*T): discounted factor under Black-Scholes methodology
# -------------------------------------------
# '''

'''
-------------------------------------------
More explanation about this web-app:
- Simulates option prices using Monte Carlo paths for the underlying asset under risk-neutral dynamics, supporting Call, Put, and Chooser options.
- Calculates the estimated option price by computing the discounted average payoff across all simulated paths, and compares it to the Black-Scholes theoretical price.
- Generates and saves an animated plot showing the evolution of each simulated path, then displays it in a Streamlit web app.
- Computes and visualizes convergence using a probability density curve centered on the Monte Carlo estimate with ±1 standard deviation, compared to the B&S price.
-------------------------------------------
'''

logo_size = 24  # Change this value to adjust size

st.markdown(
    f"""
    <a href="https://www.linkedin.com/in/louisracaud" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="{logo_size}" style="vertical-align:middle; margin-right:10px;">
        <span style="font-weight:bold; font-size:16px;">Louis Racaud</span>
    </a>
    """,
    unsafe_allow_html=True
)
st.write("---")
st.markdown("""
📩 **If you have any recommendations or suggestions for improvement**, feel free to contact me:
- 📧 Email: **racaud.louis@gmail.com**
""")