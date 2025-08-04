# app.py
# -*- coding: utf-8 -*-

"""
This is a Streamlit web application that performs a bearish technical analysis
on Nifty 50 and selected stocks.

To Run This App Locally:
1. Make sure you have Python installed.
2. Install Streamlit and other libraries:
   pip install streamlit yfinance pandas matplotlib numpy scipy
3. Save this code as `app.py`.
4. Open your terminal, navigate to the file's directory, and run:
   streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime

# --- Functions for Analysis (from the previous script) ---

def find_chart_patterns(data, order=10, K=3):
    """
    Detects Double Top and Head and Shoulders patterns.
    """
    patterns = {'double_top': [], 'head_shoulders': []}
    if len(data) < (order * 2 + 1):
        return patterns

    highs = data['High']
    lows = data['Low']
    
    peak_indices = argrelextrema(highs.values, np.greater, order=order)[0]
    valley_indices = argrelextrema(lows.values, np.less, order=order)[0]
    
    peaks = highs.iloc[peak_indices]
    valleys = lows.iloc[valley_indices]

    # Find Double Top
    for i in range(len(peaks) - 1):
        p1_idx, p2_idx = peaks.index[i], peaks.index[i+1]
        p1_val, p2_val = peaks.iloc[i], peaks.iloc[i+1]
        if abs(p1_val - p2_val) / p2_val <= K / 100:
            intervening_valleys = valleys[(valleys.index > p1_idx) & (valleys.index < p2_idx)]
            if not intervening_valleys.empty:
                patterns['double_top'].append((p1_idx, p2_idx))

    # Find Head and Shoulders
    for i in range(len(peaks) - 2):
        s1_idx, h_idx, s2_idx = peaks.index[i], peaks.index[i+1], peaks.index[i+2]
        s1_val, h_val, s2_val = peaks.iloc[i], peaks.iloc[i+1], peaks.iloc[i+2]
        if h_val > s1_val and h_val > s2_val and abs(s1_val - s2_val) / s2_val <= (K + 5) / 100:
            v1 = valleys[(valleys.index > s1_idx) & (valleys.index < h_idx)]
            v2 = valleys[(valleys.index > h_idx) & (valleys.index < s2_idx)]
            if not v1.empty and not v2.empty:
                patterns['head_shoulders'].append((s1_idx, h_idx, s2_idx))
                
    return patterns

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("Stock Market Bearish Analysis Dashboard")
st.markdown(f"An interactive tool for technical analysis. Last updated: **{datetime.now().strftime('%d-%b-%Y %H:%M')}**")

# --- Sidebar for User Input ---
st.sidebar.header("Select Stock")
stocks_to_analyze = {
    'Nifty 50': '^NSEI',
    'Adani Enterprises': 'ADANIENT.NS',
    'ITC Ltd': 'ITC.NS',
    'Titan Company': 'TITAN.NS'
}
selected_name = st.sidebar.selectbox("Choose a stock or index to analyze:", list(stocks_to_analyze.keys()))
selected_ticker = stocks_to_analyze[selected_name]

if st.sidebar.button("Analyze"):
    with st.spinner(f"Fetching and analyzing data for {selected_name}..."):
        try:
            # --- 1. Data Fetching ---
            data = yf.download(selected_ticker, period='2y', auto_adjust=True, progress=False)
            if data.empty:
                st.error(f"Could not download data for {selected_name}. Please check the ticker symbol or try again later.")
            else:
                # --- 2. Calculations ---
                data['50_SMA'] = data['Close'].rolling(window=50).mean()
                data['200_SMA'] = data['Close'].rolling(window=200).mean()
                data['SMA_Signal'] = (data['50_SMA'] < data['200_SMA']).astype(int)
                data['Death_Cross'] = data['SMA_Signal'].diff()
                death_cross_points = data[data['Death_Cross'] == 1]
                
                chart_patterns = find_chart_patterns(data.last('9M'))
                
                last_data_point = data.iloc[-1]
                last_price = last_data_point['Close']
                
                # --- 3. Display Analysis in Two Columns ---
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader(f"Bearish Analysis for {selected_name}")
                    is_bearish = False
                    
                    if not death_cross_points.empty and (data.index[-1] - death_cross_points.index[-1]).days < 180:
                        st.warning(f"**DEATH CROSS ACTIVE:** A 'Death Cross' occurred on {death_cross_points.index[-1].date()}. This is a strong long-term bearish signal.")
                        is_bearish = True
                    
                    if last_price < last_data_point['50_SMA']:
                        st.warning(f"**SHORT-TERM WEAKNESS:** The current price ({last_price:.2f}) is below the 50-Day SMA ({last_data_point['50_SMA']:.2f}).")
                        is_bearish = True
                        
                    if chart_patterns['double_top']:
                        st.error("**PATTERN ALERT:** A potential 'Double Top' bearish pattern may have formed recently.")
                        is_bearish = True
                        
                    if chart_patterns['head_shoulders']:
                        st.error("**PATTERN ALERT:** A potential 'Head and Shoulders' bearish pattern may have formed recently.")
                        is_bearish = True

                    if not is_bearish:
                        st.success("No strong immediate bearish signals detected from moving averages or chart patterns.")

                    st.subheader("Potential Downside Targets")
                    target1 = last_price * 0.98
                    target2 = last_price * 0.95
                    st.markdown(f"  - **Moderate Target (~2% Fall):** `{target1:.2f}`")
                    st.markdown(f"  - **Stronger Correction (~5% Fall):** `{target2:.2f}`")

                with col2:
                    # --- 4. Plotting ---
                    st.subheader("Price Chart & Technical Indicators")
                    plt.style.use('seaborn-v0_8-darkgrid')
                    fig, ax = plt.subplots(figsize=(16, 8))

                    ax.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.8, linewidth=1.5)
                    ax.plot(data.index, data['50_SMA'], label='50-Day SMA', color='orange', linestyle='--', linewidth=2)
                    ax.plot(data.index, data['200_SMA'], label='200-Day SMA', color='purple', linestyle='--', linewidth=2)

                    if not death_cross_points.empty:
                        ax.plot(death_cross_points.index, death_cross_points['50_SMA'], 'X', color='red', markersize=15, markeredgewidth=3, label='Death Cross')

                    for p1, p2 in chart_patterns['double_top']:
                        ax.plot([p1, p2], [data.loc[p1, 'High'], data.loc[p2, 'High']], 'o-', color='red', markersize=10, label='Double Top Pattern')
                    
                    for s1, h, s2 in chart_patterns['head_shoulders']:
                        ax.plot([s1, h, s2], [data.loc[s1, 'High'], data.loc[h, 'High'], data.loc[s2, 'High']], 'o-', color='magenta', markersize=10, label='Head & Shoulders')

                    ax.axhline(y=target1, color='darkred', linestyle=':', linewidth=2, label=f'Target 1 ({target1:.2f})')
                    ax.axhline(y=target2, color='maroon', linestyle=':', linewidth=2, label=f'Target 2 ({target2:.2f})')

                    ax.set_title(f'{selected_name} ({selected_ticker}) - Bearish Technical Analysis', fontsize=18, weight='bold')
                    ax.set_ylabel('Price (INR)')
                    
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
                    ax.grid(True)
                    
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Select a stock from the sidebar and click 'Analyze' to begin.")
