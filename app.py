# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Bearish Signal Dashboard",
    layout="wide"
)

# --- App Title ---
st.title("📉 Bearish Signal Dashboard for Hedging")
st.markdown("This tool analyzes key stocks for potential bearish reversal signals to inform hedging strategies.")

# --- Configuration ---
TARGETS = {
    "NIFTY 50": "^NSEI",
    "Adani Enterprises": "ADANIENT.NS",
    "ITC": "ITC.NS",
    "Titan": "TITAN.NS"
}
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=730) # 2 years of data for better pattern analysis

# --- Data Fetching and Caching ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_data(ticker, start, end):
    """Fetches and cleans historical stock data."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return None
    # CRITICAL FIX for yfinance multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)
    return data

# --- Pattern Recognition Function (from your example) ---
def find_bearish_patterns(data, order=10, K=5):
    """Detects Double Top and Head and Shoulders patterns."""
    patterns = {'double_top': [], 'head_shoulders': []}
    if len(data) < (order * 2 + 1):
        return patterns

    highs = data['High']
    peak_indices = argrelextrema(highs.values, np.greater, order=order)[0]
    peaks = highs.iloc[peak_indices]

    # Find Double Tops
    for i in range(len(peaks) - 1):
        p1_idx, p2_idx = peaks.index[i], peaks.index[i+1]
        p1_val, p2_val = peaks.iloc[i], peaks.iloc[i+1]
        if abs(p1_val - p2_val) / p2_val <= K/100:
            patterns['double_top'].append((p1_idx, p2_idx))

    # Find Head and Shoulders
    for i in range(len(peaks) - 2):
        s1_idx, h_idx, s2_idx = peaks.index[i], peaks.index[i+1], peaks.index[i+2]
        s1, h, s2 = peaks.iloc[i], peaks.iloc[i+1], peaks.iloc[i+2]
        if h > s1 and h > s2 and abs(s1 - s2) / s2 <= (K+5)/100:
            patterns['head_shoulders'].append((s1_idx, h_idx, s2_idx))
            
    return patterns

# --- Main Dashboard Logic ---
# Iterate through each target stock and create its analysis section
for name, ticker in TARGETS.items():
    st.header(f"Analysis for: {name} ({ticker})")
    
    data = fetch_data(ticker, START_DATE, END_DATE)

    if data is None or data.empty:
        st.warning(f"Could not retrieve data for {name}. Skipping.")
        continue

    # --- Calculate Technical Indicators using 'ta' library ---
    data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['SMA200'] = ta.trend.sma_indicator(data['Close'], window=200)
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    bb_indicator = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['BB_High'] = bb_indicator.bollinger_hband()
    
    # --- Detect Patterns ---
    bearish_patterns = find_bearish_patterns(data)

    # --- Create Columns for Layout ---
    col1, col2 = st.columns([2, 1]) # Chart is 2x wider than the summary

    with col1:
        st.subheader("Price Chart with Signals")
        
        # Create the main candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name='Price'
        )])

        # Add Moving Averages
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], name='SMA 200', line=dict(color='red', width=1.5)))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name='Upper BB', line=dict(color='purple', dash='dash', width=1)))

        # Add detected pattern markers to the chart
        for dt in bearish_patterns['double_top']:
            fig.add_trace(go.Scatter(x=dt, y=data['High'].loc[list(dt)], mode='markers', marker=dict(symbol='triangle-down', color='red', size=15), name='Double Top'))
        for hs in bearish_patterns['head_shoulders']:
            fig.add_trace(go.Scatter(x=hs, y=data['High'].loc[list(hs)], mode='markers', marker=dict(symbol='diamond', color='purple', size=15), name='Head & Shoulders'))

        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Bearish Signal Summary")
        
        # Get latest values
        latest = data.iloc[-1]
        
        # Signal 1: RSI Overbought
        if latest['RSI'] > 70:
            st.error(f"**Signal: RSI is Overbought**\nCurrent RSI: **{latest['RSI']:.2f}**. Indicates a potential for a price correction.")
        elif latest['RSI'] < 30:
            st.success(f"**Status: RSI is Oversold**\nCurrent RSI: **{latest['RSI']:.2f}**. Not a bearish signal.")
        else:
            st.info(f"**Status: RSI is Neutral**\nCurrent RSI: **{latest['RSI']:.2f}**.")
        
        # Signal 2: Death Cross (50 SMA vs 200 SMA)
        if latest['SMA50'] < latest['SMA200']:
            st.error(f"**Signal: Death Cross Active**\nSMA50 ({latest['SMA50']:.2f}) is below SMA200 ({latest['SMA200']:.2f}). This is a long-term bearish indicator.")
        else:
            st.info("**Status: No Death Cross**\nSMA50 is above SMA200.")
            
        # Signal 3: Price near Upper Bollinger Band
        if latest['Close'] >= latest['BB_High']:
             st.warning(f"**Signal: Price at Upper Bollinger Band**\nThe price is touching or has crossed the upper band, suggesting it might be overextended.")
        else:
            st.info("**Status: Price within Bollinger Bands**")

        # Signal 4: Detected Chart Patterns
        if bearish_patterns['double_top']:
            st.error(f"**Signal: Double Top Pattern Detected**\nFound {len(bearish_patterns['double_top'])} potential Double Top(s). This is a strong bearish reversal pattern.")
        if bearish_patterns['head_shoulders']:
            st.error(f"**Signal: Head & Shoulders Pattern Detected**\nFound {len(bearish_patterns['head_shoulders'])} potential H&S pattern(s). This is a bearish reversal pattern.")
        
        if not bearish_patterns['double_top'] and not bearish_patterns['head_and_shoulders']:
            st.info("**Status: No Major Bearish Patterns Detected**")

    st.markdown("---") # Add a separator between stocks
