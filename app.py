# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

# --- Page Configuration ---
st.set_page_config(
    page_title="Pro Bearish Signal Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title ---
st.title("🚨 Pro Bearish Signal Dashboard")
st.markdown("An advanced tool to detect potential bearish signals for strategic hedging.")

# --- Constants ---
TARGETS: Dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "Adani Enterprises": "ADANIENT.NS",
    "ITC": "ITC.NS",
    "Titan": "TITAN.NS"
}

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
timeframe_options: Dict[str, int] = {
    "Last 6 Months": 180,
    "Last 1 Year": 365,
    "Last 2 Years": 730,
    "Last 3 Years": 1095
}
selected_timeframe_label: str = st.sidebar.selectbox("Select Analysis Timeframe:", options=list(timeframe_options.keys()))
days_to_subtract: int = timeframe_options[selected_timeframe_label]

END_DATE: datetime = datetime.now()
START_DATE: datetime = END_DATE - timedelta(days=days_to_subtract)

# --- Core Functions ---

@st.cache_data(ttl=1800)
def fetch_data(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetches and cleans historical stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.dropna(inplace=True)
        return data
    except Exception:
        return None

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all necessary technical indicators manually using Pandas.
    This is the alternate to using the 'ta' library.
    """
    # Simple Moving Averages (SMA)
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    # Bollinger Bands
    sma20 = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['BB_High'] = sma20 + (std_dev * 2)

    # Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    # Handle potential division by zero and NaN values
    rs.replace([np.inf, -np.inf], 1e9, inplace=True)
    rs.fillna(1, inplace=True)
    data['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    return data

def find_bearish_patterns(data: pd.DataFrame, order: int = 20, K: int = 3) -> Dict[str, List[Tuple]]:
    """Detects Double Top and Head and Shoulders patterns."""
    patterns: Dict[str, List[Tuple]] = {'double_top': [], 'head_shoulders': []}
    if len(data) < (order * 2 + 1):
        return patterns

    highs = data['High']
    peak_indices = argrelextrema(highs.values, np.greater, order=order)[0]
    peaks = highs.iloc[peak_indices]

    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            p1_idx, p2_idx = peaks.index[i], peaks.index[i+1]
            p1_val, p2_val = peaks.iloc[i], peaks.iloc[i+1]
            if abs(p1_val - p2_val) / p2_val <= K / 100:
                patterns['double_top'].append((p1_idx, p2_idx))

    if len(peaks) > 2:
        for i in range(len(peaks) - 2):
            s1_idx, h_idx, s2_idx = peaks.index[i], peaks.index[i+1], peaks.index[i+2]
            s1, h, s2 = peaks.iloc[i], peaks.iloc[i+1], peaks.iloc[i+2]
            if h > s1 and h > s2 and abs(s1 - s2) / s2 <= (K + 5) / 100:
                patterns['head_shoulders'].append((s1_idx, h_idx, s2_idx))
    return patterns

def create_price_chart(data: pd.DataFrame, patterns: Dict[str, List[Tuple]]) -> go.Figure:
    """Creates the main Plotly candlestick chart with overlays."""
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Price'
    )])

    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], name='SMA 200', line=dict(color='red', width=1.5)))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name='Upper Bollinger Band', line=dict(color='purple', dash='dash', width=1)))

    for dt in patterns['double_top']:
        fig.add_trace(go.Scatter(x=dt, y=data['High'].loc[list(dt)], mode='markers', marker=dict(symbol='triangle-down', color='red', size=15), name='Double Top'))
    for hs in patterns['head_shoulders']:
        fig.add_trace(go.Scatter(x=hs, y=data['High'].loc[list(hs)], mode='markers', marker=dict(symbol='diamond', color='purple', size=15), name='Head & Shoulders'))

    fig.update_layout(height=450, xaxis_rangeslider_visible=False, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_rsi_chart(data: pd.DataFrame) -> go.Figure:
    """Creates a dedicated RSI chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="magenta")))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.update_layout(title="Relative Strength Index (RSI)", height=250, yaxis_title="RSI Value", yaxis_range=[0,100])
    return fig

def display_summary(data: pd.DataFrame, patterns: Dict[str, List[Tuple]]):
    """Displays the color-coded summary of bearish signals."""
    st.subheader("Signal Summary")
    latest = data.iloc[-1]
    signals_found = False

    if latest['RSI'] > 70:
        st.error(f"🚨 **Signal: RSI Overbought** — Current RSI is **{latest['RSI']:.2f}**. This indicates a high probability of a price correction.")
        signals_found = True
    
    if latest['SMA50'] < latest['SMA200']:
        st.error(f"🚨 **Signal: Death Cross Active** — The 50-day SMA ({latest['SMA50']:.2f}) is below the 200-day SMA ({latest['SMA200']:.2f}). This is a strong long-term bearish signal.")
        signals_found = True

    if latest['Close'] >= latest['BB_High']:
         st.warning(f"⚠️ **Warning: Price at Upper Bollinger Band** — The price is overextended, which can precede a pullback.")
         signals_found = True

    if patterns['double_top']:
        st.error(f"🚨 **Signal: Double Top Pattern Detected** — This is a classic bearish reversal pattern.")
        signals_found = True
    
    if patterns['head_shoulders']:
        st.error(f"🚨 **Signal: Head & Shoulders Pattern Detected** — This is a strong bearish reversal pattern.")
        signals_found = True

    if not signals_found:
        st.success("✅ **All Clear:** No immediate, strong bearish signals were detected based on the selected criteria.")


# --- Main Application Layout ---

for name, ticker in TARGETS.items():
    with st.expander(f"▶️ View Analysis for: {name} ({ticker})", expanded=(name == "NIFTY 50")):
        try:
            stock_data = fetch_data(ticker, START_DATE, END_DATE)
            if stock_data is None:
                st.error(f"Could not retrieve or process data for {name}. The ticker might be invalid or there might be no data for the selected period.")
                continue

            stock_data = calculate_indicators(stock_data)
            bearish_patterns = find_bearish_patterns(stock_data)

            chart_col, summary_col = st.columns([2, 1])

            with chart_col:
                st.plotly_chart(create_price_chart(stock_data, bearish_patterns), use_container_width=True)
                st.plotly_chart(create_rsi_chart(stock_data), use_container_width=True)

            with summary_col:
                display_summary(stock_data, bearish_patterns)

        except Exception as e:
            st.error(f"An unexpected error occurred while analyzing {name}: {e}")
