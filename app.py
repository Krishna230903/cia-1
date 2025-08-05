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
    page_title="Strategic Hedging Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a beautiful UI ---
st.markdown("""
<style>
    .stApp { background-color: #F0F2F6; }
    .st-emotion-cache-1r4qj8v { border-radius: 0.75rem; padding: 20px !important; }
    .st-emotion-cache-1n2qlj { border-radius: 0.5rem; padding: 10px; background-color: #FFFFFF; border: 1px solid #E0E0E0; }
    .st-emotion-cache-12fmjuu { border-radius: 0.5rem; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# --- App Title ---
st.title("🛡️ Strategic Hedging Dashboard")
st.markdown("An advanced tool to identify bearish sentiment and key price levels for hedging decisions.")

# --- Constants ---
TARGETS: Dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "Adani Enterprises": "ADANIENT.NS",
    "ITC": "ITC.NS",
    "Titan": "TITAN.NS"
}

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
timeframe_options: Dict[str, int] = { "Last 6 Months": 180, "Last 1 Year": 365, "Last 2 Years": 730 }
selected_timeframe_label: str = st.sidebar.selectbox("Select Analysis Timeframe:", options=list(timeframe_options.keys()), index=1)
days_to_subtract: int = timeframe_options[selected_timeframe_label]
END_DATE: datetime = datetime.now()
START_DATE: datetime = END_DATE - timedelta(days=days_to_subtract)

# --- Core Functions ---

@st.cache_data(ttl=1800)
def fetch_data(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Fetches and cleans historical stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty: return None
        # Robustness check for MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return None

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates all necessary technical indicators."""
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    sma20 = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['BB_High'] = sma20 + (std_dev * 2)
    
    # --- Robust RSI Calculation ---
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    # Replace inf with NaN, then fill NaN with 0 (neutral) to prevent errors
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    rs.fillna(0, inplace=True)
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def find_patterns_and_levels(data: pd.DataFrame, order: int = 20) -> Tuple[Dict[str, List], List[float]]:
    """Detects patterns and identifies key support levels."""
    patterns: Dict[str, List] = {'double_top': [], 'head_shoulders': []}
    highs = data['High']
    lows = data['Low']
    peak_indices = argrelextrema(highs.values, np.greater, order=order)[0]
    peaks = highs.iloc[peak_indices]
    
    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            p1_val, p2_val = peaks.iloc[i], peaks.iloc[i+1]
            if abs(p1_val - p2_val) / p2_val <= 0.03:
                patterns['double_top'].append((peaks.index[i], peaks.index[i+1]))
    if len(peaks) > 2:
        for i in range(len(peaks) - 2):
            s1, h, s2 = peaks.iloc[i], peaks.iloc[i+1], peaks.iloc[i+2]
            if h > s1 and h > s2 and abs(s1 - s2) / s2 <= 0.05:
                patterns['head_shoulders'].append((peaks.index[i], peaks.index[i+1], peaks.index[i+2]))

    valley_indices = argrelextrema(lows.values, np.less, order=order)[0]
    recent_lows = lows.iloc[valley_indices]
    support_levels = recent_lows[recent_lows < data['Close'].iloc[-1]].tail(2).tolist()
    return patterns, sorted(support_levels, reverse=True)

def calculate_bearish_score(data: pd.DataFrame, patterns: Dict[str, List]) -> Tuple[int, List[str]]:
    """Calculates a weighted score based on active bearish signals with NaN checks."""
    score = 0
    reasons = []
    latest = data.iloc[-1]
    
    # --- Defensive checks for NaN values before comparing ---
    if pd.notna(latest['SMA50']) and pd.notna(latest['SMA200']) and latest['SMA50'] < latest['SMA200']:
        score += 3; reasons.append("Death Cross (SMA50 < SMA200)")
    
    if pd.notna(latest['RSI']) and latest['RSI'] > 70:
        score += 2; reasons.append(f"RSI is Overbought ({latest['RSI']:.1f})")

    if pd.notna(latest['Close']) and pd.notna(latest['BB_High']) and latest['Close'] >= latest['BB_High']:
        score += 1; reasons.append("Price at Upper Bollinger Band")

    if pd.notna(latest['Close']) and pd.notna(latest['SMA50']) and pd.notna(latest['SMA200']) and latest['Close'] < latest['SMA50'] and latest['Close'] < latest['SMA200']:
        score += 1; reasons.append("Price below key moving averages")

    if patterns['double_top']:
        score += 3; reasons.append("Double Top Pattern Detected")
    if patterns['head_shoulders']:
        score += 3; reasons.append("Head & Shoulders Pattern Detected")

    return min(score, 10), reasons

def create_gauge(score: int) -> go.Figure:
    """Creates a visual gauge for the bearish confidence score."""
    # This function remains the same
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Bearish Confidence", 'font': {'size': 20}},
        number={'font': {'size': 48}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "#E0E0E0",
            'steps': [ {'range': [0, 3], 'color': 'lightgreen'}, {'range': [3, 7], 'color': 'lightyellow'}, {'range': [7, 10], 'color': 'lightcoral'}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}
        }))
    fig.update_layout(height=250, margin={'t':0, 'b':0, 'l':0, 'r':0})
    return fig

def create_main_chart(data: pd.DataFrame, patterns: Dict, supports: List[float]) -> go.Figure:
    """Creates the main candlestick chart with all overlays."""
    # This function remains the same
    fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price')])
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='SMA 50', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], name='SMA 200', line=dict(color='red', width=1.5)))
    for s_level in supports:
        fig.add_hline(y=s_level, line_dash="dash", line_color="green", annotation_text=f"Support at {s_level:.2f}")
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- Main Application Layout ---
tab_list = st.tabs([f"**{name}**" for name in TARGETS.keys()])
for i, (name, ticker) in enumerate(TARGETS.items()):
    with tab_list[i]:
        stock_data = fetch_data(ticker, START_DATE, END_DATE)
        if stock_data is None or stock_data.empty:
            st.error("Could not retrieve stock data. Please check the ticker or try again later.")
            continue
        stock_data = calculate_indicators(stock_data)
        if stock_data.empty:
            st.warning("Not enough data to perform analysis for the selected period.")
            continue
        patterns, support_levels = find_patterns_and_levels(stock_data)
        bearish_score, reasons = calculate_bearish_score(stock_data, patterns)
        summary_col, chart_col = st.columns([1, 2])
        with summary_col:
            st.subheader("Hedging Analysis")
            st.plotly_chart(create_gauge(bearish_score), use_container_width=True)
            st.markdown("##### Active Bearish Signals:")
            if reasons:
                for reason in reasons: st.markdown(f"- 📉 {reason}")
            else:
                st.markdown("- ✅ No active bearish signals found.")
            st.markdown("---")
            st.markdown("##### Key Support Levels:")
            if support_levels:
                st.metric(label="Primary Support Level", value=f"{support_levels[0]:.2f}")
                if len(support_levels) > 1: st.metric(label="Secondary Support Level", value=f"{support_levels[1]:.2f}")
                st.info("A sustained break below these levels could signal further downside.")
            else:
                st.markdown("- No clear support levels found.")
        with chart_col:
            st.plotly_chart(create_main_chart(stock_data, patterns, support_levels), use_container_width=True)
