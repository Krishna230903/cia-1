# stock_analyzer.py
# -*- coding: utf-8 -*-

"""
This Python script performs a bearish technical analysis on Nifty 50,
Adani Enterprises, ITC, and Titan.

For each security, it:
1.  Fetches the last 2 years of historical data using the yfinance library.
2.  Calculates 50-day and 200-day Simple Moving Averages (SMA).
3.  Identifies "Death Cross" events.
4.  Detects bearish chart patterns like "Double Top" and "Head and Shoulders".
5.  Prints a textual summary of all findings.
6.  Generates and saves a detailed chart visualizing all the analysis.
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import argrelextrema

def find_chart_patterns(data, order=10, K=3):
    """
    Detects Double Top, Double Bottom, and Head and Shoulders patterns.
    - 'order': How many points on each side to use for detecting a peak/valley.
    - 'K': The percentage difference allowed between peaks/valleys to be considered similar.
    """
    patterns = {'double_top': [], 'head_shoulders': []}
    
    # Ensure there's enough data
    if len(data) < (order * 2 + 1):
        return patterns

    highs = data['High']
    lows = data['Low']
    
    # Find peaks and valleys
    peak_indices = argrelextrema(highs.values, np.greater, order=order)[0]
    valley_indices = argrelextrema(lows.values, np.less, order=order)[0]
    
    peaks = highs.iloc[peak_indices]
    valleys = lows.iloc[valley_indices]

    # --- Find Double Top (Bearish) ---
    for i in range(len(peaks) - 1):
        p1_idx, p2_idx = peaks.index[i], peaks.index[i+1]
        p1_val, p2_val = peaks.iloc[i], peaks.iloc[i+1]
        
        # Check if peaks are close in price
        if abs(p1_val - p2_val) / p2_val <= K / 100:
            # Find the intervening valley
            intervening_valleys = valleys[(valleys.index > p1_idx) & (valleys.index < p2_idx)]
            if not intervening_valleys.empty:
                patterns['double_top'].append((p1_idx, p2_idx))

    # --- Find Head and Shoulders (Bearish) ---
    for i in range(len(peaks) - 2):
        s1_idx, h_idx, s2_idx = peaks.index[i], peaks.index[i+1], peaks.index[i+2]
        s1_val, h_val, s2_val = peaks.iloc[i], peaks.iloc[i+1], peaks.iloc[i+2]
        
        # Check for Head and Shoulders shape
        if h_val > s1_val and h_val > s2_val and abs(s1_val - s2_val) / s2_val <= (K + 5) / 100:
            # Find the two intervening valleys (neckline points)
            v1 = valleys[(valleys.index > s1_idx) & (valleys.index < h_idx)]
            v2 = valleys[(valleys.index > h_idx) & (valleys.index < s2_idx)]
            if not v1.empty and not v2.empty:
                patterns['head_shoulders'].append((s1_idx, h_idx, s2_idx))
                
    return patterns


def analyze_and_plot_stock(ticker, name):
    """
    Fetches, analyzes, and plots data for a given stock ticker.
    """
    try:
        print(f"--- Analyzing {name} ({ticker}) ---")

        # --- 1. Data Fetching ---
        data = yf.download(ticker, period='2y', auto_adjust=True, progress=False)
        if data.empty:
            print(f"Could not download data for {name}. Skipping.")
            return

        # --- 2. Indicator & Pattern Calculation ---
        data['50_SMA'] = data['Close'].rolling(window=50).mean()
        data['200_SMA'] = data['Close'].rolling(window=200).mean()
        
        data['SMA_Signal'] = (data['50_SMA'] < data['200_SMA']).astype(int)
        data['Death_Cross'] = data['SMA_Signal'].diff()
        death_cross_points = data[data['Death_Cross'] == 1]
        
        # Find chart patterns from the last 9 months of data
        recent_data = data.last('9M')
        chart_patterns = find_chart_patterns(recent_data)
        
        last_data_point = data.iloc[-1]
        last_price = last_data_point['Close']

        # --- 3. Textual Analysis ---
        print("\n[Bearish Analysis Summary]")
        is_bearish = False
        
        # Moving Average Signals
        if not death_cross_points.empty and (data.index[-1] - death_cross_points.index[-1]).days < 180:
             print(f"-> DEATH CROSS ACTIVE: A 'Death Cross' occurred on {death_cross_points.index[-1].date()}. This is a strong long-term bearish signal.")
             is_bearish = True
        
        if last_price < last_data_point['50_SMA']:
            print(f"-> SHORT-TERM WEAKNESS: The current price ({last_price:.2f}) is below the 50-Day SMA ({last_data_point['50_SMA']:.2f}).")
            is_bearish = True
            
        # Chart Pattern Signals
        if chart_patterns['double_top']:
            print(f"-> PATTERN ALERT: A potential 'Double Top' bearish pattern may have formed recently.")
            is_bearish = True
            
        if chart_patterns['head_shoulders']:
            print(f"-> PATTERN ALERT: A potential 'Head and Shoulders' bearish pattern may have formed recently.")
            is_bearish = True

        if not is_bearish:
            print("-> No strong immediate bearish signals detected from moving averages or chart patterns.")

        # Calculate potential downside targets
        target1 = last_price * 0.98
        target2 = last_price * 0.95
        print("\n[Potential 2-Week Downside Targets]")
        print(f"  - Moderate Target (~2% Fall): {target1:.2f}")
        print(f"  - Stronger Correction Target (~5% Fall): {target2:.2f}\n")

        # --- 4. Plotting ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(16, 9))

        ax.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.8, linewidth=1.5)
        ax.plot(data.index, data['50_SMA'], label='50-Day SMA', color='orange', linestyle='--', linewidth=2)
        ax.plot(data.index, data['200_SMA'], label='200-Day SMA', color='purple', linestyle='--', linewidth=2)

        if not death_cross_points.empty:
            ax.plot(death_cross_points.index, death_cross_points['50_SMA'], 'X', color='red', markersize=15, markeredgewidth=3, label='Death Cross')

        # Plot detected patterns
        for p1, p2 in chart_patterns['double_top']:
            ax.plot([p1, p2], [data.loc[p1, 'High'], data.loc[p2, 'High']], 'o-', color='red', markersize=10, label='Double Top Pattern')
        
        for s1, h, s2 in chart_patterns['head_shoulders']:
            ax.plot([s1, h, s2], [data.loc[s1, 'High'], data.loc[h, 'High'], data.loc[s2, 'High']], 'o-', color='magenta', markersize=10, label='Head & Shoulders Pattern')

        ax.axhline(y=target1, color='darkred', linestyle=':', linewidth=2, label=f'Target 1 ({target1:.2f})')
        ax.axhline(y=target2, color='maroon', linestyle=':', linewidth=2, label=f'Target 2 ({target2:.2f})')

        ax.set_title(f'{name} ({ticker}) - Bearish Technical Analysis', fontsize=20, weight='bold')
        ax.set_ylabel('Price (INR)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        
        # Avoid duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        ax.grid(True)

        filename = f"{ticker}_bearish_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved as '{filename}'")
        plt.close(fig)

    except Exception as e:
        print(f"An error occurred while analyzing {name}: {e}")

def main():
    """
    Main function to run the analysis for all specified stocks.
    """
    stocks_to_analyze = {
        '^NSEI': 'Nifty 50',
        'ADANIENT.NS': 'Adani Enterprises',
        'ITC.NS': 'ITC Ltd',
        'TITAN.NS': 'Titan Company'
    }

    if not os.path.exists('stock_charts'):
        os.makedirs('stock_charts')
    os.chdir('stock_charts')

    print("Starting Stock Analysis...\n")
    for ticker, name in stocks_to_analyze.items():
        analyze_and_plot_stock(ticker, name)
        print("-" * 50)
    
    print("\nAnalysis complete. All charts have been saved in the 'stock_charts' folder.")


if __name__ == '__main__':
    # To run this script, you now need to install scipy as well:
    # pip install yfinance pandas matplotlib numpy scipy
    main()
