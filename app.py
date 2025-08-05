import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Stock Technical Analysis"

# --- Stock Tickers and Time Period ---
tickers = {
    "Nifty 50": "^NSEI",
    "Adani Enterprises": "ADANIENT.NS",
    "ITC": "ITC.NS",
    "Titan": "TITAN.NS",
}
period = "1y" # Use one year of data for analysis

# --- Function to Fetch and Analyze Stock Data ---
def analyze_stock(ticker_symbol, stock_name):
    """
    Fetches stock data, performs technical analysis, and generates a plot.
    """
    # Fetch historical data
    stock_data = yf.download(ticker_symbol, period=period, progress=False)

    if stock_data.empty:
        return html.Div(f"Could not retrieve data for {stock_name}. Please check the ticker symbol.")

    # --- Technical Analysis ---
    # Moving Averages (MA)
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = exp1 - exp2
    stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    # --- Price Prediction (Simple Linear Regression for next 14 days) ---
    df_pred = stock_data[['Close']].copy()
    df_pred['Date'] = df_pred.index
    df_pred['Date'] = df_pred['Date'].map(pd.to_datetime)
    df_pred['Date_ordinal'] = df_pred['Date'].map(pd.Timestamp.toordinal)

    # Use last 60 days for trend
    X = df_pred['Date_ordinal'][-60:].values.reshape(-1, 1)
    y = df_pred['Close'][-60:].values

    model = LinearRegression()
    model.fit(X, y)

    # Predict next 14 days
    last_date = df_pred['Date_ordinal'].iloc[-1]
    future_dates_ordinal = np.array([last_date + i for i in range(1, 15)]).reshape(-1, 1)
    predicted_prices = model.predict(future_dates_ordinal)
    
    future_dates = [pd.to_datetime(pd.Timestamp.fromordinal(int(i))) for i in future_dates_ordinal.flatten()]

    # --- Create the Plot ---
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{stock_name} Candlestick Chart", "RSI", "MACD"),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Candlestick chart with Moving Averages
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Candlestick"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='50-Day MA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='200-Day MA', line=dict(color='purple')), row=1, col=1)

    # Add predicted prices to the main chart
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Prediction (14 days)', line=dict(color='cyan', dash='dot')), row=1, col=1)


    # RSI
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal_Line'], mode='lines', name='Signal Line'), row=3, col=1)

    fig.update_layout(
        height=700,
        title_text=f"Technical Analysis: {stock_name}",
        legend_title="Indicators",
        xaxis_rangeslider_visible=False,
    )

    # --- Bearish Analysis Text ---
    last_close = stock_data['Close'].iloc[-1]
    ma50 = stock_data['MA50'].iloc[-1]
    ma200 = stock_data['MA200'].iloc[-1]
    rsi = stock_data['RSI'].iloc[-1]
    macd = stock_data['MACD'].iloc[-1]
    signal = stock_data['Signal_Line'].iloc[-1]
    
    analysis_text = f"""
    ### Bearish Scenario Analysis for {stock_name}

    * **Moving Averages:** The 50-day MA ({ma50:.2f}) is currently {'above' if ma50 > ma200 else 'below'} the 200-day MA ({ma200:.2f}). A "death cross" (50-day MA crossing below 200-day MA) is a strong bearish signal. The current price ({last_close:.2f}) is trading {'above' if last_close > ma50 else 'below'} its 50-day moving average, suggesting short-term weakness.
    * **RSI:** The current RSI is {rsi:.2f}. An RSI above 70 is often considered overbought, suggesting a potential pullback. While not in extreme territory, any move towards 70 could indicate building selling pressure.
    * **MACD:** The MACD line ({macd:.2f}) is currently {'above' if macd > signal else 'below'} the Signal line ({signal:.2f}). A crossover where the MACD line goes below the Signal line is a bearish indicator.
    * **Prediction:** Based on a linear regression of the last 60 days, the predicted price for the next two weeks shows a potential continuation of the recent trend. The predicted price in 14 days is **{predicted_prices[-1]:.2f}**.

    **Disclaimer:** This is a simplified analysis and not financial advice.
    """

    return html.Div([
        dcc.Graph(figure=fig),
        dcc.Markdown(analysis_text, style={'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '5px', 'margin-top': '20px'})
    ])

# --- App Layout ---
app.layout = html.Div([
    html.H1("Stock Market Technical Analysis (Bearish Scenario)", style={'textAlign': 'center'}),
    html.P("This dashboard displays a technical analysis of selected stocks with a focus on identifying potential bearish trends. All data is fetched in real-time.", style={'textAlign': 'center'}),
    html.Hr(),
    # Generate a section for each stock
    *[html.Div([
        html.H2(stock_name, style={'textAlign': 'center'}),
        analyze_stock(ticker, stock_name)
    ], style={'padding': '20px'}) for stock_name, ticker in tickers.items()]
])

# --- Run the App ---
if __name__ == '__main__':
    app.run_server(debug=True)
