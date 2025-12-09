import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta
import base64

# Custom Modules
from preprocessing import clean_data, add_time_features, normalize_data, prepare_window_data
from model_engine import ModelEngine

# --- Page Config ---
st.set_page_config(
    page_title="AI Trading Assistant",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E88E5; text-align: center; margin-bottom: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #1E88E5; color: white;}
    .success-text {color: #2e7d32; font-weight: bold;}
    .error-text {color: #d32f2f; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def load_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, interval="1d")
        # Fix: Check if df is None first, then check if empty
        if df is None:
            return None
        if df.empty:
            return None
        df = df.reset_index()
        # Handle MultiIndex columns if present (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[0] != 'Date' and col[0] != 'Adj Close' else col[0] for col in df.columns]
            # If Date was index, it might be named 'Date' now
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def download_csv(df, filename="forecast.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="stButton">📥 Download Forecast CSV</a>'
    return href

# --- Main Interface ---
st.markdown('<div class="main-header">🚀 AI Trading Assistant & Forecaster</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data Source Selection
    source_option = st.radio("Data Source", ["Live Market Data", "Upload CSV"])
    
    df = None
    ticker_name = "Asset"
    
    if source_option == "Live Market Data":
        assets = {
            "Bitcoin (USD)": "BTC-USD", 
            "Ethereum (USD)": "ETH-USD", 
            "Gold": "GC=F", 
            "S&P 500": "^GSPC", 
            "Nifty 50": "^NSEI",
            "Tesla": "TSLA",
            "Apple": "AAPL"
        }
        selected_asset = st.selectbox("Select Asset", list(assets.keys()))
        ticker = assets[selected_asset]
        ticker_name = selected_asset
        
        if st.button("Fetch Data", key="fetch_btn"):
            with st.spinner(f"Fetching {ticker_name} data..."):
                df = load_data(ticker)
                if df is not None:
                    st.session_state['data'] = df
                    st.session_state['name'] = ticker_name
    
    else: # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV (Date, Close...)", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['data'] = df
                st.session_state['name'] = "Uploaded File"
                ticker_name = "Uploaded File"
            except Exception as e:
                st.error(f"Invalid CSV: {e}")

    # Settings
    st.markdown("---")
    st.subheader("Model Settings")
    forecast_days = st.slider("Forecast Horizon (Days)", 5, 30, 7)
    window_size = st.slider("Lookback Window (Days)", 10, 90, 60)
    
    run_analysis = st.button("🚀 Train & Forecast", type="primary")

# --- Main Area Logic ---
if 'data' in st.session_state:
    df = st.session_state['data']
    name = st.session_state.get('name', 'Asset')

    # 1. Data Preview & Checking
    if df is not None:
        try:
            # Basic cleaning & standardization
            df = clean_data(df)
            df = add_time_features(df)
            
            # Info Cards
            latest_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            change = latest_price - prev_price
            pct_change = (change / prev_price) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Price", f"${latest_price:,.2f}")
            col2.metric("24h Change", f"{change:+.2f}", f"{pct_change:+.2f}%")
            col3.metric("Data Points", len(df))
            
            # Current Chart
            fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'], name=name)])
            fig.update_layout(title=f"{name} Price History", template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Data processing error: {e}. Please ensure CSV has 'Date', 'Open', 'High', 'Low', 'Close'.")
            st.stop()
            
    # 2. Training Pipeline
    if run_analysis:
        if len(df) < window_size + 20:
            st.error(f"Insufficient data. Need at least {window_size + 20} rows.")
        else:
            with st.spinner("🤖 Training Multi-Model Pipeline (Linear Regression, Random Forest, GBM)..."):
                try:
                    # A. Preprocessing for ML
                    # Normalize Target
                    scaler, df_scaled = normalize_data(df, ['Close'])
                    
                    # Prepare Sequences
                    X, y = prepare_window_data(df_scaled, 'Close', window_size=window_size)
                    
                    # Train/Test Split (80/20)
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # B. Model Training
                    engine = ModelEngine()
                    metrics = engine.train_and_evaluate(X_train, y_train, X_test, y_test)
                    
                    # Display Leaderboard
                    st.markdown("### 🏆 Model Leaderboard")
                    results_df = pd.DataFrame(metrics).T.sort_values(by="R2", ascending=False)
                    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
                    
                    best_model, best_name, best_metrics = engine.get_best_model_info()
                    if best_model is not None and best_name is not None and best_metrics is not None:
                        st.success(f"**Best Performing Model:** {best_name} (R²: {best_metrics['R2']})")
                    else:
                        st.error("No valid models were trained successfully.")
                        st.stop()
                    
                    # C. Forecasting
                    last_window = df_scaled['Close'].values[-window_size:]
                    future_scaled = engine.forecast_future(best_model, last_window, steps=forecast_days)
                    
                    # Inverse Transform
                    future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
                    
                    # Dates
                    last_date = df['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                    
                    # D. Visualization
                    st.markdown("### 🔮 Future Forecast")
                    
                    forecast_df = pd.DataFrame({
                        "Date": future_dates,
                        "Predicted Close": future_prices
                    })
                    
                    fig_forecast = go.Figure()
                    # Historical (Last 90 days)
                    fig_forecast.add_trace(go.Scatter(x=df['Date'].tail(90), y=df['Close'].tail(90),
                                                    mode='lines', name='Historical Data',
                                                    line=dict(color='gray', width=2)))
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_prices,
                                                    mode='lines+markers', name=f'Forecast ({best_name})',
                                                    line=dict(color='#1E88E5', width=3, dash='dot')))
                    
                    fig_forecast.update_layout(title=f"{forecast_days}-Day Price Prediction", 
                                             template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Results Table
                    col_res1, col_res2 = st.columns([2, 1])
                    with col_res1:
                        st.dataframe(forecast_df, use_container_width=True)
                    with col_res2:
                        st.markdown(download_csv(forecast_df), unsafe_allow_html=True)
                        st.info("Disclaimer: AI predictions are for educational purposes only. Do not trade based solely on these numbers.")
                        
                except Exception as e:
                    st.error(f"An error occurred during training: {str(e)}")
                    st.exception(e)

else:
    st.info("👈 Please select a data source from the sidebar to begin.")
    
# Footer
st.markdown("---")
st.markdown("<center style='color: #888;'>Built with Python, Streamlit & Scikit-Learn</center>", unsafe_allow_html=True)
