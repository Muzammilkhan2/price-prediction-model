# AI Trading Assistant & Forecaster

A comprehensive Machine Learning dashboard for stock price prediction and forecasting.

## Features
- **Multi-Model Pipeline**: Trains Linear Regression, Random Forest, and Gradient Boosting models automatically.
- **Smart Forecasting**: basic time-series windowing to predict future prices (5-30 days).
- **Interactive UI**: Built with Streamlit and Plotly for professional grade visualizations.
- **Data Support**: Live data from Yahoo Finance or custom CSV uploads.

## Installation

1. **Clone the repository** (if applicable) or navigate to the folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application using:
```bash
streamlit run app.py
```
OR use the helper script:
```bash
./run.sh
```

## Structure
- `app.py`: Main application UI and logic integration.
- `model_engine.py`: Machine Learning pipeline handles training and forecasting.
- `preprocessing.py`: Data cleaning and feature engineering.
- `requirements.txt`: Python dependencies.

## Disclaimer
This tool is for educational purposes only. Do not use it for financial trading decisions.
