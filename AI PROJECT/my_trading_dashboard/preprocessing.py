import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling missing values and removing outliers.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Handle Missing Values
    # Forward fill first (appropriate for time series), then backward fill for any remaining start NaNs
    df = df.ffill().bfill()
    # Drop any rows that are still NaN (unlikely but safe)
    df = df.dropna()

    # 2. Remove Outliers using IQR method on 'Close' price
    # Note: In financial data, "outliers" might be real crashes/spikes, 
    # but for general ML stability, extreme anomalies can be capped or removed.
    # Here we will cap them to the 1st and 99th percentile to preserve data points.
    if 'Close' in df.columns:
        lower_bound = df['Close'].quantile(0.01)
        upper_bound = df['Close'].quantile(0.99)
        df['Close'] = df['Close'].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from the 'Date' column.
    """
    df = df.copy()
    
    # Ensure Date is datetime
    if 'Date' not in df.columns:
        # Try to find date in index if not a column
        if isinstance(df.index, pd.DatetimeIndex):
            df['Date'] = df.index
        else:
            raise ValueError("Date column missing and index is not datetime")
            
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature Engineering
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    
    return df

def normalize_data(df: pd.DataFrame, columns: list):
    """
    Normalizes specific numerical columns using StandardScaler.
    Returns the scaler object and the scaled dataframe.
    """
    df_scaled = df.copy()
    scaler = StandardScaler()
    
    if not columns:
        return scaler, df_scaled

    # Fit and transform
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return scaler, df_scaled

def prepare_window_data(df: pd.DataFrame, target_col: str, window_size: int = 60):
    """
    Creates sequence data (X, y) for training using a sliding window.
    X: Window of past `window_size` days
    y: Target value at t (next step)
    """
    data = df[target_col].values
    X, y = [], []
    
    # Ensure enough data
    if len(data) <= window_size:
        raise ValueError(f"Not enough data points ({len(data)}) for window size {window_size}")
        
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
        
    return np.array(X), np.array(y)
