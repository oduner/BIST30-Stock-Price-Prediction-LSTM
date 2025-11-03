import pandas as pd
import numpy as np
import os
from glob import glob
from typing import Tuple
import ta
from config_module import logger, CONFIG


def load_and_preprocess_all():
    company_index = CONFIG.get("company_index", 0)
    
    bist_files = sorted(glob("yDatas/Bist/*.csv"))
    test_files = sorted(glob("yDatas/Test/*.csv"))
    
    if not bist_files or not test_files:
        logger.error("Bist and Test files not containing .csv files!")
        raise FileNotFoundError("Bist and Test files not containing .csv files!")
    
    if company_index >= len(bist_files) or company_index >= len(test_files):
        logger.error(f"Index {company_index} is out of range. Bist: {len(bist_files)}, Test: {len(test_files)}")
        raise IndexError(f"Company index {company_index} out of range")
    
    company_code = os.path.splitext(os.path.basename(bist_files[company_index]))[0]
    bist_file = bist_files[company_index]
    test_file = test_files[company_index]
    
    logger.info(f"Processing company: {company_code}")
    logger.info(f"Bist file: {bist_file}")
    logger.info(f"Test file: {test_file}")
    
    try:
        df_train = pd.read_csv(bist_file)
        df_train["Date"] = pd.to_datetime(df_train["Date"], errors='coerce', format='mixed')
        df_train = df_train.dropna(subset=["Date"])
        
        df_test = pd.read_csv(test_file)
        df_test["Date"] = pd.to_datetime(df_test["Date"], errors='coerce', format='mixed')
        df_test = df_test.dropna(subset=["Date"])
        
        if df_train.empty or df_test.empty:
            logger.error(f"DataFrames are empty after date filtering for {company_code}")
            return pd.DataFrame(), pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing files for {company_code}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    df_train = _preprocess_dataframe(df_train, f"{company_code}_train")
    df_test = _preprocess_dataframe(df_test, f"{company_code}_test")
    
    if df_train.empty or df_test.empty:
        logger.error(f"DataFrames are empty after preprocessing for {company_code}")
        return pd.DataFrame(), pd.DataFrame()
    
    return df_train, df_test


def _add_technical_indicators(df: pd.DataFrame, name: str) -> pd.DataFrame:
    length = len(df)
    
    if length >= 20:
        df["MA_20"] = df["Close"].rolling(20).mean()
    else:
        logger.warning(f"DataFrame {name} length ({length}) < 20 for MA_20")
        df["MA_20"] = df["Close"]
    
    if length >= 14:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    else:
        logger.warning(f"DataFrame {name} length ({length}) < 14 for RSI")
        df["RSI"] = 50.0
    
    if length >= 5:
        df["Volatility"] = df["Close"].pct_change(fill_method=None).rolling(5).std()
    else:
        logger.warning(f"DataFrame {name} length ({length}) < 5 for Volatility")
        df["Volatility"] = 0.01
    
    return df


def _preprocess_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.sort_values(by="Date")
    df = df.drop_duplicates(subset='Date', keep='last')
    
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='B')
    df = df.set_index('Date').reindex(date_range).rename_axis('Date').reset_index()
    
    #df = df.dropna().reset_index(drop=True)
    
    df = _add_technical_indicators(df, name)
    
    if len(df) < 66:
        df = df.ffill().bfill()
    else:
        df = df.dropna()
    
    df = df.reset_index(drop=True)
    
    if df.empty:
        logger.error(f"DataFrame {name} is empty after feature engineering")
    
    return df


def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)