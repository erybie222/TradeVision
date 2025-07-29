import yfinance as yf
import pandas as pd
import os

def download_stock_data(tickers, start_date, end_date, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for ticker in tickers:
        print(f"⬇️  Downloading {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data.to_csv(os.path.join(save_dir, f"{ticker}.csv"))