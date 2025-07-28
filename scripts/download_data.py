import yfinance as yf
import pandas as pd
import os

def download_stock_data(tickers, start_date, end_date, save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(f"{save_dir}/{ticker}.csv")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    download_stock_data(tickers, start_date="2015-01-01", end_date="2024-12-31")